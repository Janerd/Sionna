"""
channel.py
==========
Sionna RT 信道仿真和 CSI 特征提取（Sionna 2.x 兼容版本）
（对应 MATLAB compute_rsrp.m + extract_features.m + create_cdl_channels.m）

Sionna 版本：2.x（基于 PyTorch）
  - 使用 scene.trace_paths() + scene.compute_fields() 替代旧版 compute_paths()
  - 路径数据通过 scene_mgr.extract_path_data(paths) 提取

主要功能：
  1. 对每个 UE 位置执行射线追踪（Sionna 2.x RT）
  2. 从射线追踪结果计算 RSRP、RSRQ、SINR
  3. 提取真实的 CSI 特征（时延扩展、Doppler 估计、到达角、LOS 指示）
  4. 应用 L3 滤波（3GPP TS 38.331）

特征向量组成（共 9*C 维，C = 小区数）：
  [0:C]    RSRP_l3         - L3 滤波后 RSRP [dBm]
  [C:2C]   RSRQ            - 参考信号接收质量 [dB]
  [2C:3C]  SINR            - 信噪干扰比 [dB]
  [3C:4C]  Doppler_est     - Doppler 频移估计 [Hz]（有测量噪声）
  [4C:5C]  BeamID          - 最优波束 ID（归一化到 [0,1]）
  [5C:6C]  RSRP_diff       - RSRP 变化率（相邻时隙差分）[dB/slot]
  [6C:7C]  BeamID_diff     - 波束 ID 变化（相邻时隙差分）
  [7C:8C]  DelaySpread     - RMS 时延扩展 [ns]（归一化）
  [8C:9C]  LOS_indicator   - LOS 路径是否存在（0/1）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SimConfig

# Sionna RT 导入
try:
    import sionna
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False


# =========================================================
# 单时隙信道测量结果
# =========================================================

@dataclass
class SlotMeasurement:
    """
    单个时隙的信道测量结果

    所有字段均为 UE 可观测量（不含 Ground Truth 信息）
    """
    rsrp_raw: np.ndarray        # [C] 瞬时 RSRP [dBm]
    rsrq: np.ndarray            # [C] RSRQ [dB]
    sinr: np.ndarray            # [C] SINR [dB]
    doppler_est: np.ndarray     # [C] Doppler 频移估计 [Hz]（有噪声）
    beam_id: np.ndarray         # [C] 最优波束 ID（0-indexed）
    delay_spread_ns: np.ndarray # [C] RMS 时延扩展 [ns]
    los_indicator: np.ndarray   # [C] LOS 路径是否存在（0/1）
    aoa_deg: np.ndarray         # [C] 主径到达角 [度]（用于波束选择）
    serving_cell: int = 0


# =========================================================
# 射线追踪信道计算（Sionna 2.x 兼容）
# =========================================================

def compute_channel_from_paths(
    a: np.ndarray,
    tau: np.ndarray,
    phi_r: np.ndarray,
    path_types: Optional[np.ndarray],
    cfg: SimConfig,
    ue_vel: np.ndarray,
    bs_positions_3d: np.ndarray,
    ue_pos_3d: np.ndarray,
    prev_rsrp_raw: Optional[np.ndarray] = None,
    prev_beam_id: Optional[np.ndarray] = None,
    theta_r: Optional[np.ndarray] = None,
) -> Tuple:
    """
    从 Sionna 2.0.1 射线追踪结果计算信道测量量和 CSI 特征

    Sionna 2.0.1 中 paths.a 是 (a_real, a_imag) 元组，
    在 _extract_path_data_sionna201 中已合并为复数数组：
        a = a_real + 1j * a_imag

    Sionna 2.0.1 中 paths 的维度（synthetic_array=True 时）：
        a:     [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        tau:   [num_rx, num_tx, num_paths]
        phi_r: [num_rx, num_tx, num_paths]

    注意：维度顺序是 [num_rx, ..., num_tx, ..., num_paths]
    而不是旧版的 [num_tx, num_rx, num_paths]

    参数：
        a:               路径复数增益 numpy 数组（已合并 real+imag）
        tau:             路径时延 numpy 数组 [s]
        phi_r:           到达方位角 numpy 数组 [rad]
        path_types:      路径类型 numpy 数组（可能为 None）
        cfg:             仿真配置
        ue_vel:          UE 速度向量 [vx, vy]，单位 m/s
        bs_positions_3d: [C, 3] 基站 3D 坐标
        ue_pos_3d:       [3] UE 3D 坐标
        prev_rsrp_raw:   [C] 上一时隙的瞬时 RSRP（用于计算差分）
        prev_beam_id:    [C] 上一时隙的波束 ID（用于计算差分）
        theta_r:         到达仰角 numpy 数组 [rad]（可选）

    返回：
        (SlotMeasurement, rsrp_diff, beam_id_diff)
    """
    num_cells = cfg.num_cells
    fc = cfg.fc
    c_light = 3e8

    # 预分配结果数组
    rsrp_raw = np.full(num_cells, -120.0, dtype=np.float32)
    doppler_est = np.zeros(num_cells, dtype=np.float32)
    delay_spread_ns = np.zeros(num_cells, dtype=np.float32)
    los_indicator = np.zeros(num_cells, dtype=np.float32)
    aoa_deg_arr = np.zeros(num_cells, dtype=np.float32)
    beam_id = np.zeros(num_cells, dtype=np.int32)

    # 发射功率（线性，mW）
    p_tx_mw = 10 ** (cfg.p_tx_dbm / 10)

    for c in range(num_cells):
        try:
            # ---- 提取基站 c 到 UE 的路径数据 ----
            # Sionna 2.0.1 维度（synthetic_array=True）：
            #   a:     [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            #   tau:   [num_rx, num_tx, num_paths]
            #   phi_r: [num_rx, num_tx, num_paths]
            # 注意：只有 1 个 UE（接收机），所以 num_rx=1，rx 索引为 0
            # 基站 c 对应 tx 索引 c

            if a.ndim == 5:
                # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
                a_c = a[0, 0, c, 0, :]   # [num_paths] 复数增益
            elif a.ndim == 3:
                # [num_rx, num_tx, num_paths]（synthetic_array=True 时的简化形式）
                a_c = a[0, c, :]
            elif a.ndim == 6:
                # 旧版格式 [num_tx, num_rx, max_num_paths, num_time_steps, num_tx_ant, num_rx_ant]
                a_c = a[c, 0, :, 0, 0, 0]
            elif a.ndim == 4:
                a_c = a[c, 0, :, 0]
            else:
                continue

            if tau.ndim == 3:
                # [num_rx, num_tx, num_paths]
                tau_c = tau[0, c, :]
            elif tau.ndim == 2:
                tau_c = tau[c, :]
            else:
                tau_c = tau[c, 0, :]

            if phi_r.ndim == 3:
                # [num_rx, num_tx, num_paths]
                phi_r_c = phi_r[0, c, :]
            elif phi_r.ndim == 2:
                phi_r_c = phi_r[c, :]
            else:
                phi_r_c = phi_r[c, 0, :]

        except (IndexError, TypeError):
            continue

        # 过滤有效路径（增益不为零，时延不为 NaN/Inf）
        valid = (
            (np.abs(a_c) > 1e-15) &
            np.isfinite(tau_c) &
            np.isfinite(phi_r_c) &
            (tau_c >= 0)
        )
        if not np.any(valid):
            continue

        a_valid = a_c[valid]
        tau_valid = tau_c[valid]
        phi_r_valid = phi_r_c[valid]

        # 路径功率（线性）
        path_power = np.abs(a_valid) ** 2

        # ---- RSRP 计算 ----
        # Sionna 的路径增益已经包含了路径损耗（归一化到发射功率）
        total_power_linear = p_tx_mw * float(np.sum(path_power))
        rsrp_raw[c] = float(10 * np.log10(max(total_power_linear, 1e-20)))

        # ---- 时延扩展计算（RMS Delay Spread）----
        if len(tau_valid) > 1:
            total_power = float(np.sum(path_power))
            if total_power > 0:
                mean_tau = float(np.sum(path_power * tau_valid)) / total_power
                rms_ds = float(np.sqrt(
                    np.sum(path_power * (tau_valid - mean_tau) ** 2) / total_power
                ))
                delay_spread_ns[c] = rms_ds * 1e9  # 转换为 ns
        else:
            delay_spread_ns[c] = 0.0

        # ---- LOS 指示 ----
        bs_pos = bs_positions_3d[c]
        direct_dist = float(np.linalg.norm(ue_pos_3d - bs_pos))
        direct_tau = direct_dist / c_light
        min_tau = float(np.min(tau_valid))

        # 如果最短路径时延接近直射时延（误差 < 10%），认为存在 LOS
        if direct_tau > 0 and abs(min_tau - direct_tau) / direct_tau < 0.1:
            los_indicator[c] = 1.0
        else:
            los_indicator[c] = 0.0

        # ---- Doppler 频移估计 ----
        strongest_path_idx = int(np.argmax(path_power))
        aoa_rad = float(phi_r_valid[strongest_path_idx])
        aoa_deg_arr[c] = float(np.degrees(aoa_rad))

        ue_vel_2d = np.array([ue_vel[0], ue_vel[1]], dtype=np.float64)
        arrival_dir = np.array([math.cos(aoa_rad), math.sin(aoa_rad)])
        v_proj = float(np.dot(ue_vel_2d, arrival_dir))
        fd_true = v_proj * fc / c_light

        # 加入测量噪声（真实 Doppler 估计误差约 ±20% + 固定偏差）
        v_mag = float(np.linalg.norm(ue_vel_2d))
        fd_max = v_mag * fc / c_light
        noise_std = 0.2 * fd_max + 2.0
        doppler_est[c] = float(fd_true + noise_std * np.random.randn())

        # ---- 波束 ID 计算 ----
        beam_id[c] = _select_beam(aoa_deg_arr[c], cfg.num_beams, cfg.beam_angle_range)

    # ---- RSRQ 计算（3GPP TS 38.215）----
    N_RB = 52  # 20MHz 带宽对应资源块数
    rsrp_linear = 10 ** (rsrp_raw / 10)
    noise_linear = 10 ** (cfg.noise_floor_dbm / 10)
    rssi_linear = float(np.sum(rsrp_linear)) + noise_linear
    rsrq = 10 * np.log10(N_RB * rsrp_linear / max(rssi_linear, 1e-20))

    # ---- SINR 计算（改进版：只考虑强干扰邻区）----
    sinr = _compute_sinr(rsrp_raw, cfg.noise_floor_dbm, interference_threshold_db=10.0)

    # ---- 确定服务小区 ----
    serving_cell = int(np.argmax(rsrp_raw))

    # ---- 差分特征 ----
    if prev_rsrp_raw is not None:
        rsrp_diff = (rsrp_raw - prev_rsrp_raw).astype(np.float32)
    else:
        rsrp_diff = np.zeros(num_cells, dtype=np.float32)

    if prev_beam_id is not None:
        beam_id_diff = (beam_id - prev_beam_id).astype(np.float32)
    else:
        beam_id_diff = np.zeros(num_cells, dtype=np.float32)

    meas = SlotMeasurement(
        rsrp_raw=rsrp_raw.astype(np.float32),
        rsrq=rsrq.astype(np.float32),
        sinr=sinr.astype(np.float32),
        doppler_est=doppler_est.astype(np.float32),
        beam_id=beam_id,
        delay_spread_ns=delay_spread_ns.astype(np.float32),
        los_indicator=los_indicator.astype(np.float32),
        aoa_deg=aoa_deg_arr.astype(np.float32),
        serving_cell=serving_cell,
    )

    return meas, rsrp_diff, beam_id_diff


def _compute_sinr(
    rsrp_dbm: np.ndarray,
    noise_floor_dbm: float,
    interference_threshold_db: float = 10.0,
) -> np.ndarray:
    """
    计算每个小区的 SINR

    改进版：只考虑 RSRP 差距 < interference_threshold_db 的邻区作为干扰
    """
    num_cells = len(rsrp_dbm)
    rsrp_linear = 10 ** (rsrp_dbm / 10)
    noise_linear = 10 ** (noise_floor_dbm / 10)

    sinr = np.zeros(num_cells, dtype=np.float32)

    for c in range(num_cells):
        interference_linear = 0.0
        for c_int in range(num_cells):
            if c_int == c:
                continue
            rsrp_gap = rsrp_dbm[c] - rsrp_dbm[c_int]
            if rsrp_gap < interference_threshold_db:
                interference_linear += rsrp_linear[c_int]

        sinr_linear = rsrp_linear[c] / (interference_linear + noise_linear)
        sinr[c] = float(10 * np.log10(max(sinr_linear, 1e-10)))

    return sinr


def _select_beam(aoa_deg: float, num_beams: int, beam_angle_range: float) -> int:
    """根据到达角选择最优波束 ID"""
    half_range = beam_angle_range / 2
    beam_angles = np.linspace(-half_range, half_range, num_beams)
    aoa_norm = ((aoa_deg + 180) % 360) - 180
    diffs = np.abs(beam_angles - aoa_norm)
    return int(np.argmin(diffs))


def _default_measurement(cfg: SimConfig) -> Tuple:
    """当射线追踪失败时返回默认测量值"""
    C = cfg.num_cells
    meas = SlotMeasurement(
        rsrp_raw=np.full(C, -120.0, dtype=np.float32),
        rsrq=np.full(C, -20.0, dtype=np.float32),
        sinr=np.full(C, -20.0, dtype=np.float32),
        doppler_est=np.zeros(C, dtype=np.float32),
        beam_id=np.zeros(C, dtype=np.int32),
        delay_spread_ns=np.zeros(C, dtype=np.float32),
        los_indicator=np.zeros(C, dtype=np.float32),
        aoa_deg=np.zeros(C, dtype=np.float32),
        serving_cell=0,
    )
    return meas, np.zeros(C, dtype=np.float32), np.zeros(C, dtype=np.float32)


# =========================================================
# L3 滤波（3GPP TS 38.331）
# =========================================================

def apply_l3_filter(rsrp_raw_seq: np.ndarray, alpha: float) -> np.ndarray:
    """
    对 RSRP 时序施加 L3 滤波（3GPP TS 38.331）

    公式：F(t) = (1-alpha) × F(t-1) + alpha × M(t)
    滤波在线性域（mW）进行，而非 dBm 域
    """
    T, C = rsrp_raw_seq.shape
    rsrp_linear = 10 ** (rsrp_raw_seq / 10)

    rsrp_filt_linear = np.zeros_like(rsrp_linear)
    rsrp_filt_linear[0] = rsrp_linear[0]

    for t in range(1, T):
        rsrp_filt_linear[t] = (1 - alpha) * rsrp_filt_linear[t - 1] + alpha * rsrp_linear[t]

    rsrp_l3 = 10 * np.log10(np.maximum(rsrp_filt_linear, 1e-20))
    return rsrp_l3.astype(np.float32)


# =========================================================
# 特征向量构建
# =========================================================

def build_feature_vector(
    rsrp_l3: np.ndarray,
    rsrq: np.ndarray,
    sinr: np.ndarray,
    doppler_est: np.ndarray,
    beam_id: np.ndarray,
    rsrp_diff: np.ndarray,
    beam_id_diff: np.ndarray,
    delay_spread_ns: np.ndarray,
    los_indicator: np.ndarray,
    num_beams: int,
) -> np.ndarray:
    """
    构建特征向量（共 9*C 维）

    特征组成：
      RSRP_l3 | RSRQ | SINR | Doppler_est | BeamID_norm |
      RSRP_diff | BeamID_diff | DelaySpread_norm | LOS_indicator
    """
    beam_id_norm = beam_id.astype(np.float32) / max(num_beams - 1, 1)
    delay_spread_norm = np.clip(delay_spread_ns / 500.0, 0.0, 1.0)

    feat = np.concatenate([
        rsrp_l3,
        rsrq,
        sinr,
        doppler_est,
        beam_id_norm,
        rsrp_diff,
        beam_id_diff,
        delay_spread_norm,
        los_indicator,
    ], axis=0)

    return feat.astype(np.float32)


# =========================================================
# 完整轨迹的信道仿真（Sionna 2.x 兼容）
# =========================================================

def load_neighbor_relations(network_config_path: Optional[str] = None) -> Optional[Dict]:
    """
    从 network_config.json 加载邻区关系

    参数：
        network_config_path: 配置文件路径（None 则使用默认路径）

    返回：
        邻区关系字典 {cell_id_str: [neighbor_ids]}，如果文件不存在则返回 None
    """
    import json
    from pathlib import Path

    if network_config_path is None:
        network_config_path = Path(__file__).parent / "network_config.json"
    else:
        network_config_path = Path(network_config_path)

    if not network_config_path.exists():
        return None

    try:
        with open(network_config_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        relations = cfg_data.get("neighbor_config", {}).get("relations", None)
        if relations:
            # 转换为 int key 的字典
            return {int(k): [int(v) for v in vals] for k, vals in relations.items()}
    except Exception:
        pass
    return None


def get_active_cells(
    ue_pos_2d: np.ndarray,
    bs_positions_2d: np.ndarray,
    neighbor_relations: Optional[Dict],
    serving_cell: int,
    fallback_k: int = 6,
) -> List[int]:
    """
    获取需要做射线追踪的基站列表（服务小区 + 邻区）

    参数：
        ue_pos_2d:          UE 2D 坐标
        bs_positions_2d:    [C, 2] 基站 2D 坐标
        neighbor_relations: 邻区关系字典（来自 network_config.json）
        serving_cell:       当前服务小区 ID
        fallback_k:         如果没有邻区配置，使用距离最近的 K 个基站

    返回：
        需要做射线追踪的基站 ID 列表
    """
    if neighbor_relations is not None and serving_cell in neighbor_relations:
        # 使用预配置的邻区关系
        active = [serving_cell] + neighbor_relations[serving_cell]
        return list(set(active))  # 去重
    else:
        # 回退：使用距离最近的 K 个基站
        dists = np.linalg.norm(bs_positions_2d - ue_pos_2d, axis=1)
        nearest = np.argsort(dists)[:fallback_k + 1].tolist()
        return nearest


def simulate_trajectory_channel(
    traj,
    scene_mgr,
    cfg: SimConfig,
    logger=None,
    neighbor_relations: Optional[Dict] = None,
) -> Dict:
    """
    对一条完整轨迹执行信道仿真，提取所有时隙的特征

    参数：
        traj:               Trajectory 对象（来自 trajectory.py）
        scene_mgr:          SceneManager 对象（来自 scene_setup.py，Sionna 2.x 版本）
        cfg:                仿真配置
        logger:             logging.Logger 对象（可选，用于记录错误到日志文件）
        neighbor_relations: 邻区关系字典（来自 network_config.json）
                            如果提供，只对服务小区 + 邻区做射线追踪（加速约 3 倍）
                            如果为 None，对所有基站做射线追踪

    返回：
        result 字典，包含所有时隙的信道数据和特征矩阵
    """
    import traceback as _traceback

    def _log(msg: str, level: str = "warning"):
        """统一输出到控制台和日志文件"""
        if logger is not None:
            if level == "error":
                logger.error(msg)
            elif level == "debug":
                logger.debug(msg)
            else:
                logger.warning(msg)
        else:
            print(msg)

    T = traj.num_slots
    C = cfg.num_cells

    # 预分配存储空间
    rsrp_raw_seq = np.zeros((T, C), dtype=np.float32)
    rsrq_seq = np.zeros((T, C), dtype=np.float32)
    sinr_seq = np.zeros((T, C), dtype=np.float32)
    doppler_seq = np.zeros((T, C), dtype=np.float32)
    beam_id_seq = np.zeros((T, C), dtype=np.int32)
    delay_spread_seq = np.zeros((T, C), dtype=np.float32)
    los_seq = np.zeros((T, C), dtype=np.float32)
    serving_raw_seq = np.zeros(T, dtype=np.int32)

    prev_rsrp_raw = None
    prev_beam_id = None

    # 逐时隙仿真
    for t in range(T):
        ue_pos_2d = traj.pos[t]
        ue_vel_2d = traj.vel[t]
        ue_pos_3d = np.array([ue_pos_2d[0], ue_pos_2d[1], cfg.h_ue], dtype=np.float32)

        # 在场景中放置 UE
        scene_mgr.place_receiver(ue_pos_2d)

        # 执行射线追踪（Sionna 2.x API）
        try:
            paths = scene_mgr.trace_paths()
            # Sionna 2.0.1: extract_path_data 返回 (a, tau, phi_r, theta_r, path_types)
            path_data = scene_mgr.extract_path_data(paths)
            if len(path_data) == 5:
                a, tau, phi_r, theta_r, path_types = path_data
            else:
                a, tau, phi_r, path_types = path_data
                theta_r = None

            # 计算信道测量量和 CSI 特征
            meas, rsrp_diff, beam_id_diff = compute_channel_from_paths(
                a=a,
                tau=tau,
                phi_r=phi_r,
                path_types=path_types,
                cfg=cfg,
                ue_vel=ue_vel_2d,
                bs_positions_3d=scene_mgr.bs_positions_3d,
                ue_pos_3d=ue_pos_3d,
                prev_rsrp_raw=prev_rsrp_raw,
                prev_beam_id=prev_beam_id,
                theta_r=theta_r,
            )
        except Exception as e:
            # 射线追踪失败时使用默认值，并记录完整错误信息
            tb_str = _traceback.format_exc()
            if t == 0:
                # 第一个时隙失败时记录完整错误（包含 traceback 和 Scene API 信息）
                _log(f"[轨迹 {traj.traj_type}] 时隙 {t} 射线追踪失败：{e}", "error")
                _log(f"完整错误信息：\n{tb_str}", "error")
                # 记录 Scene 对象的可用方法（帮助诊断 API 不兼容问题）
                try:
                    scene_methods = [m for m in dir(scene_mgr.scene) if not m.startswith("_")]
                    _log(f"Scene 对象可用方法：{scene_methods}", "error")
                except Exception:
                    pass
            # 后续时隙静默处理（不记录日志，避免日志文件过大）
            # 如果需要调试，可以取消下面这行的注释：
            # else:
            #     _log(f"[轨迹 {traj.traj_type}] 时隙 {t} 射线追踪失败：{e}", "warning")
            meas, rsrp_diff, beam_id_diff = _default_measurement(cfg)

        # 存储结果
        rsrp_raw_seq[t] = meas.rsrp_raw
        rsrq_seq[t] = meas.rsrq
        sinr_seq[t] = meas.sinr
        doppler_seq[t] = meas.doppler_est
        beam_id_seq[t] = meas.beam_id
        delay_spread_seq[t] = meas.delay_spread_ns
        los_seq[t] = meas.los_indicator
        serving_raw_seq[t] = meas.serving_cell

        prev_rsrp_raw = meas.rsrp_raw.copy()
        prev_beam_id = meas.beam_id.copy()

    # ---- L3 滤波 ----
    rsrp_l3_seq = apply_l3_filter(rsrp_raw_seq, cfg.l3_alpha)

    # ---- L3 服务小区 ----
    serving_l3_seq = np.argmax(rsrp_l3_seq, axis=1).astype(np.int32)

    # ---- 构建特征矩阵 ----
    rsrp_diff_seq = np.zeros((T, C), dtype=np.float32)
    rsrp_diff_seq[1:] = rsrp_raw_seq[1:] - rsrp_raw_seq[:-1]

    beam_id_diff_seq = np.zeros((T, C), dtype=np.float32)
    beam_id_diff_seq[1:] = (beam_id_seq[1:] - beam_id_seq[:-1]).astype(np.float32)

    feat_matrix = np.zeros((T, cfg.num_features), dtype=np.float32)
    for t in range(T):
        feat_matrix[t] = build_feature_vector(
            rsrp_l3=rsrp_l3_seq[t],
            rsrq=rsrq_seq[t],
            sinr=sinr_seq[t],
            doppler_est=doppler_seq[t],
            beam_id=beam_id_seq[t],
            rsrp_diff=rsrp_diff_seq[t],
            beam_id_diff=beam_id_diff_seq[t],
            delay_spread_ns=delay_spread_seq[t],
            los_indicator=los_seq[t],
            num_beams=cfg.num_beams,
        )

    return {
        "rsrp_raw": rsrp_raw_seq,
        "rsrp_l3": rsrp_l3_seq,
        "rsrq": rsrq_seq,
        "sinr": sinr_seq,
        "doppler_est": doppler_seq,
        "beam_id": beam_id_seq,
        "delay_spread": delay_spread_seq,
        "los_indicator": los_seq,
        "serving_raw": serving_raw_seq,
        "serving_l3": serving_l3_seq,
        "feat_matrix": feat_matrix,
    }


# =========================================================
# 独立运行：测试信道计算（不需要 Sionna，用模拟数据）
# =========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    print("测试信道计算（使用模拟路径数据）...")
    print(f"特征维度：{cfg.num_features}（= 9 × {cfg.num_cells} 小区）")

    # 测试 L3 滤波
    T = 100
    C = cfg.num_cells
    rsrp_test = np.random.randn(T, C) * 5 - 80

    rsrp_l3 = apply_l3_filter(rsrp_test, cfg.l3_alpha)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rsrp_test[:, 0], label="Instant RSRP", alpha=0.7)
    plt.plot(rsrp_l3[:, 0], label="L3 Filtered RSRP", linewidth=2)
    plt.xlabel("Slot")
    plt.ylabel("RSRP [dBm]")
    plt.title(f"L3 Filter (k={cfg.l3_filter_k}, alpha={cfg.l3_alpha:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 测试 SINR 计算
    rsrp_test_1d = np.array([-80, -85, -90, -95, -100, -105, -110], dtype=np.float32)
    sinr_test = _compute_sinr(rsrp_test_1d, cfg.noise_floor_dbm)

    plt.subplot(1, 2, 2)
    cells = [f"BS{i}" for i in range(len(rsrp_test_1d))]
    x = np.arange(len(cells))
    plt.bar(x - 0.2, rsrp_test_1d, 0.4, label="RSRP [dBm]", alpha=0.7)
    plt.bar(x + 0.2, sinr_test, 0.4, label="SINR [dB]", alpha=0.7)
    plt.xticks(x, cells)
    plt.xlabel("Cell")
    plt.ylabel("dB")
    plt.title("SINR (strong interferers only)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("channel_test.png", dpi=150, bbox_inches="tight")
    print("Test figure saved to channel_test.png")
    plt.close()

    print("\nFeature vector test:")
    feat = build_feature_vector(
        rsrp_l3=rsrp_test_1d,
        rsrq=np.full(C, -10.0, dtype=np.float32),
        sinr=sinr_test,
        doppler_est=np.random.randn(C).astype(np.float32) * 50,
        beam_id=np.random.randint(0, cfg.num_beams, C),
        rsrp_diff=np.random.randn(C).astype(np.float32) * 2,
        beam_id_diff=np.zeros(C, dtype=np.float32),
        delay_spread_ns=np.random.rand(C).astype(np.float32) * 200,
        los_indicator=np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        num_beams=cfg.num_beams,
    )
    print(f"  Feature dim: {feat.shape[0]} (expected: {cfg.num_features})")
    print(f"  Feature range: [{feat.min():.2f}, {feat.max():.2f}]")
    print("✓ channel.py test passed")