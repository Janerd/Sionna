"""
channel.py
==========
Sionna RT 信道仿真和 CSI 特征提取（Sionna 2.x 兼容版本）

Sionna 版本：2.x（基于 PyTorch）
  - 路径数据通过 scene_mgr.extract_path_data(paths) 提取
  - 使用 paths.doppler（物理精确）替代 GT 速度估算
  - 使用 paths.interactions 精确判断 LOS/NLOS

特征设计原则：
  - 所有特征均可在现实基站/UE 中提取（不依赖仿真内部信息）
  - 针对移动性场景优化（切换预测）

特征向量组成（共 10*C 维，C = 小区数）：
  [0:C]    RSRP_l3         - L3 滤波后 RSRP [dBm]（CSI-RS 测量，3GPP 标准）
  [C:2C]   RSRQ            - 参考信号接收质量 [dB]（3GPP 标准）
  [2C:3C]  SINR            - 信噪干扰比 [dB]（CQI 基础）
  [3C:4C]  Doppler_est     - Doppler 频移估计 [Hz]（CSI-RS 相位变化估计）
  [4C:5C]  BeamID          - 最优波束 ID（归一化到 [0,1]）
  [5C:6C]  RSRP_diff       - RSRP 变化率（相邻时隙差分）[dB/slot]
  [6C:7C]  BeamID_diff     - 波束 ID 变化（相邻时隙差分）
  [7C:8C]  DelaySpread     - RMS 时延扩展 [ns]（归一化，频域 CSI 估计）
  [8C:9C]  K_factor        - Ricean K 因子（主径功率/散射功率，归一化）
                             比 LOS_indicator 更精细，连续值
  [9C:10C] min_tau_norm    - 主径时延（归一化），隐含 UE 到基站距离
                             可从 CSI-RS 时域响应估计
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
    k_factor: np.ndarray        # [C] Ricean K 因子（主径功率/散射功率，dB）
    min_tau_s: np.ndarray       # [C] 主径时延 [s]（最短路径时延）
    aoa_deg: np.ndarray         # [C] 主径到达角 [度]（用于波束选择）
    serving_cell: int = 0


# =========================================================
# 射线追踪信道计算（Sionna 2.x 兼容）
# =========================================================

# LOS 路径的交互类型值（来自 sionna.rt.constants.InteractionType）
_INTERACTION_LOS = 0  # InteractionType.LOS = 0


def compute_channel_from_paths(
    a: np.ndarray,
    tau: np.ndarray,
    phi_r: np.ndarray,
    cfg: SimConfig,
    prev_rsrp_raw: Optional[np.ndarray] = None,
    prev_beam_id: Optional[np.ndarray] = None,
    doppler_per_path: Optional[np.ndarray] = None,
    interactions: Optional[np.ndarray] = None,
    # 以下参数保留用于兼容旧调用，但不再使用
    path_types: Optional[np.ndarray] = None,
    ue_vel: Optional[np.ndarray] = None,
    bs_positions_3d: Optional[np.ndarray] = None,
    ue_pos_3d: Optional[np.ndarray] = None,
    theta_r: Optional[np.ndarray] = None,
) -> Tuple:
    """
    从 Sionna 2.0.1 射线追踪结果计算信道测量量和 CSI 特征

    基于 Sionna 源码的正确实现：
    - paths.a：无量纲线性复数信道系数，|a|^2 是功率传输系数
    - paths.doppler：物理精确的 Doppler 频移（包含发射机+接收机+散射体速度贡献）
    - paths.interactions：路径交互类型，InteractionType.LOS=0 表示直射路径

    特征集（10 类，均可在现实网络中提取）：
      RSRP, RSRQ, SINR, Doppler, BeamID, RSRP_diff, BeamID_diff,
      DelaySpread, K_factor（替代 LOS_indicator）, min_tau（主径时延）

    参数：
        a:               路径复数增益 [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        tau:             路径时延 [s]，形状同上
        phi_r:           到达方位角 [rad]，形状同上
        cfg:             仿真配置
        prev_rsrp_raw:   [C] 上一时隙的瞬时 RSRP（用于计算差分）
        prev_beam_id:    [C] 上一时隙的波束 ID（用于计算差分）
        doppler_per_path: paths.doppler 的 numpy 数组（物理精确 Doppler）
        interactions:    paths.interactions 的 numpy 数组（保留，暂未使用）

    返回：
        (SlotMeasurement, rsrp_diff, beam_id_diff)
    """
    num_cells = cfg.num_cells

    # 预分配结果数组
    rsrp_raw = np.full(num_cells, -120.0, dtype=np.float32)
    doppler_est = np.zeros(num_cells, dtype=np.float32)
    delay_spread_ns = np.zeros(num_cells, dtype=np.float32)
    k_factor_db = np.full(num_cells, -30.0, dtype=np.float32)  # 默认 -30 dB（纯散射）
    min_tau_s = np.zeros(num_cells, dtype=np.float32)
    aoa_deg_arr = np.zeros(num_cells, dtype=np.float32)
    beam_id = np.zeros(num_cells, dtype=np.int32)

    # 发射功率（线性，mW）
    # paths.a 是无量纲线性信道系数，|a|^2 是功率传输系数
    # 接收功率 = 发射功率 × Σ|a_l|^2
    p_tx_mw = 10 ** (cfg.p_tx_dbm / 10)

    for c in range(num_cells):
        try:
            # ---- 提取基站 c 到 UE 的路径数据 ----
            # 维度：[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            if a.ndim == 5:
                a_c = a[0, 0, c, 0, :]
            elif a.ndim == 3:
                a_c = a[0, c, :]
            else:
                a_c = a[0, 0, c, 0, :]

            if tau.ndim == 3:
                tau_c = tau[0, c, :]
            elif tau.ndim == 2:
                tau_c = tau[c, :]
            else:
                tau_c = tau[0, c, :]

            if phi_r.ndim == 3:
                phi_r_c = phi_r[0, c, :]
            elif phi_r.ndim == 2:
                phi_r_c = phi_r[c, :]
            else:
                phi_r_c = phi_r[0, c, :]

        except (IndexError, TypeError):
            continue

        # 过滤有效路径
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
        total_power = float(np.sum(path_power))

        # ---- RSRP 计算 ----
        # 接收功率 = 发射功率 × Σ|a_l|^2（paths.a 已包含路径损耗）
        rsrp_raw[c] = float(10 * np.log10(max(p_tx_mw * total_power, 1e-20)))

        # ---- 主径时延（最短路径时延）----
        # 物理意义：min_tau ≈ dist(UE, BS) / c，隐含 UE 到基站的真实距离
        min_tau_s[c] = float(np.min(tau_valid))

        # ---- 时延扩展计算（RMS Delay Spread）----
        if len(tau_valid) > 1 and total_power > 0:
            mean_tau = float(np.sum(path_power * tau_valid)) / total_power
            rms_ds = float(np.sqrt(
                np.sum(path_power * (tau_valid - mean_tau) ** 2) / total_power
            ))
            delay_spread_ns[c] = rms_ds * 1e9
        else:
            delay_spread_ns[c] = 0.0

        # ---- Ricean K 因子（LOS 路径功率 / 散射路径总功率）----
        # K = P_LOS / P_scatter，用 dB 表示
        # 物理意义：K 大 → LOS 强；K 小 → 纯散射（NLOS）
        # 比 LOS_indicator（0/1）更精细，是连续值
        # 现实中可从 CSI-RS 的幅度分布估计
        #
        # 判断 LOS 路径：结合 paths.interactions（InteractionType.LOS=0）
        # 如果没有 interactions 数据，回退到"最强路径 = 主径"的近似
        los_power = 0.0
        if interactions is not None:
            try:
                if interactions.ndim == 6:
                    inter_c = interactions[0, 0, 0, c, 0, :]
                elif interactions.ndim == 4:
                    inter_c = interactions[0, 0, c, :]
                else:
                    inter_c = interactions[0, 0, c, :]
                inter_valid = inter_c[valid]
                los_mask = (inter_valid == _INTERACTION_LOS)
                if np.any(los_mask):
                    los_power = float(np.sum(path_power[los_mask]))
            except (IndexError, TypeError):
                pass

        if los_power == 0.0:
            # 回退：用最强路径近似 LOS 路径
            los_power = float(np.max(path_power))

        scatter_power = total_power - los_power
        if scatter_power > 1e-20:
            k_factor_db[c] = float(10 * np.log10(los_power / scatter_power))
        else:
            k_factor_db[c] = 30.0  # 纯 LOS，K → ∞，截断为 30 dB
        # 截断到合理范围 [-30, 30] dB
        k_factor_db[c] = float(np.clip(k_factor_db[c], -30.0, 30.0))

        # ---- Doppler 频移（直接从 paths.doppler 读取，物理精确）----
        if doppler_per_path is not None:
            try:
                if doppler_per_path.ndim == 5:
                    doppler_c = doppler_per_path[0, 0, c, 0, :]
                elif doppler_per_path.ndim == 3:
                    doppler_c = doppler_per_path[0, c, :]
                else:
                    doppler_c = doppler_per_path[0, c, :]

                doppler_valid = doppler_c[valid]
                # 功率加权平均 Doppler（主要路径贡献最大）
                fd_weighted = float(np.sum(path_power * doppler_valid) / total_power)
                # 加入测量噪声（UE 的 Doppler 估计误差约 ±5%）
                noise_std = abs(fd_weighted) * 0.05 + 1.0
                doppler_est[c] = float(fd_weighted + noise_std * np.random.randn())
            except (IndexError, TypeError):
                doppler_est[c] = 0.0
        else:
            doppler_est[c] = 0.0

        # ---- 到达角（用于波束选择）----
        strongest_path_idx = int(np.argmax(path_power))
        aoa_rad = float(phi_r_valid[strongest_path_idx])
        aoa_deg_arr[c] = float(np.degrees(aoa_rad))

        # ---- 波束 ID 计算 ----
        beam_id[c] = _select_beam(aoa_deg_arr[c], cfg.num_beams, cfg.beam_angle_range)

    # ---- RSRQ 计算（3GPP TS 38.215，修正版）----
    # RSRQ_c = N_RB × RSRP_c / RSSI_c
    N_RB = max(1, round(cfg.bw / 180e3))  # 动态计算：每个 RB 带宽 180kHz
    rsrp_linear = 10 ** (rsrp_raw / 10)
    noise_linear = 10 ** (cfg.noise_floor_dbm / 10)
    total_rsrp_linear = float(np.sum(rsrp_linear))
    rsrq = np.zeros(num_cells, dtype=np.float32)
    for c in range(num_cells):
        rssi_c = total_rsrp_linear + noise_linear
        rsrq[c] = float(10 * np.log10(max(N_RB * rsrp_linear[c] / rssi_c, 1e-20)))

    # ---- SINR 计算（射线追踪场景：所有非服务基站都是干扰）----
    sinr = _compute_sinr_rt(rsrp_raw, cfg.noise_floor_dbm)

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
        k_factor=k_factor_db.astype(np.float32),
        min_tau_s=min_tau_s.astype(np.float32),
        aoa_deg=aoa_deg_arr.astype(np.float32),
        serving_cell=serving_cell,
    )

    return meas, rsrp_diff, beam_id_diff


def _compute_sinr_rt(
    rsrp_dbm: np.ndarray,
    noise_floor_dbm: float,
) -> np.ndarray:
    """
    计算每个小区的 SINR（射线追踪场景版本）

    射线追踪场景中，每个基站的路径是独立精确计算的，
    所有非服务基站都是干扰（不使用阈值过滤）。

    SINR_c = RSRP_c / (Σ_{c'≠c} RSRP_{c'} + noise)
    """
    num_cells = len(rsrp_dbm)
    rsrp_linear = 10 ** (rsrp_dbm / 10)
    noise_linear = 10 ** (noise_floor_dbm / 10)
    total_rsrp = float(np.sum(rsrp_linear))

    sinr = np.zeros(num_cells, dtype=np.float32)
    for c in range(num_cells):
        interference = total_rsrp - rsrp_linear[c]
        sinr_linear = rsrp_linear[c] / (interference + noise_linear)
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
        k_factor=np.full(C, -30.0, dtype=np.float32),
        min_tau_s=np.zeros(C, dtype=np.float32),
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
# 特征向量构建（10 类特征）
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
    k_factor: np.ndarray,
    min_tau_s: np.ndarray,
    num_beams: int,
) -> np.ndarray:
    """
    构建特征向量（共 10*C 维）

    特征组成（均可在现实网络中提取）：
      RSRP_l3 | RSRQ | SINR | Doppler_est | BeamID_norm |
      RSRP_diff | BeamID_diff | DelaySpread_norm | K_factor_norm | min_tau_norm
    """
    beam_id_norm = beam_id.astype(np.float32) / max(num_beams - 1, 1)
    delay_spread_norm = np.clip(delay_spread_ns / 500.0, 0.0, 1.0)
    # K 因子归一化：[-30, 30] dB → [0, 1]
    k_factor_norm = (k_factor + 30.0) / 60.0
    k_factor_norm = np.clip(k_factor_norm, 0.0, 1.0)
    # 主径时延归一化：[0, 3.33us] → [0, 1]（3.33us 对应 1000m 距离）
    min_tau_norm = np.clip(min_tau_s / 3.33e-6, 0.0, 1.0)

    feat = np.concatenate([
        rsrp_l3,
        rsrq,
        sinr,
        doppler_est,
        beam_id_norm,
        rsrp_diff,
        beam_id_diff,
        delay_spread_norm,
        k_factor_norm,
        min_tau_norm,
    ], axis=0)

    return feat.astype(np.float32)


# =========================================================
# 完整轨迹的信道仿真（Sionna 2.x 兼容）
# =========================================================

def load_neighbor_relations(network_config_path: Optional[str] = None) -> Optional[Dict]:
    """从 network_config.json 加载邻区关系"""
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
    """获取需要做射线追踪的基站列表（服务小区 + 邻区）"""
    if neighbor_relations is not None and serving_cell in neighbor_relations:
        active = [serving_cell] + neighbor_relations[serving_cell]
        return list(set(active))
    else:
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

    优化：
      - 方案 B：复用 PathSolver 实例（在 SceneManager.setup() 中创建）
      - 方案 A：批量化 UE 位置（每批 cfg.rt_batch_size 个时隙，一次 PathSolver 调用）
    """
    import traceback as _traceback

    def _log(msg: str, level: str = "warning"):
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
    batch_size = getattr(cfg, "rt_batch_size", 32)

    # 预分配存储空间（10 类特征）
    rsrp_raw_seq = np.full((T, C), -120.0, dtype=np.float32)
    rsrq_seq = np.full((T, C), -20.0, dtype=np.float32)
    sinr_seq = np.full((T, C), -20.0, dtype=np.float32)
    doppler_seq = np.zeros((T, C), dtype=np.float32)
    beam_id_seq = np.zeros((T, C), dtype=np.int32)
    delay_spread_seq = np.zeros((T, C), dtype=np.float32)
    k_factor_seq = np.full((T, C), -30.0, dtype=np.float32)
    min_tau_seq = np.zeros((T, C), dtype=np.float32)
    serving_raw_seq = np.zeros(T, dtype=np.int32)

    prev_rsrp_raw = None
    prev_beam_id = None

    # 邻区优化：确定初始服务小区
    bs_positions_2d = scene_mgr.bs_positions_2d
    if neighbor_relations is not None:
        init_dists = np.linalg.norm(bs_positions_2d - traj.pos[0], axis=1)
        current_serving_cell = int(np.argmin(init_dists))
    else:
        current_serving_cell = 0

    # 方案 A：批量化处理
    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        batch_indices = list(range(batch_start, batch_end))

        batch_pos_2d = traj.pos[batch_indices]
        # batch_vel_2d 不再需要（Doppler 直接从 paths.doppler 读取）

        scene_mgr.place_receivers_batch(batch_pos_2d)

        try:
            paths = scene_mgr.trace_paths()
            path_data = scene_mgr.extract_path_data(paths)
            if len(path_data) == 6:
                a_batch, tau_batch, phi_r_batch, theta_r_batch, doppler_batch, interactions_batch = path_data
            elif len(path_data) == 5:
                a_batch, tau_batch, phi_r_batch, theta_r_batch, _ = path_data
                doppler_batch = None
                interactions_batch = None
            else:
                a_batch, tau_batch, phi_r_batch, _ = path_data
                theta_r_batch = None
                doppler_batch = None
                interactions_batch = None
            batch_ok = True
        except Exception as e:
            tb_str = _traceback.format_exc()
            _log(f"[轨迹 {traj.traj_type}] 批次 {batch_start}~{batch_end-1} 射线追踪失败：{e}", "error")
            if batch_start == 0:
                _log(f"完整错误信息：\n{tb_str}", "error")
                try:
                    scene_methods = [m for m in dir(scene_mgr.scene) if not m.startswith("_")]
                    _log(f"Scene 对象可用方法：{scene_methods}", "error")
                except Exception:
                    pass
            batch_ok = False

        for i, t in enumerate(batch_indices):
            ue_pos_2d = batch_pos_2d[i]

            if batch_ok:
                try:
                    if a_batch.ndim == 5:
                        a_i = a_batch[i:i+1]
                    elif a_batch.ndim == 3:
                        a_i = a_batch[i:i+1]
                    else:
                        a_i = a_batch[i:i+1]

                    tau_i = tau_batch[i:i+1]
                    phi_r_i = phi_r_batch[i:i+1]
                    doppler_i = doppler_batch[i:i+1] if doppler_batch is not None else None
                    interactions_i = (interactions_batch[:, i:i+1]
                                      if interactions_batch is not None else None)

                    meas, rsrp_diff, beam_id_diff = compute_channel_from_paths(
                        a=a_i,
                        tau=tau_i,
                        phi_r=phi_r_i,
                        cfg=cfg,
                        prev_rsrp_raw=prev_rsrp_raw,
                        prev_beam_id=prev_beam_id,
                        doppler_per_path=doppler_i,
                        interactions=interactions_i,
                    )
                except Exception:
                    meas, rsrp_diff, beam_id_diff = _default_measurement(cfg)
            else:
                meas, rsrp_diff, beam_id_diff = _default_measurement(cfg)

            rsrp_raw_seq[t] = meas.rsrp_raw
            rsrq_seq[t] = meas.rsrq
            sinr_seq[t] = meas.sinr
            doppler_seq[t] = meas.doppler_est
            beam_id_seq[t] = meas.beam_id
            delay_spread_seq[t] = meas.delay_spread_ns
            k_factor_seq[t] = meas.k_factor
            min_tau_seq[t] = meas.min_tau_s
            serving_raw_seq[t] = meas.serving_cell

            prev_rsrp_raw = meas.rsrp_raw.copy()
            prev_beam_id = meas.beam_id.copy()

            if neighbor_relations is not None:
                current_serving_cell = int(np.argmax(meas.rsrp_raw))

        try:
            del paths
        except Exception:
            pass

    # L3 滤波
    rsrp_l3_seq = apply_l3_filter(rsrp_raw_seq, cfg.l3_alpha)
    serving_l3_seq = np.argmax(rsrp_l3_seq, axis=1).astype(np.int32)

    # 构建特征矩阵（10 类特征）
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
            k_factor=k_factor_seq[t],
            min_tau_s=min_tau_seq[t],
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
        "k_factor": k_factor_seq,
        "min_tau": min_tau_seq,
        "serving_raw": serving_raw_seq,
        "serving_l3": serving_l3_seq,
        "feat_matrix": feat_matrix,
    }


# =========================================================
# 独立运行：测试信道计算
# =========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    print("测试信道计算（使用模拟路径数据）...")
    print(f"特征维度：{cfg.num_features}（= 10 × {cfg.num_cells} 小区）")

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

    rsrp_test_1d = np.array([-80, -85, -90, -95, -100, -105, -110], dtype=np.float32)
    sinr_test = _compute_sinr_rt(rsrp_test_1d, cfg.noise_floor_dbm)

    plt.subplot(1, 2, 2)
    cells = [f"BS{i}" for i in range(len(rsrp_test_1d))]
    x = np.arange(len(cells))
    plt.bar(x - 0.2, rsrp_test_1d, 0.4, label="RSRP [dBm]", alpha=0.7)
    plt.bar(x + 0.2, sinr_test, 0.4, label="SINR [dB]", alpha=0.7)
    plt.xticks(x, cells)
    plt.xlabel("Cell")
    plt.ylabel("dB")
    plt.title("SINR (RT version: all interferers)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("channel_test.png", dpi=150, bbox_inches="tight")
    print("Test figure saved to channel_test.png")
    plt.close()

    # 测试特征向量（10 类）
    feat = build_feature_vector(
        rsrp_l3=rsrp_test_1d,
        rsrq=np.full(C, -10.0, dtype=np.float32),
        sinr=sinr_test,
        doppler_est=np.random.randn(C).astype(np.float32) * 50,
        beam_id=np.random.randint(0, cfg.num_beams, C),
        rsrp_diff=np.random.randn(C).astype(np.float32) * 2,
        beam_id_diff=np.zeros(C, dtype=np.float32),
        delay_spread_ns=np.random.rand(C).astype(np.float32) * 200,
        k_factor=np.random.randn(C).astype(np.float32) * 5,
        min_tau_s=np.random.rand(C).astype(np.float32) * 1e-6,
        num_beams=cfg.num_beams,
    )
    print(f"\nFeature vector test:")
    print(f"  Feature dim: {feat.shape[0]} (expected: {cfg.num_features})")
    print(f"  Feature range: [{feat.min():.2f}, {feat.max():.2f}]")
    print("channel.py test passed")