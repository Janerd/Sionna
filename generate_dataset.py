"""
generate_dataset.py
===================
主数据生成脚本（Sionna 2.x 兼容版本）
（对应 MATLAB generate_dataset.m + main_simulation.m）

使用方法：
    python generate_dataset.py                    # 使用默认 UMi 配置
    python generate_dataset.py --scenario uma     # 使用 UMa 配置
    python generate_dataset.py --num-traj 120     # 生成 120 条轨迹
    python generate_dataset.py --no-gpu           # 不使用 GPU（调试用）
    python generate_dataset.py --log-file run.log # 指定日志文件

输出文件（保存到 outputs/ 目录）：
    dataset.npz          - 样本级数据集（用于 GRU 训练）
    trajectory_data.npz  - 轨迹级数据（用于策略回放）
    dataset_info.txt     - 数据集统计信息
    generate_dataset.log - 运行日志（自动生成，可用 git 追踪）

注意：
    - 标签使用"未来最优 SINR 小区"（而非 RSRP 最强小区）
    - predHorizon 改为 5 slots（200ms），比 MATLAB 的 20 slots 更合理
    - 使用 Sionna 2.x API（trace_paths + compute_fields）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import json

from config import SimConfig, get_umi_config, get_uma_config
from scene_setup import SceneManager, compute_hexagonal_bs_positions, get_bs_positions
from trajectory import Trajectory, generate_all_trajectories
from channel import simulate_trajectory_channel, load_neighbor_relations


# =========================================================
# 从 network_config.json 加载并覆盖仿真配置
# =========================================================

def apply_network_config(cfg: SimConfig, network_config_path: Path) -> SimConfig:
    """
    从 network_config.json 读取基站配置，覆盖 cfg 中的相关参数。

    这样只需修改 network_config.json，不需要同时修改 config.py。
    支持随时修改基站数量、坐标和邻区关系。

    参数：
        cfg:                  仿真配置（来自 config.py）
        network_config_path:  network_config.json 路径

    返回：
        更新后的 cfg
    """
    if not network_config_path.exists():
        return cfg

    try:
        with open(network_config_path, "r", encoding="utf-8") as f:
            net_cfg = json.load(f)
    except Exception as e:
        print(f"警告：无法读取 network_config.json（{e}），使用 config.py 默认值")
        return cfg

    bs_cfg = net_cfg.get("bs_config", {})
    positions = bs_cfg.get("positions", [])

    if not positions:
        return cfg

    # 覆盖基站数量
    num_cells = len(positions)
    if num_cells != cfg.num_cells:
        print(f"[network_config] 基站数量：{cfg.num_cells} → {num_cells}（来自 network_config.json）")
        cfg.num_cells = num_cells

    # 覆盖 ISD（如果有）
    if "isd" in bs_cfg:
        cfg.isd = float(bs_cfg["isd"])

    # 覆盖基站高度（如果有）
    if "h_bs" in bs_cfg:
        cfg.h_bs = float(bs_cfg["h_bs"])

    # 覆盖基站坐标（设置 bs_positions_override）
    cfg.bs_positions_override = np.array(
        [[float(p["x"]), float(p["y"])] for p in positions],
        dtype=np.float32,
    )
    print(f"[network_config] 已加载 {num_cells} 个基站坐标（来自 network_config.json）")

    return cfg


# =========================================================
# 日志系统设置
# =========================================================

def setup_logging(log_file: str, output_dir: Path) -> logging.Logger:
    """
    设置日志系统：同时输出到控制台和日志文件

    日志文件保存在 output_dir 下，可以直接提交到 git 仓库追踪运行记录。

    参数：
        log_file:   日志文件名（相对于 output_dir）
        output_dir: 输出目录

    返回：
        logger: 配置好的日志记录器
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_file

    # 创建 logger
    logger = logging.getLogger("generate_dataset")
    logger.setLevel(logging.DEBUG)

    # 清除已有的 handler（避免重复添加）
    logger.handlers.clear()

    # 格式：时间 + 级别 + 消息
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 handler（记录所有级别，包括 DEBUG）
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"日志文件：{log_path.absolute()}")
    return logger


class TqdmLoggingHandler(logging.Handler):
    """tqdm 兼容的日志 handler（避免进度条被日志打断）"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


# =========================================================
# 数据集构建
# =========================================================

def build_sample_dataset(
    trajectories: List[Trajectory],
    channel_results: List[Dict],
    cfg: SimConfig,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从轨迹数据构建样本级数据集

    标签 = 未来 pred_horizon slots 后的最优 SINR 小区
    （不是 RSRP 最强小区，这是与 MATLAB 版本的核心区别）
    """
    W = cfg.window_size
    H = cfg.pred_horizon
    F = cfg.num_features
    C = cfg.num_cells

    total_samples = sum(
        max(0, ch["feat_matrix"].shape[0] - W - H)
        for ch in channel_results
    )
    logger.info(f"预估样本总数：{total_samples}")

    X_raw = np.zeros((total_samples, W, F), dtype=np.float32)
    Y_cell = np.zeros(total_samples, dtype=np.int64)
    Y_sinr = np.zeros((total_samples, C), dtype=np.float32)
    meta_speed = np.zeros(total_samples, dtype=np.float32)
    meta_traj = np.zeros(total_samples, dtype=np.int32)

    sample_idx = 0

    for traj_idx, (traj, ch) in enumerate(zip(trajectories, channel_results)):
        feat_matrix = ch["feat_matrix"]
        sinr_seq = ch["sinr"]
        T = feat_matrix.shape[0]

        for slot in range(W, T - H):
            X_raw[sample_idx] = feat_matrix[slot - W : slot]

            future_slot = slot + H - 1
            future_sinr = sinr_seq[future_slot]
            Y_cell[sample_idx] = int(np.argmax(future_sinr))
            Y_sinr[sample_idx] = future_sinr

            meta_speed[sample_idx] = traj.speed_ms
            meta_traj[sample_idx] = traj_idx
            sample_idx += 1

    actual_samples = sample_idx
    X_raw = X_raw[:actual_samples]
    Y_cell = Y_cell[:actual_samples]
    Y_sinr = Y_sinr[:actual_samples]
    meta_speed = meta_speed[:actual_samples]
    meta_traj = meta_traj[:actual_samples]

    logger.info(f"实际样本数：{actual_samples}")
    return X_raw, Y_cell, Y_sinr, meta_speed, meta_traj


def create_train_val_test_split(
    meta_traj: np.ndarray,
    num_trajectories: int,
    num_speeds: int,
    traj_per_speed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按轨迹 ID 进行分层划分（train/val/test）"""
    train_traj_ids = []
    val_traj_ids = []
    test_traj_ids = []

    for s in range(num_speeds):
        start_id = s * traj_per_speed
        end_id = start_id + traj_per_speed
        traj_ids_s = list(range(start_id, end_id))

        n = len(traj_ids_s)
        n_train = round(train_ratio * n)
        n_val = round(val_ratio * n)

        train_traj_ids.extend(traj_ids_s[:n_train])
        val_traj_ids.extend(traj_ids_s[n_train : n_train + n_val])
        test_traj_ids.extend(traj_ids_s[n_train + n_val :])

    train_mask = np.isin(meta_traj, train_traj_ids)
    val_mask = np.isin(meta_traj, val_traj_ids)
    test_mask = np.isin(meta_traj, test_traj_ids)

    msg = (
        f"数据集划分：train={train_mask.sum()} 样本（{len(train_traj_ids)} 条轨迹），"
        f"val={val_mask.sum()} 样本（{len(val_traj_ids)} 条轨迹），"
        f"test={test_mask.sum()} 样本（{len(test_traj_ids)} 条轨迹）"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return train_mask, val_mask, test_mask


def build_trajectory_data(
    trajectories: List[Trajectory],
    channel_results: List[Dict],
    split_labels: List[str],
    cfg: SimConfig,
) -> List[Dict]:
    """构建轨迹级数据（用于 A3/A5/GRU 策略回放）"""
    traj_data_list = []

    for traj_idx, (traj, ch) in enumerate(zip(trajectories, channel_results)):
        td = {
            "traj_id": traj_idx,
            "speed_ms": traj.speed_ms,
            "speed_kmh": traj.speed_kmh,
            "traj_type": traj.traj_type,
            "num_slots": traj.num_slots,
            "pos": traj.pos,
            "vel": traj.vel,
            "RSRP_raw": ch["rsrp_raw"],
            "RSRP_l3": ch["rsrp_l3"],
            "RSRQ": ch["rsrq"],
            "SINR": ch["sinr"],
            "doppler": ch["doppler_est"],
            "beam_id": ch["beam_id"],
            "delay_spread": ch["delay_spread"],
            "k_factor": ch.get("k_factor", np.full((traj.num_slots, cfg.num_cells), -30.0, dtype=np.float32)),
            "min_tau": ch.get("min_tau", np.zeros((traj.num_slots, cfg.num_cells), dtype=np.float32)),
            "serving_raw": ch["serving_raw"],
            "serving_l3": ch["serving_l3"],
            "split": split_labels[traj_idx],
        }
        traj_data_list.append(td)

    return traj_data_list


# =========================================================
# 保存数据集
# =========================================================

def save_dataset(
    X_raw: np.ndarray,
    Y_cell: np.ndarray,
    Y_sinr: np.ndarray,
    meta_speed: np.ndarray,
    meta_traj: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: SimConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """保存样本级数据集到 .npz 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "dataset.npz"

    np.savez_compressed(
        save_path,
        X_raw=X_raw,
        Y_cell=Y_cell,
        Y_sinr=Y_sinr,
        split_train=train_mask,
        split_val=val_mask,
        split_test=test_mask,
        meta_speed=meta_speed,
        meta_traj=meta_traj,
        num_cells=np.array(cfg.num_cells),
        num_features=np.array(cfg.num_features),
        window_size=np.array(cfg.window_size),
        pred_horizon=np.array(cfg.pred_horizon),
        slot_duration=np.array(cfg.slot_duration),
        fc=np.array(cfg.fc),
        scenario_type=np.array(cfg.scenario_type),
    )

    size_mb = save_path.stat().st_size / 1024 / 1024
    logger.info(f"样本级数据集已保存：{save_path}（{size_mb:.1f} MB）")
    logger.info(f"  X_raw 形状：{X_raw.shape}（样本数 × 窗口 × 特征）")
    logger.info(f"  Y_cell 形状：{Y_cell.shape}，类别数：{cfg.num_cells}")


def save_trajectory_data(
    traj_data_list: List[Dict],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """保存轨迹级数据到 .npz 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "trajectory_data.npz"

    num_traj = len(traj_data_list)

    traj_ids = np.array([td["traj_id"] for td in traj_data_list])
    speed_ms = np.array([td["speed_ms"] for td in traj_data_list])
    speed_kmh = np.array([td["speed_kmh"] for td in traj_data_list])
    num_slots = np.array([td["num_slots"] for td in traj_data_list])
    splits = np.array([td["split"] for td in traj_data_list])
    traj_types = np.array([td["traj_type"] for td in traj_data_list])

    def to_object_array(field_name):
        arr = np.empty(num_traj, dtype=object)
        for i, td in enumerate(traj_data_list):
            arr[i] = td[field_name]
        return arr

    np.savez_compressed(
        save_path,
        traj_ids=traj_ids,
        speed_ms=speed_ms,
        speed_kmh=speed_kmh,
        num_slots=num_slots,
        splits=splits,
        traj_types=traj_types,
        pos=to_object_array("pos"),
        vel=to_object_array("vel"),
        RSRP_raw=to_object_array("RSRP_raw"),
        RSRP_l3=to_object_array("RSRP_l3"),
        RSRQ=to_object_array("RSRQ"),
        SINR=to_object_array("SINR"),
        doppler=to_object_array("doppler"),
        beam_id=to_object_array("beam_id"),
        delay_spread=to_object_array("delay_spread"),
        k_factor=to_object_array("k_factor"),
        min_tau=to_object_array("min_tau"),
        serving_raw=to_object_array("serving_raw"),
        serving_l3=to_object_array("serving_l3"),
    )

    size_mb = save_path.stat().st_size / 1024 / 1024
    logger.info(f"轨迹级数据已保存：{save_path}（{size_mb:.1f} MB，{num_traj} 条轨迹）")


def save_dataset_info(
    X_raw: np.ndarray,
    Y_cell: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: SimConfig,
    elapsed_time: float,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """保存数据集统计信息"""
    info_path = output_dir / "dataset_info.txt"

    lines = [
        "=" * 60,
        "Sionna RT Dataset Info (Sionna 2.x)",
        "=" * 60,
        f"Generation time: {elapsed_time:.1f} s ({elapsed_time/60:.1f} min)",
        "",
        "Scene config:",
        f"  scenario_type: {cfg.scenario_type}",
        f"  num_cells:     {cfg.num_cells}",
        f"  isd:           {cfg.isd} m",
        f"  h_bs:          {cfg.h_bs} m",
        f"  fc:            {cfg.fc/1e9:.1f} GHz",
        f"  speeds:        {cfg.speeds_kmh} km/h",
        "",
        "Dataset size:",
        f"  total samples: {len(X_raw)}",
        f"  train:         {train_mask.sum()}",
        f"  val:           {val_mask.sum()}",
        f"  test:          {test_mask.sum()}",
        f"  feature_dim:   {X_raw.shape[2]} (= 10 x {cfg.num_cells} cells)",
        f"  window_size:   {cfg.window_size} slots ({cfg.window_size*cfg.slot_duration*1000:.0f} ms)",
        f"  pred_horizon:  {cfg.pred_horizon} slots ({cfg.pred_horizon*cfg.slot_duration*1000:.0f} ms)",
        "",
        "Label distribution (test set):",
    ]

    y_test = Y_cell[test_mask]
    for c in range(cfg.num_cells):
        count = int(np.sum(y_test == c))
        ratio = count / max(len(y_test), 1) * 100
        lines.append(f"  cell_{c}: {count} samples ({ratio:.1f}%)")

    lines.extend([
        "",
        "Feature layout (10 x C dims):",
        "  [0:C]    RSRP_l3         - L3 filtered RSRP [dBm]",
        "  [C:2C]   RSRQ            - Reference signal quality [dB]",
        "  [2C:3C]  SINR            - Signal to interference+noise ratio [dB]",
        "  [3C:4C]  Doppler_est     - Doppler shift estimate [Hz] (from paths.doppler)",
        "  [4C:5C]  BeamID_norm     - Normalized beam ID [0,1]",
        "  [5C:6C]  RSRP_diff       - RSRP rate of change [dB/slot]",
        "  [6C:7C]  BeamID_diff     - Beam ID change",
        "  [7C:8C]  DelaySpread_norm- Normalized RMS delay spread",
        "  [8C:9C]  K_factor_norm   - Ricean K factor (LOS/scatter power, normalized)",
        "  [9C:10C] min_tau_norm    - Normalized first-path delay (proxy for UE-BS distance)",
        "",
        "Key differences from MATLAB version:",
        "  1. Label: best SINR cell (not best RSRP cell)",
        "  2. Features: K_factor + min_tau replace LOS_indicator; Doppler from paths.doppler",
        "  3. Channel: Sionna 2.x ray tracing (real 3D buildings)",
        "  4. SINR: all non-serving BSs are interferers (RT-correct)",
        "  5. pred_horizon: 5 slots (200ms) vs 20 slots (800ms) in MATLAB",
    ])

    info_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"数据集信息已保存：{info_path}")


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Sionna RT 数据集生成（Sionna 2.x）")
    parser.add_argument(
        "--scenario", choices=["umi", "uma"], default="umi",
        help="场景类型：umi（城市微站，默认）或 uma（城市宏站）"
    )
    parser.add_argument(
        "--num-traj", type=int, default=None,
        help="总轨迹数（覆盖配置文件中的值）"
    )
    parser.add_argument(
        "--outdir", default="outputs",
        help="输出目录（默认：outputs）"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="不使用 GPU（调试用，速度很慢）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认：42）"
    )
    parser.add_argument(
        "--log-file", default="generate_dataset.log",
        help="日志文件名（默认保存在项目根目录，不在 outdir 下）"
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    if args.scenario == "umi":
        cfg = get_umi_config()
    else:
        cfg = get_uma_config()

    # ---- 从 network_config.json 覆盖基站配置（优先级高于 config.py）----
    network_config_path = Path(__file__).parent / "network_config.json"
    cfg = apply_network_config(cfg, network_config_path)

    if args.num_traj is not None:
        cfg.num_trajectories = args.num_traj

    if args.no_gpu:
        cfg.use_gpu = False

    cfg.output_dir = args.outdir
    output_dir = Path(cfg.output_dir)

    # ---- 初始化日志 ----
    # 日志文件保存在项目根目录（与 .py 文件同级），而不是 outputs/ 下
    # 这样日志文件不受 .gitignore 的 outputs/ 排除规则影响，可以直接 git 追踪
    log_root = Path(__file__).parent
    logger = setup_logging(args.log_file, log_root)

    logger.info("=" * 60)
    logger.info("Sionna RT 数据集生成（Sionna 2.x）")
    logger.info("=" * 60)
    logger.info(f"场景：{cfg.scenario_type}，{cfg.num_cells} 小区，ISD={cfg.isd}m")
    logger.info(f"轨迹数：{cfg.num_trajectories}，速度：{cfg.speeds_kmh} km/h")
    logger.info(f"特征维度：{cfg.num_features}（= 10 × {cfg.num_cells}）")
    logger.info(f"标签：未来 {cfg.pred_horizon} slots 后的最优 SINR 小区")
    logger.info(f"输出目录：{output_dir.absolute()}")
    logger.info(f"GPU：{'是' if cfg.use_gpu else '否（CPU 模式，速度较慢）'}")
    logger.info(f"随机种子：{args.seed}")

    start_time = time.time()

    # ---- 步骤 1：初始化场景 ----
    logger.info("")
    logger.info("步骤 1/4：初始化 Sionna RT 场景...")
    scene_mgr = SceneManager(cfg)
    scene_mgr.setup()

    # 保存场景可视化（使用英文标签，避免中文字体问题）
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_mgr.visualize(
        save_path=str(output_dir / "scene_layout.png"),
        use_english=True,
    )

    # ---- 步骤 1.5：计算覆盖图和可行走网格 ----
    logger.info("")
    logger.info("步骤 1.5/4：计算覆盖图和可行走网格（约 2~10 分钟）...")

    # 1.5a：覆盖图（用于轨迹起点均匀分布）
    coverage_points = None
    try:
        coverage_points = scene_mgr.compute_coverage_points(
            cell_size=5.0,           # 5m 分辨率，精度和速度的平衡
            rsrp_threshold_dbm=-105.0,  # 只保留 RSRP > -105 dBm 的区域
        )
        logger.info(f"覆盖图计算完成：{len(coverage_points)} 个可行走点")
    except Exception as e:
        logger.warning(f"覆盖图计算失败（{e}），使用基站附近起点")
        coverage_points = None

    # 1.5b：可行走网格（用于中高速轨迹碰撞检测，避免穿越建筑物）
    walkable_grid = None
    walkable_cache = Path(cfg.output_dir) / "walkable_grid.npz"
    try:
        walkable_grid = scene_mgr.compute_walkable_grid(
            grid_size=2.0,           # 2m 分辨率，±500m 场景 = 500×500 格
            cache_path=str(walkable_cache),  # 缓存到文件，避免重复计算
        )
        logger.info(f"可行走网格计算完成：{walkable_grid['grid'].shape}，"
                    f"可行走比例：{walkable_grid['grid'].mean()*100:.1f}%")
    except Exception as e:
        logger.warning(f"可行走网格计算失败（{e}），中高速轨迹不使用碰撞检测")
        walkable_grid = None

    # ---- 步骤 2：生成轨迹 ----
    logger.info("")
    logger.info("步骤 2/4：生成 UE 轨迹（速度分级 + 碰撞检测）...")
    trajectories = generate_all_trajectories(
        cfg=cfg,
        bs_positions=scene_mgr.bs_positions_2d,
        scene_bounds=scene_mgr.scene_bounds,
        seed=args.seed,
        coverage_points=coverage_points,
        walkable_grid=walkable_grid,
    )
    logger.info(f"轨迹生成完成：共 {len(trajectories)} 条")

    # ---- 步骤 3：信道仿真 ----
    logger.info("")
    logger.info(f"步骤 3/4：执行信道仿真（{len(trajectories)} 条轨迹）...")
    logger.info("（使用 Sionna 2.x RT 射线追踪，RTX 4060 Ti 约需 1~3 小时）")

    # 加载邻区关系（如果存在，可以加速射线追踪约 3 倍）
    neighbor_relations = load_neighbor_relations()
    if neighbor_relations is not None:
        avg_neighbors = sum(len(v) for v in neighbor_relations.values()) / len(neighbor_relations)
        speedup = cfg.num_cells / (avg_neighbors + 1)
        logger.info(f"已加载邻区关系（平均邻区数：{avg_neighbors:.1f}，预计加速 {speedup:.1f}x）")
    else:
        logger.info("未找到邻区关系配置，对所有基站做射线追踪")
        logger.info("提示：运行 python network_config_tool.py 生成邻区配置可加速仿真")

    channel_results = []
    failed_trajs = 0

    for traj_idx, traj in enumerate(tqdm(trajectories, desc="信道仿真")):
        traj_start = time.time()
        try:
            ch = simulate_trajectory_channel(
                traj, scene_mgr, cfg,
                logger=logger,
                neighbor_relations=neighbor_relations,
            )
            channel_results.append(ch)
        except Exception as e:
            logger.error(f"轨迹 {traj_idx} 仿真失败：{e}")
            # 使用空结果占位，避免后续索引错误
            from channel import apply_l3_filter
            T = traj.num_slots
            C = cfg.num_cells
            empty_ch = {
                "rsrp_raw": np.full((T, C), -120.0, dtype=np.float32),
                "rsrp_l3": np.full((T, C), -120.0, dtype=np.float32),
                "rsrq": np.full((T, C), -20.0, dtype=np.float32),
                "sinr": np.full((T, C), -20.0, dtype=np.float32),
                "doppler_est": np.zeros((T, C), dtype=np.float32),
                "beam_id": np.zeros((T, C), dtype=np.int32),
                "delay_spread": np.zeros((T, C), dtype=np.float32),
                "k_factor": np.full((T, C), -30.0, dtype=np.float32),
                "min_tau": np.zeros((T, C), dtype=np.float32),
                "serving_raw": np.zeros(T, dtype=np.int32),
                "serving_l3": np.zeros(T, dtype=np.int32),
                "feat_matrix": np.zeros((T, cfg.num_features), dtype=np.float32),
            }
            channel_results.append(empty_ch)
            failed_trajs += 1

        traj_elapsed = time.time() - traj_start

        # 每 5 条轨迹记录一次进度
        if (traj_idx + 1) % 5 == 0:
            total_elapsed = time.time() - start_time
            remaining = total_elapsed / (traj_idx + 1) * (len(trajectories) - traj_idx - 1)
            logger.info(
                f"  轨迹 {traj_idx+1}/{len(trajectories)} 完成，"
                f"本条耗时 {traj_elapsed:.1f}s，"
                f"已用时 {total_elapsed/60:.1f} 分钟，"
                f"预计剩余 {remaining/60:.1f} 分钟"
            )

    if failed_trajs > 0:
        logger.warning(f"共有 {failed_trajs} 条轨迹仿真失败，已用默认值填充")

    # ---- 步骤 4：构建数据集 ----
    logger.info("")
    logger.info("步骤 4/4：构建数据集...")

    X_raw, Y_cell, Y_sinr, meta_speed, meta_traj = build_sample_dataset(
        trajectories, channel_results, cfg, logger
    )

    num_speeds = len(cfg.speeds_ms)
    traj_per_speed = cfg.num_trajectories // num_speeds
    train_mask, val_mask, test_mask = create_train_val_test_split(
        meta_traj=meta_traj,
        num_trajectories=cfg.num_trajectories,
        num_speeds=num_speeds,
        traj_per_speed=traj_per_speed,
        logger=logger,
    )

    # 确定每条轨迹的 split 标签
    split_labels = []
    for traj_idx in range(len(trajectories)):
        sample_mask = meta_traj == traj_idx
        if not np.any(sample_mask):
            split_labels.append("unused")
        elif train_mask[sample_mask][0]:
            split_labels.append("train")
        elif val_mask[sample_mask][0]:
            split_labels.append("val")
        else:
            split_labels.append("test")

    traj_data_list = build_trajectory_data(
        trajectories, channel_results, split_labels, cfg
    )

    # ---- 保存数据集 ----
    save_dataset(
        X_raw, Y_cell, Y_sinr, meta_speed, meta_traj,
        train_mask, val_mask, test_mask,
        cfg, output_dir, logger,
    )
    save_trajectory_data(traj_data_list, output_dir, logger)

    elapsed_time = time.time() - start_time
    save_dataset_info(
        X_raw, Y_cell, train_mask, val_mask, test_mask,
        cfg, elapsed_time, output_dir, logger,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"数据集生成完成！总用时：{elapsed_time/60:.1f} 分钟")
    logger.info(f"输出目录：{output_dir.absolute()}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("下一步：")
    logger.info("  1. 将 outputs/ 目录复制到公司电脑")
    logger.info("  2. 运行 GRU 训练：python train_gru_sionna.py --dataset outputs/dataset.npz")
    logger.info("  3. 运行策略评估：python replay_gru_sionna.py --trajectory-data outputs/trajectory_data.npz")


if __name__ == "__main__":
    main()