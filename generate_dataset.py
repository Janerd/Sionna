"""
generate_dataset.py
===================
主数据生成脚本（对应 MATLAB generate_dataset.m + main_simulation.m）

使用方法：
    python generate_dataset.py                    # 使用默认 UMi 配置
    python generate_dataset.py --scenario uma     # 使用 UMa 配置
    python generate_dataset.py --num-traj 120     # 生成 120 条轨迹
    python generate_dataset.py --no-gpu           # 不使用 GPU（调试用）

输出文件（保存到 outputs/ 目录）：
    dataset.npz          - 样本级数据集（用于 GRU 训练）
    trajectory_data.npz  - 轨迹级数据（用于策略回放）
    dataset_info.txt     - 数据集统计信息

数据集格式（与 MATLAB 版本兼容）：
    dataset.npz:
        X_raw:      [N, W, F] 时序特征（N=样本数，W=窗口大小，F=特征维度）
        Y_cell:     [N] 标签（未来最优 SINR 小区，0-indexed）
        Y_sinr:     [N, C] 标签（未来各小区 SINR，用于回归任务）
        split_train:[N] bool，训练集掩码
        split_val:  [N] bool，验证集掩码
        split_test: [N] bool，测试集掩码
        meta_speed: [N] 各样本对应的 UE 速度 [m/s]
        meta_traj:  [N] 各样本对应的轨迹 ID

    trajectory_data.npz:
        包含每条轨迹的完整信道数据（用于 A3/A5/GRU 策略回放）
        格式与 MATLAB trajectory_data.mat 兼容

注意：
    - 标签使用"未来最优 SINR 小区"（而非 RSRP 最强小区）
      这是与 MATLAB 版本的关键区别，使模型性能上界从 Strongest-RSRP
      提升到 Oracle-best-SINR
    - predHorizon 改为 5 slots（200ms），比 MATLAB 的 20 slots 更合理
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from config import SimConfig, get_umi_config, get_uma_config
from scene_setup import SceneManager, compute_hexagonal_bs_positions
from trajectory import Trajectory, generate_all_trajectories
from channel import simulate_trajectory_channel


# =========================================================
# 数据集构建
# =========================================================

def build_sample_dataset(
    trajectories: List[Trajectory],
    channel_results: List[Dict],
    cfg: SimConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从轨迹数据构建样本级数据集

    参数：
        trajectories:    轨迹列表
        channel_results: 每条轨迹的信道仿真结果
        cfg:             仿真配置

    返回：
        X_raw:      [N, W, F] 时序特征
        Y_cell:     [N] 标签（未来最优 SINR 小区，0-indexed）
        Y_sinr:     [N, C] 标签（未来各小区 SINR）
        meta_speed: [N] 速度 [m/s]
        meta_traj:  [N] 轨迹 ID

    关键设计决策：
        标签 = 未来 pred_horizon slots 后的最优 SINR 小区
        （不是 RSRP 最强小区，这是与 MATLAB 版本的核心区别）
    """
    W = cfg.window_size
    H = cfg.pred_horizon
    F = cfg.num_features
    C = cfg.num_cells

    # 预估样本总数
    total_samples = sum(
        max(0, ch["feat_matrix"].shape[0] - W - H)
        for ch in channel_results
    )

    print(f"预估样本总数：{total_samples}")

    X_raw = np.zeros((total_samples, W, F), dtype=np.float32)
    Y_cell = np.zeros(total_samples, dtype=np.int64)
    Y_sinr = np.zeros((total_samples, C), dtype=np.float32)
    meta_speed = np.zeros(total_samples, dtype=np.float32)
    meta_traj = np.zeros(total_samples, dtype=np.int32)

    sample_idx = 0

    for traj_idx, (traj, ch) in enumerate(zip(trajectories, channel_results)):
        feat_matrix = ch["feat_matrix"]   # [T, F]
        sinr_seq = ch["sinr"]             # [T, C]
        T = feat_matrix.shape[0]

        for slot in range(W, T - H):
            # 输入窗口：[slot-W, ..., slot-1]
            X_raw[sample_idx] = feat_matrix[slot - W : slot]

            # 标签：未来 pred_horizon slots 后的最优 SINR 小区
            # 这是关键改进：用 SINR 最强小区作为标签，而不是 RSRP 最强小区
            future_slot = slot + H - 1
            future_sinr = sinr_seq[future_slot]  # [C]
            Y_cell[sample_idx] = int(np.argmax(future_sinr))
            Y_sinr[sample_idx] = future_sinr

            meta_speed[sample_idx] = traj.speed_ms
            meta_traj[sample_idx] = traj_idx

            sample_idx += 1

    # 截取实际样本数
    actual_samples = sample_idx
    X_raw = X_raw[:actual_samples]
    Y_cell = Y_cell[:actual_samples]
    Y_sinr = Y_sinr[:actual_samples]
    meta_speed = meta_speed[:actual_samples]
    meta_traj = meta_traj[:actual_samples]

    print(f"实际样本数：{actual_samples}")
    return X_raw, Y_cell, Y_sinr, meta_speed, meta_traj


def create_train_val_test_split(
    meta_traj: np.ndarray,
    num_trajectories: int,
    num_speeds: int,
    traj_per_speed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按轨迹 ID 进行分层划分（train/val/test）

    注意：必须按轨迹划分，不能按样本随机划分，
    否则相邻时刻样本会泄漏（数据泄露）。

    参数：
        meta_traj:       [N] 各样本的轨迹 ID
        num_trajectories: 总轨迹数
        num_speeds:       速度种类数
        traj_per_speed:   每种速度的轨迹数
        train_ratio:      训练集比例
        val_ratio:        验证集比例

    返回：
        train_mask: [N] bool
        val_mask:   [N] bool
        test_mask:  [N] bool
    """
    train_traj_ids = []
    val_traj_ids = []
    test_traj_ids = []

    for s in range(num_speeds):
        # 该速度的轨迹 ID 范围
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

    print(f"数据集划分：")
    print(f"  训练集轨迹：{len(train_traj_ids)} 条，样本：{train_mask.sum()}")
    print(f"  验证集轨迹：{len(val_traj_ids)} 条，样本：{val_mask.sum()}")
    print(f"  测试集轨迹：{len(test_traj_ids)} 条，样本：{test_mask.sum()}")

    return train_mask, val_mask, test_mask


def build_trajectory_data(
    trajectories: List[Trajectory],
    channel_results: List[Dict],
    split_labels: List[str],
    cfg: SimConfig,
) -> Dict:
    """
    构建轨迹级数据（用于 A3/A5/GRU 策略回放）

    格式与 MATLAB trajectory_data.mat 兼容，
    可以直接用于现有的 Python 策略评估代码（policies.py 等）

    参数：
        trajectories:  轨迹列表
        channel_results: 信道仿真结果
        split_labels:  每条轨迹的 split 标签（'train'/'val'/'test'）
        cfg:           仿真配置

    返回：
        trajectory_data: 字典列表，每个元素对应一条轨迹
    """
    traj_data_list = []

    for traj_idx, (traj, ch) in enumerate(zip(trajectories, channel_results)):
        td = {
            "traj_id": traj_idx,
            "speed_ms": traj.speed_ms,
            "speed_kmh": traj.speed_kmh,
            "traj_type": traj.traj_type,
            "num_slots": traj.num_slots,
            "pos": traj.pos,                    # [T, 2]
            "vel": traj.vel,                    # [T, 2]
            "RSRP_raw": ch["rsrp_raw"],         # [T, C]
            "RSRP_l3": ch["rsrp_l3"],           # [T, C]
            "RSRQ": ch["rsrq"],                 # [T, C]
            "SINR": ch["sinr"],                 # [T, C]
            "doppler": ch["doppler_est"],        # [T, C]
            "beam_id": ch["beam_id"],            # [T, C]
            "delay_spread": ch["delay_spread"],  # [T, C]
            "los_indicator": ch["los_indicator"],# [T, C]
            "serving_raw": ch["serving_raw"],    # [T]
            "serving_l3": ch["serving_l3"],      # [T]
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
        # 元数据
        num_cells=np.array(cfg.num_cells),
        num_features=np.array(cfg.num_features),
        window_size=np.array(cfg.window_size),
        pred_horizon=np.array(cfg.pred_horizon),
        slot_duration=np.array(cfg.slot_duration),
        fc=np.array(cfg.fc),
        scenario_type=np.array(cfg.scenario_type),
    )

    print(f"样本级数据集已保存：{save_path}")
    print(f"  X_raw 形状：{X_raw.shape}（样本数 × 窗口 × 特征）")
    print(f"  Y_cell 形状：{Y_cell.shape}，类别数：{cfg.num_cells}")
    print(f"  文件大小：{save_path.stat().st_size / 1024 / 1024:.1f} MB")


def save_trajectory_data(
    traj_data_list: List[Dict],
    output_dir: Path,
) -> None:
    """
    保存轨迹级数据到 .npz 文件

    注意：由于每条轨迹的时隙数不同，无法直接用 np.savez 保存为统一数组。
    改为保存为 Python 对象列表（使用 numpy 的 object 数组）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "trajectory_data.npz"

    # 将轨迹数据转换为可保存的格式
    # 每个字段单独保存为 object 数组（因为各轨迹长度不同）
    num_traj = len(traj_data_list)

    # 标量字段
    traj_ids = np.array([td["traj_id"] for td in traj_data_list])
    speed_ms = np.array([td["speed_ms"] for td in traj_data_list])
    speed_kmh = np.array([td["speed_kmh"] for td in traj_data_list])
    num_slots = np.array([td["num_slots"] for td in traj_data_list])
    splits = np.array([td["split"] for td in traj_data_list])
    traj_types = np.array([td["traj_type"] for td in traj_data_list])

    # 数组字段（object 数组，每个元素是不同长度的数组）
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
        los_indicator=to_object_array("los_indicator"),
        serving_raw=to_object_array("serving_raw"),
        serving_l3=to_object_array("serving_l3"),
    )

    print(f"轨迹级数据已保存：{save_path}")
    print(f"  轨迹数：{num_traj}")
    print(f"  文件大小：{save_path.stat().st_size / 1024 / 1024:.1f} MB")


def save_dataset_info(
    X_raw: np.ndarray,
    Y_cell: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: SimConfig,
    elapsed_time: float,
    output_dir: Path,
) -> None:
    """保存数据集统计信息"""
    info_path = output_dir / "dataset_info.txt"

    lines = [
        "=" * 60,
        "Sionna RT 数据集统计信息",
        "=" * 60,
        f"生成时间：{elapsed_time:.1f} 秒",
        "",
        "场景配置：",
        f"  场景类型：{cfg.scenario_type}",
        f"  小区数量：{cfg.num_cells}",
        f"  站间距：{cfg.isd} m",
        f"  基站高度：{cfg.h_bs} m",
        f"  载波频率：{cfg.fc/1e9:.1f} GHz",
        f"  速度场景：{cfg.speeds_kmh} km/h",
        "",
        "数据集规模：",
        f"  总样本数：{len(X_raw)}",
        f"  训练集：{train_mask.sum()} 样本",
        f"  验证集：{val_mask.sum()} 样本",
        f"  测试集：{test_mask.sum()} 样本",
        f"  特征维度：{X_raw.shape[2]}（= 9 × {cfg.num_cells} 小区）",
        f"  窗口大小：{cfg.window_size} slots（{cfg.window_size*cfg.slot_duration*1000:.0f} ms）",
        f"  预测时间窗口：{cfg.pred_horizon} slots（{cfg.pred_horizon*cfg.slot_duration*1000:.0f} ms）",
        "",
        "标签分布（测试集）：",
    ]

    # 标签分布
    y_test = Y_cell[test_mask]
    for c in range(cfg.num_cells):
        count = int(np.sum(y_test == c))
        ratio = count / max(len(y_test), 1) * 100
        lines.append(f"  cell_{c}: {count} 样本（{ratio:.1f}%）")

    lines.extend([
        "",
        "特征说明（9 × C 维）：",
        "  [0:C]    RSRP_l3         - L3 滤波后 RSRP [dBm]",
        "  [C:2C]   RSRQ            - 参考信号接收质量 [dB]",
        "  [2C:3C]  SINR            - 信噪干扰比 [dB]",
        "  [3C:4C]  Doppler_est     - Doppler 频移估计 [Hz]",
        "  [4C:5C]  BeamID_norm     - 归一化波束 ID",
        "  [5C:6C]  RSRP_diff       - RSRP 变化率 [dB/slot]",
        "  [6C:7C]  BeamID_diff     - 波束 ID 变化",
        "  [7C:8C]  DelaySpread_norm- 归一化时延扩展",
        "  [8C:9C]  LOS_indicator   - LOS 指示（0/1）",
        "",
        "与 MATLAB 版本的关键区别：",
        "  1. 标签：未来最优 SINR 小区（而非 RSRP 最强小区）",
        "  2. 特征：去掉 GT 速度/方向，增加时延扩展、LOS 指示",
        "  3. 信道：Sionna RT 射线追踪（真实 3D 建筑物）",
        "  4. SINR：只考虑强干扰邻区（RSRP 差距 < 10dB）",
        "  5. 预测时间窗口：5 slots（200ms）而非 20 slots（800ms）",
    ])

    info_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"数据集信息已保存：{info_path}")


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Sionna RT 数据集生成")
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
    args = parser.parse_args()

    # ---- 加载配置 ----
    if args.scenario == "umi":
        cfg = get_umi_config()
    else:
        cfg = get_uma_config()

    if args.num_traj is not None:
        cfg.num_trajectories = args.num_traj

    if args.no_gpu:
        cfg.use_gpu = False

    cfg.output_dir = args.outdir
    output_dir = Path(cfg.output_dir)

    print("=" * 60)
    print("Sionna RT 数据集生成")
    print("=" * 60)
    print(f"场景：{cfg.scenario_type}，{cfg.num_cells} 小区，ISD={cfg.isd}m")
    print(f"轨迹数：{cfg.num_trajectories}，速度：{cfg.speeds_kmh} km/h")
    print(f"特征维度：{cfg.num_features}（= 9 × {cfg.num_cells}）")
    print(f"标签：未来 {cfg.pred_horizon} slots 后的最优 SINR 小区")
    print(f"输出目录：{output_dir.absolute()}")
    print(f"GPU：{'是' if cfg.use_gpu else '否（CPU 模式，速度较慢）'}")
    print()

    start_time = time.time()

    # ---- 初始化场景 ----
    print("步骤 1/4：初始化 Sionna RT 场景...")
    scene_mgr = SceneManager(cfg)
    scene_mgr.setup()

    # 保存场景可视化
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_mgr.visualize(save_path=str(output_dir / "scene_layout.png"))

    # ---- 生成轨迹 ----
    print("\n步骤 2/4：生成 UE 轨迹...")
    trajectories = generate_all_trajectories(
        cfg=cfg,
        bs_positions=scene_mgr.bs_positions_2d,
        scene_bounds=scene_mgr.scene_bounds,
        seed=args.seed,
    )

    # ---- 信道仿真 ----
    print(f"\n步骤 3/4：执行信道仿真（{len(trajectories)} 条轨迹）...")
    print("（使用 Sionna RT 射线追踪，RTX 4060 Ti 约需 1~3 小时）")

    channel_results = []
    for traj_idx, traj in enumerate(tqdm(trajectories, desc="信道仿真")):
        ch = simulate_trajectory_channel(traj, scene_mgr, cfg)
        channel_results.append(ch)

        # 每 10 条轨迹打印一次进度
        if (traj_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (traj_idx + 1) * (len(trajectories) - traj_idx - 1)
            print(
                f"  已完成 {traj_idx+1}/{len(trajectories)} 条，"
                f"已用时 {elapsed/60:.1f} 分钟，"
                f"预计剩余 {remaining/60:.1f} 分钟"
            )

    # ---- 构建数据集 ----
    print("\n步骤 4/4：构建数据集...")

    # 构建样本级数据集
    X_raw, Y_cell, Y_sinr, meta_speed, meta_traj = build_sample_dataset(
        trajectories, channel_results, cfg
    )

    # 数据集划分
    num_speeds = len(cfg.speeds_ms)
    traj_per_speed = cfg.num_trajectories // num_speeds
    train_mask, val_mask, test_mask = create_train_val_test_split(
        meta_traj=meta_traj,
        num_trajectories=cfg.num_trajectories,
        num_speeds=num_speeds,
        traj_per_speed=traj_per_speed,
    )

    # 确定每条轨迹的 split 标签
    split_labels = []
    for traj_idx in range(len(trajectories)):
        # 找该轨迹的任意一个样本，查看其 split
        sample_mask = meta_traj == traj_idx
        if not np.any(sample_mask):
            split_labels.append("unused")
        elif train_mask[sample_mask][0]:
            split_labels.append("train")
        elif val_mask[sample_mask][0]:
            split_labels.append("val")
        else:
            split_labels.append("test")

    # 构建轨迹级数据
    traj_data_list = build_trajectory_data(
        trajectories, channel_results, split_labels, cfg
    )

    # ---- 保存数据集 ----
    save_dataset(
        X_raw, Y_cell, Y_sinr, meta_speed, meta_traj,
        train_mask, val_mask, test_mask,
        cfg, output_dir,
    )
    save_trajectory_data(traj_data_list, output_dir)

    elapsed_time = time.time() - start_time
    save_dataset_info(
        X_raw, Y_cell, train_mask, val_mask, test_mask,
        cfg, elapsed_time, output_dir,
    )

    print("\n" + "=" * 60)
    print(f"数据集生成完成！总用时：{elapsed_time/60:.1f} 分钟")
    print(f"输出目录：{output_dir.absolute()}")
    print("=" * 60)
    print("\n下一步：")
    print("  1. 将 outputs/ 目录复制到公司电脑")
    print("  2. 运行 GRU 训练：python train_gru_sionna.py --dataset outputs/dataset.npz")
    print("  3. 运行策略评估：python replay_gru_sionna.py --trajectory-data outputs/trajectory_data.npz")


if __name__ == "__main__":
    main()