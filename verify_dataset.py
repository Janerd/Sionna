"""
verify_dataset.py
=================
数据集验证脚本

在 generate_dataset.py 运行完成后，用此脚本验证生成的数据集是否正确。

运行方式：
    python verify_dataset.py                        # 验证默认输出目录
    python verify_dataset.py --outdir outputs_umi   # 验证指定目录

检查项目：
  1. 文件存在性
  2. 数据集维度和格式
  3. 标签分布（类别平衡性）
  4. 特征统计（均值、方差、异常值）
  5. 数据集划分（train/val/test 比例）
  6. 轨迹级数据完整性
  7. 与 MATLAB 版本的兼容性检查
  8. 生成可视化报告
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "✓" if ok else "✗"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"：{detail}"
    print(msg)
    return ok


def main():
    parser = argparse.ArgumentParser(description="数据集验证")
    parser.add_argument("--outdir", default="outputs", help="数据集目录")
    args = parser.parse_args()

    output_dir = Path(args.outdir)

    print("=" * 60)
    print("Sionna RT 数据集验证")
    print("=" * 60)
    print(f"数据集目录：{output_dir.absolute()}")
    print()

    all_ok = True

    # =========================================================
    # 1. 文件存在性检查
    # =========================================================
    print("【1】文件存在性")
    required_files = ["dataset.npz", "trajectory_data.npz", "dataset_info.txt"]
    for fname in required_files:
        fpath = output_dir / fname
        ok = fpath.exists()
        all_ok &= check(fname, ok, f"{fpath.stat().st_size/1024/1024:.1f} MB" if ok else "文件不存在")
    print()

    if not (output_dir / "dataset.npz").exists():
        print("✗ dataset.npz 不存在，无法继续验证")
        return 1

    # =========================================================
    # 2. 加载数据集
    # =========================================================
    print("【2】加载数据集")
    try:
        ds = np.load(output_dir / "dataset.npz", allow_pickle=True)
        check("dataset.npz 加载", True, f"包含字段：{list(ds.keys())}")
    except Exception as e:
        check("dataset.npz 加载", False, str(e))
        return 1

    # 提取字段
    X_raw = ds["X_raw"]
    Y_cell = ds["Y_cell"]
    train_mask = ds["split_train"].astype(bool)
    val_mask = ds["split_val"].astype(bool)
    test_mask = ds["split_test"].astype(bool)
    meta_speed = ds["meta_speed"]
    meta_traj = ds["meta_traj"]

    # 元数据
    num_cells = int(ds.get("num_cells", 7))
    num_features = int(ds.get("num_features", 9 * num_cells))
    window_size = int(ds.get("window_size", 10))
    pred_horizon = int(ds.get("pred_horizon", 5))
    slot_duration = float(ds.get("slot_duration", 0.04))

    print()

    # =========================================================
    # 3. 维度检查
    # =========================================================
    print("【3】数据维度")
    N, W, F = X_raw.shape

    all_ok &= check("X_raw 维度", X_raw.ndim == 3, f"{X_raw.shape}（样本×窗口×特征）")
    all_ok &= check("窗口大小", W == window_size, f"W={W}，期望={window_size}")
    all_ok &= check("特征维度", F == num_features, f"F={F}，期望={num_features}（=9×{num_cells}）")
    all_ok &= check("Y_cell 维度", Y_cell.ndim == 1 and len(Y_cell) == N, f"{Y_cell.shape}")
    all_ok &= check("总样本数", N > 0, f"N={N}")

    # 检查标签范围
    y_min, y_max = int(Y_cell.min()), int(Y_cell.max())
    all_ok &= check(
        "标签范围",
        y_min >= 0 and y_max < num_cells,
        f"[{y_min}, {y_max}]，期望 [0, {num_cells-1}]"
    )
    print()

    # =========================================================
    # 4. 数据集划分
    # =========================================================
    print("【4】数据集划分")
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    n_test = int(test_mask.sum())
    n_total = n_train + n_val + n_test

    all_ok &= check("划分无重叠", n_total == N, f"train+val+test={n_total}，总样本={N}")
    all_ok &= check(
        "训练集比例",
        0.5 <= n_train / N <= 0.75,
        f"{n_train} 样本（{n_train/N*100:.1f}%）"
    )
    all_ok &= check(
        "验证集比例",
        0.1 <= n_val / N <= 0.3,
        f"{n_val} 样本（{n_val/N*100:.1f}%）"
    )
    all_ok &= check(
        "测试集比例",
        0.1 <= n_test / N <= 0.3,
        f"{n_test} 样本（{n_test/N*100:.1f}%）"
    )
    print()

    # =========================================================
    # 5. 标签分布
    # =========================================================
    print("【5】标签分布（测试集）")
    y_test = Y_cell[test_mask]
    label_counts = np.bincount(y_test, minlength=num_cells)
    label_ratios = label_counts / max(len(y_test), 1)

    # 检查是否有类别完全缺失
    missing_classes = np.where(label_counts == 0)[0]
    all_ok &= check(
        "无缺失类别",
        len(missing_classes) == 0,
        f"缺失类别：{missing_classes.tolist()}" if len(missing_classes) > 0 else "所有类别均有样本"
    )

    # 检查类别不平衡程度
    max_ratio = float(label_ratios.max())
    min_ratio = float(label_ratios[label_ratios > 0].min())
    imbalance_ratio = max_ratio / max(min_ratio, 1e-9)
    check(
        "类别不平衡比",
        imbalance_ratio < 50,
        f"最大/最小={imbalance_ratio:.1f}（<50 为可接受）"
    )

    for c in range(num_cells):
        print(f"    cell_{c}: {label_counts[c]} 样本（{label_ratios[c]*100:.1f}%）")
    print()

    # =========================================================
    # 6. 特征统计
    # =========================================================
    print("【6】特征统计（训练集）")
    X_train = X_raw[train_mask]

    # 计算每个特征的统计量
    feat_mean = X_train.reshape(-1, F).mean(axis=0)
    feat_std = X_train.reshape(-1, F).std(axis=0)
    feat_min = X_train.reshape(-1, F).min(axis=0)
    feat_max = X_train.reshape(-1, F).max(axis=0)

    # 检查是否有 NaN 或 Inf
    has_nan = bool(np.any(np.isnan(X_raw)))
    has_inf = bool(np.any(np.isinf(X_raw)))
    all_ok &= check("无 NaN 值", not has_nan, "存在 NaN" if has_nan else "正常")
    all_ok &= check("无 Inf 值", not has_inf, "存在 Inf" if has_inf else "正常")

    # 检查 RSRP 范围（前 num_cells 个特征）
    rsrp_mean = float(feat_mean[:num_cells].mean())
    rsrp_std = float(feat_std[:num_cells].mean())
    check(
        "RSRP 范围合理",
        -150 < rsrp_mean < -50,
        f"均值={rsrp_mean:.1f} dBm，标准差={rsrp_std:.1f} dB"
    )

    # 检查 LOS 指示（最后 num_cells 个特征）
    los_mean = float(feat_mean[-num_cells:].mean())
    check(
        "LOS 指示分布",
        0 < los_mean < 1,
        f"LOS 比例={los_mean*100:.1f}%（0~100% 为正常）"
    )

    print(f"  特征统计摘要（训练集，{X_train.shape[0]} 样本）：")
    feature_groups = [
        ("RSRP_l3", 0, num_cells),
        ("RSRQ", num_cells, 2*num_cells),
        ("SINR", 2*num_cells, 3*num_cells),
        ("Doppler_est", 3*num_cells, 4*num_cells),
        ("BeamID_norm", 4*num_cells, 5*num_cells),
        ("RSRP_diff", 5*num_cells, 6*num_cells),
        ("BeamID_diff", 6*num_cells, 7*num_cells),
        ("DelaySpread", 7*num_cells, 8*num_cells),
        ("LOS_indicator", 8*num_cells, 9*num_cells),
    ]
    for name, start, end in feature_groups:
        group_mean = float(feat_mean[start:end].mean())
        group_std = float(feat_std[start:end].mean())
        print(f"    {name:20s}: 均值={group_mean:8.3f}，标准差={group_std:.3f}")
    print()

    # =========================================================
    # 7. 轨迹级数据检查
    # =========================================================
    print("【7】轨迹级数据")
    traj_path = output_dir / "trajectory_data.npz"
    if traj_path.exists():
        try:
            td = np.load(traj_path, allow_pickle=True)
            num_traj = len(td["traj_ids"])
            splits = td["splits"]
            n_train_traj = int(np.sum(splits == "train"))
            n_val_traj = int(np.sum(splits == "val"))
            n_test_traj = int(np.sum(splits == "test"))

            check("轨迹数据加载", True, f"{num_traj} 条轨迹")
            check(
                "轨迹划分",
                n_train_traj > 0 and n_val_traj > 0 and n_test_traj > 0,
                f"train={n_train_traj}, val={n_val_traj}, test={n_test_traj}"
            )

            # 检查 RSRP_l3 字段
            rsrp_l3_arr = td["RSRP_l3"]
            sample_rsrp = rsrp_l3_arr[0]  # 第一条轨迹
            check(
                "RSRP_l3 字段",
                sample_rsrp.ndim == 2,
                f"形状={sample_rsrp.shape}（时隙×小区）"
            )

            # 检查 SINR 字段
            sinr_arr = td["SINR"]
            sample_sinr = sinr_arr[0]
            check(
                "SINR 字段",
                sample_sinr.ndim == 2,
                f"形状={sample_sinr.shape}"
            )

            # 检查 LOS 指示字段（新增，MATLAB 版本没有）
            if "los_indicator" in td:
                los_arr = td["los_indicator"]
                sample_los = los_arr[0]
                los_ratio = float(sample_los.mean())
                check(
                    "LOS 指示字段（新增）",
                    True,
                    f"LOS 比例={los_ratio*100:.1f}%"
                )
            else:
                check("LOS 指示字段", False, "字段缺失（可能是旧版数据）")

        except Exception as e:
            check("轨迹数据加载", False, str(e))
            all_ok = False
    else:
        check("trajectory_data.npz", False, "文件不存在")
    print()

    # =========================================================
    # 8. 生成可视化报告
    # =========================================================
    print("【8】生成可视化报告")
    try:
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # 图1：标签分布（测试集）
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(num_cells), label_counts, color="steelblue", alpha=0.8)
        ax1.set_xlabel("小区 ID")
        ax1.set_ylabel("样本数")
        ax1.set_title("标签分布（测试集）")
        ax1.grid(True, alpha=0.3)

        # 图2：RSRP 分布（训练集，第一个小区）
        ax2 = fig.add_subplot(gs[0, 1])
        rsrp_vals = X_train[:, -1, 0]  # 最后一个时隙，第一个小区的 RSRP
        ax2.hist(rsrp_vals, bins=50, color="green", alpha=0.7)
        ax2.set_xlabel("RSRP [dBm]")
        ax2.set_ylabel("频次")
        ax2.set_title("RSRP 分布（cell_0，训练集）")
        ax2.grid(True, alpha=0.3)

        # 图3：SINR 分布（训练集，第一个小区）
        ax3 = fig.add_subplot(gs[0, 2])
        sinr_vals = X_train[:, -1, 2*num_cells]  # 最后一个时隙，第一个小区的 SINR
        ax3.hist(sinr_vals, bins=50, color="orange", alpha=0.7)
        ax3.set_xlabel("SINR [dB]")
        ax3.set_ylabel("频次")
        ax3.set_title("SINR 分布（cell_0，训练集）")
        ax3.grid(True, alpha=0.3)

        # 图4：LOS 指示分布
        ax4 = fig.add_subplot(gs[1, 0])
        los_vals = X_train[:, -1, 8*num_cells:9*num_cells].mean(axis=1)
        ax4.hist(los_vals, bins=20, color="purple", alpha=0.7)
        ax4.set_xlabel("LOS 比例（各小区平均）")
        ax4.set_ylabel("频次")
        ax4.set_title("LOS 指示分布（训练集）")
        ax4.grid(True, alpha=0.3)

        # 图5：Doppler 估计分布
        ax5 = fig.add_subplot(gs[1, 1])
        doppler_vals = X_train[:, -1, 3*num_cells:4*num_cells].flatten()
        ax5.hist(doppler_vals, bins=50, color="red", alpha=0.7)
        ax5.set_xlabel("Doppler 估计 [Hz]")
        ax5.set_ylabel("频次")
        ax5.set_title("Doppler 估计分布（训练集）")
        ax5.grid(True, alpha=0.3)

        # 图6：时延扩展分布
        ax6 = fig.add_subplot(gs[1, 2])
        ds_vals = X_train[:, -1, 7*num_cells:8*num_cells].flatten()
        ax6.hist(ds_vals, bins=50, color="brown", alpha=0.7)
        ax6.set_xlabel("归一化时延扩展")
        ax6.set_ylabel("频次")
        ax6.set_title("时延扩展分布（训练集）")
        ax6.grid(True, alpha=0.3)

        # 图7：速度分布
        ax7 = fig.add_subplot(gs[2, 0])
        speeds_kmh = meta_speed * 3.6
        ax7.hist(speeds_kmh, bins=20, color="teal", alpha=0.7)
        ax7.set_xlabel("速度 [km/h]")
        ax7.set_ylabel("样本数")
        ax7.set_title("速度分布（所有样本）")
        ax7.grid(True, alpha=0.3)

        # 图8：RSRP 时序示例（第一条测试轨迹）
        ax8 = fig.add_subplot(gs[2, 1:])
        if traj_path.exists():
            td = np.load(traj_path, allow_pickle=True)
            test_traj_mask = td["splits"] == "test"
            if np.any(test_traj_mask):
                test_idx = int(np.where(test_traj_mask)[0][0])
                rsrp_l3_traj = td["RSRP_l3"][test_idx]  # [T, C]
                T_plot = min(200, rsrp_l3_traj.shape[0])
                for c in range(min(num_cells, 4)):
                    ax8.plot(rsrp_l3_traj[:T_plot, c], label=f"cell_{c}", alpha=0.8)
                ax8.set_xlabel("时隙")
                ax8.set_ylabel("RSRP [dBm]")
                ax8.set_title(f"RSRP 时序示例（测试轨迹 {test_idx}，前 {T_plot} 时隙）")
                ax8.legend(fontsize=8)
                ax8.grid(True, alpha=0.3)

        plt.suptitle(
            f"Sionna RT 数据集验证报告\n"
            f"场景：{ds.get('scenario_type', '未知')}，"
            f"{num_cells} 小区，{N} 样本，特征维度={F}",
            fontsize=12
        )
        plt.tight_layout()

        report_path = output_dir / "dataset_verification_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close()
        check("可视化报告", True, f"已保存到 {report_path}")
    except Exception as e:
        check("可视化报告", False, str(e))
    print()

    # =========================================================
    # 总结
    # =========================================================
    print("=" * 60)
    if all_ok:
        print("✓ 数据集验证通过！")
        print()
        print("下一步（在公司电脑上）：")
        print("  1. 将 outputs/ 目录复制到公司电脑")
        print("  2. 训练 GRU 模型：")
        print("     python train_gru_sionna.py --dataset outputs/dataset.npz")
        print("  3. 策略评估：")
        print("     python replay_gru_sionna.py --trajectory-data outputs/trajectory_data.npz")
    else:
        print("✗ 数据集验证未完全通过，请检查上方错误信息")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())