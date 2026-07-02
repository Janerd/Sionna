"""
visualize_scene_3d.py
=====================
场景、基站和 UE 轨迹可视化脚本

功能：
  1. 加载已生成的轨迹数据（outputs/trajectory_data.npz）
  2. 生成 2D 俯视图：建筑物轮廓 + 基站位置 + UE 轨迹
  3. 支持按速度筛选轨迹
  4. 支持保存图片
  5. 打印基站坐标，方便手动调整后写入 config.py

使用方法：
    # 显示所有轨迹（需要 Sionna 加载场景）
    python visualize_scene_3d.py

    # 只显示 30 km/h 轨迹
    python visualize_scene_3d.py --speed 30

    # 保存图片到 outputs/
    python visualize_scene_3d.py --save

    # 不加载 Sionna 场景（只显示基站和轨迹，无建筑物背景）
    python visualize_scene_3d.py --no-scene

    # 指定数据目录
    python visualize_scene_3d.py --datadir C:/PC_Simu/Sionna/outputs

注意：
    - 加载 Sionna 场景需要在家用电脑（有 Sionna 安装）上运行
    - 如果没有 Sionna，使用 --no-scene 参数仍可显示基站和轨迹
    - 图片保存到 outputs/ 目录
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 配置 matplotlib 字体（避免中文字体警告）
def _setup_font():
    chinese_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
    available = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_font()

# Sionna 导入（可选）
try:
    import sionna
    import sionna.rt as srt
    from sionna.rt import load_scene, PlanarArray, PathSolver
    SIONNA_AVAILABLE = True
    SIONNA_VERSION = sionna.__version__
except ImportError:
    SIONNA_AVAILABLE = False
    SIONNA_VERSION = "未安装"

from config import SimConfig, get_umi_config
from scene_setup import compute_hexagonal_bs_positions, get_bs_positions


# =========================================================
# 颜色配置
# =========================================================

# 不同速度的轨迹颜色
SPEED_COLORS = {
    30.0:  "#2196F3",   # 蓝色：30 km/h（行人/自行车）
    60.0:  "#FF9800",   # 橙色：60 km/h（城市车辆）
    120.0: "#F44336",   # 红色：120 km/h（高速车辆）
}

# 不同轨迹类型的线型
TRAJ_LINESTYLES = {
    "munich_walk": "-",
    "street_grid": "--",
    "arc":         "-.",
    "stop_and_go": ":",
    "linear":      "-",
}


# =========================================================
# 数据加载
# =========================================================

def load_trajectory_data(datadir: str) -> dict:
    """
    加载轨迹数据

    参数：
        datadir: 数据目录路径

    返回：
        包含轨迹数据的字典
    """
    traj_path = Path(datadir) / "trajectory_data.npz"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"轨迹数据文件不存在：{traj_path}\n"
            "请先运行 python generate_dataset.py 生成数据集"
        )

    print(f"加载轨迹数据：{traj_path}")
    data = np.load(traj_path, allow_pickle=True)

    num_traj = len(data["traj_ids"])
    print(f"  轨迹数：{num_traj}")
    print(f"  速度分布：{np.unique(data['speed_kmh']):.0f} km/h" if data['speed_kmh'].ndim == 0
          else f"  速度分布：{np.unique(data['speed_kmh'])} km/h")

    return data


def get_trajectories_by_speed(
    data: dict,
    speed_filter: Optional[float] = None,
    max_per_speed: int = 20,
) -> List[dict]:
    """
    从数据中提取轨迹列表

    参数：
        data:          轨迹数据字典
        speed_filter:  速度筛选（km/h），None 表示所有速度
        max_per_speed: 每种速度最多显示的轨迹数

    返回：
        轨迹列表，每个元素包含 pos, speed_kmh, traj_type
    """
    trajectories = []
    speed_counts = {}

    for i in range(len(data["traj_ids"])):
        speed_kmh = float(data["speed_kmh"][i])
        traj_type = str(data["traj_types"][i])
        pos = data["pos"][i]  # [T, 2]

        # 速度筛选
        if speed_filter is not None and abs(speed_kmh - speed_filter) > 1.0:
            continue

        # 每种速度最多显示 max_per_speed 条
        if speed_counts.get(speed_kmh, 0) >= max_per_speed:
            continue

        speed_counts[speed_kmh] = speed_counts.get(speed_kmh, 0) + 1

        trajectories.append({
            "pos": pos,
            "speed_kmh": speed_kmh,
            "traj_type": traj_type,
            "traj_id": int(data["traj_ids"][i]),
        })

    print(f"  筛选后轨迹数：{len(trajectories)}")
    for speed, count in sorted(speed_counts.items()):
        print(f"    {speed:.0f} km/h: {count} 条")

    return trajectories


# =========================================================
# 场景建筑物轮廓提取
# =========================================================

def extract_building_outlines(scene) -> List[np.ndarray]:
    """
    从 Sionna 场景中提取建筑物轮廓（2D 投影）

    参数：
        scene: Sionna Scene 对象

    返回：
        建筑物轮廓列表，每个元素是 [N, 2] 的顶点数组
    """
    outlines = []

    try:
        for obj_name, obj in scene.objects.items():
            try:
                verts = obj.vertices
                if hasattr(verts, 'numpy'):
                    verts = verts.numpy()
                else:
                    verts = np.array(verts)

                if verts.ndim != 2 or verts.shape[1] < 2 or len(verts) < 3:
                    continue

                # 过滤地面（z 坐标接近 0 的大面积物体）
                z_vals = verts[:, 2] if verts.shape[1] >= 3 else np.zeros(len(verts))
                if np.max(z_vals) < 0.5:
                    continue

                xy = verts[:, :2]

                # 用凸包近似建筑物轮廓
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(xy)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    outlines.append(xy[hull_pts])
                except Exception:
                    pass

            except Exception:
                pass

    except Exception as e:
        print(f"  建筑物轮廓提取失败：{e}")

    return outlines


# =========================================================
# 主可视化函数
# =========================================================

def visualize_scene_and_trajectories(
    cfg: SimConfig,
    trajectories: List[dict],
    scene=None,
    speed_filter: Optional[float] = None,
    save_path: Optional[str] = None,
    show_coverage: bool = True,
    show_start_end: bool = True,
    alpha_traj: float = 0.6,
) -> None:
    """
    生成场景 + 基站 + UE 轨迹的 2D 俯视图

    参数：
        cfg:            仿真配置
        trajectories:   轨迹列表
        scene:          Sionna Scene 对象（可选，用于显示建筑物）
        speed_filter:   速度筛选（km/h），None 表示所有速度
        save_path:      保存路径（None 则显示交互式窗口）
        show_coverage:  是否显示小区覆盖范围
        show_start_end: 是否显示轨迹起终点
        alpha_traj:     轨迹透明度
    """
    # 获取场景边界
    scene_bounds = (-500.0, 500.0, -500.0, 500.0)
    if scene is not None:
        try:
            bbox = scene.bbox
            scene_bounds = (
                float(bbox[0][0]), float(bbox[1][0]),
                float(bbox[0][1]), float(bbox[1][1]),
            )
        except Exception:
            pass

    xmin, xmax, ymin, ymax = scene_bounds
    scene_width = xmax - xmin
    scene_height = ymax - ymin

    # 图形尺寸：确保场景完整显示，最小 16×16 英寸
    fig_w = max(16, scene_width / 40)
    fig_h = max(16, scene_height / 40)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # ---- 1. 绘制建筑物轮廓 ----
    building_drawn = False
    if scene is not None:
        # 优先使用 Mitsuba API（与 network_config_tool.py 一致，更准确）
        try:
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.collections import PatchCollection
            all_polygons = []
            for mesh in scene.mi_scene.shapes():
                try:
                    verts = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
                    faces = mesh.faces_buffer().numpy().reshape(-1, 3)
                    if len(verts) < 3 or len(faces) == 0:
                        continue
                    if np.max(verts[:, 2]) < 0.5:
                        continue
                    for face in faces:
                        tri_xy = verts[face, :2]
                        poly = MplPolygon(tri_xy, closed=True)
                        all_polygons.append(poly)
                except Exception:
                    pass
            if all_polygons:
                pc = PatchCollection(all_polygons, alpha=0.3, facecolor="#A0A0A0",
                                     edgecolor="#606060", linewidth=0.3, zorder=1)
                ax.add_collection(pc)
                print(f"  已绘制建筑物投影（{len(all_polygons)} 个三角面）")
                building_drawn = True
        except Exception as e:
            print(f"  Mitsuba API 失败（{e}），尝试备用方法...")

        # 备用：凸包方法
        if not building_drawn:
            outlines = extract_building_outlines(scene)
            if outlines:
                for outline in outlines:
                    ax.fill(outline[:, 0], outline[:, 1],
                            alpha=0.25, color="#A0A0A0", zorder=1)
                    ax.plot(outline[:, 0], outline[:, 1],
                            "-", color="#606060", linewidth=0.5, alpha=0.6, zorder=2)
                print(f"  已绘制 {len(outlines)} 个建筑物轮廓（凸包方法）")
                building_drawn = True

    if not building_drawn:
        # 无场景时显示场景边界框
        rect = plt.Rectangle((xmin, ymin), scene_width, scene_height,
                              fill=False, linestyle="--", color="#888888",
                              linewidth=1.5, alpha=0.7, zorder=1)
        ax.add_patch(rect)
        ax.text(xmin + 10, ymax - 30, "Munich Scene boundary (±500m)",
                fontsize=10, color="#888888", va="top")

    # ---- 2. 绘制小区覆盖范围 ----
    bs_positions = get_bs_positions(cfg)
    if show_coverage:
        for c in range(cfg.num_cells):
            pos = bs_positions[c]
            circle = plt.Circle((pos[0], pos[1]), cfg.isd / 2,
                                 fill=True, facecolor="#E3F2FD",
                                 linestyle="--", edgecolor="#1565C0",
                                 alpha=0.15, linewidth=1.2, zorder=3)
            ax.add_patch(circle)

    # ---- 3. 绘制 UE 轨迹 ----
    speeds_shown = set()
    for traj in trajectories:
        speed = traj["speed_kmh"]
        traj_type = traj["traj_type"]
        pos = traj["pos"]  # [T, 2]

        color = SPEED_COLORS.get(speed, "#9E9E9E")
        linestyle = TRAJ_LINESTYLES.get(traj_type, "-")

        ax.plot(pos[:, 0], pos[:, 1],
                color=color, linestyle=linestyle,
                linewidth=1.2, alpha=alpha_traj, zorder=4)

        if show_start_end:
            ax.plot(pos[0, 0], pos[0, 1], "o",
                    color=color, markersize=6, alpha=0.9, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(pos[-1, 0], pos[-1, 1], "s",
                    color=color, markersize=6, alpha=0.9, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.5)

        speeds_shown.add(speed)

    # ---- 4. 绘制基站位置 ----
    for c in range(cfg.num_cells):
        pos = bs_positions[c]
        ax.plot(pos[0], pos[1], "r^",
                markersize=16, zorder=10, markeredgecolor="darkred",
                markeredgewidth=1.0)
        ax.annotate(
            f"BS{c}",
            xy=(pos[0], pos[1]),
            xytext=(pos[0] + 12, pos[1] + 12),
            fontsize=10, fontweight="bold", color="darkred",
            zorder=11,
        )

    # ---- 5. 坐标轴和标注 ----
    ax.set_xlabel("X [m]  (East →)", fontsize=13)
    ax.set_ylabel("Y [m]  (North ↑)", fontsize=13)
    ax.tick_params(labelsize=11)

    speed_str = f"{speed_filter:.0f} km/h" if speed_filter else "all speeds"
    ax.set_title(
        f"Munich Scene — BS Layout & UE Trajectories\n"
        f"{cfg.num_cells} BSs, ISD={cfg.isd}m, h_BS={cfg.h_bs}m  |  "
        f"Trajectories: {len(trajectories)} ({speed_str})",
        fontsize=13,
    )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, zorder=0, linewidth=0.8)

    # 坐标轴范围：场景边界 + 5% 边距
    pad_x = scene_width * 0.05
    pad_y = scene_height * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    # 中心十字线
    ax.axhline(y=0, color="k", linewidth=0.8, alpha=0.3, zorder=0)
    ax.axvline(x=0, color="k", linewidth=0.8, alpha=0.3, zorder=0)

    # ---- 6. 图例 ----
    legend_elements = []

    # 基站
    legend_elements.append(
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markeredgecolor="darkred", markersize=13, label="Base Station")
    )

    # 速度颜色
    for speed in sorted(speeds_shown):
        color = SPEED_COLORS.get(speed, "#9E9E9E")
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2,
                   label=f"UE {speed:.0f} km/h")
        )

    # 轨迹起终点
    if show_start_end:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=6, label="Traj start")
        )
        legend_elements.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markersize=6, label="Traj end")
        )

    # 覆盖范围
    if show_coverage:
        legend_elements.append(
            mpatches.Patch(facecolor="#E3F2FD", edgecolor="#1565C0",
                           linestyle="--", alpha=0.5, label=f"Coverage (ISD/2={cfg.isd/2:.0f}m)")
        )

    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存到：{save_path}")
    else:
        plt.show()

    plt.close()


# =========================================================
# 统计信息打印
# =========================================================

def print_statistics(cfg: SimConfig, trajectories: List[dict]) -> None:
    """打印轨迹统计信息"""
    print("\n" + "=" * 60)
    print("轨迹统计信息")
    print("=" * 60)

    bs_positions = get_bs_positions(cfg)

    # 按速度统计
    speed_stats = {}
    for traj in trajectories:
        speed = traj["speed_kmh"]
        if speed not in speed_stats:
            speed_stats[speed] = {"count": 0, "total_dist": 0, "total_slots": 0}
        pos = traj["pos"]
        dist = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        speed_stats[speed]["count"] += 1
        speed_stats[speed]["total_dist"] += dist
        speed_stats[speed]["total_slots"] += len(pos)

    for speed in sorted(speed_stats.keys()):
        s = speed_stats[speed]
        avg_dist = s["total_dist"] / s["count"]
        avg_slots = s["total_slots"] / s["count"]
        print(f"\n{speed:.0f} km/h ({s['count']} 条轨迹):")
        print(f"  平均距离：{avg_dist:.0f} m")
        print(f"  平均时隙：{avg_slots:.0f} slots ({avg_slots * 0.04:.1f} s)")

    # 基站位置
    print(f"\n基站位置（{cfg.num_cells} 个，ISD={cfg.isd}m）：")
    print("  可复制到 config.py 的 bs_positions_override：")
    print("  bs_positions_override = np.array([")
    for c in range(cfg.num_cells):
        pos = bs_positions[c]
        print(f"      [{pos[0]:.1f}, {pos[1]:.1f}],  # BS{c}")
    print("  ], dtype=np.float32)")

    print("=" * 60)


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="场景、基站和 UE 轨迹可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python visualize_scene_3d.py                    # 显示所有轨迹
  python visualize_scene_3d.py --speed 30         # 只显示 30 km/h 轨迹
  python visualize_scene_3d.py --speed 120        # 只显示 120 km/h 轨迹
  python visualize_scene_3d.py --save             # 保存图片
  python visualize_scene_3d.py --no-scene         # 不加载 Sionna 场景
  python visualize_scene_3d.py --max-per-speed 5  # 每种速度最多显示 5 条
        """
    )
    parser.add_argument(
        "--datadir", default="outputs",
        help="数据目录（默认：outputs）"
    )
    parser.add_argument(
        "--speed", type=float, default=None,
        help="速度筛选 [km/h]，例如 30, 60, 120（默认：显示所有速度）"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存图片到 datadir/visualization/ 目录"
    )
    parser.add_argument(
        "--no-scene", action="store_true",
        help="不加载 Sionna 场景（只显示基站和轨迹，无建筑物背景）"
    )
    parser.add_argument(
        "--max-per-speed", type=int, default=20,
        help="每种速度最多显示的轨迹数（默认：20）"
    )
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="不显示小区覆盖范围"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="轨迹透明度（0~1，默认：0.5）"
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    cfg = get_umi_config()
    print(f"配置：{cfg.num_cells} 个基站，ISD={cfg.isd}m")

    # ---- 加载轨迹数据 ----
    try:
        data = load_trajectory_data(args.datadir)
    except FileNotFoundError as e:
        print(f"错误：{e}")
        return

    trajectories = get_trajectories_by_speed(
        data,
        speed_filter=args.speed,
        max_per_speed=args.max_per_speed,
    )

    if not trajectories:
        print("没有找到符合条件的轨迹")
        return

    # ---- 打印统计信息 ----
    print_statistics(cfg, trajectories)

    # ---- 加载 Sionna 场景（可选）----
    scene = None
    if not args.no_scene:
        if SIONNA_AVAILABLE:
            print(f"\n加载 Sionna {SIONNA_VERSION} 场景...")
            try:
                from sionna.rt import load_scene, PlanarArray
                scene = load_scene(getattr(srt.scene, "munich"))

                # 设置天线阵列（PathSolver 需要）
                scene.tx_array = PlanarArray(
                    num_rows=1, num_cols=1,
                    vertical_spacing=0.5, horizontal_spacing=0.5,
                    pattern="iso", polarization="V",
                )
                scene.rx_array = PlanarArray(
                    num_rows=1, num_cols=1,
                    vertical_spacing=0.5, horizontal_spacing=0.5,
                    pattern="iso", polarization="V",
                )
                scene.frequency = cfg.fc
                print("场景加载完成")
            except Exception as e:
                print(f"场景加载失败：{e}")
                print("将不显示建筑物背景")
                scene = None
        else:
            print(f"Sionna 未安装（{SIONNA_VERSION}），不显示建筑物背景")
            print("如需显示建筑物，请安装 Sionna：pip install sionna>=2.0")

    # ---- 生成可视化 ----
    save_path = None
    if args.save:
        vis_dir = Path(args.datadir) / "visualization"
        vis_dir.mkdir(parents=True, exist_ok=True)
        speed_str = f"_{args.speed:.0f}kmh" if args.speed else "_all"
        scene_str = "_no_scene" if scene is None else ""
        save_path = str(vis_dir / f"scene_trajectories{speed_str}{scene_str}.png")

    print(f"\n生成可视化图...")
    visualize_scene_and_trajectories(
        cfg=cfg,
        trajectories=trajectories,
        scene=scene,
        speed_filter=args.speed,
        save_path=save_path,
        show_coverage=not args.no_coverage,
        show_start_end=True,
        alpha_traj=args.alpha,
    )

    if save_path:
        print(f"\n完成！图片已保存到：{save_path}")
    else:
        print("\n完成！")


if __name__ == "__main__":
    main()