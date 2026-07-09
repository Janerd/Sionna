"""
network_config_tool.py
======================
网络配置工具：交互式配置基站位置、邻区关系和 UE 轨迹预览

两种运行模式：

模式 1：GUI 模式（需要 polyscope，在家用电脑运行）
    python network_config_tool.py --gui
    - 启动官方 Sionna RT GUI，加载慕尼黑 3D 场景
    - 用 Ctrl + 左键在场景中放置基站
    - 关闭 GUI 后自动读取基站坐标
    - 计算邻区关系，预览 UE 轨迹
    - 保存到 network_config.json

模式 2：无 GUI 模式（只需要 matplotlib，在公司电脑也可运行）
    python network_config_tool.py --no-gui
    - 读取 network_config.json 中的基站位置（或六边形网格默认值）
    - 计算邻区关系
    - 预览 UE 轨迹
    - 更新 network_config.json

安装依赖（家用电脑）：
    pip install "polyscope>=2.6.0,<2.7.0" omegaconf

工作流程：
    1. 运行 --gui 模式，在 3D 场景中放置基站
    2. 关闭 GUI，查看邻区关系和轨迹预览
    3. 如需调整，修改 network_config.json 后重新运行 --no-gui
    4. 确认后运行 python generate_dataset.py 开始仿真
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 配置 matplotlib 字体
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

# =========================================================
# 配置文件路径
# =========================================================

NETWORK_CONFIG_PATH = Path(__file__).parent / "network_config.json"
SCENE_BOUNDS = (-500.0, 500.0, -500.0, 500.0)  # 慕尼黑场景范围


# =========================================================
# 配置文件读写
# =========================================================

def load_network_config() -> dict:
    """加载 network_config.json"""
    if NETWORK_CONFIG_PATH.exists():
        with open(NETWORK_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_network_config(cfg: dict) -> None:
    """保存 network_config.json"""
    with open(NETWORK_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"配置已保存到：{NETWORK_CONFIG_PATH}")


def get_bs_positions_from_config(cfg: dict) -> np.ndarray:
    """从配置文件提取基站位置 [num_cells, 2]"""
    positions = cfg.get("bs_config", {}).get("positions", [])
    if not positions:
        # 使用六边形网格默认值
        from scene_setup import compute_hexagonal_bs_positions
        from config import get_umi_config
        sim_cfg = get_umi_config()
        return compute_hexagonal_bs_positions(sim_cfg.num_cells, sim_cfg.isd)

    return np.array([[p["x"], p["y"]] for p in positions], dtype=np.float32)


# =========================================================
# 邻区关系计算
# =========================================================

def compute_neighbor_relations(
    bs_positions: np.ndarray,
    max_neighbors: int = 6,
    method: str = "distance",
    isd: float = 250.0,
) -> Dict[str, List[int]]:
    """
    计算邻区关系

    参数：
        bs_positions: [num_cells, 2] 基站 2D 坐标
        max_neighbors: 每个基站的最大邻区数
        method: "distance"（基于距离）
        isd: 站间距 [m]，用于确定邻区范围

    返回：
        邻区关系字典，key 为基站 ID（字符串），value 为邻区 ID 列表
    """
    num_cells = len(bs_positions)
    relations = {}

    # 邻区范围：2.5 个 ISD 以内的基站
    max_dist = isd * 2.5

    for i in range(num_cells):
        dists = np.linalg.norm(bs_positions - bs_positions[i], axis=1)
        dists[i] = np.inf  # 排除自身

        # 只考虑距离在 max_dist 以内的基站
        candidates = np.where(dists < max_dist)[0]

        # 按距离排序，取最近的 max_neighbors 个
        sorted_candidates = candidates[np.argsort(dists[candidates])]
        neighbors = sorted_candidates[:max_neighbors].tolist()

        relations[str(i)] = neighbors

    return relations


def print_neighbor_stats(relations: Dict[str, List[int]]) -> None:
    """打印邻区关系统计"""
    print("\n邻区关系统计：")
    neighbor_counts = [len(v) for v in relations.values()]
    print(f"  平均邻区数：{np.mean(neighbor_counts):.1f}")
    print(f"  最少邻区数：{min(neighbor_counts)}")
    print(f"  最多邻区数：{max(neighbor_counts)}")
    print(f"  总射线追踪基站数（每时隙）：{np.mean(neighbor_counts) + 1:.1f}（服务小区 + 邻区）")
    print(f"  相比全量计算的加速比：{len(relations) / (np.mean(neighbor_counts) + 1):.1f}x")


# =========================================================
# UE 轨迹预生成
# =========================================================

def generate_preview_trajectories(
    bs_positions: np.ndarray,
    scene_bounds: Tuple[float, float, float, float],
    num_per_speed: int = 3,
    speeds_kmh: List[float] = None,
    seed: int = 42,
    walkable_grid: Optional[dict] = None,
) -> List[dict]:
    """
    预生成少量示例轨迹用于可视化

    参数：
        bs_positions:  [num_cells, 2] 基站位置
        scene_bounds:  场景边界
        num_per_speed: 每种速度生成的轨迹数
        speeds_kmh:    速度列表
        seed:          随机种子
        walkable_grid: 可选，来自 compute_walkable_grid()
                       有 Sionna 时传入，中高速轨迹启用碰撞检测

    返回：
        轨迹列表
    """
    if speeds_kmh is None:
        speeds_kmh = [30.0, 60.0, 120.0]

    # 导入轨迹生成函数
    try:
        from trajectory import generate_trajectory, _get_traj_types_for_speed
        from config import get_umi_config
        cfg = get_umi_config()
    except ImportError:
        print("警告：无法导入轨迹生成模块，跳过轨迹预览")
        return []

    rng = np.random.default_rng(seed)
    trajectories = []

    for speed_kmh in speeds_kmh:
        speed_ms = speed_kmh / 3.6
        # 按速度选择轨迹类型（与实际仿真一致）
        traj_types_for_speed = _get_traj_types_for_speed(speed_kmh)
        # 取前 num_per_speed 种不重复的类型
        seen = set()
        preview_types = []
        for t in traj_types_for_speed:
            if t not in seen:
                seen.add(t)
                preview_types.append(t)
            if len(preview_types) >= num_per_speed:
                break

        # 中高速（>30 km/h）使用碰撞检测
        use_walkable = walkable_grid if speed_kmh > 30.0 else None

        for traj_type in preview_types:
            try:
                traj = generate_trajectory(
                    cfg=cfg,
                    speed_ms=speed_ms,
                    traj_type=traj_type,
                    scene_bounds=scene_bounds,
                    bs_positions=bs_positions,
                    rng=rng,
                    walkable_grid=use_walkable,
                )
                trajectories.append({
                    "pos": traj.pos,
                    "speed_kmh": speed_kmh,
                    "traj_type": traj_type,
                })
            except Exception as e:
                print(f"  轨迹生成失败（{speed_kmh} km/h, {traj_type}）：{e}")

    return trajectories


# =========================================================
# 可视化
# =========================================================

SPEED_COLORS = {
    30.0:  "#2196F3",
    60.0:  "#FF9800",
    120.0: "#F44336",
}


def visualize_network_config(
    bs_positions: np.ndarray,
    neighbor_relations: Dict[str, List[int]],
    trajectories: List[dict],
    isd: float = 250.0,
    scene_bounds: Tuple[float, float, float, float] = SCENE_BOUNDS,
    save_path: Optional[str] = None,
    scene=None,
) -> None:
    """
    可视化网络配置：建筑物投影 + 基站 + 邻区连线 + UE 轨迹

    参数：
        bs_positions:       [num_cells, 2] 基站位置
        neighbor_relations: 邻区关系字典
        trajectories:       UE 轨迹列表（按速度分色）
        isd:                站间距 [m]
        scene_bounds:       场景边界
        save_path:          保存路径（None 则显示交互式窗口）
        scene:              Sionna Scene 对象（可选，用于显示建筑物投影）
    """
    # 坐标轴范围：取场景边界和基站实际范围的并集，确保所有基站都显示
    xmin_scene, xmax_scene, ymin_scene, ymax_scene = scene_bounds
    if len(bs_positions) > 0:
        bs_margin = isd * 0.6
        xmin = min(xmin_scene, float(bs_positions[:, 0].min()) - bs_margin)
        xmax = max(xmax_scene, float(bs_positions[:, 0].max()) + bs_margin)
        ymin = min(ymin_scene, float(bs_positions[:, 1].min()) - bs_margin)
        ymax = max(ymax_scene, float(bs_positions[:, 1].max()) + bs_margin)
    else:
        xmin, xmax, ymin, ymax = xmin_scene, xmax_scene, ymin_scene, ymax_scene

    scene_width = xmax - xmin
    scene_height = ymax - ymin

    # 图形尺寸：固定 20 英寸宽，保持宽高比
    fig_w = 20
    fig_h = max(16, fig_w * scene_height / scene_width)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.tick_params(labelsize=11)

    # ---- 1. 建筑物投影（使用 Mitsuba API，绘制实际网格三角面的 2D 投影）----
    # 参考 sionna_utils.py：
    #   vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
    #   faces = mesh.faces_buffer().numpy().reshape(-1, 3)
    # 每个 face 是一个三角形，投影到 XY 平面即为建筑物俯视轮廓
    building_count = 0
    face_count = 0
    if scene is not None:
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

                    # 过滤地面（所有顶点 z 坐标都接近 0 的网格）
                    if np.max(verts[:, 2]) < 0.5:
                        continue

                    # 对每个三角面，取 XY 坐标投影
                    for face in faces:
                        tri_xy = verts[face, :2]  # [3, 2]
                        poly = MplPolygon(tri_xy, closed=True)
                        all_polygons.append(poly)
                        face_count += 1

                    building_count += 1
                except Exception:
                    pass

            if all_polygons:
                # 批量绘制所有三角面（性能更好）
                pc = PatchCollection(all_polygons, alpha=0.35, facecolor="#A0A0A0",
                                     edgecolor="#606060", linewidth=0.3, zorder=1)
                ax.add_collection(pc)
                print(f"  已绘制 {building_count} 个建筑物网格（{face_count} 个三角面）")
            else:
                print("  未找到建筑物网格（可能场景格式不同）")
        except Exception as e:
            print(f"  建筑物投影提取失败：{e}")

    if building_count == 0:
        # 无建筑物时显示场景边界框
        rect = plt.Rectangle((xmin_scene, ymin_scene),
                              xmax_scene - xmin_scene, ymax_scene - ymin_scene,
                              fill=False, linestyle="--", color="#888888",
                              linewidth=1.5, alpha=0.7, zorder=1)
        ax.add_patch(rect)
        ax.text(xmin_scene + 10, ymax_scene - 30, "Munich Scene boundary (±500m)",
                fontsize=10, color="#888888", va="top")

    # ---- 2. 小区覆盖范围 ----
    for c in range(len(bs_positions)):
        pos = bs_positions[c]
        circle = plt.Circle((pos[0], pos[1]), isd / 2,
                             fill=True, facecolor="#E3F2FD",
                             linestyle="--", edgecolor="#1565C0",
                             alpha=0.15, linewidth=1.2, zorder=3)
        ax.add_patch(circle)

    # ---- 3. 邻区连线 ----
    drawn_pairs = set()
    for cell_id_str, neighbors in neighbor_relations.items():
        cell_id = int(cell_id_str)
        if cell_id >= len(bs_positions):
            continue
        pos_i = bs_positions[cell_id]
        for nb_id in neighbors:
            if nb_id >= len(bs_positions):
                continue
            pair = tuple(sorted([cell_id, nb_id]))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            pos_j = bs_positions[nb_id]
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                    "--", color="#555555", linewidth=1.2, alpha=0.6, zorder=4)

    # ---- 4. UE 轨迹 ----
    speeds_shown = set()
    for traj in trajectories:
        speed = traj["speed_kmh"]
        pos = traj["pos"]
        color = SPEED_COLORS.get(speed, "#9E9E9E")
        ax.plot(pos[:, 0], pos[:, 1],
                color=color, linewidth=1.2, alpha=0.65, zorder=5)
        ax.plot(pos[0, 0], pos[0, 1], "o",
                color=color, markersize=7, alpha=0.9, zorder=6,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(pos[-1, 0], pos[-1, 1], "s",
                color=color, markersize=7, alpha=0.9, zorder=6,
                markeredgecolor="white", markeredgewidth=0.5)
        speeds_shown.add(speed)

    # ---- 5. 基站位置 ----
    for c in range(len(bs_positions)):
        pos = bs_positions[c]
        ax.plot(pos[0], pos[1], "r^",
                markersize=16, zorder=10,
                markeredgecolor="darkred", markeredgewidth=1.0)
        ax.annotate(
            f"BS{c}",
            xy=(pos[0], pos[1]),
            xytext=(pos[0] + 12, pos[1] + 12),
            fontsize=9, fontweight="bold", color="darkred",
            zorder=11,
        )

    # ---- 6. 坐标轴 ----
    ax.set_xlabel("X [m]  (East →)", fontsize=13)
    ax.set_ylabel("Y [m]  (North ↑)", fontsize=13)
    ax.set_title(
        f"Network Configuration — Munich Scene\n"
        f"{len(bs_positions)} BSs, ISD={isd}m  |  "
        f"Blue lines = neighbor links  |  "
        f"Trajectories: {len(trajectories)} preview",
        fontsize=13,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, zorder=0, linewidth=0.8)

    # 坐标轴范围：包含所有基站 + 5% 边距
    pad = max(scene_width, scene_height) * 0.03
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.axhline(y=0, color="k", linewidth=0.8, alpha=0.3, zorder=0)
    ax.axvline(x=0, color="k", linewidth=0.8, alpha=0.3, zorder=0)

    # ---- 7. 图例 ----
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markeredgecolor="darkred", markersize=13, label="Base Station"),
        Line2D([0], [0], color="#555555", linewidth=2.0, linestyle="--", label="Neighbor link"),
    ]
    for speed in sorted(speeds_shown):
        color = SPEED_COLORS.get(speed, "#9E9E9E")
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, label=f"UE {speed:.0f} km/h")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11,
              framealpha=0.9, markerscale=1.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"配置图已保存到：{save_path}")
    else:
        plt.show()

    plt.close()


# =========================================================
# GUI 模式（需要 polyscope）
# =========================================================

def run_gui_mode(cfg: dict) -> Optional[np.ndarray]:
    """
    启动官方 Sionna RT GUI，让用户交互式放置基站

    返回：
        基站位置数组 [num_cells, 2]，如果用户取消则返回 None
    """
    try:
        import polyscope as ps
        from sionna_rt_gui import AppHolder
        from sionna_rt_gui.config import load_config, GuiConfig
    except ImportError as e:
        print(f"错误：无法导入 sionna-rt-gui（{e}）")
        print("请安装：pip install 'polyscope>=2.6.0,<2.7.0' omegaconf")
        print("并确保 sourceCode/sionna-rt-gui/ 在 Python 路径中")
        return None

    print("=" * 60)
    print("Sionna RT GUI 基站配置工具")
    print("=" * 60)
    print("操作说明：")
    print("  Ctrl + 左键点击：在场景中添加基站（发射机）")
    print("  左键点击选中：选中后可拖动移动")
    print("  Del 键：删除选中的基站")
    print("  关闭窗口：完成配置，自动保存坐标")
    print("=" * 60)

    # 添加 sionna-rt-gui 到路径
    gui_src = Path(__file__).parent / "sourceCode" / "sionna-rt-gui" / "src"
    if gui_src.exists() and str(gui_src) not in sys.path:
        sys.path.insert(0, str(gui_src))

    try:
        # 加载 GUI 配置
        gui_cfg_path = Path(__file__).parent / "sourceCode" / "sionna-rt-gui" / "src" / "sionna_rt_gui" / "data" / "configs" / "sionna_rt_gui" / "base.yaml"
        if gui_cfg_path.exists():
            gui_cfg = load_config(str(gui_cfg_path), scene_filename="munich")
        else:
            gui_cfg = GuiConfig(scene_filename="munich")

        # 禁用示例场景（我们自己放基站）
        gui_cfg.create_example_scenario = False

        # 启动 GUI
        app = AppHolder(gui_cfg, scene_filename="munich")

        # 预加载已保存的基站位置（如果有）
        # AppHolder.app 是 SionnaRtGui 实例
        saved_positions = cfg.get("bs_config", {}).get("positions", [])
        h_bs = cfg.get("bs_config", {}).get("h_bs", 10.0)
        if saved_positions:
            try:
                gui = app.app  # SionnaRtGui 实例
                if gui is not None and gui.scene is not None:
                    # 使用 GUI 的 add_radio_device 方法，同时更新 Polyscope 可视化
                    for p in saved_positions:
                        gui.add_radio_device(
                            [float(p["x"]), float(p["y"]), float(h_bs)],
                            is_transmitter=True,
                            allow_auto_update=False,
                        )
                    print(f"  已预加载 {len(saved_positions)} 个基站位置到 3D 地图")
            except Exception as e:
                print(f"  预加载基站位置失败（{e}），请在 GUI 中手动放置")

        app.show()

        # GUI 关闭后，读取基站位置
        scene = app.app.scene
        if scene is None:
            print("警告：场景未加载")
            return None

        transmitters = scene.transmitters
        if not transmitters:
            print("警告：没有放置任何基站")
            return None

        # 提取基站坐标（只取 x, y）
        positions = []
        for name, tx in sorted(transmitters.items()):
            pos = tx.position.numpy().squeeze()
            positions.append([float(pos[0]), float(pos[1])])
            print(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        print(f"\n共放置 {len(positions)} 个基站")
        return np.array(positions, dtype=np.float32)

    except Exception as e:
        print(f"GUI 运行失败：{e}")
        import traceback
        traceback.print_exc()
        return None


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="网络配置工具：交互式配置基站位置、邻区关系和 UE 轨迹预览",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python network_config_tool.py --gui          # 启动 3D GUI 放置基站（需要 polyscope）
  python network_config_tool.py --no-gui       # 使用现有配置，只更新邻区和轨迹预览
  python network_config_tool.py --no-gui --save  # 保存可视化图片
        """
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="启动 3D GUI 放置基站（需要 polyscope）"
    )
    parser.add_argument(
        "--no-gui", action="store_true", default=False,
        help="不启动 GUI，使用现有配置"
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
        help="保存可视化图片到 outputs/network_config.png"
    )
    parser.add_argument(
        "--max-neighbors", type=int, default=6,
        help="每个基站的最大邻区数（默认：6）"
    )
    parser.add_argument(
        "--preview-trajs", type=int, default=3,
        help="每种速度预览的轨迹数（默认：3）"
    )
    args = parser.parse_args()

    # 默认：如果没有指定，根据 polyscope 是否可用决定
    if not args.gui and not args.no_gui:
        try:
            import polyscope
            args.gui = True
        except ImportError:
            args.no_gui = True

    # ---- 加载现有配置 ----
    cfg = load_network_config()
    print(f"加载配置文件：{NETWORK_CONFIG_PATH}")

    # ---- 获取基站位置 ----
    if args.gui:
        print("\n启动 GUI 模式...")
        new_positions = run_gui_mode(cfg)
        if new_positions is not None:
            # 更新配置中的基站位置
            cfg["bs_config"] = cfg.get("bs_config", {})
            cfg["bs_config"]["num_cells"] = len(new_positions)
            cfg["bs_config"]["positions"] = [
                {"id": i, "x": float(new_positions[i, 0]), "y": float(new_positions[i, 1])}
                for i in range(len(new_positions))
            ]
            bs_positions = new_positions
        else:
            print("GUI 模式失败或取消，使用现有配置")
            bs_positions = get_bs_positions_from_config(cfg)
    else:
        print("\n无 GUI 模式，使用现有配置")
        bs_positions = get_bs_positions_from_config(cfg)

    print(f"\n基站数量：{len(bs_positions)}")
    print("基站位置：")
    for i, pos in enumerate(bs_positions):
        print(f"  BS{i}: ({pos[0]:.1f}, {pos[1]:.1f}) m")

    # ---- 邻区关系：优先保留手动配置，只在必要时重新计算 ----
    isd = cfg.get("bs_config", {}).get("isd", 250.0)
    existing_relations = cfg.get("neighbor_config", {}).get("relations", None)
    existing_num_cells = len(existing_relations) if existing_relations else 0
    current_num_cells = len(bs_positions)

    if existing_relations and existing_num_cells == current_num_cells and not args.gui:
        # 基站数量没变，且不是 GUI 模式（没有重新放置基站）
        # 保留手动修改的邻区关系，不重新计算
        neighbor_relations = {int(k): [int(v) for v in vals]
                              for k, vals in existing_relations.items()}
        print(f"\n保留现有邻区关系（{existing_num_cells} 个基站，手动配置不会被覆盖）")
        print_neighbor_stats(neighbor_relations)
    else:
        # 基站数量变化，或者是 GUI 模式重新放置了基站，重新计算邻区关系
        if existing_relations and existing_num_cells != current_num_cells:
            print(f"\n基站数量变化（{existing_num_cells} → {current_num_cells}），重新计算邻区关系...")
        else:
            print(f"\n计算邻区关系（最大邻区数：{args.max_neighbors}）...")

        neighbor_relations = compute_neighbor_relations(
            bs_positions,
            max_neighbors=args.max_neighbors,
            isd=isd,
        )
        print_neighbor_stats(neighbor_relations)

        # 更新配置（只在重新计算时才写入）
        cfg["neighbor_config"] = {
            "max_neighbors": args.max_neighbors,
            "method": "distance",
            "relations": {str(k): v for k, v in neighbor_relations.items()},
        }

    # ---- 保存配置 ----
    save_network_config(cfg)

    # ---- 可视化 ----
    print("\n生成可视化图...")

    # 尝试加载 Sionna 场景（用于显示建筑物 + 碰撞检测）
    scene = None
    walkable_grid = None
    try:
        import sionna
        import sionna.rt as srt
        from sionna.rt import load_scene, PlanarArray
        print("  加载 Sionna 场景（用于显示建筑物 + 轨迹碰撞检测）...")
        scene = load_scene(getattr(srt.scene, "munich"))
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                      vertical_spacing=0.5, horizontal_spacing=0.5,
                                      pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                      vertical_spacing=0.5, horizontal_spacing=0.5,
                                      pattern="iso", polarization="V")
        print("  场景加载完成")

        # 计算可行走网格（用于中高速轨迹碰撞检测）
        try:
            from matplotlib.path import Path as MplPath
            print("  计算可行走网格（用于轨迹碰撞检测）...")
            xmin, xmax, ymin, ymax = SCENE_BOUNDS
            grid_size = 2.0
            W = int(np.ceil((xmax - xmin) / grid_size))
            H = int(np.ceil((ymax - ymin) / grid_size))
            walkable = np.ones((H, W), dtype=bool)

            for mesh in scene.mi_scene.shapes():
                try:
                    verts = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
                    faces = mesh.faces_buffer().numpy().reshape(-1, 3)
                    if len(verts) < 3 or len(faces) == 0:
                        continue
                    if np.max(verts[:, 2]) < 0.5:
                        continue
                    for face in faces:
                        tri = verts[face, :2]
                        tx_min, ty_min = tri.min(axis=0)
                        tx_max, ty_max = tri.max(axis=0)
                        ix_min = max(0, int((tx_min - xmin) / grid_size))
                        ix_max = min(W - 1, int((tx_max - xmin) / grid_size) + 1)
                        iy_min = max(0, int((ty_min - ymin) / grid_size))
                        iy_max = min(H - 1, int((ty_max - ymin) / grid_size) + 1)
                        if ix_min > ix_max or iy_min > iy_max:
                            continue
                        tri_path = MplPath(np.vstack([tri, tri[0]]))
                        gx = np.arange(ix_min, ix_max + 1)
                        gy = np.arange(iy_min, iy_max + 1)
                        gxx, gyy = np.meshgrid(gx, gy)
                        cx = xmin + (gxx + 0.5) * grid_size
                        cy = ymin + (gyy + 0.5) * grid_size
                        pts = np.column_stack([cx.ravel(), cy.ravel()])
                        inside = tri_path.contains_points(pts).reshape(gyy.shape)
                        walkable[iy_min:iy_max+1, ix_min:ix_max+1] &= ~inside
                except Exception:
                    pass

            walkable_grid = {
                "grid": walkable,
                "origin": (xmin, ymin),
                "grid_size": grid_size,
            }
            print(f"  可行走网格计算完成（可行走比例：{walkable.mean()*100:.1f}%）")
        except Exception as e:
            print(f"  可行走网格计算失败（{e}），轨迹不使用碰撞检测")
            walkable_grid = None

    except Exception as e:
        print(f"  场景加载失败（{e}），不显示建筑物背景")

    # ---- 预生成 UE 轨迹（有 Sionna 时使用碰撞检测）----
    print(f"\n预生成示例轨迹（每种速度 {args.preview_trajs} 条）...")
    trajectories = generate_preview_trajectories(
        bs_positions=bs_positions,
        scene_bounds=SCENE_BOUNDS,
        num_per_speed=args.preview_trajs,
        speeds_kmh=cfg.get("trajectory_config", {}).get("speeds_kmh", [30.0, 60.0, 120.0]),
        seed=cfg.get("trajectory_config", {}).get("seed", 42),
        walkable_grid=walkable_grid,
    )
    print(f"  生成了 {len(trajectories)} 条示例轨迹"
          + ("（中高速使用碰撞检测）" if walkable_grid is not None else ""))

    save_path = None
    if args.save:
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        save_path = str(output_dir / "network_config.png")

    visualize_network_config(
        bs_positions=bs_positions,
        neighbor_relations=neighbor_relations,
        trajectories=trajectories,
        isd=isd,
        scene_bounds=SCENE_BOUNDS,
        save_path=save_path,
        scene=scene,
    )

    # ---- 打印下一步操作 ----
    print("\n" + "=" * 60)
    print("配置完成！下一步：")
    print("1. 查看可视化图，确认基站位置和轨迹是否合理")
    print("2. 如需调整基站位置，修改 network_config.json 中的 bs_config.positions")
    print("   然后重新运行：python network_config_tool.py --no-gui")
    print("3. 确认后运行仿真：python generate_dataset.py")
    print("=" * 60)

    # 打印 bs_positions_override 代码（方便复制到 config.py）
    print("\n如需在 config.py 中手动指定基站位置，复制以下代码：")
    print("bs_positions_override = np.array([")
    for i, pos in enumerate(bs_positions):
        print(f"    [{pos[0]:.1f}, {pos[1]:.1f}],  # BS{i}")
    print("], dtype=np.float32)")


if __name__ == "__main__":
    main()