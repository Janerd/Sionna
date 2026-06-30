"""
scene_setup.py
==============
Sionna RT 场景和基站布局（对应 MATLAB generate_cell_layout.m）

Sionna 版本：2.0.1（基于 PyTorch，不需要 TensorFlow）

Sionna 2.0.1 正确 API（基于源码分析）：
  - 射线追踪：PathSolver()(scene, ...)  而非 scene.trace_paths() 或 scene.compute_paths()
  - 路径对象：paths.a = (a_real, a_imag)，paths.tau，paths.phi_r，paths.theta_r
  - 天线阵列：scene.tx_array 和 scene.rx_array 必须在调用 PathSolver 前设置
  - PlanarArray：pattern 参数为字符串，可选 "iso", "dipole", "hw_dipole", "tr38901"
  - Transmitter/Receiver：position 参数为 mi.Point3f 或可转换为 mi.Point3f 的列表

主要功能：
  1. 加载 Sionna 内置的慕尼黑 3D 城市场景
  2. 在场景中放置基站（六边形网格布局）
  3. 提供场景可视化功能

慕尼黑场景坐标系说明：
  - 坐标原点在场景中心（Frauenkirche 附近）
  - 场景范围约 ±500m（东西方向）× ±500m（南北方向）
  - 坐标单位：米（m）
  - X 轴：东西方向（正方向向东）
  - Y 轴：南北方向（正方向向北）
  - Z 轴：高度（正方向向上）

基站部署策略：
  - 默认：六边形网格（ISD=200m），中心在 (0,0)
  - 推荐：根据慕尼黑场景的真实街道布局手动指定基站位置
    → 在 config.py 中设置 bs_positions_override 参数
    → 或者运行 python scene_setup.py 查看场景图后手动调整
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# =========================================================
# matplotlib 中文字体配置（解决 CJK 字符警告）
# =========================================================
import matplotlib
import matplotlib.pyplot as plt


def _setup_matplotlib_font():
    """配置 matplotlib 支持中文显示，避免 CJK 字符警告"""
    chinese_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "FangSong",
        "KaiTi",
        "Arial Unicode MS",
    ]
    available = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            return font
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


_FONT_NAME = _setup_matplotlib_font()

# =========================================================
# Sionna RT 导入（Sionna 2.0.1 API）
# =========================================================
try:
    import sionna
    import sionna.rt as srt
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    import mitsuba as mi
    import drjit as dr
    SIONNA_AVAILABLE = True
    SIONNA_VERSION = sionna.__version__
except ImportError:
    SIONNA_AVAILABLE = False
    SIONNA_VERSION = "未安装"
    print("警告：Sionna 未安装，场景功能不可用。请参考 setup.md 安装。")

from config import SimConfig


# =========================================================
# 慕尼黑场景的真实坐标信息
# =========================================================

# Sionna 内置慕尼黑场景的大致范围（单位：米）
MUNICH_SCENE_BOUNDS = {
    "xmin": -500.0,
    "xmax": 500.0,
    "ymin": -500.0,
    "ymax": 500.0,
}

# 慕尼黑场景中的典型街道方向（用于生成沿街道的 UE 轨迹）
MUNICH_STREET_DIRECTIONS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

# 慕尼黑场景中适合放置基站的位置（路灯杆/建筑物顶部）
MUNICH_SUGGESTED_BS_POSITIONS = {
    7: np.array([
        [0.0,    0.0],
        [200.0,  0.0],
        [100.0,  173.0],
        [-100.0, 173.0],
        [-200.0, 0.0],
        [-100.0, -173.0],
        [100.0,  -173.0],
    ], dtype=np.float32),
}


# =========================================================
# 基站位置计算
# =========================================================

def compute_hexagonal_bs_positions(
    num_cells: int,
    isd: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    计算六边形网格基站位置

    参数：
        num_cells: 小区数量（7 或 19）
        isd: 站间距 [m]
        center: 中心小区坐标 [m]

    返回：
        positions: [num_cells, 2] 基站 2D 坐标 [m]
    """
    positions = []
    positions.append([center[0], center[1]])

    if num_cells >= 7:
        for i in range(6):
            angle = math.radians(30 + 60 * i)
            x = center[0] + isd * math.cos(angle)
            y = center[1] + isd * math.sin(angle)
            positions.append([x, y])

    if num_cells >= 19:
        for i in range(6):
            angle = math.radians(60 * i)
            x = center[0] + isd * math.sqrt(3) * math.cos(angle)
            y = center[1] + isd * math.sqrt(3) * math.sin(angle)
            positions.append([x, y])

        for i in range(6):
            angle = math.radians(30 + 60 * i)
            x = center[0] + isd * 2 * math.cos(angle)
            y = center[1] + isd * 2 * math.sin(angle)
            positions.append([x, y])

    positions = np.array(positions[:num_cells], dtype=np.float32)
    return positions


def get_bs_positions(cfg: SimConfig) -> np.ndarray:
    """
    获取基站位置

    优先级：
    1. cfg.bs_positions_override（如果设置了手动指定的位置）
    2. 六边形网格（默认）
    """
    if hasattr(cfg, "bs_positions_override") and cfg.bs_positions_override is not None:
        pos = np.array(cfg.bs_positions_override, dtype=np.float32)
        if pos.shape == (cfg.num_cells, 2):
            print(f"使用手动指定的基站位置（{cfg.num_cells} 个基站）")
            return pos
        else:
            print(f"警告：bs_positions_override 形状 {pos.shape} 不匹配，使用默认六边形网格")

    return compute_hexagonal_bs_positions(cfg.num_cells, cfg.isd)


# =========================================================
# Sionna 2.0.1 射线追踪辅助函数
# =========================================================

def _get_paths_sionna201(scene, cfg: SimConfig, solver=None):
    """
    Sionna 2.0.1 射线追踪 API

    正确用法（基于源码 path_solver.py）：
        solver = PathSolver()
        paths = solver(scene, max_depth=..., samples_per_src=..., ...)

    注意：
    - PathSolver 是独立的类，不是 scene 的方法
    - scene.tx_array 和 scene.rx_array 必须在调用前设置
    - scene.all_set(radio_map=False) 会检查是否设置了 tx_array/rx_array
    - solver 参数：传入已有实例可复用（方案 B 优化），None 则新建
    """
    if solver is None:
        solver = PathSolver()
    paths = solver(
        scene,
        max_depth=cfg.max_reflections + cfg.max_diffractions,
        samples_per_src=cfg.num_samples_per_ray,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=cfg.max_diffractions > 0,
        edge_diffraction=False,
    )
    return paths


def _extract_path_data_sionna201(paths):
    """
    从 Sionna 2.0.1 paths 对象提取路径数据

    Sionna 2.0.1 中 Paths 对象的属性（基于源码 paths.py）：
        paths.a:            (a_real, a_imag) 元组，[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        paths.tau:          路径时延 [s]，形状同上
        paths.phi_r:        到达方位角 [rad]，形状同上
        paths.theta_r:      到达仰角 [rad]，形状同上
        paths.doppler:      Doppler 频移 [Hz]，形状同上（物理精确，包含所有散射体贡献）
        paths.interactions: 路径交互类型，[max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
                            InteractionType.LOS=0, SPECULAR=1, DIFFRACTION=2, ...

    注意：
    - paths.a 是 (a_real, a_imag) 元组，不是复数张量
    - paths.doppler 已经包含了完整的 Doppler 计算（发射机+接收机+散射体速度贡献）
    - paths.interactions 可以直接判断 LOS/NLOS，比时延误差判断更准确
    """
    try:
        # paths.a 是 (a_real, a_imag) 元组
        a_real, a_imag = paths.a
        a_real_np = a_real.numpy()
        a_imag_np = a_imag.numpy()
        # 合并为复数数组
        a = a_real_np + 1j * a_imag_np

        tau = paths.tau.numpy()
        phi_r = paths.phi_r.numpy()
        theta_r = paths.theta_r.numpy()

        # Doppler 频移（物理精确，直接从 paths 读取）
        try:
            doppler = paths.doppler.numpy()
        except AttributeError:
            doppler = None

        # 路径交互类型（用于精确判断 LOS/NLOS）
        try:
            interactions = paths.interactions.numpy()
        except AttributeError:
            interactions = None

        return a, tau, phi_r, theta_r, doppler, interactions

    except Exception as e:
        raise RuntimeError(f"路径数据提取失败：{e}")


# =========================================================
# Sionna RT 场景设置
# =========================================================

class SceneManager:
    """
    管理 Sionna RT 场景和基站布局（Sionna 2.0.1 兼容版本）

    Sionna 2.0.1 正确使用方式（基于源码）：
    ─────────────────────────────────────────────────────
    1. 加载场景：scene = load_scene(sionna.rt.scene.munich)
    2. 设置天线阵列：
       scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
       scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    3. 添加发射机和接收机：
       tx = Transmitter(name="bs_0", position=[x, y, z])
       scene.add(tx)
       rx = Receiver(name="ue", position=[x, y, z])
       scene.add(rx)
    4. 执行射线追踪：
       solver = PathSolver()
       paths = solver(scene, max_depth=3, samples_per_src=1e6)
    5. 提取路径数据：
       a_real, a_imag = paths.a  # 注意：a 是元组，不是复数张量
       tau = paths.tau
       phi_r = paths.phi_r

    使用方法：
        scene_mgr = SceneManager(cfg)
        scene_mgr.setup()
        scene_mgr.visualize_with_scene(save_path="outputs/scene_map.png")
        paths = scene_mgr.trace_paths()
        a, tau, phi_r, theta_r, types = scene_mgr.extract_path_data(paths)
    """

    SCENE_NAME = "munich"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.scene = None
        self.bs_positions_2d: Optional[np.ndarray] = None
        self.bs_positions_3d: Optional[np.ndarray] = None
        self._is_setup = False
        self._current_ue_pos: Optional[np.ndarray] = None
        self._scene_bbox: Optional[Tuple] = None
        self._path_solver: Optional[object] = None  # PathSolver 实例

    def setup(self) -> None:
        """初始化场景：加载 3D 城市模型，放置基站"""
        if not SIONNA_AVAILABLE:
            raise RuntimeError(
                f"Sionna 未安装（当前版本：{SIONNA_VERSION}），无法设置场景。\n"
                "请运行：pip install sionna>=2.0"
            )

        print(f"正在加载 Sionna {SIONNA_VERSION} 场景：{self.SCENE_NAME}...")
        print("（第一次运行需要下载场景文件，约 50MB，请确保网络连接正常）")

        self.scene = load_scene(getattr(srt.scene, self.SCENE_NAME))
        print("场景加载完成")

        # 尝试获取场景实际边界
        try:
            bbox = self.scene.bbox
            self._scene_bbox = (
                float(bbox[0][0]), float(bbox[1][0]),
                float(bbox[0][1]), float(bbox[1][1]),
            )
            print(f"场景边界：X=[{self._scene_bbox[0]:.0f}, {self._scene_bbox[1]:.0f}]m, "
                  f"Y=[{self._scene_bbox[2]:.0f}, {self._scene_bbox[3]:.0f}]m")
        except Exception:
            self._scene_bbox = None

        # 设置天线阵列（必须在 PathSolver 调用前设置）
        # Sionna 2.0.1 中 PlanarArray 的 pattern 参数为字符串
        # 可选值："iso", "dipole", "hw_dipole", "tr38901"
        tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        self.scene.tx_array = tx_array
        self.scene.rx_array = rx_array
        print("天线阵列已设置（SISO，全向天线）")

        # 设置载波频率
        self.scene.frequency = self.cfg.fc
        print(f"载波频率：{self.cfg.fc/1e9:.1f} GHz")

        # 获取基站位置
        self.bs_positions_2d = get_bs_positions(self.cfg)

        # 3D 坐标
        self.bs_positions_3d = np.column_stack([
            self.bs_positions_2d,
            np.full(self.cfg.num_cells, self.cfg.h_bs, dtype=np.float32),
        ])

        # 在场景中放置基站
        self._place_transmitters()

        # 创建 PathSolver 实例（复用，避免每次重新创建）
        self._path_solver = PathSolver()

        self._is_setup = True
        print(f"基站布局完成：{self.cfg.num_cells} 个基站，ISD={self.cfg.isd}m")
        print(f"基站高度：{self.cfg.h_bs}m")
        print(f"提示：运行 scene_mgr.visualize_with_scene() 可查看基站在地图上的位置")

        # 记录 Scene 对象的可用方法（帮助诊断 API 兼容性）
        scene_methods = [m for m in dir(self.scene) if not m.startswith("_")]
        print(f"Scene 可用方法（共 {len(scene_methods)} 个）：{scene_methods}")

    def _place_transmitters(self) -> None:
        """在 Sionna 2.0.1 场景中放置基站"""
        for tx_name in list(self.scene.transmitters.keys()):
            self.scene.remove(tx_name)

        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_3d[c]
            tx = Transmitter(
                name=f"bs_{c}",
                position=[float(pos[0]), float(pos[1]), float(pos[2])],
                orientation=[0.0, 0.0, 0.0],
            )
            self.scene.add(tx)

        print(f"已放置 {self.cfg.num_cells} 个基站")

    def place_receiver(self, ue_pos_2d: np.ndarray) -> None:
        """
        在场景中放置单个 UE（接收机）

        参数：
            ue_pos_2d: UE 的 2D 坐标 [x, y]，单位 m
        """
        for rx_name in list(self.scene.receivers.keys()):
            self.scene.remove(rx_name)

        ue_pos_3d = [float(ue_pos_2d[0]), float(ue_pos_2d[1]), self.cfg.h_ue]
        self._current_ue_pos = np.array(ue_pos_3d, dtype=np.float32)

        rx = Receiver(
            name="ue",
            position=ue_pos_3d,
            orientation=[0.0, 0.0, 0.0],
        )
        self.scene.add(rx)

    def place_receivers_batch(self, ue_positions_2d: np.ndarray) -> None:
        """
        在场景中批量放置多个 UE（方案 A 优化）

        参数：
            ue_positions_2d: [N, 2] UE 的 2D 坐标数组，单位 m

        注意：
            - 批量放置后调用 trace_paths() 会一次计算所有 UE 的路径
            - 路径数据形状：a[N, 1, C, 1, num_paths]，tau[N, C, num_paths]
            - 处理完后调用 clear_receivers() 清理
        """
        # 清除所有现有接收机
        for rx_name in list(self.scene.receivers.keys()):
            self.scene.remove(rx_name)

        # 批量添加接收机
        for i, pos_2d in enumerate(ue_positions_2d):
            ue_pos_3d = [float(pos_2d[0]), float(pos_2d[1]), self.cfg.h_ue]
            rx = Receiver(
                name=f"ue_{i}",
                position=ue_pos_3d,
                orientation=[0.0, 0.0, 0.0],
            )
            self.scene.add(rx)

    def clear_receivers(self) -> None:
        """清除场景中所有接收机（批量处理后调用）"""
        for rx_name in list(self.scene.receivers.keys()):
            self.scene.remove(rx_name)

    def trace_paths(self):
        """
        执行射线追踪（Sionna 2.0.1 API，方案 B：复用 PathSolver 实例）

        使用 self._path_solver 复用实例，避免每次重新初始化的开销。

        返回：
            paths: Sionna 2.0.1 Paths 对象
        """
        if not self._is_setup:
            raise RuntimeError("请先调用 setup() 初始化场景")

        # 方案 B：复用 PathSolver 实例（self._path_solver 在 setup() 中创建）
        return _get_paths_sionna201(self.scene, self.cfg, solver=self._path_solver)

    def extract_path_data(self, paths):
        """
        从路径对象提取数据（Sionna 2.0.1 API）

        返回：
            a:          路径复数增益 numpy 数组（a_real + 1j * a_imag）
            tau:        路径时延 numpy 数组 [s]
            phi_r:      到达方位角 numpy 数组 [rad]
            theta_r:    到达仰角 numpy 数组 [rad]
            path_types: 路径类型 numpy 数组（可能为 None）
        """
        return _extract_path_data_sionna201(paths)

    @property
    def scene_bounds(self) -> Tuple[float, float, float, float]:
        """
        返回场景的 2D 边界 [xmin, xmax, ymin, ymax]

        优先使用从场景对象获取的实际边界，
        否则使用慕尼黑场景的默认范围 ±500m。
        注意：不再使用 ISD 估算，避免基站数量变化时边界不一致。
        """
        if self._scene_bbox is not None:
            return self._scene_bbox
        # 慕尼黑场景默认范围 ±500m
        return (-500.0, 500.0, -500.0, 500.0)

    def update_bs_positions(self, new_positions_2d: np.ndarray) -> None:
        """更新基站位置"""
        if new_positions_2d.shape != (self.cfg.num_cells, 2):
            raise ValueError(
                f"基站位置数组形状错误：期望 ({self.cfg.num_cells}, 2)，"
                f"实际 {new_positions_2d.shape}"
            )

        self.bs_positions_2d = new_positions_2d.astype(np.float32)
        self.bs_positions_3d = np.column_stack([
            self.bs_positions_2d,
            np.full(self.cfg.num_cells, self.cfg.h_bs, dtype=np.float32),
        ])

        if self._is_setup:
            self._place_transmitters()
            print(f"基站位置已更新")

    def visualize(self, save_path: Optional[str] = None, use_english: bool = True) -> None:
        """可视化基站布局（仅显示基站位置和覆盖范围，无地图背景）"""
        # 图形尺寸根据场景范围自动调整，确保不漏显示
        xmin, xmax, ymin, ymax = self.scene_bounds
        scene_width = xmax - xmin
        scene_height = ymax - ymin
        fig_size = max(12, scene_width / 50)  # 每 50m 对应 1 英寸，最小 12 英寸
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * scene_height / scene_width))

        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            ax.plot(pos[0], pos[1], "r^", markersize=10, zorder=5)
            ax.annotate(f"BS{c}", xy=(pos[0], pos[1]),
                        xytext=(pos[0] + 8, pos[1] + 8), fontsize=7)

        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            circle = plt.Circle((pos[0], pos[1]), self.cfg.isd / 2,
                                 fill=False, linestyle="--", color="blue", alpha=0.3)
            ax.add_patch(circle)

        ax.set_xlabel("X [m] (East)")
        ax.set_ylabel("Y [m] (North)")
        ax.set_title(f"BS Layout ({self.cfg.num_cells} cells, ISD={self.cfg.isd}m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # 使用完整场景范围，确保所有基站都显示
        ax.set_xlim(xmin - 20, xmax + 20)
        ax.set_ylim(ymin - 20, ymax + 20)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"场景图已保存到：{save_path}")
        else:
            plt.show()
        plt.close()

    def visualize_with_scene(self, save_path: Optional[str] = None) -> None:
        """
        可视化基站布局叠加在场景俯视图上

        Sionna 2.0.1 中获取建筑物轮廓的方式：
          - scene.objects：包含所有场景物体（建筑物、地面等）
          - 每个物体的 vertices 属性包含顶点坐标（mi.TensorXf 格式）
        """
        if not self._is_setup or self.scene is None:
            print("警告：场景未初始化，只显示基站位置（无地图背景）")
            self.visualize(save_path=save_path)
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # 尝试绘制建筑物轮廓
        building_count = 0
        try:
            for obj_name, obj in self.scene.objects.items():
                try:
                    # Sionna 2.0.1 中物体顶点通过 .vertices 属性获取
                    # 类型为 mi.TensorXf，需要 .numpy() 转换
                    verts = obj.vertices
                    if hasattr(verts, 'numpy'):
                        verts = verts.numpy()
                    else:
                        verts = np.array(verts)

                    if verts.ndim == 2 and verts.shape[1] >= 2 and len(verts) >= 3:
                        xy = verts[:, :2]
                        z_vals = verts[:, 2] if verts.shape[1] >= 3 else np.zeros(len(verts))
                        if np.max(z_vals) < 0.5:
                            continue

                        from scipy.spatial import ConvexHull
                        try:
                            hull = ConvexHull(xy)
                            hull_pts = np.append(hull.vertices, hull.vertices[0])
                            ax.fill(xy[hull_pts, 0], xy[hull_pts, 1],
                                    alpha=0.25, color="#8B8B8B", zorder=1)
                            ax.plot(xy[hull_pts, 0], xy[hull_pts, 1],
                                    "-", color="#555555", linewidth=0.4, alpha=0.7, zorder=2)
                            building_count += 1
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            print(f"  建筑物轮廓绘制失败：{e}")

        if building_count > 0:
            print(f"已绘制 {building_count} 个建筑物轮廓")
        else:
            print("注意：无法自动获取建筑物轮廓，只显示基站位置")
            xmin, xmax, ymin, ymax = self.scene_bounds
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  fill=False, linestyle=":", color="gray",
                                  linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
            ax.text(0.5, 0.5,
                    "Munich Scene\n(Building outlines unavailable)\n"
                    "Use scene.render() for 3D view",
                    ha="center", va="center", fontsize=11, color="#666666",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        # 绘制基站位置
        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            ax.plot(pos[0], pos[1], "r^", markersize=14, zorder=10,
                    label="BS" if c == 0 else "")
            ax.annotate(f"BS{c}", xy=(pos[0], pos[1]),
                        xytext=(pos[0] + 8, pos[1] + 8),
                        fontsize=9, fontweight="bold", color="red",
                        zorder=11)

        # 绘制小区覆盖范围
        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            circle = plt.Circle((pos[0], pos[1]), self.cfg.isd / 2,
                                 fill=False, linestyle="--", color="blue",
                                 alpha=0.4, linewidth=1.5, zorder=5)
            ax.add_patch(circle)

        ax.set_xlabel("X [m] (East →)")
        ax.set_ylabel("Y [m] (North ↑)")
        ax.set_title(
            f"Munich Scene - BS Layout\n"
            f"{self.cfg.num_cells} cells, ISD={self.cfg.isd}m, h_BS={self.cfg.h_bs}m\n"
            f"Red triangles = BS positions | Blue circles = coverage (ISD/2)"
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, zorder=0)

        xmin, xmax, ymin, ymax = self.scene_bounds
        # 使用完整场景范围 + 10% 边距，确保所有基站都显示
        ax.set_xlim(xmin - abs(xmin) * 0.1, xmax + abs(xmax) * 0.1)
        ax.set_ylim(ymin - abs(ymin) * 0.1, ymax + abs(ymax) * 0.1)

        ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
        ax.text(xmax * 0.95, 5, "E", fontsize=10, ha="right")
        ax.text(5, ymax * 0.95, "N", fontsize=10, va="top")

        # 打印基站坐标
        print("\n当前基站坐标（可复制到 config.py 的 bs_positions_override）：")
        print("bs_positions_override = np.array([")
        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            print(f"    [{pos[0]:.1f}, {pos[1]:.1f}],  # BS{c}")
        print("], dtype=np.float32)")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n场景地图已保存到：{save_path}")
            print("请查看图片，确认基站位置是否在街道上（而非建筑物内）")
            print("如需调整，请修改 config.py 中的 bs_positions_override 参数")
        else:
            plt.show()
        plt.close()

    def get_street_aligned_bounds(self) -> Tuple[float, float, float, float]:
        """返回适合生成街道对齐轨迹的场景边界"""
        xmin, xmax, ymin, ymax = self.scene_bounds
        margin = max(50.0, self.cfg.isd * 0.5)
        return (xmin + margin, xmax - margin, ymin + margin, ymax - margin)

    def compute_walkable_grid(
        self,
        grid_size: float = 2.0,
        cache_path: Optional[str] = None,
    ) -> dict:
        """
        从建筑物网格提取可行走区域（街道 = True，建筑物内 = False）。

        使用与建筑物投影相同的 Mitsuba API：
            mesh.vertex_positions_buffer() + mesh.faces_buffer()

        参数：
            grid_size:   网格分辨率 [m]（默认 2m，±500m 场景 = 500×500 格）
            cache_path:  缓存文件路径（None 则不缓存）

        返回：
            {
              "grid":      np.ndarray [H, W] bool，True = 可行走（街道）
              "origin":    (xmin, ymin)，网格左下角坐标 [m]
              "grid_size": float，每格大小 [m]
            }
        """
        if not self._is_setup or self.scene is None:
            raise RuntimeError("请先调用 setup() 初始化场景")

        # 尝试从缓存加载
        if cache_path is not None:
            cache_file = Path(cache_path)
            if cache_file.exists():
                try:
                    data = np.load(cache_file)
                    print(f"从缓存加载可行走网格：{cache_file}")
                    return {
                        "grid": data["grid"],
                        "origin": tuple(data["origin"]),
                        "grid_size": float(data["grid_size"]),
                    }
                except Exception:
                    pass

        print(f"计算可行走网格（分辨率 {grid_size}m）...")

        xmin, xmax, ymin, ymax = self.scene_bounds
        W = int(np.ceil((xmax - xmin) / grid_size))
        H = int(np.ceil((ymax - ymin) / grid_size))
        origin = (xmin, ymin)

        # 初始化：全部可行走
        walkable = np.ones((H, W), dtype=bool)

        # 从建筑物网格提取占地，标记为不可行走
        building_count = 0
        try:
            from matplotlib.path import Path as MplPath

            for mesh in self.scene.mi_scene.shapes():
                try:
                    verts = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
                    faces = mesh.faces_buffer().numpy().reshape(-1, 3)

                    if len(verts) < 3 or len(faces) == 0:
                        continue
                    # 过滤地面
                    if np.max(verts[:, 2]) < 0.5:
                        continue

                    # 取建筑物的 XY 占地范围（用所有顶点的 XY 坐标）
                    xy = verts[:, :2]
                    bx_min, by_min = xy.min(axis=0)
                    bx_max, by_max = xy.max(axis=0)

                    # 只处理场景范围内的建筑物
                    if bx_max < xmin or bx_min > xmax or by_max < ymin or by_min > ymax:
                        continue

                    # 对每个三角面，标记其覆盖的网格格子为不可行走
                    for face in faces:
                        tri = verts[face, :2]  # [3, 2]
                        # 三角形的包围盒
                        tx_min, ty_min = tri.min(axis=0)
                        tx_max, ty_max = tri.max(axis=0)

                        # 转换为网格索引
                        ix_min = max(0, int((tx_min - xmin) / grid_size))
                        ix_max = min(W - 1, int((tx_max - xmin) / grid_size) + 1)
                        iy_min = max(0, int((ty_min - ymin) / grid_size))
                        iy_max = min(H - 1, int((ty_max - ymin) / grid_size) + 1)

                        if ix_min > ix_max or iy_min > iy_max:
                            continue

                        # 用 matplotlib.path 做精确的点在三角形内判断
                        tri_path = MplPath(np.vstack([tri, tri[0]]))  # 闭合路径

                        # 生成候选格子的中心点
                        gx = np.arange(ix_min, ix_max + 1)
                        gy = np.arange(iy_min, iy_max + 1)
                        gxx, gyy = np.meshgrid(gx, gy)
                        cx = xmin + (gxx + 0.5) * grid_size
                        cy = ymin + (gyy + 0.5) * grid_size
                        pts = np.column_stack([cx.ravel(), cy.ravel()])

                        inside = tri_path.contains_points(pts)
                        inside_2d = inside.reshape(gyy.shape)

                        walkable[iy_min:iy_max+1, ix_min:ix_max+1] &= ~inside_2d

                    building_count += 1
                except Exception:
                    pass

        except Exception as e:
            print(f"  可行走网格计算失败：{e}，返回全可行走网格")

        walkable_ratio = walkable.mean() * 100
        print(f"  可行走网格：{W}×{H}，可行走比例：{walkable_ratio:.1f}%，"
              f"处理了 {building_count} 个建筑物网格")

        result = {
            "grid": walkable,
            "origin": origin,
            "grid_size": grid_size,
        }

        # 保存缓存
        if cache_path is not None:
            try:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    grid=walkable,
                    origin=np.array(origin),
                    grid_size=np.array(grid_size),
                )
                print(f"  可行走网格已缓存到：{cache_path}")
            except Exception as e:
                print(f"  缓存保存失败：{e}")

        return result

    def compute_coverage_points(
        self,
        cell_size: float = 5.0,
        rsrp_threshold_dbm: float = -105.0,
        h_ue: Optional[float] = None,
    ) -> np.ndarray:
        """
        使用 RadioMapSolver 计算覆盖图，提取信号覆盖区域的坐标点。

        这些点代表"有信号的街道区域"，用于轨迹起点的均匀采样，
        避免起点落在建筑物内或信号盲区。

        参数：
            cell_size:           覆盖图分辨率 [m]（越小越精细，越慢）
            rsrp_threshold_dbm:  RSRP 阈值 [dBm]，低于此值的点不作为起点
            h_ue:                UE 高度 [m]（None 则使用 cfg.h_ue）

        返回：
            coverage_points: [N, 2] 覆盖区域的 2D 坐标数组
        """
        if not self._is_setup or self.scene is None:
            raise RuntimeError("请先调用 setup() 初始化场景")

        if h_ue is None:
            h_ue = self.cfg.h_ue

        print(f"计算覆盖图（分辨率 {cell_size}m，RSRP 阈值 {rsrp_threshold_dbm} dBm）...")

        try:
            from sionna.rt import RadioMapSolver
            solver = RadioMapSolver()
            radio_map = solver(
                self.scene,
                cell_size=(cell_size, cell_size),
                center=(0.0, 0.0, h_ue),
                orientation=(0.0, 0.0, 0.0),
                size=None,  # 使用场景默认大小
                measurement_surface="xy",
                samples_per_tx=int(1e6),
                max_depth=self.cfg.max_reflections,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=True,
                diffraction=self.cfg.max_diffractions > 0,
            )

            # 提取路径增益（dB）
            # radio_map.path_gain 形状：[num_tx, num_cells_y, num_cells_x]
            path_gain = radio_map.path_gain.numpy()  # [num_tx, H, W]

            # 取所有基站的最大路径增益（最强基站覆盖）
            max_path_gain = np.max(path_gain, axis=0)  # [H, W]

            # 转换为 RSRP（dBm）
            p_tx_mw = 10 ** (self.cfg.p_tx_dbm / 10)
            rsrp_dbm = 10 * np.log10(np.maximum(p_tx_mw * max_path_gain, 1e-20))

            # 提取覆盖区域的坐标
            # radio_map 的坐标系：中心为 (0,0)，cell_size 为分辨率
            H, W = rsrp_dbm.shape
            xmin_map = -W * cell_size / 2
            ymin_map = -H * cell_size / 2

            coverage_mask = rsrp_dbm > rsrp_threshold_dbm
            iy, ix = np.where(coverage_mask)

            # 转换为世界坐标
            x_coords = xmin_map + (ix + 0.5) * cell_size
            y_coords = ymin_map + (iy + 0.5) * cell_size

            coverage_points = np.column_stack([x_coords, y_coords]).astype(np.float32)

            print(f"  覆盖图大小：{W}×{H}，覆盖点数：{len(coverage_points)}")
            print(f"  RSRP 范围：[{rsrp_dbm.min():.1f}, {rsrp_dbm.max():.1f}] dBm")
            print(f"  覆盖率：{coverage_mask.mean()*100:.1f}%")

            return coverage_points

        except Exception as e:
            print(f"  覆盖图计算失败（{e}），回退到基站附近起点")
            # 回退：返回基站位置作为起点候选
            return self.bs_positions_2d.copy()


# =========================================================
# 独立运行：查看场景地图和基站位置
# =========================================================

if __name__ == "__main__":
    from config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    print("=" * 60)
    print("慕尼黑场景可视化工具")
    print("=" * 60)
    print(f"场景类型：{cfg.scenario_type}")
    print(f"小区数量：{cfg.num_cells}")
    print(f"站间距：{cfg.isd} m")
    print(f"Sionna 版本：{SIONNA_VERSION}")

    positions = get_bs_positions(cfg)
    print(f"\n基站位置（六边形网格，ISD={cfg.isd}m）：")
    for i in range(len(positions)):
        print(f"  BS{i}: ({positions[i, 0]:.1f}, {positions[i, 1]:.1f}) m")

    if SIONNA_AVAILABLE:
        print(f"\n正在加载 Sionna {SIONNA_VERSION} 场景...")
        scene_mgr = SceneManager(cfg)
        scene_mgr.setup()

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(exist_ok=True)

        print("\n生成场景地图（含建筑物轮廓）...")
        scene_mgr.visualize_with_scene(
            save_path=str(output_dir / "scene_map_with_bs.png")
        )

        scene_mgr.visualize(
            save_path=str(output_dir / "scene_layout.png"),
            use_english=True,
        )

        print("\n" + "=" * 60)
        print("下一步：")
        print("1. 查看 outputs/scene_map_with_bs.png")
        print("2. 确认基站是否在街道上（红色三角形应在街道交叉口）")
        print("3. 如需调整，在 config.py 中添加：")
        print("   bs_positions_override = np.array([...], dtype=np.float32)")
        print("4. 重新运行此脚本验证调整结果")
        print("5. 确认后运行 python generate_dataset.py 生成数据集")
        print("=" * 60)
    else:
        print(f"\n⚠ Sionna 未安装（{SIONNA_VERSION}），跳过场景加载")
        print("  请运行：pip install sionna>=2.0")