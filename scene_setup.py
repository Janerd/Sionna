"""
scene_setup.py
==============
Sionna RT 场景和基站布局（对应 MATLAB generate_cell_layout.m）

Sionna 版本：2.x（基于 PyTorch，不需要 TensorFlow）
  - 2.x 中使用 scene.trace_paths() 而非 scene.compute_paths()

主要功能：
  1. 加载 Sionna 内置的慕尼黑 3D 城市场景
  2. 在场景中放置基站
  3. 提供场景可视化功能

慕尼黑场景坐标系说明：
  - Sionna 内置慕尼黑场景的坐标原点在场景中心
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

UE 轨迹策略：
  - 默认：随机生成（street_grid 类型，沿网格道路移动）
  - 推荐：基于慕尼黑真实道路网络生成
    → 轨迹应沿街道方向移动（避免穿越建筑物）
    → 在路口处转弯（LOS/NLOS 突变的关键位置）
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
# Sionna RT 导入（Sionna 2.x API）
# =========================================================
try:
    import sionna
    import sionna.rt as srt
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
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
# 这些值基于 Sionna 文档和场景文件，实际运行时可通过
# scene.bbox 属性获取精确范围
MUNICH_SCENE_BOUNDS = {
    "xmin": -500.0,
    "xmax": 500.0,
    "ymin": -500.0,
    "ymax": 500.0,
}

# 慕尼黑场景中的典型街道方向（用于生成沿街道的 UE 轨迹）
# 慕尼黑市中心街道主要沿东西（0°）和南北（90°）方向
# 以及对角线方向（约 45°、135°）
MUNICH_STREET_DIRECTIONS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

# 慕尼黑场景中适合放置基站的位置（路灯杆/建筑物顶部）
# 这些位置基于慕尼黑市中心的典型街道交叉口
# 坐标单位：米，相对于场景中心
# 注意：实际部署时应根据 scene_layout.png 可视化结果调整
MUNICH_SUGGESTED_BS_POSITIONS = {
    # 7 小区方案（UMi，ISD≈200m）
    7: np.array([
        [0.0,    0.0],    # BS0: 场景中心
        [200.0,  0.0],    # BS1: 东侧
        [100.0,  173.0],  # BS2: 东北
        [-100.0, 173.0],  # BS3: 西北
        [-200.0, 0.0],    # BS4: 西侧
        [-100.0, -173.0], # BS5: 西南
        [100.0,  -173.0], # BS6: 东南
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

    注意：
        六边形网格是理想化的部署方案。
        在真实慕尼黑场景中，建议根据街道布局手动调整基站位置，
        使基站位于街道交叉口或建筑物顶部，而不是建筑物内部。
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
    2. MUNICH_SUGGESTED_BS_POSITIONS（如果有预设的慕尼黑位置）
    3. 六边形网格（默认）

    参数：
        cfg: 仿真配置

    返回：
        positions: [num_cells, 2] 基站 2D 坐标 [m]
    """
    # 检查是否有手动指定的位置
    if hasattr(cfg, "bs_positions_override") and cfg.bs_positions_override is not None:
        pos = np.array(cfg.bs_positions_override, dtype=np.float32)
        if pos.shape == (cfg.num_cells, 2):
            print(f"使用手动指定的基站位置（{cfg.num_cells} 个基站）")
            return pos
        else:
            print(f"警告：bs_positions_override 形状 {pos.shape} 不匹配，使用默认六边形网格")

    # 使用六边形网格（默认）
    return compute_hexagonal_bs_positions(cfg.num_cells, cfg.isd)


# =========================================================
# Sionna 2.x 射线追踪辅助函数
# =========================================================

def _get_paths_sionna2(scene, cfg: SimConfig):
    """Sionna 2.x 射线追踪 API 封装"""
    try:
        paths = scene.trace_paths(
            max_depth=cfg.max_reflections + cfg.max_diffractions,
            num_samples=cfg.num_samples_per_ray,
            diffraction=cfg.max_diffractions > 0,
            scattering=False,
        )
        scene.compute_fields(paths, check_scene=False)
        return paths
    except AttributeError:
        try:
            paths = scene.compute_paths(
                max_depth=cfg.max_reflections + cfg.max_diffractions,
                num_samples=cfg.num_samples_per_ray,
                diffraction=cfg.max_diffractions > 0,
                scattering=False,
            )
            return paths
        except AttributeError as e:
            raise RuntimeError(
                f"Sionna API 不兼容（版本 {SIONNA_VERSION}）。\n"
                f"错误：{e}\n"
                f"请确认已安装 Sionna 2.x：pip install sionna>=2.0"
            )


def _extract_path_data_sionna2(paths, num_cells: int):
    """从 Sionna 2.x paths 对象提取路径数据"""
    import torch

    try:
        a = paths.a
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()

        tau = paths.tau
        if isinstance(tau, torch.Tensor):
            tau = tau.cpu().numpy()

        phi_r = paths.phi_r
        if isinstance(phi_r, torch.Tensor):
            phi_r = phi_r.cpu().numpy()

        try:
            path_types = paths.types
            if isinstance(path_types, torch.Tensor):
                path_types = path_types.cpu().numpy()
        except AttributeError:
            path_types = None

        return a, tau, phi_r, path_types

    except Exception as e:
        raise RuntimeError(f"路径数据提取失败：{e}")


# =========================================================
# Sionna RT 场景设置
# =========================================================

class SceneManager:
    """
    管理 Sionna RT 场景和基站布局（Sionna 2.x 兼容版本）

    关于基站位置和 UE 轨迹的重要说明：
    ─────────────────────────────────────────────────────
    慕尼黑场景是真实的 3D 城市地图，基站和 UE 的位置需要
    与真实街道布局对应，否则会出现：
      - 基站放在建筑物内部（射线追踪结果异常）
      - UE 轨迹穿越建筑物（物理上不合理）

    推荐做法：
    1. 先运行 scene_mgr.visualize_with_scene() 查看场景地图
    2. 根据地图选择合适的基站位置（街道交叉口）
    3. 在 config.py 中设置 bs_positions_override
    4. UE 轨迹使用 street_grid 类型（沿街道方向移动）

    使用方法：
        scene_mgr = SceneManager(cfg)
        scene_mgr.setup()
        # 查看场景地图和基站位置
        scene_mgr.visualize_with_scene(save_path="outputs/scene_map.png")
        # 执行射线追踪
        paths = scene_mgr.trace_paths()
        a, tau, phi_r, types = scene_mgr.extract_path_data(paths)
    """

    SCENE_NAME = "munich"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.scene = None
        self.bs_positions_2d: Optional[np.ndarray] = None
        self.bs_positions_3d: Optional[np.ndarray] = None
        self._is_setup = False
        self._current_ue_pos: Optional[np.ndarray] = None
        self._scene_bbox: Optional[Tuple] = None  # 场景实际边界

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
                float(bbox[0][0]), float(bbox[1][0]),  # xmin, xmax
                float(bbox[0][1]), float(bbox[1][1]),  # ymin, ymax
            )
            print(f"场景边界：X=[{self._scene_bbox[0]:.0f}, {self._scene_bbox[1]:.0f}]m, "
                  f"Y=[{self._scene_bbox[2]:.0f}, {self._scene_bbox[3]:.0f}]m")
        except Exception:
            self._scene_bbox = None

        # 获取基站位置
        self.bs_positions_2d = get_bs_positions(self.cfg)

        # 3D 坐标
        self.bs_positions_3d = np.column_stack([
            self.bs_positions_2d,
            np.full(self.cfg.num_cells, self.cfg.h_bs, dtype=np.float32),
        ])

        self._place_transmitters()

        self._is_setup = True
        print(f"基站布局完成：{self.cfg.num_cells} 个基站，ISD={self.cfg.isd}m")
        print(f"基站高度：{self.cfg.h_bs}m")
        print(f"提示：运行 scene_mgr.visualize_with_scene() 可查看基站在地图上的位置")

    def _place_transmitters(self) -> None:
        """在 Sionna 2.x 场景中放置基站"""
        for tx_name in list(self.scene.transmitters.keys()):
            self.scene.remove(tx_name)

        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_3d[c]

            antenna = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="V",
            )

            tx = Transmitter(
                name=f"bs_{c}",
                position=pos.tolist(),
                orientation=[0.0, 0.0, 0.0],
            )
            tx.antenna = antenna
            self.scene.add(tx)

        print(f"已放置 {self.cfg.num_cells} 个基站")

    def place_receiver(self, ue_pos_2d: np.ndarray) -> None:
        """在场景中放置 UE"""
        for rx_name in list(self.scene.receivers.keys()):
            self.scene.remove(rx_name)

        ue_pos_3d = [float(ue_pos_2d[0]), float(ue_pos_2d[1]), self.cfg.h_ue]
        self._current_ue_pos = np.array(ue_pos_3d, dtype=np.float32)

        antenna = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )

        rx = Receiver(
            name="ue",
            position=ue_pos_3d,
            orientation=[0.0, 0.0, 0.0],
        )
        rx.antenna = antenna
        self.scene.add(rx)

    def trace_paths(self):
        """执行射线追踪（Sionna 2.x API）"""
        if not self._is_setup:
            raise RuntimeError("请先调用 setup() 初始化场景")
        return _get_paths_sionna2(self.scene, self.cfg)

    def extract_path_data(self, paths):
        """从路径对象提取数据"""
        return _extract_path_data_sionna2(paths, self.cfg.num_cells)

    @property
    def scene_bounds(self) -> Tuple[float, float, float, float]:
        """
        返回场景的 2D 边界 [xmin, xmax, ymin, ymax]

        优先使用从场景对象获取的实际边界，
        否则使用基于 ISD 的估算值。
        """
        if self._scene_bbox is not None:
            return self._scene_bbox
        margin = self.cfg.isd * 2.5
        return (-margin, margin, -margin, margin)

    def update_bs_positions(self, new_positions_2d: np.ndarray) -> None:
        """
        更新基站位置

        参数：
            new_positions_2d: [num_cells, 2] 新的基站 2D 坐标 [m]

        使用场景：
            1. 查看 visualize_with_scene() 生成的地图
            2. 根据街道布局确定合适的基站位置
            3. 调用此函数更新位置
        """
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
        """
        可视化基站布局（仅显示基站位置和覆盖范围，无地图背景）

        参数：
            save_path:   保存路径（None 则显示交互式窗口）
            use_english: 使用英文标签（避免中文字体问题）
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            ax.plot(pos[0], pos[1], "r^", markersize=12, zorder=5)
            ax.annotate(f"BS{c}", xy=(pos[0], pos[1]),
                        xytext=(pos[0] + 5, pos[1] + 5), fontsize=8)

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

        xmin, xmax, ymin, ymax = self.scene_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"场景图已保存到：{save_path}")
        else:
            plt.show()
        plt.close()

    def visualize_with_scene(self, save_path: Optional[str] = None) -> None:
        """
        可视化基站布局叠加在场景俯视图上

        使用 Sionna 2.x 的 scene.render() 生成场景俯视图，
        然后叠加基站位置。

        Sionna 2.x 中获取建筑物轮廓的方式：
          - scene.objects：包含所有场景物体（建筑物、地面等）
          - 每个物体的 mesh 属性包含顶点和面片信息
          - 通过 mi.traverse(scene.mi_scene) 可以访问 Mitsuba 场景树

        参数：
            save_path: 保存路径（None 则显示交互式窗口）
        """
        if not self._is_setup or self.scene is None:
            print("警告：场景未初始化，只显示基站位置（无地图背景）")
            self.visualize(save_path=save_path)
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # ---- 尝试多种方式获取建筑物轮廓 ----
        building_count = 0

        # 方式1：通过 scene.objects 获取物体顶点（Sionna 2.x 主要方式）
        try:
            import torch
            for obj_name, obj in self.scene.objects.items():
                try:
                    # Sionna 2.x 中物体的顶点通过 .vertices 属性获取
                    # 形状：[N, 3]，坐标单位为米
                    verts = obj.vertices
                    if isinstance(verts, torch.Tensor):
                        verts = verts.cpu().numpy()
                    else:
                        verts = np.array(verts)

                    if verts.ndim == 2 and verts.shape[1] >= 2 and len(verts) >= 3:
                        xy = verts[:, :2]
                        # 过滤掉地面（z 坐标接近 0 的大面积物体）
                        z_vals = verts[:, 2] if verts.shape[1] >= 3 else np.zeros(len(verts))
                        if np.max(z_vals) < 0.5:  # 地面，跳过
                            continue

                        # 绘制建筑物轮廓（凸包近似）
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
            print(f"  方式1失败：{e}")

        # 方式2：通过 scene.mi_scene 访问 Mitsuba 场景（备用）
        if building_count == 0:
            try:
                import mitsuba as mi
                mi_scene = self.scene.mi_scene
                # 遍历 Mitsuba 场景中的所有形状
                for shape in mi_scene.shapes():
                    try:
                        # 获取顶点位置
                        params = mi.traverse(shape)
                        verts_key = [k for k in params.keys() if "vertex_positions" in k]
                        if verts_key:
                            verts = np.array(params[verts_key[0]]).reshape(-1, 3)
                            if len(verts) >= 3 and np.max(verts[:, 2]) > 0.5:
                                xy = verts[:, :2]
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
                print(f"  方式2失败：{e}")

        if building_count > 0:
            print(f"已绘制 {building_count} 个建筑物轮廓")
        else:
            # 方式3：如果都失败，用文字提示并显示场景边界框
            print("注意：无法自动获取建筑物轮廓")
            print("  建议：使用 scene.render() 生成 3D 渲染图查看场景")
            xmin, xmax, ymin, ymax = self.scene_bounds
            # 绘制场景边界
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
        ax.set_xlim(xmin * 1.1, xmax * 1.1)
        ax.set_ylim(ymin * 1.1, ymax * 1.1)

        # 添加坐标轴说明
        ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
        ax.text(xmax * 0.95, 5, "E", fontsize=10, ha="right")
        ax.text(5, ymax * 0.95, "N", fontsize=10, va="top")

        # 打印基站坐标（方便手动调整）
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
        """
        返回适合生成街道对齐轨迹的场景边界

        慕尼黑场景中，UE 轨迹应限制在街道区域内，
        避免进入建筑物密集区域。
        保守估计：使用场景边界的 80%
        """
        xmin, xmax, ymin, ymax = self.scene_bounds
        margin = max(50.0, self.cfg.isd * 0.5)
        return (xmin + margin, xmax - margin, ymin + margin, ymax - margin)


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

    # 计算基站位置
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

        # 生成场景地图（叠加基站位置）
        print("\n生成场景地图（含建筑物轮廓）...")
        scene_mgr.visualize_with_scene(
            save_path=str(output_dir / "scene_map_with_bs.png")
        )

        # 生成简单布局图
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