"""
scene_setup.py
==============
Sionna RT 场景和基站布局（对应 MATLAB generate_cell_layout.m）

主要功能：
  1. 加载 Sionna 内置的慕尼黑 3D 城市场景
  2. 在场景中放置基站（六边形网格布局）
  3. 提供场景可视化功能

与 MATLAB 版本的主要区别：
  - 使用真实 3D 建筑物模型（慕尼黑市中心）
  - 基站位置基于六边形网格，但受限于场景范围
  - LOS/NLOS 由建筑物遮挡自然决定，不是概率模型
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Sionna RT 导入
# 注意：如果 import 失败，说明 Sionna 未正确安装，请参考 setup.md
try:
    import sionna
    from sionna.rt import load_scene, Transmitter, PlanarArray
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("警告：Sionna 未安装，场景功能不可用。请参考 setup.md 安装。")

from config import SimConfig


# =========================================================
# 基站位置计算（六边形网格）
# =========================================================

def compute_hexagonal_bs_positions(
    num_cells: int,
    isd: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    计算六边形网格基站位置（对应 MATLAB generate_cell_layout.m）

    参数：
        num_cells: 小区数量（7 或 19）
        isd: 站间距 [m]
        center: 中心小区坐标 [m]

    返回：
        positions: [num_cells, 2] 基站 2D 坐标 [m]

    六边形网格布局：
        7 小区：1 中心 + 6 邻区（第一圈）
        19 小区：1 中心 + 6 邻区（第一圈）+ 12 邻区（第二圈）
    """
    positions = []

    # 中心小区
    positions.append([center[0], center[1]])

    if num_cells >= 7:
        # 第一圈：6 个邻区，均匀分布在 60° 间隔
        for i in range(6):
            angle = math.radians(30 + 60 * i)  # 从 30° 开始，每 60° 一个
            x = center[0] + isd * math.cos(angle)
            y = center[1] + isd * math.sin(angle)
            positions.append([x, y])

    if num_cells >= 19:
        # 第二圈：12 个邻区
        # 第二圈的基站位置：6 个在 ISD*sqrt(3) 距离，6 个在 ISD*2 距离
        # 实际上第二圈有两种类型的位置
        for i in range(6):
            # 类型1：在第一圈两个邻区之间（距离 ISD*sqrt(3)）
            angle = math.radians(60 * i)
            x = center[0] + isd * math.sqrt(3) * math.cos(angle)
            y = center[1] + isd * math.sqrt(3) * math.sin(angle)
            positions.append([x, y])

        for i in range(6):
            # 类型2：在第一圈邻区的延伸方向（距离 ISD*2）
            angle = math.radians(30 + 60 * i)
            x = center[0] + isd * 2 * math.cos(angle)
            y = center[1] + isd * 2 * math.sin(angle)
            positions.append([x, y])

    positions = np.array(positions[:num_cells], dtype=np.float32)
    return positions


# =========================================================
# Sionna RT 场景设置
# =========================================================

class SceneManager:
    """
    管理 Sionna RT 场景和基站布局

    使用方法：
        scene_mgr = SceneManager(cfg)
        scene_mgr.setup()
        scene = scene_mgr.scene
        bs_positions = scene_mgr.bs_positions_2d
    """

    # Sionna 内置场景名称
    # 'munich'：慕尼黑市中心，密集城区，有真实建筑物
    # 'etoile'：巴黎凯旋门周边
    # 'simple_street_canyon'：简单街道峡谷（用于快速测试）
    SCENE_NAME = "munich"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.scene = None
        self.bs_positions_2d: np.ndarray | None = None  # [num_cells, 2]
        self.bs_positions_3d: np.ndarray | None = None  # [num_cells, 3]
        self._is_setup = False

    def setup(self) -> None:
        """
        初始化场景：加载 3D 城市模型，放置基站

        注意：第一次运行时需要从网络下载场景文件（约 50MB），
        之后会缓存到本地。
        """
        if not SIONNA_AVAILABLE:
            raise RuntimeError("Sionna 未安装，无法设置场景。请参考 setup.md。")

        print(f"正在加载 Sionna 场景：{self.SCENE_NAME}...")
        print("（第一次运行需要下载场景文件，约 50MB，请确保网络连接正常）")

        # 加载内置场景
        # sionna.rt.scene.munich 是慕尼黑市中心的真实 3D 建筑模型
        self.scene = load_scene(getattr(sionna.rt.scene, self.SCENE_NAME))

        print(f"场景加载完成")

        # 计算基站位置
        # 慕尼黑场景的中心坐标约为 (0, 0)，场景范围约 ±500m
        # 对于 UMi 场景（ISD=200m），7 个基站可以放在场景内
        self.bs_positions_2d = compute_hexagonal_bs_positions(
            num_cells=self.cfg.num_cells,
            isd=self.cfg.isd,
            center=(0.0, 0.0),
        )

        # 3D 坐标（加上基站高度）
        self.bs_positions_3d = np.column_stack([
            self.bs_positions_2d,
            np.full(self.cfg.num_cells, self.cfg.h_bs, dtype=np.float32),
        ])

        # 在场景中放置基站
        self._place_transmitters()

        self._is_setup = True
        print(f"基站布局完成：{self.cfg.num_cells} 个基站，ISD={self.cfg.isd}m")
        print(f"基站高度：{self.cfg.h_bs}m")

    def _place_transmitters(self) -> None:
        """在 Sionna 场景中放置基站（发射机）"""
        # 清除已有的发射机
        for tx_name in list(self.scene.transmitters.keys()):
            self.scene.remove(tx_name)

        # 放置每个基站
        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_3d[c]

            # 创建天线阵列（单天线 SISO，后续可扩展为 MIMO）
            # 如果需要 MIMO，可以改为：
            # antenna = PlanarArray(num_rows=2, num_cols=2, ...)
            antenna = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",      # 全向天线（简化模型）
                polarization="V",   # 垂直极化
            )

            # 创建发射机
            tx = Transmitter(
                name=f"bs_{c}",
                position=pos.tolist(),
                orientation=[0.0, 0.0, 0.0],  # 无旋转
            )
            tx.antenna = antenna

            self.scene.add(tx)

        print(f"已放置 {self.cfg.num_cells} 个基站")

    def place_receiver(self, ue_pos_2d: np.ndarray) -> None:
        """
        在场景中放置 UE（接收机）

        参数：
            ue_pos_2d: UE 的 2D 坐标 [x, y]，单位 m
        """
        from sionna.rt import Receiver

        # 清除已有的接收机
        for rx_name in list(self.scene.receivers.keys()):
            self.scene.remove(rx_name)

        # UE 3D 坐标
        ue_pos_3d = [float(ue_pos_2d[0]), float(ue_pos_2d[1]), self.cfg.h_ue]

        # 创建 UE 天线（单天线）
        antenna = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )

        # 创建接收机
        rx = Receiver(
            name="ue",
            position=ue_pos_3d,
            orientation=[0.0, 0.0, 0.0],
        )
        rx.antenna = antenna

        self.scene.add(rx)

    def compute_paths(self) -> object:
        """
        执行射线追踪，计算所有基站到 UE 的路径

        返回：
            paths: Sionna RT 路径对象，包含所有路径的时延、功率、角度等信息

        注意：
            - 必须先调用 place_receiver() 放置 UE
            - 使用 GPU 加速（RTX 4060 Ti 的 RT Core）
            - 每次调用约需 10~50ms（取决于场景复杂度）
        """
        if not self._is_setup:
            raise RuntimeError("请先调用 setup() 初始化场景")

        paths = self.scene.compute_paths(
            max_depth=self.cfg.max_reflections + self.cfg.max_diffractions,
            num_samples=self.cfg.num_samples_per_ray,
            # 同时计算反射和衍射路径
            diffraction=self.cfg.max_diffractions > 0,
            scattering=False,  # 不计算散射（计算量太大）
        )

        return paths

    @property
    def scene_bounds(self) -> Tuple[float, float, float, float]:
        """
        返回场景的 2D 边界 [xmin, xmax, ymin, ymax]

        用于生成 UE 轨迹时限制范围
        """
        # 慕尼黑场景的大致范围（根据实际场景调整）
        # 保守估计：±400m
        margin = self.cfg.isd * 2.5
        return (-margin, margin, -margin, margin)

    def visualize(self, save_path: str | None = None) -> None:
        """
        可视化场景（基站位置 + 建筑物轮廓）

        参数：
            save_path: 如果指定，保存图片到该路径；否则显示交互式窗口
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # 绘制基站位置
        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            ax.plot(pos[0], pos[1], "r^", markersize=12, zorder=5)
            ax.annotate(
                f"BS{c}",
                xy=(pos[0], pos[1]),
                xytext=(pos[0] + 5, pos[1] + 5),
                fontsize=8,
            )

        # 绘制小区覆盖范围（近似圆形）
        for c in range(self.cfg.num_cells):
            pos = self.bs_positions_2d[c]
            circle = plt.Circle(
                (pos[0], pos[1]),
                self.cfg.isd / 2,
                fill=False,
                linestyle="--",
                color="blue",
                alpha=0.3,
            )
            ax.add_patch(circle)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(f"基站布局（{self.cfg.num_cells} 个小区，ISD={self.cfg.isd}m）")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(["基站位置"], loc="upper right")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"场景图已保存到：{save_path}")
        else:
            plt.show()

        plt.close()


# =========================================================
# 独立运行：测试场景设置
# =========================================================

if __name__ == "__main__":
    from config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    print("测试场景设置...")
    print(f"场景类型：{cfg.scenario_type}")
    print(f"小区数量：{cfg.num_cells}")
    print(f"站间距：{cfg.isd} m")

    # 计算基站位置（不需要 Sionna）
    positions = compute_hexagonal_bs_positions(cfg.num_cells, cfg.isd)
    print(f"\n基站位置（前 3 个）：")
    for i in range(min(3, len(positions))):
        print(f"  BS{i}: ({positions[i, 0]:.1f}, {positions[i, 1]:.1f}) m")

    # 如果 Sionna 可用，测试完整场景设置
    if SIONNA_AVAILABLE:
        print("\n正在测试 Sionna 场景设置...")
        scene_mgr = SceneManager(cfg)
        scene_mgr.setup()
        print("✓ 场景设置成功")

        # 保存可视化
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(exist_ok=True)
        scene_mgr.visualize(save_path=str(output_dir / "scene_layout.png"))
    else:
        print("\n⚠ Sionna 未安装，跳过场景设置测试")
        print("  请参考 setup.md 安装 Sionna 后重新运行")