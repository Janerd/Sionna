"""
config.py
=========
仿真全局参数配置（对应 MATLAB config.m）

修改此文件即可调整仿真场景，无需修改其他代码。

与 MATLAB 版本的主要区别：
  - 使用 Sionna 内置的慕尼黑 3D 城市场景（真实建筑物）
  - 场景类型改为 'umi'（城市微站，ISD=200m），更具挑战性
  - 去掉了 Ground Truth 速度/方向特征
  - 增加了射线追踪相关参数
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimConfig:
    """仿真全局参数配置类"""

    # =========================================================
    # 网络拓扑参数
    # =========================================================

    # 场景类型：
    #   'umi'  - 城市微站（UMi Street Canyon），ISD=200m，基站高度10m
    #            这是本项目的主要场景，A3 在此场景下会自然失效
    #   'uma'  - 城市宏站（UMa），ISD=500m，基站高度25m
    #            对应 MATLAB 的 complex 场景，A3 表现较好
    scenario_type: str = "umi"

    # 小区数量
    # umi 场景：19 个小区（1 中心 + 两圈邻区），ISD=250m
    #   覆盖范围约 ±430m，与慕尼黑场景（±500m）匹配
    # uma 场景：19 个小区（1 中心 + 两圈邻区），ISD=500m
    num_cells: int = 19

    # 站间距 [m]
    # umi: 250m（密集城区，19 个基站均匀覆盖 ±500m 场景）
    # uma: 500m（对应 MATLAB complex 场景）
    isd: float = 250.0

    # 基站天线高度 [m]
    # umi: 10m（路灯杆高度，低于建筑物，LOS/NLOS 切换频繁）
    # uma: 25m（高于建筑物，覆盖广）
    h_bs: float = 10.0

    # UE 天线高度 [m]
    h_ue: float = 1.5

    # =========================================================
    # 无线信道参数
    # =========================================================

    # 载波频率 [Hz]（5G NR Sub-6G，3.5GHz 频段）
    fc: float = 3.5e9

    # 系统带宽 [Hz]
    bw: float = 20e6

    # 基站发射功率 [dBm]
    # umi: 30dBm（微站，低功率）
    # uma: 46dBm（宏站，高功率）
    p_tx_dbm: float = 30.0

    # UE 噪声系数 [dB]
    noise_figure_db: float = 7.0

    # 热噪声功率谱密度 [dBm/Hz]
    noise_density_dbm_hz: float = -174.0

    # =========================================================
    # 射线追踪参数（Sionna RT 特有）
    # =========================================================

    # 最大反射次数
    # 2次反射可以捕捉主要的多径效应（直射 + 一次反射 + 二次反射）
    # 增加到3次会更真实，但计算时间增加约3倍
    max_reflections: int = 2

    # 最大衍射次数（建筑物边缘衍射）
    # 1次衍射可以捕捉建筑物角落的绕射效应
    max_diffractions: int = 1

    # 射线追踪采样数（每次计算的射线数量）
    # 越多越精确，但计算时间越长
    # RTX 4060 Ti 推荐：1e6（约 10~50ms/次）
    # 如果显存不足，可以降低到 5e5
    num_samples_per_ray: int = 1_000_000

    # 批量射线追踪：每批同时处理的 UE 数量（方案 A 优化）
    # 批量化可将 GPU 利用率从 ~5% 提升到 ~80%，速度提升 5~10 倍
    # 内存估算（RTX 4060 Ti 16GB）：
    #   batch_size=32：约 4~8 GB 显存（安全）
    #   batch_size=64：约 8~12 GB 显存（较安全）
    #   batch_size=128：约 12~16 GB 显存（可能 OOM）
    # 建议：16GB 显存使用 32，如果稳定可以尝试 64
    rt_batch_size: int = 32

    # 是否使用 GPU 加速射线追踪
    # True：使用 RTX 4060 Ti 的 RT Core 加速（推荐）
    # False：CPU 计算（速度慢约 100 倍）
    use_gpu: bool = True

    # =========================================================
    # UE 运动参数
    # =========================================================

    # 速度场景 [km/h]
    # 注意：去掉了 3 km/h（低速场景切换极少，对研究意义不大）
    # 重点关注中高速场景（切换频繁，A3 更容易失效）
    speeds_kmh: List[float] = field(default_factory=lambda: [30.0, 60.0, 120.0])

    @property
    def speeds_ms(self) -> List[float]:
        """速度 [m/s]"""
        return [v / 3.6 for v in self.speeds_kmh]

    # =========================================================
    # 仿真时间参数
    # =========================================================

    # 测量周期 [s]（对应 3GPP A3 事件的 TTT 典型值 40ms）
    # 每个测量周期 = 一次 RSRP 测量 + 一次切换判决
    slot_duration: float = 40e-3  # 40ms

    # 每条轨迹的测量周期数（速度自适应，见 generate_dataset.py）
    # 目标：UE 穿越约 1.5 个 ISD 的距离
    # 30 km/h：约 1000 slots（40s，穿越 333m）
    # 60 km/h：约 500 slots（20s，穿越 333m）
    # 120 km/h：约 250 slots（10s，穿越 333m）
    # 注意：实际 slot 数由 generate_dataset.py 根据速度自动计算
    num_slots_base: int = 1000  # 基准值，实际值自动调整

    # =========================================================
    # 数据集参数
    # =========================================================

    # 总轨迹条数（每种速度各 num_trajectories/num_speeds 条）
    # 3种速度 × 40条 = 120条
    # 每种速度 40 条：train=24, val=8, test=8
    num_trajectories: int = 120

    # 时序预测输入窗口长度 [时隙]
    # 使用过去 10 个时隙（400ms）的特征预测未来目标小区
    window_size: int = 10

    # 预测时间窗口（提前量）[时隙]
    # 改为 5 slots（200ms），比 MATLAB 的 20 slots（800ms）更合理
    # 理由：200ms 后的目标小区对当前切换决策更有指导意义
    pred_horizon: int = 5

    # 输出目录
    output_dir: str = "outputs"

    # =========================================================
    # 基站位置手动指定（可选）
    # =========================================================

    # 手动指定基站位置（覆盖六边形网格默认值）
    # 格式：[[x0, y0], [x1, y1], ...] 单位：米
    # 坐标系：慕尼黑场景中心为原点，X 向东，Y 向北
    #
    # 使用方法：
    #   1. 运行 python scene_setup.py 查看场景地图
    #   2. 根据地图确定街道交叉口的坐标
    #   3. 在此处填入坐标
    #
    # 示例（19 小区，ISD=250m，均匀覆盖慕尼黑场景 ±500m）：
    # 运行 python visualize_scene_3d.py 查看基站在地图上的位置
    # 根据地图调整坐标，使基站落在街道交叉口
    # bs_positions_override = np.array([
    #     [   0.0,    0.0],  # BS0:  场景中心
    #     [ 250.0,    0.0],  # BS1:  东
    #     [ 125.0,  217.0],  # BS2:  东北
    #     [-125.0,  217.0],  # BS3:  西北
    #     [-250.0,    0.0],  # BS4:  西
    #     [-125.0, -217.0],  # BS5:  西南
    #     [ 125.0, -217.0],  # BS6:  东南
    #     [ 433.0,    0.0],  # BS7:  东（第二圈）
    #     [ 375.0,  217.0],  # BS8:  东北（第二圈）
    #     [ 217.0,  375.0],  # BS9:  北偏东（第二圈）
    #     [   0.0,  433.0],  # BS10: 北（第二圈）
    #     [-217.0,  375.0],  # BS11: 北偏西（第二圈）
    #     [-375.0,  217.0],  # BS12: 西北（第二圈）
    #     [-433.0,    0.0],  # BS13: 西（第二圈）
    #     [-375.0, -217.0],  # BS14: 西南（第二圈）
    #     [-217.0, -375.0],  # BS15: 南偏西（第二圈）
    #     [   0.0, -433.0],  # BS16: 南（第二圈）
    #     [ 217.0, -375.0],  # BS17: 南偏东（第二圈）
    #     [ 375.0, -217.0],  # BS18: 东南（第二圈）
    # ], dtype=np.float32)
    #
    # 设置为 None 则使用六边形网格（默认）
    bs_positions_override: object = None

    # =========================================================
    # 特征参数
    # =========================================================

    # 波束数量（每个小区）
    num_beams: int = 8

    # 波束覆盖角度范围 [度]（均匀分布在此范围内）
    beam_angle_range: float = 120.0  # ±60°

    # =========================================================
    # L3 滤波参数（3GPP TS 38.331）
    # =========================================================

    # L3 滤波系数 k
    # alpha = 1/2^k，k=4 对应 alpha=0.0625（3GPP 默认值）
    # F(t) = (1-alpha)*F(t-1) + alpha*M(t)
    l3_filter_k: int = 4

    @property
    def l3_alpha(self) -> float:
        """L3 滤波系数 alpha"""
        return 1.0 / (2 ** self.l3_filter_k)

    # =========================================================
    # 切换参数（A3 基线，用于对比）
    # =========================================================

    # A3 事件偏置 [dB]
    a3_offset_db: float = 3.0

    # 触发时间 TTT [s]
    a3_ttt_s: float = 80e-3  # 80ms

    # 迟滞量 [dB]
    a3_hysteresis_db: float = 2.0

    # =========================================================
    # 派生参数（自动计算）
    # =========================================================

    @property
    def wavelength(self) -> float:
        """波长 [m]"""
        return 3e8 / self.fc

    @property
    def noise_floor_dbm(self) -> float:
        """噪声底 [dBm]"""
        import math
        return self.noise_density_dbm_hz + 10 * math.log10(self.bw) + self.noise_figure_db

    @property
    def num_features(self) -> int:
        """特征向量维度

        特征组成（每个小区 C 个，共 10 类）：
          RSRP_l3(C) + RSRQ(C) + SINR(C) + Doppler_est(C) + BeamID(C)
          + RSRP_diff(C) + BeamID_diff(C) + DelaySpread(C)
          + K_factor(C) + min_tau(C)
        共 10*C 维

        变更说明：
          - 删除 LOS_indicator（0/1 二值，信息量有限）
          - 新增 K_factor（Ricean K 因子，连续值，比 LOS_indicator 更精细）
          - 新增 min_tau（主径时延，隐含 UE 到基站距离）
          - 所有特征均可在现实网络中提取（不依赖仿真内部信息）
        """
        return 10 * self.num_cells

    @property
    def a3_ttt_slots(self) -> int:
        """A3 TTT 对应的时隙数"""
        return max(1, round(self.a3_ttt_s / self.slot_duration))


# =========================================================
# 预定义场景配置
# =========================================================

def get_umi_config() -> SimConfig:
    """
    城市微站场景（UMi Street Canyon）
    
    这是本项目的主要场景：
    - 19 个基站，ISD=250m，均匀覆盖慕尼黑场景（±500m）
    - 基站高度 10m（路灯杆高度，低于建筑物）
    - LOS/NLOS 切换频繁（UE 转弯时发生）
    - 多径结构复杂（街道反射）
    - A3 在此场景下会自然失效
    """
    cfg = SimConfig()
    cfg.scenario_type = "umi"
    cfg.num_cells = 19
    cfg.isd = 250.0
    cfg.h_bs = 10.0
    cfg.p_tx_dbm = 30.0
    cfg.speeds_kmh = [30.0, 60.0, 120.0]
    cfg.num_trajectories = 120
    return cfg


def get_uma_config() -> SimConfig:
    """
    城市宏站场景（UMa）
    
    对应 MATLAB complex 场景：
    - ISD=500m，基站高度25m（高于建筑物）
    - LOS/NLOS 切换较少
    - A3 表现较好（用于对比）
    """
    cfg = SimConfig()
    cfg.scenario_type = "uma"
    cfg.num_cells = 19
    cfg.isd = 500.0
    cfg.h_bs = 25.0
    cfg.p_tx_dbm = 46.0
    cfg.speeds_kmh = [30.0, 60.0, 120.0]
    cfg.num_trajectories = 120
    return cfg


# 默认使用 UMi 场景
DEFAULT_CONFIG = get_umi_config()


if __name__ == "__main__":
    # 打印配置信息
    cfg = DEFAULT_CONFIG
    print("=" * 50)
    print("仿真配置")
    print("=" * 50)
    print(f"场景类型:       {cfg.scenario_type}")
    print(f"小区数量:       {cfg.num_cells}")
    print(f"站间距:         {cfg.isd} m")
    print(f"基站高度:       {cfg.h_bs} m")
    print(f"载波频率:       {cfg.fc/1e9:.1f} GHz")
    print(f"发射功率:       {cfg.p_tx_dbm} dBm")
    print(f"噪声底:         {cfg.noise_floor_dbm:.1f} dBm")
    print(f"速度场景:       {cfg.speeds_kmh} km/h")
    print(f"总轨迹数:       {cfg.num_trajectories}")
    print(f"窗口大小:       {cfg.window_size} slots ({cfg.window_size*cfg.slot_duration*1000:.0f} ms)")
    print(f"预测时间窗口:   {cfg.pred_horizon} slots ({cfg.pred_horizon*cfg.slot_duration*1000:.0f} ms)")
    print(f"特征维度:       {cfg.num_features}")
    print(f"射线追踪反射:   {cfg.max_reflections} 次")
    print(f"射线追踪衍射:   {cfg.max_diffractions} 次")
    print(f"使用 GPU:       {cfg.use_gpu}")