"""
trajectory.py
=============
UE 轨迹生成（对应 MATLAB generate_ue_trajectory.m）

主要功能：
  1. 生成多种类型的 UE 运动轨迹
  2. 轨迹沿慕尼黑真实街道方向移动（避免穿越建筑物）
  3. 支持速度自适应的轨迹长度

慕尼黑场景中的轨迹设计原则：
  ─────────────────────────────────────────────────────
  慕尼黑市中心的街道主要沿以下方向：
    - 东西方向（0°/180°）：主干道
    - 南北方向（90°/270°）：主干道
    - 对角线方向（约 45°/135°）：部分街道

  UE 轨迹应：
    1. 沿街道方向移动（不穿越建筑物）
    2. 在路口处转弯（LOS/NLOS 突变的关键位置）
    3. 起点在街道上（不在建筑物内）

  轨迹类型推荐：
    - street_grid：最真实，沿网格街道移动，在路口转弯
    - munich_walk：专为慕尼黑场景设计，沿主要街道方向移动
    - linear/arc：简化轨迹，用于基础验证

  注意：
    当前轨迹生成不感知建筑物位置（不读取场景几何）。
    轨迹是否穿越建筑物由射线追踪自动处理（穿越建筑物的路径
    会被遮挡，RSRP 会大幅下降）。
    如需严格避免穿越建筑物，需要读取场景几何并做碰撞检测，
    这会显著增加复杂度，当前版本不实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import SimConfig


# =========================================================
# 轨迹数据结构
# =========================================================

@dataclass
class Trajectory:
    """
    UE 轨迹数据

    字段说明：
        pos:        [T, 2] UE 坐标序列 [x, y]，单位 m
        vel:        [T, 2] UE 速度向量序列 [vx, vy]，单位 m/s
        speed_ms:   UE 速度大小 [m/s]
        speed_kmh:  UE 速度大小 [km/h]
        traj_type:  轨迹类型字符串
        num_slots:  时隙总数
    """
    pos: np.ndarray        # [T, 2]
    vel: np.ndarray        # [T, 2]
    speed_ms: float
    speed_kmh: float
    traj_type: str
    num_slots: int


# =========================================================
# 慕尼黑场景的街道方向
# =========================================================

# 慕尼黑市中心主要街道方向（度）
# 这些方向基于慕尼黑市中心的典型街道布局
MUNICH_STREET_DIRECTIONS_DEG = [
    0.0,    # 东西方向（向东）
    90.0,   # 南北方向（向北）
    180.0,  # 东西方向（向西）
    270.0,  # 南北方向（向南）
    45.0,   # 东北方向（部分街道）
    135.0,  # 西北方向（部分街道）
    225.0,  # 西南方向（部分街道）
    315.0,  # 东南方向（部分街道）
]

# 慕尼黑场景中典型的街道间距（米）
# 慕尼黑市中心街道间距约 50~120m
MUNICH_BLOCK_LENGTH_M = 80.0  # 典型街区长度


# =========================================================
# 轨迹生成函数
# =========================================================

def generate_trajectory(
    cfg: SimConfig,
    speed_ms: float,
    traj_type: str,
    scene_bounds: Tuple[float, float, float, float],
    bs_positions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Trajectory:
    """
    生成单条 UE 轨迹

    参数：
        cfg:          仿真配置
        speed_ms:     UE 速度 [m/s]
        traj_type:    轨迹类型（见下方说明）
        scene_bounds: 场景边界 [xmin, xmax, ymin, ymax]
        bs_positions: [num_cells, 2] 基站位置（用于选择起点）
        rng:          随机数生成器（用于可复现性）

    返回：
        Trajectory 对象

    轨迹类型：
        'linear'       - 直线匀速（基础验证）
        'random'       - 随机游走（方向缓慢变化）
        'arc'          - 圆弧转弯
        'street_grid'  - 沿道路网格移动（推荐，最接近真实城市移动）
                         UE 沿水平/垂直方向移动，在路口随机转弯
                         转弯时会经历 LOS/NLOS 突变（A3 失效的关键场景）
        'munich_walk'  - 慕尼黑街道行走（新增）
                         专为慕尼黑场景设计，沿主要街道方向移动
                         街道间距约 80m，转弯概率更高
        'stop_and_go'  - 停留-移动交替（模拟等红灯）
                         在小区边界附近停留，测试 A3 的乒乓切换
    """
    if rng is None:
        rng = np.random.default_rng()

    # 计算轨迹时隙数（速度自适应）
    # 目标：UE 穿越约 4 个 ISD 的距离，确保经历多次切换
    # 30 km/h：约 4320 slots（约 3 分钟，穿越 1000m）
    # 60 km/h：约 2160 slots（约 1.5 分钟，穿越 1000m）
    # 120 km/h：约 1080 slots（约 45 秒，穿越 1000m）
    target_dist = cfg.isd * 4.0
    dist_per_slot = speed_ms * cfg.slot_duration
    num_slots = min(int(math.ceil(target_dist / dist_per_slot)), 5000)
    num_slots = max(num_slots, cfg.window_size + cfg.pred_horizon + 10)

    # 选择起点（对齐到街道网格交叉口，避免起点在建筑物内）
    # 慕尼黑街道间距约 80m，交叉口在 (n×80, m×80) 处
    # 在基站附近选择最近的街道交叉口作为起点
    STREET_GRID_SIZE = 80.0  # 慕尼黑典型街区长度 [m]
    anchor_idx = rng.integers(0, len(bs_positions))
    anchor_pos = bs_positions[anchor_idx]

    # 对齐到最近的街道交叉口
    x0 = round(float(anchor_pos[0]) / STREET_GRID_SIZE) * STREET_GRID_SIZE
    y0 = round(float(anchor_pos[1]) / STREET_GRID_SIZE) * STREET_GRID_SIZE

    # 加小随机偏移（±15m），模拟 UE 不在正中心，但仍在街道上
    x0 += float(rng.uniform(-15, 15))
    y0 += float(rng.uniform(-15, 15))

    # 限制起点在场景边界内
    xmin, xmax, ymin, ymax = scene_bounds
    margin = 30.0  # 距边界至少 30m（慕尼黑场景边缘建筑物密集）
    x0 = float(np.clip(x0, xmin + margin, xmax - margin))
    y0 = float(np.clip(y0, ymin + margin, ymax - margin))

    # 根据轨迹类型生成轨迹
    if traj_type == "linear":
        pos, vel = _generate_linear(x0, y0, speed_ms, num_slots, cfg.slot_duration, rng)
    elif traj_type == "random":
        pos, vel = _generate_random(x0, y0, speed_ms, num_slots, cfg.slot_duration, rng)
    elif traj_type == "arc":
        pos, vel = _generate_arc(x0, y0, speed_ms, num_slots, cfg.slot_duration, rng)
    elif traj_type == "street_grid":
        pos, vel = _generate_street_grid(
            x0, y0, speed_ms, num_slots, cfg.slot_duration, scene_bounds, rng
        )
    elif traj_type == "munich_walk":
        pos, vel = _generate_munich_walk(
            x0, y0, speed_ms, num_slots, cfg.slot_duration, scene_bounds, rng
        )
    elif traj_type == "stop_and_go":
        pos, vel = _generate_stop_and_go(
            x0, y0, speed_ms, num_slots, cfg.slot_duration, scene_bounds, rng
        )
    else:
        raise ValueError(
            f"未知轨迹类型：{traj_type}。"
            f"可选：linear, random, arc, street_grid, munich_walk, stop_and_go"
        )

    return Trajectory(
        pos=pos.astype(np.float32),
        vel=vel.astype(np.float32),
        speed_ms=speed_ms,
        speed_kmh=speed_ms * 3.6,
        traj_type=traj_type,
        num_slots=num_slots,
    )


# =========================================================
# 各轨迹类型的实现
# =========================================================

def _generate_linear(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """直线匀速轨迹"""
    direction = float(rng.random()) * 2 * math.pi
    vx = speed_ms * math.cos(direction)
    vy = speed_ms * math.sin(direction)

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    for t in range(num_slots):
        pos[t] = [x0 + vx * t * dt, y0 + vy * t * dt]
        vel[t] = [vx, vy]

    return pos, vel


def _generate_random(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """随机游走轨迹（方向随机缓慢变化）"""
    direction = float(rng.random()) * 2 * math.pi
    dir_change_std = 0.1

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    pos[0] = [x0, y0]
    vel[0] = [speed_ms * math.cos(direction), speed_ms * math.sin(direction)]

    for t in range(1, num_slots):
        direction += dir_change_std * float(rng.standard_normal())
        vx = speed_ms * math.cos(direction)
        vy = speed_ms * math.sin(direction)
        pos[t] = pos[t - 1] + np.array([vx, vy]) * dt
        vel[t] = [vx, vy]

    return pos, vel


def _generate_arc(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """圆弧轨迹（直线 → 圆弧转弯 → 直线）"""
    direction = float(rng.random()) * 2 * math.pi
    turn_start = round(num_slots * 0.3)
    turn_end = round(num_slots * 0.7)
    turn_angle = math.pi / 2 * (2 * float(rng.random()) - 1)
    turn_rate = turn_angle / ((turn_end - turn_start) * dt)

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    pos[0] = [x0, y0]
    vel[0] = [speed_ms * math.cos(direction), speed_ms * math.sin(direction)]

    for t in range(1, num_slots):
        if turn_start <= t < turn_end:
            direction += turn_rate * dt
        vx = speed_ms * math.cos(direction)
        vy = speed_ms * math.sin(direction)
        pos[t] = pos[t - 1] + np.array([vx, vy]) * dt
        vel[t] = [vx, vy]

    return pos, vel


def _generate_street_grid(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    scene_bounds: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    道路网格轨迹（推荐用于慕尼黑场景）

    UE 沿水平或垂直方向移动，在路口随机转弯（左转/右转/直行）。
    这种轨迹会导致 UE 在转弯时经历 LOS/NLOS 突变，
    是 A3 最容易失效的场景。

    在慕尼黑场景中：
    - 街道间距约 60~100m
    - 转弯时 UE 从一条街道进入另一条街道
    - LOS/NLOS 状态会在转弯时突变
    """
    xmin, xmax, ymin, ymax = scene_bounds
    margin = 30.0

    # 初始方向：只能是 0°, 90°, 180°, 270°（沿主要街道方向）
    directions = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    direction = float(rng.choice(directions))

    # 街区长度（慕尼黑典型值：60~100m）
    block_length = MUNICH_BLOCK_LENGTH_M
    dist_to_next_turn = block_length * (0.5 + float(rng.random()) * 0.5)
    dist_traveled = 0.0

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    pos[0] = [x0, y0]
    vel[0] = [speed_ms * math.cos(direction), speed_ms * math.sin(direction)]

    for t in range(1, num_slots):
        vx = speed_ms * math.cos(direction)
        vy = speed_ms * math.sin(direction)

        new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        # 边界反弹
        if new_pos[0] < xmin + margin or new_pos[0] > xmax - margin:
            direction = math.pi - direction
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        if new_pos[1] < ymin + margin or new_pos[1] > ymax - margin:
            direction = -direction
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        pos[t] = new_pos
        vel[t] = [vx, vy]

        dist_traveled += speed_ms * dt

        # 到达路口：随机转弯
        if dist_traveled >= dist_to_next_turn:
            turn_choice = float(rng.random())
            if turn_choice < 0.25:
                direction += math.pi / 2   # 左转 90°
            elif turn_choice < 0.50:
                direction -= math.pi / 2   # 右转 90°
            # else: 直行

            direction = direction % (2 * math.pi)
            dist_to_next_turn = block_length * (0.5 + float(rng.random()) * 0.5)
            dist_traveled = 0.0

    return pos, vel


def _generate_munich_walk(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    scene_bounds: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    慕尼黑街道行走轨迹（新增，专为慕尼黑场景设计）

    与 street_grid 的区别：
    1. 支持慕尼黑的对角线街道方向（45°/135°等）
    2. 更高的转弯概率（模拟城市中频繁转弯）
    3. 街区长度更接近慕尼黑实际值（约 80m）
    4. 转弯后有短暂的"过渡段"（模拟路口减速）

    这种轨迹特别适合测试：
    - 转弯时的 LOS/NLOS 突变（A3 TTT 失效）
    - 多次穿越小区边界（切换频率高）
    - 不同方向的 Doppler 频移变化
    """
    xmin, xmax, ymin, ymax = scene_bounds
    margin = 30.0

    # 慕尼黑街道方向（弧度）
    street_dirs_rad = [math.radians(d) for d in MUNICH_STREET_DIRECTIONS_DEG]

    # 初始方向：随机选择一个街道方向
    direction = float(rng.choice(street_dirs_rad))

    # 街区长度（慕尼黑典型值）
    block_length = MUNICH_BLOCK_LENGTH_M
    dist_to_next_turn = block_length * (0.3 + float(rng.random()) * 0.7)
    dist_traveled = 0.0

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    pos[0] = [x0, y0]
    vel[0] = [speed_ms * math.cos(direction), speed_ms * math.sin(direction)]

    # 转弯过渡状态
    in_turn = False
    turn_slots_remaining = 0
    next_direction = direction

    for t in range(1, num_slots):
        # 处理转弯过渡（模拟路口减速/转弯）
        if in_turn:
            turn_slots_remaining -= 1
            if turn_slots_remaining <= 0:
                in_turn = False
                direction = next_direction

        vx = speed_ms * math.cos(direction)
        vy = speed_ms * math.sin(direction)

        new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        # 边界处理：转向最近的街道方向
        if new_pos[0] < xmin + margin or new_pos[0] > xmax - margin:
            # 选择一个向内的街道方向
            if new_pos[0] < xmin + margin:
                direction = 0.0  # 向东
            else:
                direction = math.pi  # 向西
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        if new_pos[1] < ymin + margin or new_pos[1] > ymax - margin:
            if new_pos[1] < ymin + margin:
                direction = math.pi / 2  # 向北
            else:
                direction = 3 * math.pi / 2  # 向南
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        pos[t] = new_pos
        vel[t] = [vx, vy]

        dist_traveled += speed_ms * dt

        # 到达路口：随机转弯（慕尼黑场景转弯概率更高）
        if dist_traveled >= dist_to_next_turn and not in_turn:
            turn_choice = float(rng.random())

            if turn_choice < 0.35:
                # 左转：选择当前方向左侧 90° 的街道方向
                next_direction = (direction + math.pi / 2) % (2 * math.pi)
                # 对齐到最近的街道方向
                next_direction = _snap_to_street_direction(next_direction, street_dirs_rad)
                in_turn = True
                turn_slots_remaining = max(1, int(0.5 / dt))  # 0.5s 转弯过渡
            elif turn_choice < 0.70:
                # 右转
                next_direction = (direction - math.pi / 2) % (2 * math.pi)
                next_direction = _snap_to_street_direction(next_direction, street_dirs_rad)
                in_turn = True
                turn_slots_remaining = max(1, int(0.5 / dt))
            elif turn_choice < 0.80:
                # U 形转弯（偶尔）
                next_direction = (direction + math.pi) % (2 * math.pi)
                in_turn = True
                turn_slots_remaining = max(1, int(1.0 / dt))
            # else: 直行（30% 概率）

            dist_to_next_turn = block_length * (0.3 + float(rng.random()) * 0.7)
            dist_traveled = 0.0

    return pos, vel


def _snap_to_street_direction(direction_rad: float, street_dirs_rad: List[float]) -> float:
    """将方向对齐到最近的街道方向"""
    diffs = [abs(math.sin(direction_rad - d)) for d in street_dirs_rad]
    return street_dirs_rad[int(np.argmin(diffs))]


def _generate_stop_and_go(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    scene_bounds: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    停留-移动交替轨迹（模拟等红灯）

    UE 会在小区边界附近停留一段时间，然后快速移动。
    这会导致 A3 在停留期间频繁触发和取消，产生乒乓切换。

    在慕尼黑场景中：
    - 停留位置模拟路口等红灯
    - 移动方向沿街道方向
    - 停留时间：5~15 秒
    - 移动时间：10~30 秒
    """
    xmin, xmax, ymin, ymax = scene_bounds
    margin = 30.0

    # 初始方向：沿街道方向
    street_dirs_rad = [math.radians(d) for d in [0.0, 90.0, 180.0, 270.0]]
    direction = float(rng.choice(street_dirs_rad))

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    pos[0] = [x0, y0]
    vel[0] = [speed_ms * math.cos(direction), speed_ms * math.sin(direction)]

    is_moving = True
    state_timer = 0
    state_duration = int((10 + float(rng.random()) * 10) / dt)

    for t in range(1, num_slots):
        state_timer += 1

        if state_timer >= state_duration:
            is_moving = not is_moving
            state_timer = 0
            if is_moving:
                # 开始移动：随机选择街道方向
                direction = float(rng.choice(street_dirs_rad))
                state_duration = int((10 + float(rng.random()) * 20) / dt)
            else:
                state_duration = int((5 + float(rng.random()) * 10) / dt)

        if is_moving:
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
        else:
            vx, vy = 0.0, 0.0

        new_pos = pos[t - 1] + np.array([vx, vy]) * dt
        new_pos[0] = float(np.clip(new_pos[0], xmin + margin, xmax - margin))
        new_pos[1] = float(np.clip(new_pos[1], ymin + margin, ymax - margin))

        pos[t] = new_pos
        vel[t] = [vx, vy]

    return pos, vel


# =========================================================
# 批量生成轨迹
# =========================================================

def generate_all_trajectories(
    cfg: SimConfig,
    bs_positions: np.ndarray,
    scene_bounds: Tuple[float, float, float, float],
    seed: int = 42,
) -> List[Trajectory]:
    """
    生成所有轨迹

    参数：
        cfg:          仿真配置
        bs_positions: [num_cells, 2] 基站位置
        scene_bounds: 场景边界 [xmin, xmax, ymin, ymax]
        seed:         随机种子

    返回：
        trajectories: 所有轨迹的列表

    轨迹分配策略（针对慕尼黑场景优化）：
        - munich_walk：最真实，专为慕尼黑设计（40%）
        - street_grid：沿网格街道移动（30%）
        - arc：圆弧转弯（15%）
        - stop_and_go：停留-移动交替（10%）
        - linear：直线（5%，用于基础验证）
    """
    rng = np.random.default_rng(seed)

    num_speeds = len(cfg.speeds_ms)
    traj_per_speed = cfg.num_trajectories // num_speeds

    # 轨迹类型分配（针对慕尼黑场景优化）
    # munich_walk 和 street_grid 最真实，分配最多
    traj_types = [
        "munich_walk",   # 40%：慕尼黑街道行走
        "street_grid",   # 20%：网格街道
        "munich_walk",   # 重复以增加比例
        "arc",           # 15%：圆弧转弯
        "street_grid",   # 重复
        "stop_and_go",   # 10%：停留-移动
        "munich_walk",   # 重复
        "linear",        # 5%：直线
        "munich_walk",   # 重复
        "street_grid",   # 重复
    ]

    trajectories = []

    print(f"开始生成轨迹：{cfg.num_trajectories} 条，{num_speeds} 种速度")
    print(f"场景边界：X=[{scene_bounds[0]:.0f}, {scene_bounds[1]:.0f}]m, "
          f"Y=[{scene_bounds[2]:.0f}, {scene_bounds[3]:.0f}]m")

    for s_idx, speed_ms in enumerate(cfg.speeds_ms):
        speed_kmh = cfg.speeds_kmh[s_idx]
        print(f"\n速度：{speed_kmh:.0f} km/h ({speed_ms:.2f} m/s)")

        for t_idx in range(traj_per_speed):
            traj_type = traj_types[t_idx % len(traj_types)]

            traj = generate_trajectory(
                cfg=cfg,
                speed_ms=speed_ms,
                traj_type=traj_type,
                scene_bounds=scene_bounds,
                bs_positions=bs_positions,
                rng=rng,
            )

            trajectories.append(traj)

            if (t_idx + 1) % 5 == 0 or t_idx == traj_per_speed - 1:
                print(
                    f"  轨迹 {t_idx+1}/{traj_per_speed} ({traj_type}): "
                    f"{traj.num_slots} slots, "
                    f"距离≈{speed_ms * traj.num_slots * cfg.slot_duration:.0f}m"
                )

    print(f"\n轨迹生成完成：共 {len(trajectories)} 条")
    return trajectories


# =========================================================
# 独立运行：测试轨迹生成
# =========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config import DEFAULT_CONFIG
    from scene_setup import compute_hexagonal_bs_positions

    cfg = DEFAULT_CONFIG
    bs_positions = compute_hexagonal_bs_positions(cfg.num_cells, cfg.isd)
    # 使用固定的慕尼黑场景边界 ±500m，与 scene_setup.py 保持一致
    scene_bounds = (-500.0, 500.0, -500.0, 500.0)

    print("测试轨迹生成（慕尼黑场景）...")

    rng = np.random.default_rng(42)
    traj_types = ["linear", "street_grid", "munich_walk", "arc", "stop_and_go"]
    speed_ms = 30.0 / 3.6

    fig, axes = plt.subplots(1, len(traj_types), figsize=(25, 5))

    for i, traj_type in enumerate(traj_types):
        traj = generate_trajectory(
            cfg=cfg,
            speed_ms=speed_ms,
            traj_type=traj_type,
            scene_bounds=scene_bounds,
            bs_positions=bs_positions,
            rng=rng,
        )

        ax = axes[i]
        ax.plot(traj.pos[:, 0], traj.pos[:, 1], "b-", linewidth=0.8, alpha=0.7)
        ax.plot(traj.pos[0, 0], traj.pos[0, 1], "go", markersize=8, label="Start")
        ax.plot(traj.pos[-1, 0], traj.pos[-1, 1], "rs", markersize=8, label="End")

        for c in range(cfg.num_cells):
            ax.plot(bs_positions[c, 0], bs_positions[c, 1], "r^", markersize=8)

        ax.set_title(f"{traj_type}\n({traj.num_slots} slots)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        print(f"  {traj_type}: {traj.num_slots} slots, "
              f"距离≈{speed_ms * traj.num_slots * cfg.slot_duration:.0f}m")

    plt.tight_layout()
    plt.savefig("trajectory_examples.png", dpi=150, bbox_inches="tight")
    print("\n轨迹示例已保存到 trajectory_examples.png")
    plt.close()