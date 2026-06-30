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

def _is_walkable(pos: np.ndarray, walkable_grid: Optional[dict]) -> bool:
    """
    检查位置是否可行走（不在建筑物内）

    参数：
        pos:           [2] UE 2D 坐标 [x, y]
        walkable_grid: compute_walkable_grid() 的返回值，None 则跳过检查

    返回：
        True = 可行走（街道），False = 不可行走（建筑物内）
    """
    if walkable_grid is None:
        return True
    grid = walkable_grid["grid"]
    xmin, ymin = walkable_grid["origin"]
    gs = walkable_grid["grid_size"]
    ix = int((pos[0] - xmin) / gs)
    iy = int((pos[1] - ymin) / gs)
    H, W = grid.shape
    if 0 <= ix < W and 0 <= iy < H:
        return bool(grid[iy, ix])
    return True  # 边界外默认可行走


def _find_walkable_direction(
    pos: np.ndarray,
    speed_ms: float,
    dt: float,
    current_direction: float,
    street_dirs_rad: List[float],
    walkable_grid: Optional[dict],
    rng: np.random.Generator,
    look_ahead_steps: int = 5,
) -> float:
    """
    当当前方向被建筑物阻挡时，找到一个可行走的街道方向。

    策略：
    1. 优先尝试左转/右转 90°（不允许 U 形转弯，避免来回踱步）
    2. 向前看 look_ahead_steps 步，确保方向真正畅通
    3. 选择畅通步数最多的方向

    参数：
        look_ahead_steps: 向前预测的步数（越多越不容易走进死胡同）
    """
    # 候选方向：左转、右转（不包含 U 形，避免来回踱步）
    candidates = [
        (current_direction + math.pi / 2) % (2 * math.pi),   # 左转 90°
        (current_direction - math.pi / 2) % (2 * math.pi),   # 右转 90°
    ]
    # 加入其他街道方向（排除 U 形）
    u_turn = (current_direction + math.pi) % (2 * math.pi)
    for d in street_dirs_rad:
        # 排除 U 形方向（与当前方向相差 > 135°）
        angle_diff = abs(math.atan2(math.sin(d - current_direction),
                                    math.cos(d - current_direction)))
        if angle_diff < 2.5:  # < 143°，不是 U 形
            if d not in candidates:
                candidates.append(d)

    # 对每个候选方向，计算向前 look_ahead_steps 步的畅通步数
    best_direction = None
    best_clear_steps = -1

    for direction in candidates:
        clear_steps = 0
        test_pos = pos.copy()
        for _ in range(look_ahead_steps):
            test_pos = test_pos + np.array([
                speed_ms * math.cos(direction),
                speed_ms * math.sin(direction),
            ]) * dt
            if _is_walkable(test_pos, walkable_grid):
                clear_steps += 1
            else:
                break

        if clear_steps > best_clear_steps:
            best_clear_steps = clear_steps
            best_direction = direction

    if best_direction is not None and best_clear_steps > 0:
        return best_direction

    # 所有方向都被阻挡（极少发生），允许 U 形转弯
    if _is_walkable(pos + np.array([math.cos(u_turn), math.sin(u_turn)]) * speed_ms * dt,
                    walkable_grid):
        return u_turn

    # 最后兜底：随机选一个街道方向
    return float(rng.choice(street_dirs_rad))


def generate_trajectory(
    cfg: SimConfig,
    speed_ms: float,
    traj_type: str,
    scene_bounds: Tuple[float, float, float, float],
    bs_positions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    walkable_grid: Optional[dict] = None,
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

    walkable_grid 参数（可选）：
        来自 scene_mgr.compute_walkable_grid()，用于碰撞检测。
        提供后，中高速轨迹（street_grid/munich_walk/arc/linear）
        会避免进入建筑物内部。
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
        pos, vel = _generate_linear(x0, y0, speed_ms, num_slots, cfg.slot_duration, rng,
                                    walkable_grid=walkable_grid)
    elif traj_type == "random":
        pos, vel = _generate_random(x0, y0, speed_ms, num_slots, cfg.slot_duration, rng)
    elif traj_type == "arc":
        pos, vel = _generate_arc(x0, y0, speed_ms, num_slots, cfg.slot_duration, rng,
                                  walkable_grid=walkable_grid)
    elif traj_type == "street_grid":
        pos, vel = _generate_street_grid(
            x0, y0, speed_ms, num_slots, cfg.slot_duration, scene_bounds, rng,
            walkable_grid=walkable_grid,
        )
    elif traj_type == "munich_walk":
        pos, vel = _generate_munich_walk(
            x0, y0, speed_ms, num_slots, cfg.slot_duration, scene_bounds, rng,
            walkable_grid=walkable_grid,
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
    walkable_grid: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """直线匀速轨迹（支持碰撞检测：遇到建筑物时转向）"""
    street_dirs_rad = [math.radians(d) for d in [0.0, 90.0, 180.0, 270.0]]
    direction = float(rng.choice(street_dirs_rad))

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    pos[0] = [x0, y0]
    vel[0] = [speed_ms * math.cos(direction), speed_ms * math.sin(direction)]

    for t in range(1, num_slots):
        vx = speed_ms * math.cos(direction)
        vy = speed_ms * math.sin(direction)
        new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        # 碰撞检测：遇到建筑物时转向
        if walkable_grid is not None and not _is_walkable(new_pos, walkable_grid):
            direction = _find_walkable_direction(
                pos[t - 1], speed_ms, dt, direction, street_dirs_rad, walkable_grid, rng
            )
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        pos[t] = new_pos
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
    walkable_grid: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """圆弧轨迹（直线 → 圆弧转弯 → 直线，支持碰撞检测）"""
    street_dirs_rad = [math.radians(d) for d in [0.0, 90.0, 180.0, 270.0]]
    direction = float(rng.choice(street_dirs_rad))
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
        new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        # 碰撞检测
        if walkable_grid is not None and not _is_walkable(new_pos, walkable_grid):
            direction = _find_walkable_direction(
                pos[t - 1], speed_ms, dt, direction, street_dirs_rad, walkable_grid, rng
            )
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        pos[t] = new_pos
        vel[t] = [vx, vy]

    return pos, vel


def _generate_street_grid(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    scene_bounds: Tuple[float, float, float, float],
    rng: np.random.Generator,
    walkable_grid: Optional[dict] = None,
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

        # 碰撞检测：遇到建筑物时转向
        if walkable_grid is not None and not _is_walkable(new_pos, walkable_grid):
            direction = _find_walkable_direction(
                pos[t - 1], speed_ms, dt, direction,
                [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
                walkable_grid, rng,
            )
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
    walkable_grid: Optional[dict] = None,
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

        # 碰撞检测：遇到建筑物时立即转向
        if walkable_grid is not None and not _is_walkable(new_pos, walkable_grid):
            direction = _find_walkable_direction(
                pos[t - 1], speed_ms, dt, direction, street_dirs_rad, walkable_grid, rng
            )
            in_turn = False  # 取消当前转弯过渡
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

# =========================================================
# 速度分级的轨迹类型配置
# =========================================================

# 不同速度对应的轨迹类型分配（物理上更合理）
# 低速（≤30 km/h）：行人/自行车，频繁转弯，可以停留
# 中速（≤60 km/h）：城市车辆，路口转弯，偶尔停留
# 高速（>60 km/h）：快速路/高速，少转弯，主要直线和大弧度
TRAJ_TYPES_BY_SPEED = {
    "low": [        # ≤30 km/h：行人/自行车
        "munich_walk",   # 35%：慕尼黑街道行走（频繁转弯）
        "stop_and_go",   # 20%：停留-移动（等红灯）
        "munich_walk",
        "street_grid",   # 20%：网格街道
        "munich_walk",
        "stop_and_go",
        "munich_walk",
        "street_grid",
        "arc",           # 5%：圆弧（骑车转弯）
        "munich_walk",
    ],
    "mid": [        # 30~60 km/h：城市车辆
        "street_grid",   # 30%：网格街道（路口转弯）
        "munich_walk",   # 30%：慕尼黑街道
        "street_grid",
        "arc",           # 20%：圆弧转弯（匝道/环岛）
        "munich_walk",
        "street_grid",
        "stop_and_go",   # 10%：停留（红灯）
        "arc",
        "munich_walk",
        "street_grid",
    ],
    "high": [       # >60 km/h：快速路/高速
        "linear",        # 30%：直线（高速公路）
        "arc",           # 40%：大弧度转弯（匝道）
        "linear",
        "arc",
        "linear",
        "arc",
        "street_grid",   # 20%：网格（城市快速路）
        "linear",
        "arc",
        "street_grid",
    ],
}


def _get_traj_types_for_speed(speed_kmh: float) -> List[str]:
    """根据速度返回对应的轨迹类型列表"""
    if speed_kmh <= 30.0:
        return TRAJ_TYPES_BY_SPEED["low"]
    elif speed_kmh <= 60.0:
        return TRAJ_TYPES_BY_SPEED["mid"]
    else:
        return TRAJ_TYPES_BY_SPEED["high"]


def generate_all_trajectories(
    cfg: SimConfig,
    bs_positions: np.ndarray,
    scene_bounds: Tuple[float, float, float, float],
    seed: int = 42,
    coverage_points: Optional[np.ndarray] = None,
    walkable_grid: Optional[dict] = None,
) -> List[Trajectory]:
    """
    生成所有轨迹

    参数：
        cfg:             仿真配置
        bs_positions:    [num_cells, 2] 基站位置
        scene_bounds:    场景边界 [xmin, xmax, ymin, ymax]
        seed:            随机种子
        coverage_points: [N, 2] 可选，覆盖图中的可行走点坐标
                         如果提供，轨迹起点从这些点中选择（更均匀的分布）
                         如果为 None，从基站附近的街道交叉口选择
        walkable_grid:   可选，来自 scene_mgr.compute_walkable_grid()
                         提供后，中高速轨迹会避免进入建筑物内部

    返回：
        trajectories: 所有轨迹的列表

    轨迹分配策略（速度分级，物理上更合理）：
        低速（≤30 km/h）：行人/自行车，频繁转弯，可以停留
        中速（≤60 km/h）：城市车辆，路口转弯，偶尔停留（使用碰撞检测）
        高速（>60 km/h）：快速路/高速，少转弯（使用碰撞检测）
    """
    rng = np.random.default_rng(seed)

    num_speeds = len(cfg.speeds_ms)
    traj_per_speed = cfg.num_trajectories // num_speeds

    trajectories = []

    print(f"开始生成轨迹：{cfg.num_trajectories} 条，{num_speeds} 种速度")
    print(f"场景边界：X=[{scene_bounds[0]:.0f}, {scene_bounds[1]:.0f}]m, "
          f"Y=[{scene_bounds[2]:.0f}, {scene_bounds[3]:.0f}]m")
    if coverage_points is not None:
        print(f"使用覆盖图起点：{len(coverage_points)} 个可行走点")
    if walkable_grid is not None:
        print(f"使用可行走网格：中高速轨迹启用碰撞检测")

    for s_idx, speed_ms in enumerate(cfg.speeds_ms):
        speed_kmh = cfg.speeds_kmh[s_idx]
        traj_types = _get_traj_types_for_speed(speed_kmh)
        print(f"\n速度：{speed_kmh:.0f} km/h ({speed_ms:.2f} m/s) "
              f"→ 轨迹类型：{set(traj_types)}")

        # 中高速（>30 km/h）使用碰撞检测，低速不使用（行人可以进入建筑物附近）
        use_walkable = walkable_grid if speed_kmh > 30.0 else None

        for t_idx in range(traj_per_speed):
            traj_type = traj_types[t_idx % len(traj_types)]

            # 如果有覆盖图起点，使用覆盖图中的点作为基站位置的替代
            # 这样轨迹起点更均匀地分布在整个场景中
            if coverage_points is not None and len(coverage_points) > 0:
                effective_bs_positions = coverage_points
            else:
                effective_bs_positions = bs_positions

            traj = generate_trajectory(
                cfg=cfg,
                speed_ms=speed_ms,
                traj_type=traj_type,
                scene_bounds=scene_bounds,
                bs_positions=effective_bs_positions,
                rng=rng,
                walkable_grid=use_walkable,
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


# trajectory.py 不再包含独立运行入口
# 轨迹可视化已合并到 network_config_tool.py
# 运行：python network_config_tool.py --no-gui --save
