"""
trajectory.py
=============
UE 轨迹生成（对应 MATLAB generate_ue_trajectory.m）

主要功能：
  1. 生成多种类型的 UE 运动轨迹
  2. 轨迹沿道路网格生成（模拟真实城市移动）
  3. 支持速度自适应的轨迹长度

与 MATLAB 版本的主要区别：
  - 增加了"street_grid"轨迹类型（沿道路网格移动，含转弯）
  - 增加了"stop_and_go"轨迹类型（停留-移动交替，模拟等红灯）
  - 轨迹范围受场景边界限制（不会跑出慕尼黑场景）
  - 去掉了 Ground Truth 速度/方向的直接输出（改为估计值）
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

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
# 轨迹生成函数
# =========================================================

def generate_trajectory(
    cfg: SimConfig,
    speed_ms: float,
    traj_type: str,
    scene_bounds: Tuple[float, float, float, float],
    bs_positions: np.ndarray,
    rng: np.random.Generator | None = None,
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
        'linear'      - 直线匀速（基础验证，对应 MATLAB）
        'random'      - 随机游走（方向缓慢变化，对应 MATLAB）
        'arc'         - 圆弧转弯（对应 MATLAB）
        'street_grid' - 沿道路网格移动（新增，最真实）
                        UE 沿水平/垂直方向移动，在路口随机转弯
                        这是城市场景中最常见的移动模式
        'stop_and_go' - 停留-移动交替（新增）
                        模拟 UE 在路口等红灯后继续移动
                        会在小区边界附近停留，测试 A3 的乒乓切换
    """
    if rng is None:
        rng = np.random.default_rng()

    # 计算轨迹时隙数（速度自适应）
    # 目标：UE 穿越约 1.5 个 ISD 的距离
    target_dist = cfg.isd * 1.5
    dist_per_slot = speed_ms * cfg.slot_duration
    num_slots = min(int(math.ceil(target_dist / dist_per_slot)), 5000)
    num_slots = max(num_slots, cfg.window_size + cfg.pred_horizon + 10)

    # 选择起点（在某个基站附近随机偏移）
    anchor_idx = rng.integers(0, len(bs_positions))
    anchor_pos = bs_positions[anchor_idx]
    r0 = 0.45 * cfg.isd * math.sqrt(float(rng.random()))
    theta0 = float(rng.random()) * 2 * math.pi
    x0 = float(anchor_pos[0]) + r0 * math.cos(theta0)
    y0 = float(anchor_pos[1]) + r0 * math.sin(theta0)

    # 限制起点在场景边界内
    xmin, xmax, ymin, ymax = scene_bounds
    margin = 20.0  # 距边界至少 20m
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
    elif traj_type == "stop_and_go":
        pos, vel = _generate_stop_and_go(
            x0, y0, speed_ms, num_slots, cfg.slot_duration, scene_bounds, rng
        )
    else:
        raise ValueError(f"未知轨迹类型：{traj_type}。可选：linear, random, arc, street_grid, stop_and_go")

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
    """直线匀速轨迹（对应 MATLAB linear）"""
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
    """
    随机游走轨迹（对应 MATLAB random）
    方向随机缓慢变化，模拟城市行人/车辆
    """
    direction = float(rng.random()) * 2 * math.pi
    dir_change_std = 0.1  # 方向变化标准差 [rad/时隙]

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
    """
    圆弧轨迹（对应 MATLAB arc）
    分三段：直线 → 圆弧转弯 → 直线
    """
    direction = float(rng.random()) * 2 * math.pi
    turn_start = round(num_slots * 0.3)
    turn_end = round(num_slots * 0.7)
    turn_angle = math.pi / 2 * (2 * float(rng.random()) - 1)  # ±90°
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
    道路网格轨迹（新增，最接近真实城市移动）

    UE 沿水平或垂直方向移动，在路口随机转弯（左转/右转/直行）。
    这种轨迹会导致 UE 在转弯时经历 LOS/NLOS 突变，
    是 A3 最容易失效的场景。

    道路网格间距约为 50~100m（城市街道典型值）。
    """
    xmin, xmax, ymin, ymax = scene_bounds
    margin = 20.0

    # 初始方向：只能是 0°, 90°, 180°, 270°（沿道路方向）
    directions = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    direction = float(rng.choice(directions))

    # 路口间距（模拟城市街道间距）
    block_length = 60.0  # m（约 60m 一个路口）
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

        # 边界反弹（遇到场景边界时转向）
        if new_pos[0] < xmin + margin or new_pos[0] > xmax - margin:
            direction = math.pi - direction  # 水平反弹
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        if new_pos[1] < ymin + margin or new_pos[1] > ymax - margin:
            direction = -direction  # 垂直反弹
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
            new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        pos[t] = new_pos
        vel[t] = [vx, vy]

        # 更新行驶距离
        dist_traveled += speed_ms * dt

        # 到达路口：随机转弯
        if dist_traveled >= dist_to_next_turn:
            # 转弯选择：左转(25%)、右转(25%)、直行(50%)
            turn_choice = float(rng.random())
            if turn_choice < 0.25:
                direction += math.pi / 2  # 左转 90°
            elif turn_choice < 0.50:
                direction -= math.pi / 2  # 右转 90°
            # else: 直行，方向不变

            # 归一化方向到 [0, 2π)
            direction = direction % (2 * math.pi)

            # 重置到下一个路口的距离
            dist_to_next_turn = block_length * (0.5 + float(rng.random()) * 0.5)
            dist_traveled = 0.0

    return pos, vel


def _generate_stop_and_go(
    x0: float, y0: float,
    speed_ms: float, num_slots: int, dt: float,
    scene_bounds: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    停留-移动交替轨迹（新增）

    模拟 UE 在路口等红灯后继续移动的场景。
    UE 会在小区边界附近停留一段时间，然后快速移动。
    这会导致 A3 在停留期间频繁触发和取消，产生乒乓切换。

    停留时间：5~15 秒（对应 125~375 个时隙）
    移动时间：10~30 秒（对应 250~750 个时隙）
    """
    xmin, xmax, ymin, ymax = scene_bounds
    margin = 20.0

    direction = float(rng.random()) * 2 * math.pi

    pos = np.zeros((num_slots, 2))
    vel = np.zeros((num_slots, 2))

    pos[0] = [x0, y0]
    vel[0] = [speed_ms * math.cos(direction), speed_ms * math.sin(direction)]

    # 停留-移动状态机
    is_moving = True
    state_timer = 0
    # 初始移动时间：10~20 秒
    state_duration = int((10 + float(rng.random()) * 10) / dt)

    for t in range(1, num_slots):
        state_timer += 1

        # 状态切换
        if state_timer >= state_duration:
            is_moving = not is_moving
            state_timer = 0
            if is_moving:
                # 开始移动：随机选择新方向
                direction = float(rng.random()) * 2 * math.pi
                # 移动时间：10~30 秒
                state_duration = int((10 + float(rng.random()) * 20) / dt)
            else:
                # 开始停留：5~15 秒
                state_duration = int((5 + float(rng.random()) * 10) / dt)

        if is_moving:
            vx = speed_ms * math.cos(direction)
            vy = speed_ms * math.sin(direction)
        else:
            vx, vy = 0.0, 0.0

        new_pos = pos[t - 1] + np.array([vx, vy]) * dt

        # 边界限制
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
    生成所有轨迹（对应 MATLAB generate_dataset.m 中的轨迹循环）

    参数：
        cfg:          仿真配置
        bs_positions: [num_cells, 2] 基站位置
        scene_bounds: 场景边界
        seed:         随机种子（保证可复现性）

    返回：
        trajectories: 所有轨迹的列表

    轨迹分配策略：
        - 每种速度各 num_trajectories/num_speeds 条
        - 轨迹类型循环分配：street_grid, random, arc, stop_and_go, linear
        - 优先使用 street_grid（最真实），其次是 random 和 arc
    """
    rng = np.random.default_rng(seed)

    num_speeds = len(cfg.speeds_ms)
    traj_per_speed = cfg.num_trajectories // num_speeds

    # 轨迹类型分配（按优先级排列）
    # street_grid 最真实，分配最多；stop_and_go 用于测试乒乓切换
    traj_types = ["street_grid", "random", "arc", "stop_and_go", "linear"]

    trajectories = []
    traj_count = 0

    print(f"开始生成轨迹：{cfg.num_trajectories} 条，{num_speeds} 种速度")

    for s_idx, speed_ms in enumerate(cfg.speeds_ms):
        speed_kmh = cfg.speeds_kmh[s_idx]
        print(f"\n速度：{speed_kmh:.0f} km/h ({speed_ms:.2f} m/s)")

        for t_idx in range(traj_per_speed):
            traj_count += 1
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
    scene_bounds = (-cfg.isd * 2.5, cfg.isd * 2.5, -cfg.isd * 2.5, cfg.isd * 2.5)

    print("测试轨迹生成...")

    # 生成各类型轨迹示例
    rng = np.random.default_rng(42)
    traj_types = ["linear", "random", "arc", "street_grid", "stop_and_go"]
    speed_ms = 30.0 / 3.6  # 30 km/h

    fig, axes = plt.subplots(1, len(traj_types), figsize=(20, 4))

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
        ax.plot(traj.pos[0, 0], traj.pos[0, 1], "go", markersize=8, label="起点")
        ax.plot(traj.pos[-1, 0], traj.pos[-1, 1], "rs", markersize=8, label="终点")

        # 绘制基站
        for c in range(cfg.num_cells):
            ax.plot(bs_positions[c, 0], bs_positions[c, 1], "r^", markersize=8)

        ax.set_title(f"{traj_type}\n({traj.num_slots} slots)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        print(f"  {traj_type}: {traj.num_slots} slots, 距离≈{speed_ms * traj.num_slots * cfg.slot_duration:.0f}m")

    plt.tight_layout()
    plt.savefig("trajectory_examples.png", dpi=150, bbox_inches="tight")
    print("\n轨迹示例已保存到 trajectory_examples.png")
    plt.close()