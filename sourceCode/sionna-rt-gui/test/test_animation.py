#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from sionna_rt_gui.animation import Trajectory, LoopingMode


def check_position_and_direction(
    traj: Trajectory, pos: np.ndarray, direction: np.ndarray
):
    pos, direction = traj.current_position_and_direction()
    assert np.allclose(pos, pos)
    assert np.allclose(direction, direction)


def test_trajectory_basics():
    traj = Trajectory(looping_mode_i=LoopingMode.Mirror.value)
    assert traj.distance == 0.0
    assert traj.total_distance() == 0.0

    traj.add_point([1.0, 2.0, 3.0])
    assert len(traj) == 1
    assert traj.total_distance() == 0.0
    check_position_and_direction(traj, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0])

    traj.add_point([1.0, 2.0, 6.0])
    assert len(traj) == 2
    # Snaps to the latest point
    assert traj.distance == 3.0
    assert traj.total_distance() == 3.0
    check_position_and_direction(traj, [1.0, 2.0, 6.0], [0.0, 0.0, 1.0])

    traj.add_point([1.0, 4.0, 6.0])
    assert len(traj) == 3
    assert traj.distance == (3.0 + 2.0)

    traj.distance = 2.0
    check_position_and_direction(traj, [1.0, 2.0, 5.0], [0.0, 0.0, 1.0])
    traj.distance = 4.0
    check_position_and_direction(traj, [1.0, 3.0, 6.0], [0.0, 1.0, 0.0])

    traj.backward = True
    traj.distance = 2.0
    check_position_and_direction(traj, [1.0, 2.0, 5.0], [0.0, 0.0, -1.0])

    traj.clear()
    assert len(traj) == 0
    assert traj.total_distance() == 0.0
