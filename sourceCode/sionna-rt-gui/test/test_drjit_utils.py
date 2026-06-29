#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import drjit as dr
import mitsuba as mi

from sionna_rt_gui.drjit_util import drjit_cleanup
from sionna_rt_gui.ps_utils import get_array_ptr, get_array_size_bytes


def test_drjit_cleanup():
    drjit_cleanup()
    assert len(dr.kernel_history()) == 0


def test_array_getters():
    arr = dr.full(mi.Float, 0.5, 124)
    ptr, shape, _, nbytes = get_array_ptr(arr)
    assert isinstance(ptr, int) and ptr > 0
    assert shape == (124,)
    assert nbytes == 124 * 4

    assert get_array_size_bytes(arr) == 124 * 4
