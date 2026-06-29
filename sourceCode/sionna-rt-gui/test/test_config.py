#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os

from sionna_rt_gui import CONFIGS_DIR
from sionna_rt_gui.config import load_config


def test_example_config():
    load_config(os.path.join(CONFIGS_DIR, "example.yaml"))
