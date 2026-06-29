#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from sionna_rt_gui.config import GuiConfig
from sionna_rt_gui.sionna_utils import get_built_in_scenes


def test_get_built_in_scenes():
    scenes = get_built_in_scenes()
    assert len(scenes) > 0
    assert "etoile" in scenes

    cfg = GuiConfig(config_path="__memory__")
    assert cfg.default_scene_filename in scenes
