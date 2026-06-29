#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import logging
import os
import sys


def add_project_root_to_path():
    lib_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if lib_path not in sys.path:
        sys.path.append(lib_path)


if __name__ == "__main__":
    add_project_root_to_path()

from sionna_rt_gui import AppHolder, DEFAULT_CONFIG_PATH
from sionna_rt_gui.config import load_config


def main():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Interactive Sionna RT GUI"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the GUI configuration file to use.",
    )
    parser.add_argument(
        "scene",
        type=str,
        nargs="?",
        default=None,
        help="Path to the Sionna RT scene to load (.xml file or name of a built-in scene).",
    )
    watch_group = parser.add_mutually_exclusive_group()
    watch_group.add_argument(
        "--watch", action="store_true", dest="watch", default=False
    )
    watch_group.add_argument("--no-watch", action="store_false", dest="watch")
    args = parser.parse_args()

    cfg_overrides = {
        "use_live_reload": args.watch,
    }
    cfg = load_config(args.config, scene_filename=args.scene)

    # Configure logging
    logging.basicConfig(
        level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- Initialization
    app = AppHolder(cfg, scene_filename=args.scene, overrides=cfg_overrides)

    # --- Running loop
    app.show()


if __name__ == "__main__":
    main()
