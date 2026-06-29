#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import polyscope as ps


def tick():
    pass


def main():
    # Pre-init settings
    ps.set_program_name("Hi")
    # Window size and position will be loaded from the last run from `.polyscope.ini`
    ps.set_use_prefs_file(True)
    ps.set_enable_vsync(True)
    ps.set_max_fps(-1)
    ps.set_user_gui_is_on_right_side(False)
    ps.set_verbosity(1)
    ps.set_build_default_gui_panels(False)
    # ps.set_background_color(self.cfg.background_color)
    ps.set_ground_plane_mode("none")
    ps.set_window_resizable(True)
    # ps.set_window_size(*self.cfg.default_resolution)
    ps.set_give_focus_on_show(True)
    ps.set_transparency_mode("pretty")

    ps.set_up_dir("z_up")
    ps.set_front_dir("y_front")

    # This creates an empty ImGui window for some reason
    # ps.set_user_callback(tick)
    ps.init()
    ps.show()


if __name__ == "__main__":
    main()
