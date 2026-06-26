"""
verify_env.py
=============
环境验证脚本

运行方式：
    python verify_env.py

检查项目：
  1. Python 版本
  2. CUDA 和 GPU 可用性
  3. PyTorch 安装和 GPU 支持
  4. Sionna 安装
  5. 其他依赖包
  6. 磁盘空间
  7. 快速功能测试（不需要下载场景）
"""

from __future__ import annotations

import sys
import os
import platform
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> bool:
    """打印检查结果"""
    status = "✓" if ok else "✗"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"：{detail}"
    print(msg)
    return ok


def main():
    print("=" * 60)
    print("Sionna RT 环境验证")
    print("=" * 60)
    print(f"操作系统：{platform.system()} {platform.release()}")
    print(f"Python：{sys.version}")
    print()

    all_ok = True

    # =========================================================
    # 1. Python 版本检查
    # =========================================================
    print("【1】Python 版本")
    py_ver = sys.version_info
    py_ok = py_ver >= (3, 10)
    all_ok &= check(
        "Python >= 3.10",
        py_ok,
        f"当前版本 {py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    )
    print()

    # =========================================================
    # 2. CUDA 检查
    # =========================================================
    print("【2】CUDA 和 GPU")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            check("nvidia-smi 可用", True, gpu_info)
        else:
            check("nvidia-smi 可用", False, "未找到 NVIDIA GPU 或驱动未安装")
            all_ok = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        check("nvidia-smi 可用", False, "nvidia-smi 命令不可用")
        all_ok = False
    print()

    # =========================================================
    # 3. PyTorch 检查
    # =========================================================
    print("【3】PyTorch")
    try:
        import torch
        torch_ok = True
        check("torch 已安装", True, f"版本 {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        all_ok &= check(
            "CUDA 可用",
            cuda_available,
            f"{'是' if cuda_available else '否（请检查 CUDA 安装）'}"
        )

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            check("GPU 信息", True, f"{gpu_name}，显存 {gpu_mem:.1f} GB")

            # 简单的 GPU 计算测试
            try:
                x = torch.randn(1000, 1000, device="cuda")
                y = torch.mm(x, x)
                del x, y
                torch.cuda.empty_cache()
                check("GPU 计算测试", True, "矩阵乘法正常")
            except Exception as e:
                check("GPU 计算测试", False, str(e))
                all_ok = False

            # CUDA 版本
            cuda_ver = torch.version.cuda
            check("CUDA 版本", True, f"{cuda_ver}")
    except ImportError:
        check("torch 已安装", False, "未安装，请运行：pip install torch --index-url https://download.pytorch.org/whl/cu121")
        all_ok = False
    print()

    # =========================================================
    # 4. Sionna 检查
    # =========================================================
    print("【4】Sionna")
    try:
        import sionna
        sionna_ver = sionna.__version__
        check("sionna 已安装", True, f"版本 {sionna_ver}")

        # 检查 Sionna RT 模块
        try:
            from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
            check("sionna.rt 模块", True, "射线追踪模块可用")
        except ImportError as e:
            check("sionna.rt 模块", False, str(e))
            all_ok = False

        # 检查内置场景是否可访问（不下载）
        try:
            munich_path = getattr(sionna.rt.scene, "munich", None)
            check(
                "内置场景（munich）",
                munich_path is not None,
                "可访问" if munich_path else "不可访问，可能需要下载"
            )
        except Exception as e:
            check("内置场景（munich）", False, str(e))

    except ImportError:
        check("sionna 已安装", False, "未安装，请运行：pip install sionna")
        all_ok = False
    print()

    # =========================================================
    # 5. 其他依赖包
    # =========================================================
    print("【5】其他依赖包")
    deps = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("h5py", "h5py"),
        ("sklearn", "scikit-learn"),
    ]

    for import_name, pkg_name in deps:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "未知版本")
            check(f"{pkg_name}", True, f"版本 {ver}")
        except ImportError:
            check(f"{pkg_name}", False, f"未安装，请运行：pip install {pkg_name}")
            all_ok = False
    print()

    # =========================================================
    # 6. 磁盘空间检查
    # =========================================================
    print("【6】磁盘空间")
    try:
        import shutil
        sionna_dir = Path("C:/Users/haojia/Sionna")
        total, used, free = shutil.disk_usage(sionna_dir.drive + "\\")
        free_gb = free / 1024**3
        space_ok = free_gb >= 10.0
        all_ok &= check(
            "可用磁盘空间 >= 10GB",
            space_ok,
            f"当前可用 {free_gb:.1f} GB"
        )
    except Exception as e:
        check("磁盘空间检查", False, str(e))
    print()

    # =========================================================
    # 7. 项目文件检查
    # =========================================================
    print("【7】项目文件")
    project_dir = Path("C:/Users/haojia/Sionna")
    required_files = [
        "config.py",
        "scene_setup.py",
        "trajectory.py",
        "channel.py",
        "generate_dataset.py",
    ]

    for fname in required_files:
        fpath = project_dir / fname
        check(f"{fname}", fpath.exists(), "存在" if fpath.exists() else "缺失")
    print()

    # =========================================================
    # 8. 快速功能测试（不需要 Sionna）
    # =========================================================
    print("【8】快速功能测试")

    # 测试 config.py
    try:
        sys.path.insert(0, str(project_dir))
        from config import DEFAULT_CONFIG, get_umi_config
        cfg = get_umi_config()
        check("config.py", True, f"UMi 配置，特征维度={cfg.num_features}")
    except Exception as e:
        check("config.py", False, str(e))
        all_ok = False

    # 测试 trajectory.py（不需要 Sionna）
    try:
        from trajectory import generate_trajectory
        import numpy as np
        from config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG
        bs_pos = np.array([[0, 0], [200, 0], [-200, 0]], dtype=np.float32)
        bounds = (-500, 500, -500, 500)
        traj = generate_trajectory(cfg, 30/3.6, "street_grid", bounds, bs_pos)
        check("trajectory.py", True, f"street_grid 轨迹，{traj.num_slots} slots")
    except Exception as e:
        check("trajectory.py", False, str(e))
        all_ok = False

    # 测试 channel.py（不需要 Sionna）
    try:
        from channel import apply_l3_filter, build_feature_vector, _compute_sinr
        import numpy as np
        from config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG
        C = cfg.num_cells

        # 测试 L3 滤波
        rsrp_test = np.random.randn(50, C).astype(np.float32) * 5 - 80
        rsrp_l3 = apply_l3_filter(rsrp_test, cfg.l3_alpha)
        assert rsrp_l3.shape == (50, C)

        # 测试特征向量构建
        feat = build_feature_vector(
            rsrp_l3=rsrp_l3[0],
            rsrq=np.full(C, -10.0, dtype=np.float32),
            sinr=np.full(C, 5.0, dtype=np.float32),
            doppler_est=np.zeros(C, dtype=np.float32),
            beam_id=np.zeros(C, dtype=np.int32),
            rsrp_diff=np.zeros(C, dtype=np.float32),
            beam_id_diff=np.zeros(C, dtype=np.float32),
            delay_spread_ns=np.zeros(C, dtype=np.float32),
            los_indicator=np.zeros(C, dtype=np.float32),
            num_beams=cfg.num_beams,
        )
        assert feat.shape[0] == cfg.num_features
        check("channel.py", True, f"特征维度={feat.shape[0]}，L3 滤波正常")
    except Exception as e:
        check("channel.py", False, str(e))
        all_ok = False

    print()

    # =========================================================
    # 总结
    # =========================================================
    print("=" * 60)
    if all_ok:
        print("✓ 所有检查通过！可以运行 generate_dataset.py")
        print()
        print("运行命令：")
        print("  cd C:\\Users\\haojia\\Sionna")
        print("  python generate_dataset.py")
        print()
        print("预计运行时间（RTX 4060 Ti）：")
        print("  UMi 场景，60 条轨迹：约 1~2 小时")
        print("  UMa 场景，120 条轨迹：约 3~5 小时")
    else:
        print("✗ 部分检查未通过，请根据上方提示修复后重新运行")
        print()
        print("常见问题解决：")
        print("  1. CUDA 不可用：更新 NVIDIA 驱动，重新安装 PyTorch")
        print("  2. Sionna 未安装：pip install sionna")
        print("  3. 其他包缺失：pip install -r requirements.txt")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())