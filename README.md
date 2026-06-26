# Sionna RT 移动性仿真项目

## 项目目标

用 Sionna RT（射线追踪）替代 MATLAB 的 CDL 信道模型，在真实 3D 城市场景中
生成更真实的仿真数据，用于研究 UE 侧 CSI 特征辅助切换决策。

## 与 MATLAB 项目的对应关系

```
MATLAB                          →  Python/Sionna
─────────────────────────────────────────────────
config.m                        →  config.py
generate_cell_layout.m          →  scene_setup.py（基站布局）
generate_ue_trajectory.m        →  trajectory.py
compute_rsrp.m + CDL            →  channel.py（Sionna RT 射线追踪）
extract_features.m              →  features.py
generate_dataset.m              →  generate_dataset.py（主脚本）
results/dataset.mat             →  outputs/dataset.npz
results/trajectory_data.mat     →  outputs/trajectory_data.npz
```

## 环境要求

- Windows 10/11，64位
- Python 3.10 或 3.11（推荐 3.11）
- NVIDIA GPU：RTX 4060 Ti（8GB 或 16GB 显存均可）
- CUDA 12.1 或更高版本
- 磁盘空间：至少 10GB（用于数据集和模型）

## 安装步骤

详见 `setup.md`

## 运行步骤

1. 安装环境（见 setup.md）
2. 验证环境：`python verify_env.py`
3. 生成数据集：`python generate_dataset.py`
4. 验证数据集：`python verify_dataset.py`

## 目录结构

```
C:\Users\haojia\Sionna\
├── README.md           # 本文件
├── setup.md            # 环境安装说明
├── requirements.txt    # Python 依赖
├── config.py           # 仿真参数配置（对应 MATLAB config.m）
├── scene_setup.py      # 场景和基站布局
├── trajectory.py       # UE 轨迹生成
├── channel.py          # Sionna RT 信道仿真和特征提取
├── features.py         # CSI 特征计算
├── generate_dataset.py # 主数据生成脚本
├── verify_env.py       # 环境验证脚本
├── verify_dataset.py   # 数据集验证脚本
└── outputs/            # 生成的数据集（运行后自动创建）
    ├── dataset.npz
    └── trajectory_data.npz
```

## 关键改进（相比 MATLAB 版本）

1. **真实 3D 城市场景**：使用 Sionna 内置的慕尼黑城市场景（真实建筑物布局）
2. **真实射线追踪**：计算每条路径的时延、功率、到达角，而不是统计模型均值
3. **真实 LOS/NLOS**：由建筑物遮挡自然决定，不是概率模型
4. **真实多径结构**：反射、衍射路径自然产生，不是 CDL 固定参数
5. **真实 CSI 特征**：时延扩展、Doppler 估计、到达角、LOS 指示等
6. **去掉 Ground Truth 信息**：不使用精确速度/方向，只用 UE 可观测量