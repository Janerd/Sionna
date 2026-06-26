# 环境安装说明

## 重要说明：Sionna 版本

本项目使用 **Sionna 2.x**（基于 PyTorch，不需要 TensorFlow）。

版本对应关系：
- Sionna 0.x / 1.x：基于 TensorFlow（旧版，不推荐）
- **Sionna 2.x：基于 PyTorch（本项目使用此版本）**

---

## 第一步：确认 CUDA 版本

打开命令提示符（Win+R → cmd），运行：

```
nvidia-smi
```

查看右上角显示的 CUDA Version，需要 **12.1 或更高**。

如果 CUDA 版本低于 12.1，需要先更新 NVIDIA 驱动：
- 访问 https://www.nvidia.com/drivers
- 选择 RTX 4060 Ti，下载最新驱动并安装

---

## 第二步：安装 Miniconda（如果还没有）

下载地址：https://docs.conda.io/en/latest/miniconda.html

选择 Windows 64-bit Python 3.11 版本，安装时勾选"Add to PATH"。

安装完成后，打开"Anaconda Prompt"（开始菜单搜索）。

---

## 第三步：创建虚拟环境

在 Anaconda Prompt 中运行：

```bash
# 创建名为 sionna 的虚拟环境，Python 3.11
conda create -n sionna python=3.11 -y

# 激活环境
conda activate sionna

# 验证 Python 版本
python --version
# 应该显示：Python 3.11.x
```

---

## 第四步：安装 PyTorch（GPU 版本）

```bash
# 安装 PyTorch 2.x，支持 CUDA 12.6（RTX 4060 Ti 推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir

# 验证 GPU 可用
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda, torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
# 应该显示：CUDA: 12.x True
#           GPU: NVIDIA GeForce RTX 4060 Ti
```

如果显示 CUDA available: False，说明 CUDA 版本不匹配，尝试：
```bash
# CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 第五步：安装 Sionna 2.x

```bash
# 安装 Sionna 2.x（基于 PyTorch，不需要 TensorFlow）
pip install sionna

# 验证安装
python -c "import sionna; print('Sionna version:', sionna.__version__)"
# 应该显示：Sionna version: 2.x.x
```

**注意**：
- Sionna 2.x 不需要 TensorFlow，只需要 PyTorch
- 如果安装的是 Sionna 1.x（需要 TensorFlow），请卸载后重新安装：
  ```bash
  pip uninstall sionna -y
  pip install sionna>=2.0
  ```

如果安装失败，尝试指定版本：
```bash
pip install sionna==2.0.1
```

---

## 第六步：安装 Sionna RT 模块（如果需要单独安装）

Sionna 2.x 中，RT（射线追踪）模块已包含在主包中。
如果遇到 `from sionna.rt import ...` 报错，尝试：

```bash
# 确认 sionna.rt 可用
python -c "from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray; print('sionna.rt OK')"
```

---

## 第七步：安装其他依赖

```bash
# 安装项目其他依赖
pip install -r requirements.txt
```

---

## 第八步：验证完整环境

```bash
# 切换到项目目录
cd C:\Users\haojia\Sionna

# 运行环境验证脚本
python verify_env.py
```

如果所有检查都通过，显示"✓ 所有检查通过"，即可开始运行仿真。

---

## 运行数据生成

```bash
cd C:\Users\haojia\Sionna

# 生成数据集（日志自动保存到 outputs/generate_dataset.log）
python generate_dataset.py

# 或者指定参数
python generate_dataset.py --scenario umi --num-traj 60 --outdir outputs
```

预计运行时间（RTX 4060 Ti）：
- UMi 场景，60 条轨迹：约 1~2 小时
- UMa 场景，120 条轨迹：约 3~5 小时

---

## 常见问题

### 问题1：AttributeError: 'Scene' object has no attribute 'compute_paths'

这是 Sionna 版本问题。Sionna 2.x 使用 `trace_paths()` 而非 `compute_paths()`。
本项目代码已针对 Sionna 2.x 更新，请确认安装的是 Sionna 2.x：

```bash
python -c "import sionna; print(sionna.__version__)"
# 应该显示 2.x.x
```

如果显示 1.x.x，请升级：
```bash
pip install --upgrade sionna
```

### 问题2：jitc_llvm_init(): LLVM API initialization failed

这是 Sionna 2.x 的已知警告，不影响功能，可以忽略。

### 问题3：CUDA out of memory

减小 `config.py` 中的 `num_samples_per_ray`（默认 1e6，可改为 5e5）：
```python
cfg.num_samples_per_ray = 500_000
```

### 问题4：场景加载失败（网络问题）

Sionna 内置场景需要从网络下载，确保网络连接正常。
如果下载失败，可以手动下载后放到 Sionna 缓存目录：
`C:\Users\haojia\.sionna\scenes\`

### 问题5：matplotlib 中文字体警告

代码已自动处理，会优先使用系统中文字体（微软雅黑等），
如果没有中文字体则自动切换为英文标签，不影响功能。

### 问题6：import sionna 报错 "DLL load failed"

这通常是 CUDA 运行时库缺失，安装 CUDA Toolkit：
- 访问 https://developer.nvidia.com/cuda-downloads
- 选择 Windows → x86_64 → 12.x → exe(local)
- 下载并安装

---

## 卸载和重建环境

如果环境出现问题，可以完全重建：

```bash
conda deactivate
conda remove -n sionna --all -y
# 然后从第三步重新开始