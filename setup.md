# 环境安装说明

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
# 安装 PyTorch 2.x，支持 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证 GPU 可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# 应该显示：CUDA available: True
#           GPU: NVIDIA GeForce RTX 4060 Ti
```

如果显示 CUDA available: False，说明 CUDA 版本不匹配，尝试：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 第五步：安装 Sionna

```bash
# 安装 Sionna（当前最新版本 1.x）
pip install sionna

# 验证安装
python -c "import sionna; print('Sionna version:', sionna.__version__)"
```

**注意**：Sionna 1.x 基于 DrJit，不再依赖 TensorFlow，安装更简单。
如果安装失败，尝试指定版本：
```bash
pip install sionna==1.0.0
```

---

## 第六步：安装其他依赖

```bash
# 安装项目其他依赖
pip install -r C:\Users\haojia\Sionna\requirements.txt
```

---

## 第七步：验证完整环境

```bash
# 切换到项目目录
cd C:\Users\haojia\Sionna

# 运行环境验证脚本
python verify_env.py
```

如果所有检查都通过，显示"✓ 环境验证通过"，即可开始运行仿真。

---

## 常见问题

### 问题1：pip install sionna 报错 "No module named drjit"

```bash
pip install drjit
pip install sionna
```

### 问题2：CUDA out of memory

减小 `config.py` 中的 `num_samples_per_ray`（默认 1e6，可改为 5e5）。

### 问题3：场景加载失败

Sionna 内置场景需要从网络下载，确保网络连接正常。
如果下载失败，可以手动下载后放到 Sionna 缓存目录：
`C:\Users\haojia\.sionna\scenes\`

### 问题4：import sionna 报错 "DLL load failed"

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