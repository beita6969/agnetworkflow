# GPU配置修复指南
# GPU Configuration Fix Guide

**问题**: `CUDA available: False` - GPU驱动无法访问
**解决**: 配置环境变量并验证GPU可用性

---

## 🚀 快速修复（推荐）

### 方法1: 使用自动修复脚本

已为您创建了自动修复脚本 `fix_gpu_and_start.sh`，它会：
- ✓ 自动检测GPU硬件
- ✓ 自动查找NVIDIA驱动库
- ✓ 配置环境变量
- ✓ 验证GPU可用性
- ✓ 启动AIME训练

**使用步骤:**

```bash
# 1. 上传脚本到服务器
scp fix_gpu_and_start.sh root@YOUR_SERVER:/root/integration/

# 2. SSH到服务器
ssh root@YOUR_SERVER

# 3. 进入目录并运行
cd /root/integration
chmod +x fix_gpu_and_start.sh
./fix_gpu_and_start.sh
```

**脚本会自动:**
1. 检查GPU硬件
2. 查找NVIDIA驱动库路径
3. 设置LD_LIBRARY_PATH
4. 验证nvidia-smi和PyTorch CUDA
5. 启动训练（后台运行）

**查看训练进度:**
```bash
tail -f /root/integration/aime_training_gpu.log
```

---

## 🔧 手动修复（备选）

如果自动脚本无法运行，可以手动执行以下步骤：

### 步骤1: 检查GPU硬件

```bash
# 检查GPU是否存在
lspci | grep -i nvidia
# 应该显示: NVIDIA GPU型号信息

# 检查GPU设备文件
ls -l /dev/nvidia*
# 应该显示: nvidia0, nvidiactl等设备文件
```

### 步骤2: 查找NVIDIA驱动库

```bash
# 查找libnvidia-ml.so库
find /usr /opt -name "libnvidia-ml.so*" 2>/dev/null

# 常见位置:
# - /usr/lib64-nvidia/
# - /usr/lib/x86_64-linux-gnu/
# - /usr/local/cuda/lib64/
# - /opt/conda/lib/
```

### 步骤3: 设置环境变量

假设找到库在 `/usr/lib64-nvidia/`：

```bash
# 方案A: 临时设置（当前session）
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 方案B: 永久设置（推荐）
echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 步骤4: 验证GPU可用

```bash
# 测试nvidia-smi
nvidia-smi
# 应该显示: GPU信息、驱动版本、CUDA版本

# 测试PyTorch CUDA支持
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
# 应该显示: CUDA available: True, Device count: 1 (或更多)
```

### 步骤5: 启动训练

```bash
cd /root/integration

# 设置Python路径
export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH

# 启动训练（后台）
nohup python3 -u deep_train_real_workflow.py \
    --config aime_config.yaml \
    > aime_training.log 2>&1 &

# 等待几秒
sleep 5

# 查看初始日志
tail -50 aime_training.log

# 实时监控
tail -f aime_training.log
```

---

## ⚠️ 常见问题

### 问题1: nvidia-smi仍然失败

**症状:**
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

**解决:**
```bash
# 检查驱动是否安装
ls /usr/lib64-nvidia/libnvidia-ml.so*

# 如果文件存在但nvidia-smi失败，检查设备权限
ls -l /dev/nvidia*

# 尝试重新加载驱动模块
sudo modprobe nvidia

# 如果仍然失败，可能需要重启服务器
sudo reboot
```

### 问题2: PyTorch仍然检测不到CUDA

**症状:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**可能原因:**
1. **PyTorch版本与CUDA版本不匹配**

检查CUDA版本:
```bash
nvcc --version  # 或
nvidia-smi | grep "CUDA Version"
```

重新安装匹配的PyTorch:
```bash
# CUDA 12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **LD_LIBRARY_PATH没有生效**

在Python中检查:
```python
import os
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
```

3. **需要重启Python进程**

```bash
# 退出Python
exit()

# 重新设置环境变量
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

# 重新进入Python测试
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 问题3: 训练启动后立即崩溃

**检查日志:**
```bash
tail -100 aime_training.log | grep -i error
```

**常见错误:**
- **Out of memory**: 减少配置中的batch_size或env_num
- **Module not found**: 检查PYTHONPATH设置
- **API error**: 检查OpenAI API密钥

---

## 📊 成功启动的标志

训练成功启动后，您应该在日志中看到:

```
================================================================================
  REAL WORKFLOW DEEP INTEGRATION TRAINING
================================================================================

Device: cuda                          ← ✓ 使用GPU
PyTorch version: 2.8.0+cu126
CUDA available: True                  ← ✓ CUDA可用

================================================================================
Loading Trainable Qwen Policy
================================================================================
[TrainableQwenPolicy] Loading Qwen model from /root/models/Qwen2.5-7B-Instruct...
[TrainableQwenPolicy] Device: cuda    ← ✓ 模型加载到GPU
Loading checkpoint shards: 100%|████████| 4/4
[TrainableQwenPolicy] Model loaded successfully

================================================================================
Starting AIME Training
================================================================================
[AIMEEvaluator] Loaded 30 AIME problems
[DeepWorkflowEnv] Dataset: AIME
Epoch 1/50, Episode 1/10              ← ✓ 训练开始
```

**GPU使用确认:**
```bash
# 在另一个终端监控GPU
watch -n 1 nvidia-smi

# 您应该看到:
# - Python进程使用GPU
# - 显存占用 16-20GB
# - GPU利用率 > 0%
```

---

## 🎯 完整启动命令总结

**一次性完整命令** (复制粘贴到服务器终端):

```bash
# 进入目录
cd /root/integration

# 设置所有必要的环境变量
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH

# 验证GPU
echo "=== GPU验证 ==="
nvidia-smi | head -20
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 启动训练
echo ""
echo "=== 启动训练 ==="
nohup python3 -u deep_train_real_workflow.py --config aime_config.yaml > aime_training.log 2>&1 &
echo "训练已启动，进程ID: $!"

# 等待并查看初始日志
sleep 5
tail -50 aime_training.log
```

---

## 📝 监控命令

```bash
# 实时查看训练日志
tail -f aime_training.log

# 查看Pass@K分数
grep "Pass@K" aime_training.log | tail -20

# 查看Epoch进度
grep "Epoch" aime_training.log | tail -10

# 查看GPU使用
nvidia-smi

# 查看训练进程
ps aux | grep deep_train_real_workflow.py | grep -v grep

# 停止训练
kill $(ps aux | grep 'deep_train_real_workflow.py --config aime' | grep -v grep | awk '{print $2}')
```

---

## ✅ 验证清单

训练启动前，确认:
- [ ] GPU硬件检测: `lspci | grep -i nvidia` 有输出
- [ ] nvidia-smi工作: `nvidia-smi` 显示GPU信息
- [ ] PyTorch CUDA可用: `python3 -c "import torch; print(torch.cuda.is_available())"` 返回True
- [ ] Qwen模型已下载: `/root/models/Qwen2.5-7B-Instruct/` 目录存在且包含4个.safetensors文件
- [ ] AIME数据集已下载: `/root/AFlow/data/AIME_2024.jsonl` 文件存在
- [ ] 配置文件正确: `aime_config.yaml` 中的API密钥已设置

---

**创建时间**: 2025-10-19
**目的**: 修复CUDA GPU驱动配置问题并成功启动AIME训练
**状态**: 已准备好使用
