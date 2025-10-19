# 🚀 AIME训练 - GPU快速启动
# AIME Training - GPU Quick Start

**当前问题**: `CUDA available: False` - GPU驱动配置问题
**解决方案**: 使用自动修复脚本或手动配置
**状态**: ✅ 修复工具已准备就绪

---

## ⚡ 1分钟快速启动

### 在您的服务器上运行:

```bash
# 方法1: 自动修复（推荐）
cd /root/integration
./fix_gpu_and_start.sh

# 方法2: 快速手动修复
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH
cd /root/integration
nohup python3 -u deep_train_real_workflow.py --config aime_config.yaml > aime_training.log 2>&1 &
```

### 监控训练:
```bash
tail -f aime_training.log          # 查看日志
nvidia-smi                         # 查看GPU使用
grep "Pass@K" aime_training.log    # 查看分数
```

---

## 📦 您现在拥有的文件

### 本地 Mac 上:
```
/Users/zhangmingda/Desktop/agent worflow/integration/
├── fix_gpu_and_start.sh        ← ✅ 自动修复脚本
├── GPU_FIX_README.md            ← ✅ 详细修复指南
├── QUICK_START_GPU.md           ← ✅ 这个文件
├── aime_config.yaml             ← AIME训练配置
├── deep_train_real_workflow.py  ← 主训练脚本
└── ... (其他训练文件)
```

### 需要上传到服务器:
- `fix_gpu_and_start.sh` (自动修复脚本)
- 或直接在服务器上手动执行命令

---

## 🔄 完整部署流程

### 步骤1: 上传修复脚本到服务器

```bash
# 在您的Mac上执行
cd "/Users/zhangmingda/Desktop/agent worflow/integration"

# 上传脚本到服务器（替换YOUR_SERVER为实际服务器地址）
scp fix_gpu_and_start.sh root@YOUR_SERVER:/root/integration/
```

### 步骤2: 在服务器上运行

```bash
# SSH到服务器
ssh root@YOUR_SERVER

# 运行自动修复脚本
cd /root/integration
chmod +x fix_gpu_and_start.sh
./fix_gpu_and_start.sh
```

### 步骤3: 验证训练已启动

脚本会自动:
- ✓ 检测GPU (lspci)
- ✓ 查找NVIDIA驱动库
- ✓ 配置LD_LIBRARY_PATH
- ✓ 验证nvidia-smi
- ✓ 验证PyTorch CUDA
- ✓ 启动训练（后台）
- ✓ 显示初始日志

---

## ✅ 成功标志

### 您应该看到:

```
===================================================================
  AIME Training - GPU Configuration and Startup
===================================================================

[1/6] 检查GPU硬件...
  ✓ NVIDIA GPU硬件检测成功
  00:00.0 VGA compatible controller: NVIDIA Corporation ...

[2/6] 查找NVIDIA驱动库...
  ✓ 找到NVIDIA库: /usr/lib64-nvidia

[3/6] 设置环境变量...
  ✓ LD_LIBRARY_PATH=/usr/lib64-nvidia:...

[4/6] 验证nvidia-smi...
  ✓ nvidia-smi 工作正常
  A100-SXM4-40GB, 40960 MiB, 535.xx

[5/6] 验证PyTorch CUDA支持...
  ✓ PyTorch CUDA支持正常
  CUDA available: True              ← 重要!
  CUDA version: 12.6
  Device count: 1

[6/6] 启动AIME训练...
  ✓ 训练进程已启动 (PID: 12345)

===================================================================
  训练已成功启动！
===================================================================
```

### 在训练日志中:

```
Device: cuda                        ← ✓ 使用GPU
CUDA available: True                ← ✓ CUDA可用
[TrainableQwenPolicy] Device: cuda  ← ✓ Qwen模型在GPU上
Loading checkpoint shards: 100%     ← ✓ 模型加载中
[AIMEEvaluator] Loaded 30 problems  ← ✓ 数据集已加载
Epoch 1/50, Episode 1/10           ← ✓ 训练开始
```

---

## 🛠 如果自动脚本失败

### 手动4步修复:

```bash
# 步骤1: 找到NVIDIA驱动库
find /usr /opt -name "libnvidia-ml.so*" 2>/dev/null | head -1 | xargs dirname
# 记住输出路径，例如: /usr/lib64-nvidia

# 步骤2: 设置环境变量（替换路径为步骤1的输出）
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH

# 步骤3: 验证GPU
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# 两个命令都应该成功

# 步骤4: 启动训练
cd /root/integration
nohup python3 -u deep_train_real_workflow.py --config aime_config.yaml > aime_training.log 2>&1 &
```

---

## 📊 监控训练

### 实时日志:
```bash
tail -f /root/integration/aime_training.log
# Ctrl+C 退出监控
```

### Pass@K分数:
```bash
grep "Pass@K" /root/integration/aime_training.log | tail -20
```

### GPU使用:
```bash
watch -n 1 nvidia-smi
# 应该看到:
# - Python进程使用GPU
# - 显存使用 16-20GB
# - GPU利用率 > 0%
```

### 训练进度:
```bash
grep "Epoch" /root/integration/aime_training.log | tail -10
```

---

## 🎯 预期训练时间

**配置** (aime_config.yaml):
- 数据集: 30个AIME问题 (24训练 / 6测试)
- Epochs: 50
- Episodes per epoch: 10
- 每轮使用gpt-4o-mini推理

**时间估算**:
- 每个问题: ~1.5分钟
- 每个episode: 24题 × 1.5分钟 = 36分钟
- 每个epoch: 10 episodes × 36分钟 = 6小时
- **总训练时间: 50 epochs × 6小时 = 300小时 ≈ 12.5天**

**成本估算** (gpt-4o-mini):
- 输入: $0.15 / 1M tokens
- 输出: $0.60 / 1M tokens
- **预计总成本: $8-15 USD**

---

## 🔴 停止训练

### 如果需要停止:

```bash
# 查找进程ID
ps aux | grep 'deep_train_real_workflow.py --config aime' | grep -v grep

# 停止训练
kill <PID>

# 或者使用一行命令
kill $(ps aux | grep 'deep_train_real_workflow.py --config aime' | grep -v grep | awk '{print $2}')
```

---

## 📁 重要文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 训练日志 | `/root/integration/aime_training.log` | 主要日志文件 |
| 配置文件 | `/root/integration/aime_config.yaml` | 训练配置 |
| AIME数据集 | `/root/AFlow/data/AIME_2024.jsonl` | 30个问题 |
| Qwen模型 | `/root/models/Qwen2.5-7B-Instruct/` | 15GB模型 |
| 修复脚本 | `/root/integration/fix_gpu_and_start.sh` | 自动修复脚本 |

---

## 🆘 需要帮助?

### 查看详细指南:
```bash
cat GPU_FIX_README.md    # 完整修复指南
```

### 常见问题:

**Q: nvidia-smi失败?**
A: 检查 `lspci | grep -i nvidia`，如果有GPU但nvidia-smi失败，可能需要 `sudo reboot`

**Q: PyTorch检测不到CUDA?**
A: 重新安装PyTorch: `pip3 install torch --index-url https://download.pytorch.org/whl/cu126`

**Q: 训练启动后立即崩溃?**
A: 查看日志 `tail -100 aime_training.log | grep -i error`

---

**创建时间**: 2025-10-19
**状态**: ✅ 就绪 - 随时可以启动训练
**下一步**: 运行 `fix_gpu_and_start.sh` 或手动执行4步修复
