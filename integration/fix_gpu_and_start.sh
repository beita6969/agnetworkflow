#!/bin/bash
# GPU配置修复和训练启动脚本
# Fix GPU configuration and start training

echo "==================================================================="
echo "  AIME Training - GPU Configuration and Startup"
echo "  AIME训练 - GPU配置与启动"
echo "==================================================================="
echo ""

# 1. 检查GPU硬件
echo "[1/6] 检查GPU硬件..."
if lspci | grep -i nvidia > /dev/null; then
    echo "  ✓ NVIDIA GPU硬件检测成功"
    lspci | grep -i nvidia | head -3
else
    echo "  ✗ 未检测到NVIDIA GPU硬件"
    echo "  请确认服务器是否有NVIDIA GPU"
    exit 1
fi
echo ""

# 2. 查找NVIDIA驱动库
echo "[2/6] 查找NVIDIA驱动库..."
NVIDIA_LIB_PATHS=(
    "/usr/lib64-nvidia"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/lib64"
    "/usr/local/nvidia/lib64"
    "/opt/conda/lib"
)

FOUND_LIB=""
for path in "${NVIDIA_LIB_PATHS[@]}"; do
    if [ -f "$path/libnvidia-ml.so" ] || [ -f "$path/libnvidia-ml.so.1" ]; then
        FOUND_LIB="$path"
        echo "  ✓ 找到NVIDIA库: $path"
        break
    fi
done

if [ -z "$FOUND_LIB" ]; then
    echo "  ⚠ 在常见路径未找到libnvidia-ml.so，尝试搜索..."
    FOUND_LIB=$(find /usr /opt -name "libnvidia-ml.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$FOUND_LIB" ]; then
        echo "  ✓ 找到NVIDIA库: $FOUND_LIB"
    else
        echo "  ✗ 未找到NVIDIA驱动库"
        echo "  可能需要安装NVIDIA驱动: apt-get install nvidia-driver-535"
        exit 1
    fi
fi
echo ""

# 3. 设置环境变量
echo "[3/6] 设置环境变量..."
export LD_LIBRARY_PATH="$FOUND_LIB:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

echo "  ✓ LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  ✓ CUDA_HOME=$CUDA_HOME"
echo ""

# 4. 验证nvidia-smi
echo "[4/6] 验证nvidia-smi..."
if nvidia-smi > /dev/null 2>&1; then
    echo "  ✓ nvidia-smi 工作正常"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "  ✗ nvidia-smi 失败"
    echo "  尝试手动设置LD_LIBRARY_PATH后再运行"
    exit 1
fi
echo ""

# 5. 验证PyTorch CUDA
echo "[5/6] 验证PyTorch CUDA支持..."
CUDA_TEST=$(python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1)

if echo "$CUDA_TEST" | grep "CUDA available: True" > /dev/null; then
    echo "  ✓ PyTorch CUDA支持正常"
    echo "$CUDA_TEST" | sed 's/^/    /'
else
    echo "  ✗ PyTorch无法访问CUDA"
    echo "$CUDA_TEST" | sed 's/^/    /'
    echo ""
    echo "  可能的解决方案:"
    echo "  1. 重新安装PyTorch with CUDA:"
    echo "     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
    echo "  2. 检查CUDA版本是否匹配"
    exit 1
fi
echo ""

# 6. 启动训练
echo "[6/6] 启动AIME训练..."
echo "  配置文件: aime_config.yaml"
echo "  日志文件: aime_training_gpu.log"
echo ""

# 进入integration目录
cd "$(dirname "$0")"

# 设置Python路径
export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH

# 检查配置文件
if [ ! -f "aime_config.yaml" ]; then
    echo "  ✗ 未找到aime_config.yaml"
    exit 1
fi

# 启动训练（后台运行）
echo "  正在后台启动训练进程..."
nohup python3 -u deep_train_real_workflow.py \
    --config aime_config.yaml \
    > aime_training_gpu.log 2>&1 &

TRAIN_PID=$!
sleep 3

# 检查进程是否还在运行
if ps -p $TRAIN_PID > /dev/null; then
    echo "  ✓ 训练进程已启动 (PID: $TRAIN_PID)"
    echo ""
    echo "==================================================================="
    echo "  训练已成功启动！"
    echo "==================================================================="
    echo ""
    echo "监控训练进度:"
    echo "  tail -f aime_training_gpu.log"
    echo ""
    echo "查看GPU使用:"
    echo "  nvidia-smi"
    echo ""
    echo "停止训练:"
    echo "  kill $TRAIN_PID"
    echo ""
    echo "查看初始日志:"
    sleep 2
    tail -50 aime_training_gpu.log
else
    echo "  ✗ 训练进程启动失败"
    echo ""
    echo "查看错误日志:"
    cat aime_training_gpu.log
    exit 1
fi
