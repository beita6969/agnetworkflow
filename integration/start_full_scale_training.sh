#!/bin/bash
#
# Large-Scale Training Launch Script
# 大规模训练启动脚本
#
# This script starts training on the COMPLETE HumanEval dataset (131 training problems)
# 此脚本在完整HumanEval数据集上启动训练（131个训练问题）

echo "========================================================================"
echo "  Large-Scale Training - Complete HumanEval Dataset"
echo "  大规模训练 - 完整HumanEval数据集"
echo "========================================================================"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU is required for this training."
    echo "错误：未找到nvidia-smi。此训练需要GPU。"
    exit 1
fi

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
echo ""

# Check CUDA
echo "Checking PyTorch CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Check if configuration file exists
CONFIG_FILE="deep_config_full_scale.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    echo "错误：未找到配置文件：$CONFIG_FILE"
    exit 1
fi

echo "Configuration file: $CONFIG_FILE"
echo ""

# Check if Qwen model exists
MODEL_PATH="/root/models/Qwen2.5-7B-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "WARNING: Qwen model not found at $MODEL_PATH"
    echo "警告：未找到Qwen模型：$MODEL_PATH"
    echo "Please download the model first or update the model_path in $CONFIG_FILE"
    echo "请先下载模型或更新配置文件中的model_path"
    exit 1
fi

echo "Qwen model found: $MODEL_PATH"
MODEL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
echo "Model size: $MODEL_SIZE"
echo ""

# Display training configuration
echo "========================================================================"
echo "  Training Configuration"
echo "  训练配置"
echo "========================================================================"
echo "Total epochs: 30"
echo "Episodes per epoch: 5"
echo "Training problems per episode: 131 (COMPLETE training set)"
echo "Estimated time per episode: ~70 minutes"
echo "Estimated time per epoch: ~5.8 hours"
echo "Estimated total time: ~174 hours (~7.25 days)"
echo ""
echo "Output directory: ./output/full_scale_training"
echo "Log file: full_scale_training.log"
echo ""

# Confirmation
echo "========================================================================"
echo "  Ready to start large-scale training"
echo "  准备开始大规模训练"
echo "========================================================================"
echo ""
echo "This training will:"
echo "  - Train on ALL 131 training problems (not random sample)"
echo "  - Run for 30 epochs"
echo "  - Take approximately 7.25 days to complete"
echo "  - Save checkpoints every 3 epochs"
echo ""
read -p "Press ENTER to start training, or Ctrl+C to cancel... "
echo ""

# Start training
echo "Starting training..."
echo "训练开始..."
echo ""

nohup python3 deep_train_real_workflow.py \
    --config "$CONFIG_FILE" \
    > full_scale_training.log 2>&1 &

TRAIN_PID=$!

echo "Training started with PID: $TRAIN_PID"
echo "训练已启动，进程ID：$TRAIN_PID"
echo ""

# Wait a moment and check if training started successfully
sleep 5

if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "Training is running successfully!"
    echo "训练正在成功运行！"
    echo ""
    echo "Monitor training with:"
    echo "  tail -f full_scale_training.log"
    echo ""
    echo "Check training progress:"
    echo "  grep 'Pass@' full_scale_training.log | tail -20"
    echo ""
    echo "Stop training:"
    echo "  kill $TRAIN_PID"
    echo ""

    # Show initial output
    echo "========================================================================"
    echo "  Initial Training Output"
    echo "========================================================================"
    tail -50 full_scale_training.log
    echo ""
    echo "========================================================================"
    echo "Training log: full_scale_training.log"
    echo "========================================================================"
else
    echo "ERROR: Training failed to start. Check full_scale_training.log for details."
    echo "错误：训练启动失败。请查看full_scale_training.log了解详情。"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 full_scale_training.log
    exit 1
fi
