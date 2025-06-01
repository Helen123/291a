#!/bin/bash

# PPO Training Startup Script
# Usage: bash start_ppo_training.sh

echo "🚀 Starting PPO Training on GPU1..."

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegeneration

# Set environment variables
export CUDA_VISIBLE_DEVICES=1
export HF_ALLOW_CODE_EVAL=1

# Check GPU status
echo "📊 GPU Status:"
nvidia-smi --id=1 --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv

# Check CUDA availability
echo "🔧 CUDA Check:"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Start training
echo "🎯 Starting PPO Training..."
echo "📝 Logs will be saved to: ppo_training.log"
echo "💡 Use 'tail -f ppo_training.log' to monitor progress"

# Run training with logging
nohup python ppo_trainer.py > ppo_training.log 2>&1 &

# Get process ID
PID=$!
echo "🔄 Training process started with PID: $PID"

# Save PID for later reference
echo $PID > ppo_training.pid

echo "✅ PPO Training started successfully!"
echo ""
echo "📋 Useful commands:"
echo "  Monitor logs:     tail -f ppo_training.log"
echo "  Check process:    ps -p $PID"
echo "  Stop training:    kill $PID"
echo "  GPU usage:        watch -n 1 nvidia-smi"
echo "" 