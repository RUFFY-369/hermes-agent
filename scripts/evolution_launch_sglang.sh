#!/bin/bash
# scripts/evolution_launch_sglang.sh
# Configured for LoRA hot-swapping and flexible GPU configs.
# Supports single GPU (RTX 3090/4090) up to multi-GPU (H100 nodes).

MODEL_PATH=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
PORT=${2:-30000}
TP_SIZE=${3:-1}  # Default to TP=1 for single-GPU setups

echo "🚀 Launching SGLang Inference Server..."
echo "📍 Model: $MODEL_PATH"
echo "🛠️ TP: $TP_SIZE | Port: $PORT"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tp "$TP_SIZE" \
    --enable-lora \
    --lora-target-modules all \
    --max-loras-per-batch 8 \
    --max-lora-rank 64 \
    --mem-fraction-static 0.50 \
    --max-running-requests 128 \
    --chunked-prefill-size 4096
