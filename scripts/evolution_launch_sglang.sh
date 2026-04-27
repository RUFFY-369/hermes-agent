#!/bin/bash
# scripts/launch_sglang.sh (H100 Optimized)
# Configured for Tensor Parallelism and Maximum Throughput

MODEL_PATH=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
PORT=${2:-30000}
TP_SIZE=${3:-4} # Default to TP=4 for H100 node

echo "🚀 Launching SGLang Cluster Engine..."
echo "📍 Model: $MODEL_PATH"
echo "🛠️ TP: $TP_SIZE | Max Requests: 128 | Static Mem: 0.90"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tp "$TP_SIZE" \
    --enable-lora \
    --max-loras-per-batch 8 \
    --max-lora-rank 64 \
    --lora-target-modules all \
    --mem-fraction-static 0.90 \
    --max-running-requests 128 \
    --enable-lora-overlap-loading \
    --chunked-prefill-size 4096
