#!/bin/bash
set -e

# ==========================================
# GPU 功耗限制
# ==========================================
echo "[INFO] Setting GPU power limit to 300W per GPU..."
sudo nvidia-smi -pl 300 || echo "[WARN] Cannot set power limit (no sudo). Proceeding anyway."
nvidia-smi --query-gpu=index,temperature.gpu,power.limit,memory.free --format=csv,noheader

# 系統記憶體最佳化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_TRIM_THRESHOLD_=0

# ==========================================
# Inpaint Embedding Fine-tuning
# 14B 模型 (bfloat16) ≈ 28GB，單張 4090 (24GB) 放不下
# 使用 2 張 GPU (CUDA_VISIBLE_DEVICES=2,3)，--multi_gpu 自動分散
# ==========================================
CUDA_VISIBLE_DEVICES=1,2,3 conda run -n sean-uni3c-diffsynth python train_inpaint.py \
    --train_data  data/dl3dv_manifest_all.json \
    --output_dir  checkpoints/inpaint \
    --max_steps   2000 \
    --save_every  200 \
    --learning_rate 1e-4 \
    --grad_accum  4 \
    --nframe      25 \
    --max_area    131072 \
    --seed        1024 \
    --gradient_checkpointing \
    --multi_gpu \
    --vram_per_gpu 16
