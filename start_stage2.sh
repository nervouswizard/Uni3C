#!/bin/bash
set -e

# ==========================================
# GPU 功耗限制（防止 3x 4090 同時滿載導致硬關機）
# 預設 450W × 3 = 1350W；限制到 300W × 3 = 900W
# ==========================================
echo "[INFO] Setting GPU power limit to 300W per GPU..."
sudo nvidia-smi -pl 300 || echo "[WARN] Cannot set power limit (no sudo). Proceeding anyway."
nvidia-smi --query-gpu=index,temperature.gpu,power.limit,memory.free --format=csv,noheader

# 系統記憶體最佳化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_TRIM_THRESHOLD_=0

# NCCL 設定（延長超時、避免 watchdog 誤殺）
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_ASYNC_ERROR_HANDLING=1

# ==========================================
# 模式 A：單 GPU + Sequential CPU Offload（最穩定，速度較慢）
# 建議先用此模式確認程式可以正常執行
# ==========================================
CUDA_VISIBLE_DEVICES=2 systemd-run --user --scope -p MemoryMax=2G python cam_control.py \
    --reference_image "data/tcp/human.png" \
    --render_path "outputs/tcp/0406/human_down" \
    --output_path "outputs/tcp/0406/human_down/result.mp4" \
    --prompt "The video features a human." \
    --max_area 258048 \
    --sequential_offload

# ==========================================
# 模式 B：3 GPU + FSDP + SP（速度較快，需確認電源充足）
# 確認模式 A 可以跑完後，再改用此模式
# ==========================================
# CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
#     --nproc_per_node=3 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=localhost:29500 \
#     cam_control.py \
#     --reference_image "data/tcp/human.png" \
#     --render_path "outputs/0405/human" \
#     --output_path "outputs/0405/human/result.mp4" \
#     --prompt "The video features a human." \
#     --max_area 258048 \
#     --nframe 9 \
#     --enable_sp \
#     --fsdp \
#     --gradient_checkpointing
