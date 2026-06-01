#!/bin/bash
set -e

# ============================================================
# PCD Completion 推論腳本
#
# 使用方式：
#   bash start_pcd_completion.sh [SUBJECT] [TRAJ] [GPU]
#
# 範例：
#   bash start_pcd_completion.sh human left 2
#   bash start_pcd_completion.sh tiger right 0
#
# 流程：
#   Stage 1（需先完成）：bash start.sh          → 生成 render 資料夾
#   Stage 2（本腳本）  ：bash start_pcd_completion.sh → 補洞補全推論
# ============================================================

SUBJECT="${1:-human}"        # 對象名稱（對應 data/tcp/input_image/ 與 outputs/tcp/）
TRAJ="${2:-left}"            # 軌跡名稱（對應 Stage 1 的 --traj_type）
GPU="${3:-2}"                # 使用的 GPU index

DATE="$(date +%Y%m%d_%H%M)"

REFERENCE_IMAGE="data/tcp/input_image/${SUBJECT}.png"
RENDER_PATH="outputs/tcp/$(date +%m%d)/${SUBJECT}/${TRAJ}"
OUTPUT_PATH="outputs/tcp/$(date +%m%d)/${SUBJECT}/${TRAJ}/result_pcd_completion_${DATE}.mp4"

# ── 若 png 不存在則試 jpg / jpeg ────────────────────────────
if [ ! -f "$REFERENCE_IMAGE" ]; then
    for ext in jpg jpeg JPG JPEG; do
        candidate="data/tcp/input_image/${SUBJECT}.${ext}"
        if [ -f "$candidate" ]; then
            REFERENCE_IMAGE="$candidate"
            break
        fi
    done
fi

echo "[INFO] Subject      : $SUBJECT"
echo "[INFO] Trajectory   : $TRAJ"
echo "[INFO] GPU          : $GPU"
echo "[INFO] Reference    : $REFERENCE_IMAGE"
echo "[INFO] Render path  : $RENDER_PATH"
echo "[INFO] Output       : $OUTPUT_PATH"

if [ ! -f "$REFERENCE_IMAGE" ]; then
    echo "[ERROR] Reference image not found: $REFERENCE_IMAGE"
    exit 1
fi

if [ ! -d "$RENDER_PATH" ]; then
    echo "[ERROR] Render path not found: $RENDER_PATH"
    echo "        Please run Stage 1 (start.sh) first."
    exit 1
fi

# ── 環境設定 ────────────────────────────────────────────────
echo "[INFO] Setting GPU power limit to 300W..."
sudo nvidia-smi -pl 300 || echo "[WARN] Cannot set power limit. Proceeding anyway."
nvidia-smi --query-gpu=index,temperature.gpu,power.limit,memory.free --format=csv,noheader

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_TRIM_THRESHOLD_=0

# ── 模式 A：單 GPU + Sequential CPU Offload（最穩定）────────
# pcd_guidance_scale : 點雲 loss 強度（建議 0.5~2.0，從 1.0 開始調）
# latent_lr          : latent 優化的 learning rate（建議 0.01~0.05）
CUDA_VISIBLE_DEVICES="$GPU" python cam_control.py \
    --reference_image "$REFERENCE_IMAGE" \
    --render_path     "$RENDER_PATH" \
    --output_path     "$OUTPUT_PATH" \
    --prompt          "This video describes a slow and stable camera movement with high quality and high definition." \
    --max_area        258048 \
    --sequential_offload \
    --pcd_completion \
    --pcd_guidance_scale 1.0 \
    --latent_lr          0.02

echo "[INFO] Done → $OUTPUT_PATH"

# ── 模式 B：3 GPU + FSDP + SP（速度快，電源充足時使用）────
# 取消下方注解並注解掉模式 A 即可切換
# ─────────────────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
#     --nproc_per_node=3 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=localhost:29500 \
#     cam_control.py \
#     --reference_image "$REFERENCE_IMAGE" \
#     --render_path     "$RENDER_PATH" \
#     --output_path     "$OUTPUT_PATH" \
#     --prompt          "This video describes a slow and stable camera movement with high quality and high definition." \
#     --max_area        258048 \
#     --enable_sp \
#     --fsdp \
#     --gradient_checkpointing \
#     --pcd_completion \
#     --pcd_guidance_scale 1.0 \
#     --latent_lr          0.02
