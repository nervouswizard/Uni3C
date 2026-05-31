#!/bin/bash
# 準備 inpaint 訓練資料（自蒸餾方式）
#
# 流程：
#   1. 對每張參考圖 + 每種軌跡，執行 cam_render.py 產生 render.mp4 / render_mask.mp4
#   2. 執行原版 cam_control.py（不加 --inpaint_mode）產生 pseudo GT 影片
#   3. 輸出 train_manifest.json
#
# 使用方式：
#   bash prepare_train_data.sh
#
# 需先準備：
#   data/train/images/    <- 放入要用的參考圖（.png 或 .jpg）

set -e

IMAGES_DIR="data/train/images"
OUTPUT_BASE="data/train/renders"
MANIFEST_PATH="data/train/manifest.json"
GPU=0

# 軌跡列表，可依需求增減
TRAJS=("orbit" "tour" "swing1" "swing2" "free1" "free2")

# 每張圖使用哪些 prompt（index 對應 IMAGES_DIR 中的圖，留空則用預設 prompt）
DEFAULT_PROMPT="This video describes a slow and stable camera movement with high quality."

mkdir -p "$OUTPUT_BASE"
echo "[" > "$MANIFEST_PATH"
first_entry=true

for img_path in "$IMAGES_DIR"/*.png "$IMAGES_DIR"/*.jpg; do
    [ -f "$img_path" ] || continue
    img_name=$(basename "${img_path%.*}")

    for traj in "${TRAJS[@]}"; do
        render_dir="$OUTPUT_BASE/${img_name}_${traj}"
        gt_video="$render_dir/gt.mp4"

        echo "======================================"
        echo "Image: $img_name  Traj: $traj"
        echo "======================================"

        # Step 1: 產生 V_pcd render
        CUDA_VISIBLE_DEVICES=$GPU python cam_render.py \
            --reference_image "$img_path" \
            --output_path "$render_dir" \
            --traj_type "$traj" \
            --nframe 81

        # Step 2: 用原版 cam_control 產生 pseudo GT
        CUDA_VISIBLE_DEVICES=$GPU systemd-run --user --scope -p MemoryMax=2G \
            python cam_control.py \
                --reference_image "$img_path" \
                --render_path "$render_dir" \
                --output_path "$gt_video" \
                --prompt "$DEFAULT_PROMPT" \
                --max_area 258048 \
                --sequential_offload

        # Step 3: 加入 manifest
        if [ "$first_entry" = false ]; then
            echo "," >> "$MANIFEST_PATH"
        fi
        first_entry=false

        cat >> "$MANIFEST_PATH" <<EOF
    {
        "reference_image": "$img_path",
        "render_path": "$render_dir",
        "gt_video": "$gt_video",
        "prompt": "$DEFAULT_PROMPT"
    }
EOF

        echo "Done: $render_dir"
    done
done

echo "]" >> "$MANIFEST_PATH"
echo ""
echo "Manifest written to: $MANIFEST_PATH"
echo "Total samples: $(grep -c 'render_path' "$MANIFEST_PATH")"
