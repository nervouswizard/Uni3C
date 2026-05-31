"""
Prepares training data from DL3DV-10K for train_inpaint.py.

For each DL3DV scene, this script:
  1. Reads camera poses from transforms.json (NeRFStudio format) or COLMAP sparse
  2. Selects clips of N consecutive frames
  3. Runs Depth-Pro on the first frame → builds metric point cloud
  4. Renders V_pcd (render.mp4 + render_mask.mp4) under actual DL3DV camera trajectory
  5. Exports gt_video.mp4 from the original DL3DV frames
  6. Writes train_manifest.json for train_inpaint.py

Actual DL3DV-10K directory structure (after download):
    dl3dv_root/
    └── <scene_hash>/
        └── colmap/               ← extra 'colmap/' subdirectory
            ├── transforms.json   ← camera poses + intrinsics
            ├── images/           ← full resolution
            ├── images_4/         ← 4× downsampled
            └── images_8/         ← 8× downsampled (recommended)

Download (requires HuggingFace login + access approval):
    hf auth login
    hf download DL3DV/DL3DV-10K-Sample --repo-type dataset \
        --include "*/colmap/transforms.json" "*/colmap/images_8/*" \
        --local-dir /data/DL3DV-10K

Coordinate conventions (important):
  - transforms.json stores c2w matrices in OpenGL convention (x-right, y-up, z-backward)
  - point_rendering() expects w2c matrices in OpenCV convention (x-right, y-down, z-forward)
  - This script handles the conversion automatically.

Scale normalization:
  - Depth-Pro gives metric depth (meters).
  - DL3DV COLMAP is metric-ish but may have an arbitrary global scale.
  - We normalize the camera trajectory so total motion ≈ 0.3 × scene depth.
  - Use --scale_factor to override if the rendering looks wrong.

Usage:
    python prepare_dl3dv.py \\
        --dl3dv_root /data/DL3DV-10K \\
        --output_dir data/dl3dv_renders \\
        --manifest   data/dl3dv_manifest.json \\
        --nframe 81 --clip_stride 81 --max_scenes 50

    # Then train:
    python train_inpaint.py \\
        --train_data data/dl3dv_manifest.json \\
        --output_dir checkpoints/inpaint
"""

import argparse
import json
import os
import struct
import warnings
from pathlib import Path

import einops
import numpy as np
import torch
import trimesh
import warnings
from PIL import Image, ImageOps
from pytorch3d.renderer import PointsRasterizationSettings
from torchvision.transforms import ToTensor, ToPILImage
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

import third_party.depth_pro.depth_pro as depth_pro
from src.pointcloud import point_rendering
from src.utils import points_padding, np_points_padding, set_initial_camera

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Camera loaders
# ──────────────────────────────────────────────────────────────

def load_transforms_json(scene_dir):
    """Load cameras from NeRFStudio transforms.json.

    Returns:
        frames: list of dicts, each with keys:
            'file_path', 'c2w_opencv' (4×4 np.float32), 'K' (3×3 np.float32)
        image_dir: Path to image folder
    """
    tf_path = Path(scene_dir) / "transforms.json"
    with open(tf_path) as f:
        meta = json.load(f)

    # Find image directory (prefer what's actually downloaded)
    first_fp = meta["frames"][0]["file_path"]
    original_img_subdir = Path(first_fp).parent  # e.g. "images" or "images_8"
    img_dir = Path(scene_dir) / original_img_subdir
    if not img_dir.exists():
        # Fall back to whatever resolution is available
        for candidate in ["images_8", "images_4", "images_2", "images"]:
            if (Path(scene_dir) / candidate).exists():
                img_dir = Path(scene_dir) / candidate
                break
    actual_img_subdir = img_dir.name  # e.g. "images_8"

    # Intrinsics (scene-level or per-frame)
    def get_K(entry, meta):
        w = entry.get("w", meta.get("w", None))
        h = entry.get("h", meta.get("h", None))
        fl_x = entry.get("fl_x", meta.get("fl_x", None))
        fl_y = entry.get("fl_y", meta.get("fl_y", None))
        cx   = entry.get("cx", meta.get("cx", w / 2 if w else None))
        cy   = entry.get("cy", meta.get("cy", h / 2 if h else None))
        if fl_x is None:
            fov = meta.get("camera_angle_x", None)
            fl_x = fl_y = (w / 2) / np.tan(fov / 2) if (w and fov) else None
        return np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)

    frames = []
    for entry in meta["frames"]:
        # Remap to the actually-available image subdirectory
        raw_fp = Path(entry["file_path"])
        remapped = Path(actual_img_subdir) / raw_fp.name
        fp = Path(scene_dir) / remapped
        if not fp.suffix:
            for ext in [".png", ".jpg", ".jpeg"]:
                if fp.with_suffix(ext).exists():
                    fp = fp.with_suffix(ext)
                    break

        c2w_opengl = np.array(entry["transform_matrix"], dtype=np.float32)  # 4×4, OpenGL

        # OpenGL → OpenCV: flip y and z axes of the rotation part
        # c2w_opencv = c2w_opengl @ diag(1, -1, -1, 1)
        flip = np.diag([1, -1, -1, 1]).astype(np.float32)
        c2w_opencv = c2w_opengl @ flip

        K = get_K(entry, meta)
        frames.append({"file_path": str(fp), "c2w_opencv": c2w_opencv, "K": K})

    return frames, img_dir


def load_colmap_cameras(scene_dir):
    """Load cameras from COLMAP sparse/0 (text format preferred, binary as fallback)."""
    sparse_dir = Path(scene_dir) / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = Path(scene_dir) / "sparse"

    # Try text first
    cam_txt = sparse_dir / "cameras.txt"
    img_txt = sparse_dir / "images.txt"

    if cam_txt.exists() and img_txt.exists():
        return _load_colmap_txt(scene_dir, cam_txt, img_txt)

    raise FileNotFoundError(
        f"No COLMAP text cameras found in {sparse_dir}. "
        "Convert to text with: colmap model_converter --input_path sparse/0 "
        "--output_path sparse/0 --output_type TXT"
    )


def _load_colmap_txt(scene_dir, cam_txt, img_txt):
    # Parse cameras.txt
    cameras = {}
    with open(cam_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            w, h   = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            # PINHOLE: fx fy cx cy
            if model in ("PINHOLE", "SIMPLE_PINHOLE"):
                fx = params[0]; fy = params[1] if model == "PINHOLE" else params[0]
                cx = params[2] if model == "PINHOLE" else w / 2
                cy = params[3] if model == "PINHOLE" else h / 2
            else:
                fx = fy = params[0]; cx = w / 2; cy = h / 2
            cameras[cam_id] = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

    # Parse images.txt (COLMAP format: w2c via quaternion + translation)
    frames = []
    img_dir = Path(scene_dir) / "images"
    with open(img_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz      = map(float, parts[5:8])
        cam_id = int(parts[8])
        name   = parts[9]

        # Quaternion → rotation matrix (COLMAP: w2c)
        R = _quat_to_R(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3,  3] = t
        c2w = np.linalg.inv(w2c)  # COLMAP w2c → c2w (OpenCV convention)

        K = cameras.get(cam_id, cameras[list(cameras.keys())[0]])
        fp = str(img_dir / name)
        frames.append({"file_path": fp, "c2w_opencv": c2w, "K": K})
        i += 2  # skip the point list line

    frames.sort(key=lambda x: x["file_path"])
    return frames, img_dir


def _quat_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float32)


def detect_format(scene_dir):
    scene_dir = Path(scene_dir)
    # DL3DV actual structure: <scene>/colmap/transforms.json
    if (scene_dir / "colmap" / "transforms.json").exists():
        return "nerf_dl3dv"
    # flat NeRFStudio layout
    if (scene_dir / "transforms.json").exists():
        return "nerf"
    # COLMAP inside colmap/ subdirectory
    if (scene_dir / "colmap" / "sparse").exists():
        return "colmap_dl3dv"
    if (scene_dir / "sparse").exists():
        return "colmap"
    return None


def load_scene_cameras(scene_dir):
    fmt = detect_format(scene_dir)
    if fmt == "nerf_dl3dv":
        return load_transforms_json(str(Path(scene_dir) / "colmap"))
    elif fmt == "nerf":
        return load_transforms_json(scene_dir)
    elif fmt == "colmap_dl3dv":
        return load_colmap_cameras(str(Path(scene_dir) / "colmap"))
    elif fmt == "colmap":
        return load_colmap_cameras(scene_dir)
    else:
        raise ValueError(f"No recognised camera format in {scene_dir}")


# ──────────────────────────────────────────────────────────────
# Rendering a single clip
# ──────────────────────────────────────────────────────────────

def render_clip(clip_frames, depth_model, depth_transform, output_dir,
                nframe, target_hw, scale_factor, device, raster_radius=0.008):
    """Render V_pcd for one clip and export gt_video.

    clip_frames: list of frame dicts (length >= nframe), sorted by time
    Returns True on success.
    """
    os.makedirs(output_dir, exist_ok=True)
    clip_frames = clip_frames[:nframe]
    if len(clip_frames) < nframe:
        return False

    # ── Reference image (frame 0) ──────────────────────────
    ref_path = clip_frames[0]["file_path"]
    if not os.path.exists(ref_path):
        return False

    image = Image.open(ref_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    h, w = target_hw
    image = image.resize((w, h), Image.Resampling.BICUBIC)
    validation_image = ToTensor()(image)[None]  # [1,3,h,w]

    # ── Depth estimation ────────────────────────────────────
    with torch.no_grad():
        depth_np = np.array(image)
        depth_t  = depth_transform(depth_np)
        pred     = depth_model.infer(depth_t, f_px=None)
        depth    = pred["depth"][None, None]          # [1,1,H,W]
        focallen = pred["focallength_px"].item()

    # ── Intrinsics (use Depth-Pro focal for consistency) ────
    K0 = torch.tensor([
        [focallen, 0, w / 2],
        [0, focallen, h / 2],
        [0, 0, 1],
    ], dtype=torch.float32)

    # ── Relative camera poses (all relative to frame 0) ────
    c2w_0_np = clip_frames[0]["c2w_opencv"]          # reference c2w in OpenCV
    w2c_0_np = np.linalg.inv(c2w_0_np)

    # Scale normalization: total camera path ≈ 0.3 × depth_avg
    depth_avg = float(torch.median(depth).item())
    positions = np.stack([f["c2w_opencv"][:3, 3] for f in clip_frames])
    diffs     = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_motion = float(diffs.sum()) + 1e-6
    auto_scale = (depth_avg * 0.3) / total_motion
    s = scale_factor if scale_factor is not None else auto_scale

    w2cs_list = []
    K_list    = []
    for f in clip_frames:
        c2w_i   = f["c2w_opencv"].copy()
        c2w_i[:3, 3] *= s                            # scale translation
        w2c_rel = w2c_0_np.copy()
        w2c_rel[:3, 3] *= s
        w2c_i   = np.linalg.inv(c2w_i) @ np.linalg.inv(w2c_rel)  # = w2c_i @ c2w_0
        w2cs_list.append(torch.tensor(w2c_i, dtype=torch.float32))
        K_list.append(K0)

    w2cs    = torch.stack(w2cs_list)   # [F,4,4]
    K_stack = torch.stack(K_list)      # [F,3,3]

    # ── Render V_pcd ────────────────────────────────────────
    raster_settings = PointsRasterizationSettings(
        image_size=(h, w), radius=raster_radius, points_per_pixel=8
    )
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        control_imgs, render_masks = point_rendering(
            K=K_stack.float(),
            w2cs=w2cs.float(),
            depth=depth.float(),
            image=(validation_image.float() * 2 - 1),
            raster_settings=raster_settings,
            device=device,
            background_color=[0, 0, 0],
            sobel_threshold=0.35,
        )

    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=nframe)
    render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=nframe)

    render_video, mask_video = [], []
    for i in range(nframe):
        render_video.append(ToPILImage()((control_imgs[0][:, i] + 1) / 2))
        mask_video.append(ToPILImage()(render_masks[0][:, i]))

    export_to_video(render_video, f"{output_dir}/render.mp4", fps=16)
    export_to_video(mask_video,   f"{output_dir}/render_mask.mp4", fps=16)

    # ── Save cam_info.json (same format as cam_render.py) ───
    cam_info_data = {
        "intrinsic": K0.numpy().tolist(),
        "extrinsic": [w.numpy().tolist() for w in w2cs_list],
        "height": h,
        "width":  w,
    }
    with open(f"{output_dir}/cam_info.json", "w") as _f:
        json.dump(cam_info_data, _f)

    # ── GT video from original frames ───────────────────────
    gt_frames = []
    for f in clip_frames:
        fp = f["file_path"]
        if not os.path.exists(fp):
            return False
        img = Image.open(fp).convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
        gt_frames.append(img)
    export_to_video(gt_frames, f"{output_dir}/gt_video.mp4", fps=16)

    # ── Save reference image ─────────────────────────────────
    image.save(f"{output_dir}/reference_image.png")

    return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl3dv_root",   required=True,  help="Root directory of DL3DV-10K")
    parser.add_argument("--output_dir",   required=True,  help="Where to save rendered clips")
    parser.add_argument("--manifest",     required=True,  help="Output manifest JSON path")
    parser.add_argument("--nframe",       default=81,   type=int,   help="Frames per clip")
    parser.add_argument("--clip_stride",  default=81,   type=int,   help="Gap between clip starts (frames)")
    parser.add_argument("--target_height",default=480,  type=int)
    parser.add_argument("--target_width", default=768,  type=int)
    parser.add_argument("--max_scenes",   default=None, type=int,   help="Limit number of scenes (for debugging)")
    parser.add_argument("--max_clips_per_scene", default=5, type=int)
    parser.add_argument("--scale_factor", default=None, type=float,
                        help="Override auto scale. If rendering shows no parallax, try 0.5-2.0")
    parser.add_argument("--prompt",       default="This video describes a slow and stable camera movement.",
                        help="Default prompt for all clips")
    parser.add_argument("--gpu",          default=0, type=int)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    target_hw = (args.target_height, args.target_width)

    # ── Load depth model ──────────────────────────────────
    print("Loading Depth-Pro...")
    d_model, d_transform = depth_pro.create_model_and_transforms(device=device)
    d_model = d_model.eval()

    # ── Scan scenes ───────────────────────────────────────
    dl3dv_root = Path(args.dl3dv_root)
    scene_dirs = sorted([
        d for d in dl3dv_root.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")   # skip .cache, .git, etc.
        and d.name != "*"                # skip literal '*' dir from failed hf download
        and detect_format(d)
    ])
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]
    print(f"Found {len(scene_dirs)} scenes with recognised camera format")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    total_clips = 0

    for scene_dir in scene_dirs:
        print(f"\n── Scene: {scene_dir.name}")
        try:
            frames, _ = load_scene_cameras(str(scene_dir))
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Check images exist
        frames = [f for f in frames if os.path.exists(f["file_path"])]
        if len(frames) < args.nframe:
            print(f"  Skipping: only {len(frames)} frames (need {args.nframe})")
            continue

        clips_this_scene = 0
        start = 0
        while start + args.nframe <= len(frames) and clips_this_scene < args.max_clips_per_scene:
            clip = frames[start:start + args.nframe]
            clip_id   = f"{scene_dir.name}_start{start:05d}"
            out_dir   = os.path.join(args.output_dir, scene_dir.name, f"clip_{start:05d}")

            print(f"  Clip {clip_id}...", end=" ", flush=True)
            try:
                ok = render_clip(
                    clip_frames=clip,
                    depth_model=d_model,
                    depth_transform=d_transform,
                    output_dir=out_dir,
                    nframe=args.nframe,
                    target_hw=target_hw,
                    scale_factor=args.scale_factor,
                    device=device,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                start += args.clip_stride
                continue

            if ok:
                manifest.append({
                    "reference_image": f"{out_dir}/reference_image.png",
                    "render_path":     out_dir,
                    "gt_video":        f"{out_dir}/gt_video.mp4",
                    "prompt":          args.prompt,
                })
                total_clips += 1
                clips_this_scene += 1
                print("OK")
            else:
                print("SKIP (missing files)")

            start += args.clip_stride

    # ── Write manifest ────────────────────────────────────
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total clips prepared: {total_clips}")
    print(f"Manifest written to:  {args.manifest}")
    print(f"\nNext step:")
    print(f"  python train_inpaint.py \\")
    print(f"      --train_data {args.manifest} \\")
    print(f"      --output_dir checkpoints/inpaint \\")
    print(f"      --max_steps 2000 --gradient_checkpointing")


if __name__ == "__main__":
    main()
