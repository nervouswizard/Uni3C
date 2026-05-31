"""
Fine-tunes the inpaint patch_embedding of PCDController.

Only the patch_embedding (Conv3d, ~2M params) is updated; everything else is frozen.
The model learns to fill point-cloud holes by seeing render_latent + render_mask
as direct backbone input rather than only through the ControlNet side branch.

Dataset JSON format (--train_data):
    [
        {
            "reference_image": "data/tcp/input_image/lotus.png",
            "render_path":     "outputs/lotus/orbit",      <- folder with render.mp4, render_mask.mp4
            "gt_video":        "data/tcp/gt_videos/lotus_orbit.mp4",
            "prompt":          "A lotus flower."
        },
        ...
    ]

Quick start:
    python train_inpaint.py \
        --train_data data/train_manifest.json \
        --output_dir checkpoints/inpaint \
        --max_steps 2000 \
        --learning_rate 1e-4 \
        --gradient_checkpointing
"""

import argparse
import json
import os
import random

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from src.camera import get_camera_embedding
from src.dataset import load_dataset
from src.models.pcd_controller import PCDController
from src.pipelines.pipeline_pcd import PCDControllerPipeline
from src.utils import load_video, set_seed


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class InpaintDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, nframe, max_area, vae_scale_spatial=8, vae_scale_temporal=4):
        with open(manifest_path) as f:
            self.items = json.load(f)
        self.nframe = nframe
        self.max_area = max_area
        self.vae_ss = vae_scale_spatial
        self.vae_st = vae_scale_temporal

    def __len__(self):
        return len(self.items)

    def _resize_hw(self, h, w):
        mod = self.vae_ss * 2  # patch_size spatial = 2
        h = round(np.sqrt(self.max_area * h / w)) // mod * mod
        w = round(np.sqrt(self.max_area * w / h)) // mod * mod
        return h, w

    def __getitem__(self, idx):
        item = self.items[idx]

        # reference image
        img = Image.open(item["reference_image"]).convert("RGB")
        height, width = self._resize_hw(img.height, img.width)
        img = img.resize((width, height))
        img_t = ToTensor()(img) * 2 - 1  # [3,H,W] in [-1,1]

        # render video + mask (truncate to nframe to match gt_video)
        render_frames = load_video(f"{item['render_path']}/render.mp4")[:self.nframe]
        render_video = torch.stack([ToTensor()(f) for f in render_frames], dim=0) * 2 - 1
        render_video = F.interpolate(render_video, size=(height, width), mode='bicubic')[None]
        render_video[0, 0] = img_t  # first frame = reference image
        render_video = torch.clip(render_video, -1, 1)

        mask_frames = load_video(f"{item['render_path']}/render_mask.mp4")[:self.nframe]
        render_mask = torch.stack([ToTensor()(f) for f in mask_frames], dim=0)[:, 0:1]
        render_mask = F.interpolate(render_mask, size=(height, width), mode='nearest')[None]
        render_mask[render_mask < 0.5] = 0
        render_mask[render_mask >= 0.5] = 1

        # rearrange to [b,c,f,h,w]
        render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")
        render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")

        # ground truth video
        gt_frames = load_video(item["gt_video"])
        nf = min(self.nframe, len(gt_frames))
        gt_video = torch.stack([ToTensor()(f) for f in gt_frames[:nf]], dim=0) * 2 - 1
        gt_video = F.interpolate(gt_video, size=(height, width), mode='bicubic')[None]
        gt_video = torch.clip(gt_video, -1, 1)
        gt_video = einops.rearrange(gt_video, "b f c h w -> b c f h w")

        return {
            "reference_image": img,
            "render_video": render_video.squeeze(0),   # [c,f,H,W]
            "render_mask":  render_mask.squeeze(0),    # [1,f,H,W]
            "gt_video":     gt_video.squeeze(0),       # [c,f,H,W]
            "prompt":       item.get("prompt", ""),
            "height":       height,
            "width":        width,
            "render_path":  item["render_path"],       # for cam_info.json loading
        }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

@torch.no_grad()
def encode_video(vae, video, latents_mean, latents_std):
    """video: [b,c,f,H,W] float32 on GPU → normalised latent"""
    from diffusers.pipelines.wan.pipeline_wan_video2video import retrieve_latents
    lat = retrieve_latents(vae.encode(video), sample_mode="argmax")
    return (lat - latents_mean) * latents_std


def get_latent_stats(vae, device):
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
    std  = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
    return mean, std


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",   required=True,          help="Path to JSON manifest")
    parser.add_argument("--output_dir",   required=True,          help="Directory for checkpoints")
    parser.add_argument("--resume_from",  default=None,           help="Path to patch_embedding .pth to resume from")
    parser.add_argument("--max_steps",    default=2000, type=int)
    parser.add_argument("--save_every",   default=200,  type=int)
    parser.add_argument("--learning_rate",default=1e-4, type=float)
    parser.add_argument("--grad_accum",   default=4,    type=int, help="Gradient accumulation steps")
    parser.add_argument("--nframe",       default=81,   type=int)
    parser.add_argument("--max_area",     default=480*768, type=int)
    parser.add_argument("--seed",         default=1024, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Split model across all GPUs in CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=2,3)")
    parser.add_argument("--vram_per_gpu", default=21, type=int,
                        help="Max VRAM (GiB) to use per GPU for model shards (default 21)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0")  # primary device is always cuda:0 within visible set

    # ── Load models ──────────────────────────
    base_model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    cfg = OmegaConf.load(hf_hub_download(repo_id="ewrfcas/Uni3C", filename="config.json"))

    print("Loading transformer (to CPU first)...")
    transformer = PCDController.from_pretrained(
        base_model_id, subfolder="transformer",
        controlnet_cfg=cfg.controlnet_cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    transformer.build_controlnet(model_path="controlnet.pth")
    # build_controlnet creates modules in float32; cast to match transformer dtype
    for name in ["controlnet", "controlnet_patch_embedding",
                 "controlnet_mask_embedding", "controlnet_rope"]:
        if hasattr(transformer, name):
            getattr(transformer, name).to(torch.bfloat16)
    transformer.build_inpaint_embedding(checkpoint_path=args.resume_from)
    transformer.freeze_except_inpaint()
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        # controlnet is frozen — disable gc to avoid keyword-arg incompatibility
        transformer.controlnet.gradient_checkpointing = False

    if args.multi_gpu:
        from accelerate import dispatch_model, infer_auto_device_map
        n_gpus = torch.cuda.device_count()
        # GPU 0 holds only a small transformer slice so it has room for
        # VAE + text_encoder + image_encoder (~6 GiB) + encode activations (~5 GiB).
        # GPU 1+ carry the bulk of the 28 GiB transformer.
        gpu0_for_transformer = 8  # fixed: leaves ~16 GiB on GPU 0 for aux + activations
        max_memory = {0: f"{gpu0_for_transformer}GiB"}
        max_memory.update({i: f"{args.vram_per_gpu}GiB" for i in range(1, n_gpus)})
        max_memory["cpu"] = "80GiB"
        print(f"Distributing transformer across {n_gpus} GPUs "
              f"(GPU0: {gpu0_for_transformer}GiB for transformer + ~16GiB for aux/activations; "
              f"GPU1+: {args.vram_per_gpu}GiB each)...")
        device_map = infer_auto_device_map(
            transformer,
            max_memory=max_memory,
            no_split_module_classes=["WanTransformerBlock", "WanXControlNet"],
            dtype=torch.bfloat16,
        )
        # patch_embedding (trainable) must stay on GPU 0
        device_map["patch_embedding"] = 0
        transformer = dispatch_model(transformer, device_map=device_map)
    else:
        transformer = transformer.to(device)

    transformer.train()

    # T5 text encoder (~15GB) and CLIP image encoder stay on CPU to avoid OOM.
    # Only VAE goes to the last GPU (small shard + 1.5GB VAE fits comfortably).
    if args.multi_gpu and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        vae_device = torch.device(f"cuda:{n_gpus - 1}")
        enc_device = torch.device("cpu")   # T5 + CLIP on CPU
        print(f"VAE → {vae_device} | T5/CLIP → CPU (too large for GPU)")
    else:
        vae_device = device
        enc_device = device

    aux_device = vae_device  # VAE encode happens here

    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32).to(vae_device)
    vae.eval().requires_grad_(False)
    latents_mean, latents_std = get_latent_stats(vae, vae_device)

    print("Loading text encoder (CPU)...")
    tokenizer     = AutoTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder  = UMT5EncoderModel.from_pretrained(base_model_id, subfolder="text_encoder",
                                                     torch_dtype=torch.bfloat16).to(enc_device)
    text_encoder.eval().requires_grad_(False)

    print("Loading image encoder (CPU)...")
    image_encoder = CLIPVisionModel.from_pretrained(base_model_id, subfolder="image_encoder",
                                                    torch_dtype=torch.float32).to(enc_device)
    image_encoder.eval().requires_grad_(False)
    image_processor = CLIPImageProcessor.from_pretrained(base_model_id, subfolder="image_processor")

    scheduler = UniPCMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    scheduler.set_timesteps(1000, device=device)

    # ── Build pipeline (for encode_prompt / encode_image helpers) ──
    pipe = PCDControllerPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder,
        image_encoder=image_encoder, image_processor=image_processor,
        transformer=transformer, vae=vae, scheduler=scheduler,
    )

    # ── Dataset & optimizer ──────────────────
    def collate_fn(batch):
        """Keep PIL Images and strings as lists; collate tensors normally."""
        from PIL import Image as PILImage
        result = {}
        for key in batch[0]:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], (PILImage.Image, str)):
                result[key] = vals
            else:
                result[key] = torch.utils.data.dataloader.default_collate(vals)
        return result

    dataset = InpaintDataset(args.train_data, nframe=args.nframe, max_area=args.max_area)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                          num_workers=2, collate_fn=collate_fn)

    optimizer = AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=1e-2,
    )

    # ── TensorBoard ──────────────────────────
    log_dir = os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs → {log_dir}  (run: tensorboard --logdir {log_dir})")

    # ── Training ─────────────────────────────
    step = 0
    running_loss = 0.0
    best_loss = float("inf")
    loss_window = []        # sliding window for smooth loss tracking
    optimizer.zero_grad()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            height = batch["height"].item()
            width  = batch["width"].item()

            # Videos → aux_device for VAE encode; mask/latents → device for transformer
            render_video = batch["render_video"].to(aux_device, torch.float32)
            render_mask  = batch["render_mask"].to(device, torch.float32)
            gt_video     = batch["gt_video"].to(aux_device, torch.float32)
            prompt       = batch["prompt"][0]
            ref_image    = batch["reference_image"]  # PIL Image list

            # Encode render + gt to latent space (on aux_device, then move to device)
            with torch.no_grad():
                render_latent = encode_video(vae, render_video, latents_mean, latents_std).to(device)
                torch.cuda.empty_cache()
                gt_latent     = encode_video(vae, gt_video, latents_mean, latents_std).to(device)
                torch.cuda.empty_cache()

                # I2V condition — replicate pipeline's prepare_latents logic exactly.
                # Wan2.1 temporal mask has vae_scale_factor_temporal channels (=4), not 1.
                # condition shape: [1, 4+16=20, f_lat, h, w]
                num_lat_frames = gt_latent.shape[2]
                lat_h = height // pipe.vae_scale_factor_spatial
                lat_w = width  // pipe.vae_scale_factor_spatial
                vst = pipe.vae_scale_factor_temporal  # 4

                # Encode first-frame-only video as latent condition
                nf_pixel = args.nframe
                first_frame_vid = torch.cat([
                    gt_video[:, :, 0:1],
                    torch.zeros_like(gt_video[:, :, 1:])
                ], dim=2)  # [1, 3, nframe, H, W]
                latent_condition = encode_video(vae, first_frame_vid, latents_mean, latents_std).to(device)
                torch.cuda.empty_cache()

                # Build 4-channel temporal mask
                mask_pix = torch.ones(1, 1, nf_pixel, lat_h, lat_w)
                mask_pix[:, :, 1:] = 0.0
                first_mask = mask_pix[:, :, 0:1].repeat_interleave(vst, dim=2)  # [1,1,4,h,w]
                mask_pix = torch.cat([first_mask, mask_pix[:, :, 1:]], dim=2)   # [1,1,4+(nf-1),h,w]
                mask_lat = mask_pix.view(1, -1, vst, lat_h, lat_w).transpose(1, 2)  # [1,4,f_lat,h,w]
                mask_lat = mask_lat.to(device)

                condition = torch.cat([mask_lat, latent_condition], dim=1)  # [1,20,f_lat,h,w]

                # Prompt and image embeddings (T5/CLIP on CPU, result moved to device)
                prompt_embeds, _ = pipe.encode_prompt(
                    prompt=prompt, negative_prompt=None,
                    do_classifier_free_guidance=False,
                    device=enc_device,
                )
                prompt_embeds = prompt_embeds.to(device)
                image_embeds = pipe.encode_image(ref_image[0], enc_device).to(device)

                # Sample random timestep and add noise to gt_latent
                t_idx = random.randint(0, len(scheduler.timesteps) - 1)
                t     = scheduler.timesteps[t_idx].to(device)
                noise = torch.randn_like(gt_latent)
                noisy_latent = scheduler.add_noise(gt_latent.float(), noise, t.reshape(1)).to(torch.bfloat16)

            # ── Camera embedding: use real Plücker rays if cam_info.json exists ──
            add_channels = cfg.controlnet_cfg.get("add_channels", 1)
            render_path_str = batch["render_path"][0]
            cam_info_path = os.path.join(render_path_str, "cam_info.json")
            if add_channels > 1 and os.path.exists(cam_info_path):
                _ci = json.load(open(cam_info_path))
                _w2cs = torch.tensor(np.array(_ci["extrinsic"]),  dtype=torch.float32, device=device)[:args.nframe]
                _K    = torch.tensor(np.array(_ci["intrinsic"]), dtype=torch.float32)
                _K[0, :] = _K[0, :] / _ci["width"]  * width
                _K[1, :] = _K[1, :] / _ci["height"] * height
                _nf   = _w2cs.shape[0]
                _K    = _K[None].repeat(_nf, 1, 1).to(device)
                camera_embedding_train = get_camera_embedding(
                    _K, _w2cs, _nf, height, width, normalize=True
                )  # [1, 6, _nf, H, W]
                # Trim/pad to match render_mask frame count
                _tf = render_mask.shape[2]
                if camera_embedding_train.shape[2] > _tf:
                    camera_embedding_train = camera_embedding_train[:, :, :_tf]
            elif add_channels > 1:
                camera_embedding_train = torch.zeros(
                    1, add_channels - 1, *render_mask.shape[2:],
                    device=device, dtype=render_mask.dtype,
                )
            else:
                camera_embedding_train = None

            # Forward pass (only patch_embedding is trainable)
            latent_model_input = torch.cat([noisy_latent, condition.to(torch.bfloat16)], dim=1)
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(1),
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds.repeat(1, 1, 1).to(torch.bfloat16),
                render_latent=render_latent.to(torch.bfloat16),
                render_mask=render_mask,
                camera_embedding=camera_embedding_train,
                return_dict=False,
            )[0]  # [1,16,f_lat,h_lat,w_lat]

            # ── Masked loss: only penalise hole regions (render_mask=0) ──────
            # Downsample mask from pixel space to latent space
            mask_latent = F.interpolate(
                render_mask.float(),
                size=noise_pred.shape[-3:],
                mode='nearest',
            ).to(device)
            hole_mask = (1.0 - mask_latent).expand_as(noise_pred)  # 1 = hole, 0 = covered
            n_holes = hole_mask.sum().clamp(min=1.0)
            loss = (F.mse_loss(noise_pred.float(), noise.float(), reduction='none') * hole_mask
                    ).sum() / n_holes
            loss = loss / args.grad_accum
            loss.backward()

            step_loss = loss.item() * args.grad_accum
            running_loss += step_loss
            loss_window.append(step_loss)
            if len(loss_window) > 50:
                loss_window.pop(0)

            # Log per-step loss to TensorBoard
            writer.add_scalar("train/loss_step", step_loss, step)

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            step += 1

            if step % 20 == 0:
                avg20  = running_loss / 20
                smooth = sum(loss_window) / len(loss_window)
                print(f"Step {step:5d}/{args.max_steps}  "
                      f"loss(20-avg)={avg20:.4f}  "
                      f"loss(50-smooth)={smooth:.4f}  "
                      f"best={best_loss:.4f}")
                writer.add_scalar("train/loss_20avg",   avg20,  step)
                writer.add_scalar("train/loss_50smooth", smooth, step)
                running_loss = 0.0

                # Save best model based on 50-step smooth loss
                if smooth < best_loss:
                    best_loss = smooth
                    best_path = os.path.join(args.output_dir, "inpaint_emb_best.pth")
                    transformer.save_inpaint_embedding(best_path)
                    print(f"  ↑ New best ({best_loss:.4f}) → saved to {best_path}")
                    writer.add_scalar("train/best_loss", best_loss, step)

            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"inpaint_emb_step{step:05d}.pth")
                transformer.save_inpaint_embedding(ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    writer.close()

    # Final save
    final_path = os.path.join(args.output_dir, "inpaint_emb_final.pth")
    transformer.save_inpaint_embedding(final_path)
    print(f"\nTraining complete.")
    print(f"  Final checkpoint : {final_path}")
    print(f"  Best checkpoint  : {os.path.join(args.output_dir, 'inpaint_emb_best.pth')}  (loss={best_loss:.4f})")
    print(f"  TensorBoard logs : tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
