import argparse
import json
import os
import warnings

import einops
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from diffusers.utils import export_to_video
from pytorch3d.renderer import PointsRasterizationSettings
from torchvision.transforms import ToTensor, ToPILImage

import third_party.depth_pro.depth_pro as depth_pro
from src.pointcloud import point_rendering
from src.utils import traj_map, points_padding, np_points_padding, set_initial_camera, build_cameras
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")


def main():
    torch.set_grad_enabled(False)
    # == parse configs ==
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_image", default=None, type=str, required=True,
                        help="the path of input image")
    parser.add_argument("--output_path", default="outputs/temp", type=str,
                        help="output folder's path")
    parser.add_argument("--traj_type", default="custom_frame9", type=str,
                        choices=["custom", "custom_frame9", "free1", "free2", "free3", "free4", "free5", "swing1", "swing2", "orbit", "test1", "test_right", "test_up", "test_down"],
                        help="custom refers to a custom trajectory, while the others are pre-defined camera trajectories (see traj_map for details)")
    parser.add_argument("--nframe", default=81, type=int, help="Total number of frames")
    parser.add_argument("--d_r", default=2.0, type=float,
                        help="Camera distance, default is 1.0, range 0.25 to 2.5")
    parser.add_argument("--d_theta", default=0.0, type=float,
                        help="Vertical rotation, <0 up, >0 down, range -90 to 30; generally not recommended to angle too much downwards")
    parser.add_argument("--d_phi", default=180.0, type=float,
                        help="Horizontal rotation, <0 right, >0 left, supports 360 degrees; range -360 to 360")
    parser.add_argument("--x_offset", default=0.0, type=float,
                        help="Horizontal translation, <0 left, >0 right, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--y_offset", default=0.0, type=float,
                        help="Vertical translation, <0 up, >0 down, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--z_offset", default=0.0, type=float,
                        help="Forward and backward translation, <0 back, >0 forward, range -0.5 to 0.5 is ok; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--focal_length", default=1.0, type=float,
                        help="Focal length, range 0.25 to 2.5; changing focal length zooms in and out")
    parser.add_argument("--start_elevation", default=5.0, type=float,
                        help="Initial angle, no exceptions to change")
    parser.add_argument("--input_pcd", default=None, type=str,
                        help="Optional custom point cloud (.ply) path")

    args = parser.parse_args()
    device = "cuda"
    print("Init depth model")
    depth_model, depth_transform = depth_pro.create_model_and_transforms(device=device)
    depth_model = depth_model.eval()

    print("Init mask model")
    seg_net = TracerUniversalB7(device=device, batch_size=1,
                                model_path=hf_hub_download(repo_id="Carve/tracer_b7", filename="tracer_b7.pth")).eval()

    # == motion definition ==
    if args.traj_type == "custom" or args.traj_type == "custom_frame9":
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = \
            "free", args.x_offset, args.y_offset, args.z_offset, args.d_theta, args.d_phi, args.d_r
    else:
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(args.traj_type)

    # load image
    image = Image.open(args.reference_image).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w_origin, h_origin = image.size
    # 不使用預設的解析度列表，改為直接將原圖等比例縮放到大約 512 的基準高度/寬度
    base_res = 512
    if w_origin > h_origin:
        width = int(base_res * (w_origin / h_origin))
        height = base_res
    else:
        width = base_res
        height = int(base_res * (h_origin / w_origin))
        
    # 確保高寬能被 16 整除 (避免 FFmpeg macro_block_size 警告並確保影片相容性)
    width = (width // 16) * 16
    height = (height // 16) * 16
    
    print(f"Image: {args.reference_image.split('/')[-1]}, Resolution: {h_origin}x{w_origin}->{height}x{width}")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    validation_image = ToTensor()(image)[None]  # [1,c,h,w], 0~1

    os.makedirs(args.output_path, exist_ok=True)

    # inference depth
    with torch.no_grad():
        depth_image = np.array(image)
        depth_image = depth_transform(depth_image)
        prediction = depth_model.infer(depth_image, f_px=None)
        depth = prediction["depth"]  # Depth in [m].
        depth = depth[None, None]
        focallength_px = prediction["focallength_px"].item()  # Focal length in pixels.
        K = torch.tensor([[focallength_px, 0, width / 2],
                          [0, focallength_px, height / 2],
                          [0, 0, 1]], dtype=torch.float32)
        K_inv = K.inverse()
        intrinsic = K[None].repeat(args.nframe, 1, 1)

    # get pointcloud
    points2d = torch.stack(torch.meshgrid(torch.arange(width, dtype=torch.float32),
                                          torch.arange(height, dtype=torch.float32), indexing="xy"), -1)  # [h,w,2]
    points3d = points_padding(points2d).reshape(height * width, 3)  # [hw,3]
    points3d = (K_inv @ points3d.T * depth.reshape(1, height * width).cpu()).T
    colors = ((depth_image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(height * width, 3)
    points3d = points3d.cpu().numpy()
    colors = colors.cpu().numpy()

    # inference foreground mask
    with torch.no_grad():
        origin_w_, origin_h_ = image.size
        image_pil = image.resize((512, 512))
        fg_mask = seg_net([image_pil])[0]
        fg_mask = fg_mask.resize((origin_w_, origin_h_))
    fg_mask = np.array(fg_mask)
    fg_mask = fg_mask > 127.5
    fg_mask = torch.tensor(fg_mask)
    if fg_mask.float().mean() < 0.05:
        fg_mask[...] = True
    depth_avg = torch.median(depth[0, 0, fg_mask]).item()

    w2c_0, c2w_0 = set_initial_camera(args.start_elevation, depth_avg)

    # convert points3d to the world coordinate
    points3d = (c2w_0.numpy()[:3] @ np_points_padding(points3d).T).T
    pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
    _ = pcd.export(f"{args.output_path}/pcd.ply")

    # build camera viewpoints according to d_theta，d_phi, d_r
    w2cs, c2ws, intrinsic = build_cameras(cam_traj=cam_traj,
                                          w2c_0=w2c_0,
                                          c2w_0=c2w_0,
                                          intrinsic=intrinsic,
                                          nframe=args.nframe,
                                          focal_length=args.focal_length,
                                          d_theta=d_theta,
                                          d_phi=d_phi,
                                          d_r=d_r,
                                          radius=depth_avg,
                                          x_offset=x_offset,
                                          y_offset=y_offset,
                                          z_offset=z_offset)

    # === 讀取自訂點雲 ===
    custom_points = None
    custom_colors = None
    if args.input_pcd is not None:
        print(f"Loading custom point cloud from {args.input_pcd}")
        custom_pcd = trimesh.load(args.input_pcd)
        custom_points = torch.tensor(custom_pcd.vertices, dtype=torch.float32).to(device)
        # ========== 保持自訂點雲原樣，改為根據點雲計算相機參數 ==========
        
        # 由於此檔案的影像輸入值域為 [-1, 1]，我們需將 0~255 的點雲顏色正規化為 [-1, 1]
        custom_colors = np.array(custom_pcd.colors[:, :3])
        custom_colors = torch.tensor(custom_colors, dtype=torch.float32).to(device) / 255.0 * 2.0 - 1.0

        # 將自訂點雲的座標系轉換為 PyTorch3D 相機座標 (X向左, Y向上, Z向前) 以符合原始腳本邏輯
        # 我們假設點雲原本是 OpenGL 座標系 (Z向螢幕內，因此大部分為負值)
        # 由於您發現圖片上下顛倒，因此我們除了調整 Z 軸，也反轉 Y 軸
        if custom_points[:, 2].mean() < -0.2:
            custom_points[:, 2] = -custom_points[:, 2]  # Z 軸反轉
            custom_points[:, 1] = -custom_points[:, 1]  # Y 軸反轉 (修正上下顛倒)
            
        # ====== 修正點雲的長寬比例以符合原圖 ======
        # 計算目前點雲的長寬範圍
        min_pt = custom_points.min(dim=0)[0]
        max_pt = custom_points.max(dim=0)[0]
        cur_w = (max_pt[0] - min_pt[0]).item()
        cur_h = (max_pt[1] - min_pt[1]).item()
        
        if cur_h > 0:
            cur_aspect = cur_w / cur_h
            target_aspect = width / height
            
            # 如果目前比例跟目標比例差太多 (例如本來是 1:1 但原圖是 16:9)
            if abs(cur_aspect - target_aspect) > 0.05:
                if cur_aspect < target_aspect:
                    # 點雲太窄，需要拉長 X 軸
                    scale_x = target_aspect / cur_aspect
                    custom_points[:, 0] *= scale_x
                    print(f"調整點雲形狀: X 軸拉伸 {scale_x:.2f} 倍以符合原圖 {width}x{height} 的比例")
                else:
                    # 點雲太扁，需要拉長 Y 軸
                    scale_y = cur_aspect / target_aspect
                    custom_points[:, 1] *= scale_y
                    print(f"調整點雲形狀: Y 軸拉伸 {scale_y:.2f} 倍以符合原圖 {width}x{height} 的比例")

        # 重新計算因為可能被縮放過的長寬
        min_pt_new = custom_points.min(dim=0)[0]
        max_pt_new = custom_points.max(dim=0)[0]
        cur_w_new = (max_pt_new[0] - min_pt_new[0]).item()
        cur_h_new = (max_pt_new[1] - min_pt_new[1]).item()
        
        center_custom = custom_points.mean(dim=0).cpu()
        
        # 為了讓點雲投影到畫面上剛好符合第一幀原圖的大小
        # 相似三角形公式: 投影高度 = (3D高度 / 相機距離) * 焦距
        # 因此，為了讓投影高度 == 視窗高度 (height)，我們這樣反推距離：
        if cur_h_new > 0 and height > 0:
            new_radius = (cur_h_new * focallength_px) / height
        else:
            scale_custom = torch.max(custom_points.max(dim=0)[0] - custom_points.min(dim=0)[0]).item()
            new_radius = scale_custom * 1.5 if scale_custom > 0 else 1.5
            
        depth_avg = new_radius
        
        print(f"自訂點雲中心: {center_custom.numpy()}, 實際大小: {cur_w_new:.3f}x{cur_h_new:.3f}, 自動調整相機距離: {new_radius:.3f}")
        
        # 設定新的世界中心點與相機位置 c2w_0
        # 因為已經把點雲調整成適合的 Z 軸方向，我們就把相機放在點雲的正前方
        c2w_0_custom = torch.tensor([
            [1.0, 0.0, 0.0, center_custom[0].item()],
            [0.0, 1.0, 0.0, center_custom[1].item()],
            [0.0, 0.0, 1.0, center_custom[2].item() - new_radius], # 相機位置在 Z軸退後
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        # 第一幀如果想要給定一點初始仰角
        # 注意: 這裡已經不再顛倒 Y 軸，所以仰角旋轉方向可能也需要微調
        elevation_rad = np.deg2rad(args.start_elevation)
        R_elevation = torch.tensor([
            [1, 0, 0, 0],
            [0, np.cos(elevation_rad), -np.sin(elevation_rad), 0],
            [0, np.sin(elevation_rad), np.cos(elevation_rad), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        
        # 將仰角加入，計算繞著原點 (0,0,0) 的初始相機位置
        c2w_0_origin = R_elevation @ torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -new_radius],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        
        # 產生真正的初始相機 (平移到自訂點雲中心)
        c2w_0 = c2w_0_origin.clone()
        c2w_0[:3, -1] += center_custom
        w2c_0 = c2w_0.inverse()
        
        # 重新生成相機軌跡
        # 由於原本的 build_cameras 假設相機圍繞 [0, 0, 0] 旋轉
        # 我們為了支援預設所有的複雜軌跡(如 swing)，改為直接呼叫 build_cameras
        # 但我們餵給它的是「繞著原點」的初始相機 c2w_0_origin
        w2cs_base, c2ws_base, intrinsic = build_cameras(cam_traj=cam_traj,
                                                        w2c_0=c2w_0_origin.inverse(),
                                                        c2w_0=c2w_0_origin,
                                                        intrinsic=intrinsic,
                                                        nframe=args.nframe,
                                                        focal_length=args.focal_length,
                                                        d_theta=d_theta,
                                                        d_phi=d_phi,
                                                        d_r=d_r,
                                                        radius=new_radius,
                                                        x_offset=x_offset,
                                                        y_offset=y_offset,
                                                        z_offset=z_offset)

        # 所有的相機軌跡算完後，統一平移到 center_custom 的位置，才能對齊真正的點雲中心
        w2cs_list_new = []
        c2ws_list_new = []
        for i in range(args.nframe):
            c2w_i = c2ws_base[i].clone()
            c2w_i[:3, -1] += center_custom
            w2cs_list_new.append(c2w_i.inverse())
            c2ws_list_new.append(c2w_i)
            
        w2cs = torch.stack(w2cs_list_new, dim=0)
        c2ws = torch.stack(c2ws_list_new, dim=0)

    # save camera infos
    w2cs_list = w2cs.cpu().numpy().tolist()
    camera_infos = {"intrinsic": K.cpu().numpy().tolist(), "extrinsic": w2cs_list, "height": height, "width": width}
    with open(f"{args.output_path}/cam_info.json", "w") as writer:
        json.dump(camera_infos, writer, indent=2)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        control_imgs, render_masks = point_rendering(K=intrinsic.float(),
                                                     w2cs=w2cs.float(),
                                                     depth=depth.float(),
                                                     image=validation_image.float() * 2 - 1,
                                                     raster_settings=PointsRasterizationSettings(image_size=(height, width),
                                                                                                 radius=0.008,
                                                                                                 points_per_pixel=8),
                                                     device=device,
                                                     background_color=[0, 0, 0],
                                                     sobel_threshold=0.35,
                                                     sam_mask=None,
                                                     custom_points=custom_points,
                                                     custom_colors=custom_colors)

    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=args.nframe)
    render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=args.nframe)

    render_video = []
    mask_video = []
    control_imgs = control_imgs.to(torch.float32)
    for i in range(args.nframe):
        img = ToPILImage()((control_imgs[0][:, i] + 1) / 2)
        render_video.append(img)
        mask = ToPILImage()(render_masks[0][:, i])
        mask_video.append(mask)

    export_to_video(render_video, f"{args.output_path}/render.mp4", fps=16)
    export_to_video(mask_video, f"{args.output_path}/render_mask.mp4", fps=16)

    print("Rendering finished.")


if __name__ == "__main__":
    main()
