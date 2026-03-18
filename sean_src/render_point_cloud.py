import os
import sys
import torch
import numpy as np
from PIL import Image
import einops
from torchvision.transforms import ToTensor, ToPILImage
from diffusers.utils import export_to_video
import trimesh
from pytorch3d.renderer import PointsRasterizationSettings

# 確保可以載入專案內的模組
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pointcloud import point_rendering
from src.utils import set_initial_camera, build_cameras, traj_map, points_padding, np_points_padding

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 圖片與深度圖的路徑
    img_path = '../data/tcp/human.png'
    depth_path = '../data/tcp/depth/dense_0210_05_human_vis.jpg'
    output_path = 'outputs/tcp_render'
    os.makedirs(output_path, exist_ok=True)
    
    # 載入原圖
    print(f"Loading image from {img_path}")
    image = Image.open(img_path).convert("RGB")
    w_origin, h_origin = image.size
    
    # 調整尺寸以符合網路和渲染的需求，通常長寬要是 16 的倍數
    base_res = 512
    if w_origin > h_origin:
        width = int(base_res * (w_origin / h_origin))
        height = base_res
    else:
        width = base_res
        height = int(base_res * (h_origin / w_origin))
        
    width = (width // 16) * 16
    height = (height // 16) * 16
    
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    validation_image = ToTensor()(image)[None].to(device)  # [1, c, h, w], 0~1
    
    # 載入深度圖
    print(f"Loading depth map from {depth_path}")
    depth_img = Image.open(depth_path).convert("L")
    depth_img = depth_img.resize((width, height), Image.Resampling.BICUBIC)
    depth_np = np.array(depth_img).astype(np.float32)
    
    # 轉換深度可視化 (0-255) 為真實的深度數值。
    # 假設這是一張視差圖 (disparity map)，通常越亮越近 (數值越大)，所以 depth = 1 / disparity
    disp = depth_np / 255.0
    disp = np.clip(disp, 0.05, 1.0) # 避免除以零
    depth_np = 1.0 / disp
    
    # 縮放深度到合理的範圍（例如中位數為 2.0 公尺）
    target_median = 2.0
    current_median = np.median(depth_np)
    depth_np = depth_np * (target_median / current_median)
    
    depth = torch.from_numpy(depth_np)[None, None].to(device) # [1, 1, h, w]
    
    # 設定相機內參 (Intrinsic matrix)，預估一個合理焦距
    focal_length_px = max(width, height) # 視角大約 60 度
    K = torch.tensor([[focal_length_px, 0, width / 2],
                      [0, focal_length_px, height / 2],
                      [0, 0, 1]], dtype=torch.float32)
    K_inv = K.inverse()
    
    nframe = 60  # 輸出的總幀數
    intrinsic = K[None].repeat(nframe, 1, 1)
    
    # 將深度與圖片結合轉換成 3D 點雲 (單純用來導出 PLY，渲染會由 point_rendering 接手)
    points2d = torch.stack(torch.meshgrid(torch.arange(width, dtype=torch.float32),
                                          torch.arange(height, dtype=torch.float32), indexing="xy"), -1)  # [h,w,2]
    points3d_raw = points_padding(points2d).reshape(height * width, 3)  # [hw,3]
    points3d_raw = (K_inv @ points3d_raw.T * torch.tensor(depth_np).reshape(1, height * width)).T
    colors_raw = np.array(image).reshape(height * width, 3)
    
    depth_avg = target_median
    start_elevation = 5.0
    w2c_0, c2w_0 = set_initial_camera(start_elevation, depth_avg)
    
    # 導出點雲 (.ply) 供外部檢視
    points3d_world = (c2w_0.numpy()[:3] @ np_points_padding(points3d_raw.numpy()).T).T
    pcd = trimesh.PointCloud(vertices=points3d_world, colors=colors_raw)
    pcd.export(f"{output_path}/pcd.ply")
    print(f"Saved point cloud to {output_path}/pcd.ply")
    
    # 設定相機移動軌跡，這裡設定水平旋轉 -60 度
    cam_traj = "free"
    d_theta = 0.0
    d_phi = -60.0
    d_r = 1.0
    x_offset = y_offset = z_offset = 0.0
    
    w2cs, c2ws, intrinsic = build_cameras(cam_traj=cam_traj,
                                          w2c_0=w2c_0,
                                          c2w_0=c2w_0,
                                          intrinsic=intrinsic,
                                          nframe=nframe,
                                          focal_length=1.0,
                                          d_theta=d_theta,
                                          d_phi=d_phi,
                                          d_r=d_r,
                                          radius=depth_avg,
                                          x_offset=x_offset,
                                          y_offset=y_offset,
                                          z_offset=z_offset)
                                          
    print("Start rendering...")
    with torch.no_grad():
        # point_rendering 函式會自動生成點雲並用 PyTorch3D 渲染出來
        control_imgs, render_masks = point_rendering(K=intrinsic.float(),
                                                     w2cs=w2cs.float(),
                                                     depth=depth.float(),
                                                     image=validation_image.float() * 2 - 1, # [-1, 1] 範圍
                                                     raster_settings=PointsRasterizationSettings(image_size=(height, width),
                                                                                                 radius=0.008,
                                                                                                 points_per_pixel=8),
                                                     device=device,
                                                     background_color=[0, 0, 0],
                                                     sobel_threshold=0.35,
                                                     sam_mask=None)

    # 整理渲染出來的影像
    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=nframe)
    
    render_video = []
    control_imgs = control_imgs.to(torch.float32)
    for i in range(nframe):
        # 轉換回 0~1 的範圍並變成 PIL Image
        img = ToPILImage()((control_imgs[0][:, i] + 1) / 2)
        render_video.append(img)

    video_path = f"{output_path}/render.mp4"
    export_to_video(render_video, video_path, fps=15)
    print(f"Rendering finished. Video saved to {video_path}")

if __name__ == "__main__":
    main()
