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
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    
    # 圖片與深度圖的路徑
    img_path = '../data/tcp/human.png'
    depth_path = '../data/tcp/depth/ComfyUI_temp_hbbhc_00002_.png'
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
    validation_image = ToTensor()(image)[None]  # [1, c, h, w], 0~1
    
    # 載入深度圖
    print(f"Loading depth map from {depth_path}")
    depth_img = Image.open(depth_path).convert("L")
    depth_img = depth_img.resize((width, height), Image.Resampling.BICUBIC)
    depth_np = np.array(depth_img).astype(np.float32)
    
    # ---------------------------------------------------------
    # 去除背景邏輯
    # 深度圖中越亮 (數值越大) 代表越近，越暗 (數值越小) 代表越遠 (背景)
    # 我們設定一個 Threshold，如果低於此值就當作背景，並將該點排除
    # ---------------------------------------------------------
    bg_threshold = 20.0  # 像素值低於 20 視為背景 (可依實際圖片調整)
    fg_mask = depth_np > bg_threshold  # 建立前景 Mask (True 為前景，False 為背景)
    
    # 轉換深度可視化 (0-255) 為真實的深度數值。
    # 將背景的像素值先補為一個非零的小值，避免除以零產生無窮大，稍後會將其設為 0
    depth_np_safe = depth_np.copy()
    depth_np_safe[~fg_mask] = 255.0 # 先暫時當作最亮(最近)來處理視差，反正最後會丟掉
    
    disp = depth_np_safe / 255.0
    disp = np.clip(disp, 0.05, 1.0) # 避免除以零
    depth_np = 1.0 / disp
    
    # 計算中位數縮放時，只考慮前景部分的深度！這樣中位數才會準確
    target_median = 2.0
    if fg_mask.sum() > 0:
        current_median = np.median(depth_np[fg_mask])
    else:
        current_median = np.median(depth_np)
        
    depth_np = depth_np * (target_median / current_median)

    # --- 新增這段來控制點雲厚度 ---
    thickness_factor = 0.3  # <1.0 代表壓扁點雲，>1.0 代表拉長點雲
    depth_np = target_median + (depth_np - target_median) * thickness_factor
    
    # 將背景區域的深度設為 0 讓後續能辨識過濾掉
    # 將背景設定為 0 (全黑)，代表這是背景區域
    depth_np[~fg_mask] = 0.0
    
    # 建立深度圖張量前，保留一個只有前景的 valid_mask 給點雲導出使用
    valid_mask = depth_np > 0.0
    
    depth = torch.from_numpy(depth_np)[None, None].to(device) # [1, 1, h, w]
    
    # 設定相機內參 (Intrinsic matrix)，預估一個合理焦距
    focal_length_px = max(width, height) # 視角大約 60 度
    K = torch.tensor([[focal_length_px, 0, width / 2],
                      [0, focal_length_px, height / 2],
                      [0, 0, 1]], dtype=torch.float32)
    K_inv = K.inverse()
    
    nframe = 80  # 輸出的總幀數
    intrinsic = K[None].repeat(nframe, 1, 1)
    
    # 將深度與圖片結合轉換成 3D 點雲 (單純用來導出 PLY)
    points2d = torch.stack(torch.meshgrid(torch.arange(width, dtype=torch.float32),
                                          torch.arange(height, dtype=torch.float32), indexing="xy"), -1)  # [h,w,2]
    points3d_raw = points_padding(points2d).reshape(height * width, 3)  # [hw,3]
    points3d_raw = (K_inv @ points3d_raw.T * torch.tensor(depth_np).reshape(1, height * width)).T
    colors_raw = np.array(image).reshape(height * width, 3)
    
    # ==== 在導出 PLY 時，過濾掉背景點 ====
    valid_mask_1d = valid_mask.reshape(-1)
    points3d_raw_valid = points3d_raw[valid_mask_1d]
    colors_raw_valid = colors_raw[valid_mask_1d]
    
    print(f"Original points: {points3d_raw.shape[0]}, after removing background: {points3d_raw_valid.shape[0]}")
    
    depth_avg = target_median
    start_elevation = 5.0
    w2c_0, c2w_0 = set_initial_camera(start_elevation, depth_avg)
    
    # 導出點雲 (.ply) 供外部檢視
    points3d_world = (c2w_0.numpy()[:3] @ np_points_padding(points3d_raw_valid.numpy()).T).T
    pcd = trimesh.PointCloud(vertices=points3d_world, colors=colors_raw_valid)
    pcd.export(f"{output_path}/pcd.ply")
    print(f"Saved point cloud to {output_path}/pcd.ply")
    
    # ==== 自訂相機移動軌跡：向左90度 -> 回正 -> 向右90度 -> 回正 ====
    # 我們不使用 build_cameras，而是自己手刻角度軌跡
    w2cs_list = []
    c2ws_list = []
    
    steps = nframe - 1
    phis_custom = []
    for i in range(nframe):
        if i == 0:
            phis_custom.append(0.0)
            continue
            
        progress = i / steps
        if progress <= 0.25: # 0 -> 0.25：0度轉到 -90度 (向左)
            phi = -90.0 * (progress / 0.25)
        elif progress <= 0.5: # 0.25 -> 0.5：-90度轉回 0度
            phi = -90.0 * (1 - (progress - 0.25) / 0.25)
        elif progress <= 0.75: # 0.5 -> 0.75：0度轉到 90度 (向右)
            phi = 90.0 * ((progress - 0.5) / 0.25)
        else: # 0.75 -> 1.0：90度轉回 0度
            phi = 90.0 * (1 - (progress - 0.75) / 0.25)
        phis_custom.append(phi)
        
    for i in range(nframe):
        d_phi_cur = phis_custom[i]
        d_phi_rad = np.deg2rad(d_phi_cur)
        
        # 繞 Y 軸旋轉矩陣
        R_phi = torch.tensor([[np.cos(d_phi_rad), 0, np.sin(d_phi_rad), 0],
                              [0, 1, 0, 0],
                              [-np.sin(d_phi_rad), 0, np.cos(d_phi_rad), 0],
                              [0, 0, 0, 1]], dtype=torch.float32)
                              
        # 為了讓點雲維持在畫面中央，相機必須繞著點雲的中心點旋轉。
        # 由於 set_initial_camera 會將相機沿著原點向 Z 軸負方向退後 depth_avg 距離，
        # 所以點雲的中心就在原點 (0,0,0)。
        # 因此，旋轉矩陣 R_phi 應該直接應用於相機位置與姿態上，形成環繞原點的效果。
        # R_phi @ c2w_0 就是讓相機繞著 Y 軸轉圈。
        c2w_1 = R_phi @ c2w_0
        w2c_1 = c2w_1.inverse()
        
        c2ws_list.append(c2w_1)
        w2cs_list.append(w2c_1)

    w2cs = torch.stack(w2cs_list, dim=0)
    c2ws = torch.stack(c2ws_list, dim=0)
    
    # 確保內參矩陣匹配總幀數
    intrinsic = K[None].repeat(nframe, 1, 1)

    # ==========================
    # 分批渲染來避免 OOM 錯誤
    # ==========================
    print("Start rendering...")
    
    from src.pointcloud import get_boundaries_mask, PointsZbufRenderer
    from pytorch3d.renderer import PointsRasterizer, AlphaCompositor, PerspectiveCameras
    from pytorch3d.structures import Pointclouds
    
    # 參數設定
    render_settings = PointsRasterizationSettings(image_size=(height, width),
                                                radius=0.008,
                                                points_per_pixel=8)
    
    with torch.no_grad():
        # --- 預先計算 3D 點雲 ---
        contract = 8.0
        depth_ts = depth.clone()
        mid_depth = torch.median(depth_ts.reshape(-1), dim=0)[0] * contract
        depth_ts[depth_ts > mid_depth] = ((2 * mid_depth) - (mid_depth ** 2 / (depth_ts[depth_ts > mid_depth] + 1e-6)))

        # 由於深度可能會有 0 (背景)，1/depth 可能會出現無窮大或極大值
        # 為了不影響 boundary_mask 的判斷，我們先將深度為 0 的地方補為一個較大數值或原本的中位數
        depth_for_mask = depth_ts.clone()
        depth_for_mask[depth_for_mask == 0.0] = target_median
        
        point_depth = einops.rearrange(depth_ts[0], "c h w -> (h w) c")
        disp = 1 / (depth_for_mask + 1e-7)
        boundary_mask = get_boundaries_mask(disp, sobel_threshold=0.35).reshape(-1).cpu()

        x = torch.arange(width).float() + 0.5
        y = torch.arange(height).float() + 0.5
        points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).to(device)
        points = einops.rearrange(points, "w h c -> (h w) c")
        
        c2w_0_inv = c2ws[0].to(device).inverse()
        points_3d = (c2ws[0].to(device) @ points_padding((K.to(device).inverse() @ points_padding(points).T).T * point_depth).T).T[:, :3]

        validation_image_adj = validation_image.float() * 2 - 1
        colors = einops.rearrange(validation_image_adj[0], "c h w -> (h w) c")

        # 將深度為 0 的點過濾掉
        bg_mask = (point_depth.cpu().reshape(-1) == 0.0)
        combined_mask = boundary_mask | bg_mask
        
        points_3d = points_3d.cpu()[~combined_mask]
        colors = colors.cpu()[~combined_mask]
        # ---------------------
        
        all_control_imgs = []
        
        print(f"Total rendered points: {points_3d.shape[0]}")
        
        # 分批渲染，每次 10 幀
        batch_size = 10
        for start_idx in range(0, nframe, batch_size):
            end_idx = min(start_idx + batch_size, nframe)
            batch_nframe = end_idx - start_idx
            print(f"Rendering frame {start_idx} to {end_idx - 1}...")
            
            batch_w2cs = w2cs[start_idx:end_idx].clone().to(device)
            batch_c2ws = batch_w2cs.inverse()
            batch_intrinsic = intrinsic[start_idx:end_idx].clone().to(device)
            
            # 建立這一批次的點雲
            point_cloud = Pointclouds(points=[points_3d.to(device)], features=[colors.to(device)]).extend(batch_nframe)
            
            # 轉換 opencv 座標系為 opengl 座標系
            batch_c2ws[:, :, 0] = - batch_c2ws[:, :, 0]
            batch_c2ws[:, :, 1] = - batch_c2ws[:, :, 1]
            batch_w2cs = batch_c2ws.inverse()

            focal_length_batch = torch.stack([batch_intrinsic[:, 0, 0], batch_intrinsic[:, 1, 1]], dim=1)
            principal_point_batch = torch.stack([batch_intrinsic[:, 0, 2], batch_intrinsic[:, 1, 2]], dim=1)
            image_shapes_batch = torch.tensor([[height, width]]).repeat(batch_nframe, 1)
            
            cameras = PerspectiveCameras(focal_length=focal_length_batch, principal_point=principal_point_batch,
                                         R=batch_c2ws[:, :3, :3], T=batch_w2cs[:, :3, -1], in_ndc=False,
                                         image_size=image_shapes_batch, device=device)

            renderer = PointsZbufRenderer(
                rasterizer=PointsRasterizer(cameras=cameras, raster_settings=render_settings),
                compositor=AlphaCompositor(background_color=[0, 0, 0])
            )

            render_rgbs, _ = renderer(point_cloud)  # rgb:[f,h,w,3]
            render_rgbs = einops.rearrange(render_rgbs, "f h w c -> f c h w")  # [f,3,h,w]
            
            # 第一幀我們選擇強制替換回原圖，以保持清晰度
            if start_idx == 0:
                render_rgbs[0:1] = validation_image_adj
                
            all_control_imgs.append(render_rgbs.cpu())
            torch.cuda.empty_cache()

    control_imgs = torch.cat(all_control_imgs, dim=0) # [nframe, 3, h, w]
    control_imgs = einops.rearrange(control_imgs, "f c h w -> 1 c f h w")
    
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
