import trimesh
import torch
import numpy as np
import os
from torchvision.transforms import ToPILImage
from diffusers.utils import export_to_video

# We will just patch cam_render.py to do auto-alignment
with open("cam_render.py", "r") as f:
    code = f.read()

# Replace the custom points loading part
old_code = """        # 由於此檔案的影像輸入值域為 [-1, 1]，我們需將 0~255 的點雲顏色正規化為 [-1, 1]
        custom_colors = np.array(custom_pcd.colors[:, :3])
        custom_colors = torch.tensor(custom_colors, dtype=torch.float32).to(device) / 255.0 * 2.0 - 1.0"""

new_code = """        # 由於此檔案的影像輸入值域為 [-1, 1]，我們需將 0~255 的點雲顏色正規化為 [-1, 1]
        custom_colors = np.array(custom_pcd.colors[:, :3])
        custom_colors = torch.tensor(custom_colors, dtype=torch.float32).to(device) / 255.0 * 2.0 - 1.0

        # ========== Auto Align Custom Point Cloud to Reference ==========
        # 1. 將自訂點雲的座標系轉換為與 points3d 相同的方向 
        # 假設自訂點雲為 OpenGL 相機座標 (X向右, Y向上, Z向後)
        custom_points[:, 1] = -custom_points[:, 1]
        custom_points[:, 2] = -custom_points[:, 2]
        
        # 轉換到世界座標
        c2w_0_tensor = c2w_0.to(device)
        custom_points = (c2w_0_tensor[:3, :3] @ custom_points.T).T + c2w_0_tensor[:3, -1]
        
        # 2. 對齊中心點與尺度
        ref_points = torch.tensor(points3d, dtype=torch.float32).to(device)
        
        center_ref = ref_points.mean(dim=0)
        center_custom = custom_points.mean(dim=0)
        
        scale_ref = torch.max(ref_points.max(dim=0)[0] - ref_points.min(dim=0)[0])
        scale_custom = torch.max(custom_points.max(dim=0)[0] - custom_points.min(dim=0)[0])
        
        custom_points = (custom_points - center_custom) * (scale_ref / scale_custom) + center_ref
        print(f"Aligned Custom Points: center={center_custom.cpu().numpy()} -> {center_ref.cpu().numpy()}, scale={scale_custom.item():.2f} -> {scale_ref.item():.2f}")
"""

patched_code = code.replace(old_code, new_code)
with open("cam_render_patched.py", "w") as f:
    f.write(patched_code)

