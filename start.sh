CUDA_VISIBLE_DEVICES=2 python cam_render.py --reference_image "data/tcp/human.png" \
                     --output_path "outputs/tcp/0406/human_down" \
                     --traj_type "test_down" \
                     --input_pcd "data/tcp/point_clouds/human.ply"