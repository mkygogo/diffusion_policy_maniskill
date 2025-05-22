from pprint import pprint
import h5py
import sys
import os
from PIL import Image
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_policy.utils import load_demo_dataset

h5file_path = '../demos/PickCube-v1/motionplanning/pickcube-trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5'

# with h5py.File(h5file_path, "r") as f:
#     print(list(f.keys()))  # 列出顶级Group
#     print(f["episodes/episode_0/observations"].shape) 

def print_keys_with_type(d, indent=0):
    for key, value in d.items():
        value_type = " (dict)" if isinstance(value, dict) else ""
        print("  " * indent + f"{key}{value_type}")
        if isinstance(value, dict):
            print_keys_with_type(value, indent + 1)

def save_images(image_array, output_dir="./imgs"):
    # 确保输入数组的形状正确
    if image_array.ndim != 4 or image_array.shape[3] != 3:
        raise ValueError("输入数组形状应为 (N, H, W, 3)")

    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(image_array.shape[0]):
        # 将数组转换为Pillow图像对象（假设值范围是0-255或0-1）
        img_data = image_array[i]
        if img_data.max() <= 1.0:  # 如果是0-1浮点数，转换为0-255
            img_data = (img_data * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)
        
        img = Image.fromarray(img_data)

        img.save(os.path.join(output_dir, f"img_{i}.png"))
    
    print(f"已保存 {image_array.shape[0]} 张图片到 {output_dir}")

trajectories = load_demo_dataset(h5file_path, num_traj=1, concat=False)

print_keys_with_type(trajectories['observations'][0])

print(type(trajectories))
pprint(trajectories.keys()) #['observations', 'actions']
pprint(trajectories['observations'][0].keys()) #['agent', 'extra', 'sensor_param', 'sensor_data']
pprint(trajectories['actions'][0].shape) 
pprint(trajectories['observations'][0]['agent'].keys()) #['qpos', 'qvel']
pprint(trajectories['observations'][0]['agent']['qpos'].shape)
pprint(trajectories['observations'][0]['agent']['qpos'][0])
pprint(trajectories['observations'][0]['agent']['qvel'].shape)
pprint(trajectories['observations'][0]['agent']['qvel'][37])
pprint(trajectories['observations'][0]['extra'].keys()) #['is_grasped', 'tcp_pose', 'goal_pos']
pprint(trajectories['observations'][0]['extra']['is_grasped'].shape)
pprint(trajectories['observations'][0]['extra']['tcp_pose'].shape)
pprint(trajectories['observations'][0]['extra']['goal_pos'].shape)
pprint(trajectories['observations'][0]['sensor_param'].keys()) #['base_camera']
pprint(trajectories['observations'][0]['sensor_param']['base_camera'].keys()) #['extrinsic_cv', 'cam2world_gl', 'intrinsic_cv']
pprint(trajectories['observations'][0]['sensor_param']['base_camera']['extrinsic_cv'].shape)
pprint(trajectories['observations'][0]['sensor_param']['base_camera']['cam2world_gl'].shape)
pprint(trajectories['observations'][0]['sensor_param']['base_camera']['intrinsic_cv'].shape)
pprint(trajectories['observations'][0]['sensor_data'].keys()) #['base_camera']

pprint(trajectories['observations'][0]['sensor_data']['base_camera']['rgb'].shape) 
#save_images(trajectories['observations'][0]['sensor_data']['base_camera']['rgb'])


