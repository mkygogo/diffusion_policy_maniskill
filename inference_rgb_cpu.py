from train_rgbd import Agent, Args
#from train import Agent, Args
import tyro
import torch
import numpy as np

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from PIL import Image
from datetime import datetime

def save_obs_img(imgs):
    
    # 假设 obs["rgb"] 可能是 numpy 或 PyTorch Tensor
    images = imgs
    # 如果是 PyTorch Tensor，转换为 numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    # 如果是 float 类型（0-1），转换为 uint8（0-255）
    if images.dtype == np.float32 or images.dtype == np.float64:
        images = (images * 255).astype(np.uint8)
    # 确保形状是 (2, H, W, C)
    if images.shape[0] == 2:
        img1, img2 = images[0], images[1]
    else:
        raise ValueError("obs['rgb'] 的形状不是 (2, H, W, C)")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 使用 PIL 保存
    Image.fromarray(img1).save(f"obs_imgs/{current_time}-image1.png")
    Image.fromarray(img2).save(f"obs_imgs/{current_time}-image2.png")



if __name__ == "__main__":
    args = tyro.cli(Args)
    args.env_id = "PickCube-v1"
    args.obs_mode = "rgb"
    args.control_mode = "pd_ee_delta_pos" 
    args.obs_horizon = 2
    args.act_horizon = 8
    args.pred_horizon = 16

    args.num_eval_envs = 1
    args.sim_backend = "physx_cpu"
    args.ckpt_path = "runs/diffusion_policy-PickCube-v1-rgb-1000_motionplanning_demos-1/checkpoints/best_eval_success_at_end.pt"
    #args.ckpt_path = "runs/diffusion_policy-PickCube-v1-state-100_motionplanning_demos-1/checkpoints/best_eval_success_at_end.pt"

    other_kwargs = dict(obs_horizon=args.obs_horizon)
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", 
                      obs_mode=args.obs_mode, render_mode="rgb_array", 
                      human_render_camera_configs=dict(shader_pack="default"))
    
    env = gym.make(
        "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        reconfiguration_freq=1,
        **env_kwargs  
    )
    env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
    #要保存视频需要render_mode="rgb_array"
    video_dir = f'runs/infer_rgb_cpu/videos'
    env = RecordEpisode( env,output_dir=video_dir,save_trajectory=False,
                    info_on_video=True, source_type="diffusion_policy",
                        source_desc="diffusion_policy evaluation rollout",
                    )

    # env = make_eval_envs(
    #     args.env_id,
    #     args.num_eval_envs,
    #     args.sim_backend,
    #     env_kwargs,
    #     other_kwargs,
    #     video_dir=f"runs/infer_rgb/videos" if args.capture_video else None,
    #     wrappers=[FlattenRGBDObservationWrapper],
    # )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    agent = Agent(env, args).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval() 

    obs, _ = env.reset(seed=1) # reset with a seed for determinism
    #print("Observation:", obs)
    print("Observation shape:", obs["state"].shape)  
    print("Observation shape:", obs["rgb"].shape)  
    # 转换后的字典
    expanded_obs = {
        "state": np.expand_dims(obs["state"], axis=0),  # 添加第一个维度 (1, 2, 29)
        "rgb": np.expand_dims(obs["rgb"], axis=0)       # 添加第一个维度 (1, 2, 128, 128, 3)
    }
    # 验证形状
    #print("New Obs shape:", expanded_obs["state"].shape)  # 输出: (1, 2, 29)
    #print("New Obs shape:",expanded_obs["rgb"].shape)    # 输出: (1, 2, 128, 128, 3)

    save_obs_img(obs["rgb"])

    done = False
    while not done:
        obs_tenser = {
            "state": torch.from_numpy(expanded_obs["state"]).float().to(device),
            "rgb": torch.from_numpy(expanded_obs["rgb"]).float().to(device)
        }   
        #print("obs_tensor shape:", obs_tensor.shape)
        with torch.no_grad():
            actions = agent.get_action(obs_tenser)#(obs_tensor)
        action_to_execute = actions[:, 0, :].cpu().numpy() 
        
        obs, rew, terminated, truncated, info = env.step(action_to_execute)
        save_obs_img(obs['rgb'])
        expanded_obs = {
            "state": np.expand_dims(obs["state"], axis=0),  
            "rgb": np.expand_dims(obs["rgb"], axis=0)       
        }

        #print(f"terminated :{terminated} truncated:{truncated}")
        done = terminated #or truncated
        #env.render()  # a display is required to render

    env.close()   

   