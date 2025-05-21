# inference.py
import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import mani_skill.envs
import torch
import numpy as np
import time
from typing import List
from dataclasses import dataclass, field
import tyro
from train_rgbd import Agent, Args  # 直接引用训练代码中的定义
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.utils import convert_obs, build_state_obs_extractor

run_name = "infer"

def obs_process_fn(obs_dict):
    print("Observation keys:", obs_dict.keys())
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray):
            print(f"{k} shape: {v.shape} dtype: {v.dtype}")
        else:
            print(f"{k} type: {type(v)}")   

    processed = dict()
    
    # 提取视觉观测
    if 'rgb' in obs_dict:
        rgb = obs_dict['rgb']
        if rgb.shape == (1, 2, 128, 128, 3):  # 批量+多摄像头情况
            # 移除批处理维度并转置 -> [num_cameras=2, C=3, H, W]
            processed['rgb'] = np.transpose(rgb[0], (0, 3, 1, 2))
        else:
            raise ValueError(f"Unsupported rgb shape: {rgb.shape}")
    
    if 'state' in obs_dict:
        state = obs_dict['state']
        if state.shape == (1, 2, 29):  # 批量+历史观测
            # 移除批处理维度 -> [obs_horizon=2, state_dim=29]
            processed['state'] = state[0]
        else:
            raise ValueError(f"Unsupported state shape: {state.shape}") 
    # 状态观测处理
    #state_obs_extractor = build_state_obs_extractor(args.env_id)
    #processed['state'] = state_obs_extractor(obs_dict).astype(np.float32)
    
    print("Processed observation structure:")
    for k, v in processed.items():
        print(f"- {k}: shape={v.shape}, dtype={v.dtype}")
    
    return processed

def main(args: Args):

    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", 
                        render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"))
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    env = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    print("Action space shape:", env.single_action_space.shape)
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    checkpoint = torch.load("runs/diffusion_policy-PickCube-v1-rgb-100_motionplanning_demos-1/checkpoints/best_eval_success_at_end.pt", map_location=device)
    
    # 创建Agent实例（结构与训练时完全一致）
    agent = Agent(env, args).to(device)
    
    # 加载EMA权重（推荐）
    agent.load_state_dict(checkpoint["ema_agent"], strict=True)
    agent.eval()

    # 初始化观测历史
    obs_history = []
    obs, _ = env.reset()
    processed_obs = obs_process_fn(obs)
    obs_history.append(obs)

    done = False
    while not done:
        # 构建观测序列 (长度=obs_horizon)
        if len(obs_history) < args.obs_horizon:
            # 不足时用第一个观测填充
            obs_seq = {
                'rgb': np.stack(obs_history[0]['rgb'] * args.obs_horizon),
                'state': np.stack(obs_history[0]['state'] * args.obs_horizon)
            }
        else:
            obs_seq = {
                'rgb': np.stack([obs['rgb'] for obs in obs_history[-args.obs_horizon:]]),
                'state': np.stack([obs['state'] for obs in obs_history[-args.obs_horizon:]])
            }
        
        # 转换为PyTorch张量
        obs_tensor = {
            'rgb': torch.from_numpy(obs_seq['rgb']).float().to(device),
            'state': torch.from_numpy(obs_seq['state']).float().to(device)
        }
        
        # 获取动作
        with torch.no_grad():
            action_sequence = agent.get_action(obs_tensor)
        
        # 执行动作
        action = action_sequence[0, 0].cpu().numpy()
        next_obs, _, terminated, truncated, _ = env.step(action)
        
        # 更新观测历史
        processed_next_obs = obs_process_fn(next_obs)
        obs_history.append(processed_next_obs)
        env.render()

    env.close()

if __name__ == "__main__":
    # 从训练参数加载配置（必须与训练时一致）
    args = tyro.cli(Args)
    args.env_id = "PickCube-v1"
    args.obs_mode = "rgb"
    args.control_mode = "pd_ee_delta_pos" 
    args.obs_horizon = 2
    args.act_horizon = 8
    args.pred_horizon = 16
    args.num_eval_envs = 1
    args.sim_backend = "physx_cpu"
    args.max_episode_steps = 100
    
    main(args)