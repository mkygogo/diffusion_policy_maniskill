from train import Agent, Args
import tyro
import torch
import numpy as np
import mani_skill.envs
import gymnasium as gym
from collections import defaultdict
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode


def cpu_make_env(env_id, record_video, env_kwargs = dict(), other_kwargs=dict()):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        if record_video:
            video_dir = f'runs/infer_cpu/videos'
            env = RecordEpisode( env,output_dir=video_dir,save_trajectory=False,
                        info_on_video=True, source_type="diffusion_policy",
                            source_desc="diffusion_policy evaluation rollout", )        
        
        return env
    return thunk

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.env_id = "PickCube-v1"
    args.obs_mode = "state"
    args.control_mode = "pd_ee_delta_pos" 
    args.obs_horizon = 2
    args.act_horizon = 8
    args.pred_horizon = 16

    args.num_eval_envs = 1
    args.sim_backend = "physx_cpu"
    args.ckpt_path = "runs/diffusion_policy-PickCube-v1-state-100_motionplanning_demos-1/checkpoints/best_eval_success_at_end.pt"

    other_kwargs = dict(obs_horizon=args.obs_horizon)
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", 
                      obs_mode=args.obs_mode, render_mode="human", 
                      human_render_camera_configs=dict(shader_pack="default"))

    #env创建方法一：非并行模式的env可以human方式渲染，但是不能保存视频
    env = gym.make(
        "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        reconfiguration_freq=1,
        **env_kwargs  
    )
    env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True) #ignore_terminations如果哦是true，任务没有完成就环境关闭了
    #要保存视频需要render_mode="rgb_array"
    # video_dir = f'runs/infer_cpu1/videos'
    # env = RecordEpisode( env,output_dir=video_dir,save_trajectory=False,
    #                 info_on_video=True, source_type="diffusion_policy",
    #                     source_desc="diffusion_policy evaluation rollout",
    #                 )

    #env方法创建二：这种方法生成的env是可以并行的，在推理过程没有必要，
    #vector_cls = gym.vector.SyncVectorEnv if args.num_eval_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
    #env = vector_cls([cpu_make_env(args.env_id, True, env_kwargs, other_kwargs) for _ in range(args.num_eval_envs)])
    #env = gym.vector.SyncVectorEnv([cpu_make_env(args.env_id, True, env_kwargs, other_kwargs)])
    #print(env.single_observation_space.shape)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    agent = Agent(env, args).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    obs, _ = env.reset(seed=10) # reset with a seed for determinism
    print("Observation:", obs)
    print("Observation shape:", obs.shape)  
    obs_expanded = obs[np.newaxis, :, :]
    print("obs_expanded shape:", obs_expanded.shape)

    done = False
    while not done:
       
        #obs_tensor = torch.from_numpy(obs).float().to(device)
        obs_tensor = torch.from_numpy(obs_expanded).float().to(device)
        #print("obs_tensor shape:", obs_tensor.shape)
        with torch.no_grad():
            actions = agent.get_action(obs_tensor)
        action_to_execute = actions[:, 0, :].cpu().numpy() 
        
        obs, rew, terminated, truncated, info = env.step(action_to_execute)
        obs_expanded = obs[np.newaxis, :, :]

        #print(f"terminated :{terminated} truncated:{truncated}")
        done = terminated #or truncated
        env.render()  # a display is required to render

    env.close()