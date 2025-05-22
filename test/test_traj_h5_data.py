from mani_skill.utils import common, io_utils, wrappers
import h5py
import json
import gymnasium as gym
import os


if __name__=="__main__":
    traj_path = "../demos/PickCube-v1/motionplanning/pickcube-trajectory.h5"
    # Load trajectory metadata json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    #print(f"json:{json.dumps(json_data, indent=4)}")

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()
    env_kwargs["obs_mode"] = "rgb"
    env_kwargs["sim_backend"] = "cpu"
    env_kwargs["num_envs"] = 1

    print(f"{json.dumps(env_kwargs, indent=4)}")

    ori_h5_file = h5py.File(traj_path, "r")
    env = gym.make(env_id, **env_kwargs)
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    parts = ori_traj_name.split(".")
    if len(parts) > 1:
        ori_traj_name = parts[0]
    suffix = "{}.{}.{}".format(
        env.unwrapped.obs_mode,
        env.unwrapped.control_mode,
        env.unwrapped.backend.sim_backend,
    )    
    new_traj_name = ori_traj_name + "." + suffix
    print(f"new_traj_name:{new_traj_name}")
    ori_env = gym.make(env_id, **ori_env_kwargs)
    #output_h5_path = env._h5_file.filename
    #print(f"output_h5_path:{output_h5_path}")

    
