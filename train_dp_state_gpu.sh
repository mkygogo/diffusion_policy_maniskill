seed=1
demos=100
python train.py --env-id PickCube-v1 \
  --demo-path demos/PickCube-v1/motionplanning/trajectory_gpu.state.pd_joint_pos.physx_cuda.h5 \
  --control-mode "pd_joint_pos" --sim-backend "physx_cuda" --num-demos ${demos} --max_episode_steps 100 \
  --total_iters 30000 \
  --exp-name diffusion_policy-PickCube-v1-state-${demos}_motionplanning_demos-${seed} \
  --track # track training on wandb
