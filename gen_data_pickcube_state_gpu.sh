python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ./demos/PickCube-v1/motionplanning/trajectory_gpu.h5 \
  --use-first-env-state -c pd_joint_pos -o state \
  --save-traj --num-envs 10 -b gpu