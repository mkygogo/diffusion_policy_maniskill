python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/PickCube-v1/motionplanning/pickcube-trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu