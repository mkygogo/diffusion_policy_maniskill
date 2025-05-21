python -m mani_skill.examples.motionplanning.panda.run --env-id PickCube-v1 \
    --traj-name="pickcube-trajectory" --only-count-success --save-video -n 1 \
    --shader="rt" # generate sample videos
mv demos/PickCube-v1/motionplanning/0.mp4 demos/PickCube-v1/motionplanning/sample.mp4
python -m mani_skill.examples.motionplanning.panda.run --env-id PickCube-v1 --traj-name="pickcube-trajectory" -n 1000 --only-count-success