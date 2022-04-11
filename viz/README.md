# Visualisation 
This package containes code for visualisation scripts, like creating autonomous drives.

## Video creator

### Creating on-policy video from (autonomous) drive:
```bash
python -m viz.video_creator --dataset-folder=/gpfs/space/projects/Bolt/dataset/<drive-name> --video-type=driving
```

### Creating off-policy video using trained model:
```bash
python -m viz.video_creator --dataset-folder=/gpfs/space/projects/Bolt/dataset/<drive-name> --video-type=prediction --model-path <path to pytorch model> --model-type pilotnet-conditional --output-modality waypoints
```

Use `--output-modality` parameter to use different prediction targets like steering angle or trajectory waypoints.

Use `--model-type` parameter to use different model architectures like `pilotnet-conditional` and `pilotnet-control`.

### Running visual backprop:
Visual backprop is implementation of network visualisation method from 'VisualBackProp: efficient visualization of CNNs' paper: 
https://arxiv.org/abs/1611.05418

Use one of the following commands to run visual backprop interactively. Use `k` key for forwad and `j` key for backwards movements. 

To create video from visual backprop, just add `--video` parameter.

```bash
# Nvidia camera input with Pilotnet model
python -m viz.visual_backprop --model-name 20220215010503_aug-none --input-modality nvidia-camera --dataset-name 2022-01-28-14-47-23_e2e_rec_elva_forward --model-type pilotnet
# Nvidia camera input with Pilotnet Conditional model
python -m viz.visual_backprop --input-modality nvidia-camera --output-modality waypoints --model-type pilotnet-conditional --model-name 20220331132219_waypoints10-c-2 --dataset-name 2021-10-26-10-49-06_e2e_rec_ss20_elva
# Nvidia camera input with Pilotnet Control model
python -m viz.visual_backprop --input-modality nvidia-camera --output-modality waypoints --model-type pilotnet-control --model-name 20220405185427_waypoints10-control --dataset-name 2021-10-26-10-49-06_e2e_rec_ss20_elva

# Ouster lidar input with old Pilotnet model
python -m viz.visual_backprop --model-name 20211124202413_lidar-v4 --input-modality ouster-lidar --dataset-name 2021-10-26-10-49-06_e2e_rec_ss20_elva --model-type pilotnet-old
```