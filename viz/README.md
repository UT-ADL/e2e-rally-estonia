# This packege containes code for visualisation scripts, like creating autonomous drives.

## Video creator

### Creating on-policy video from (autonomous) drive:
```bash
python -m viz.video_creator --dataset-folder=/gpfs/space/projects/Bolt/dataset/<drive-name> --video-type=driving
```

### Creating off-policy video using trained model:
```bash
python -m viz.video_creator --dataset-folder=/gpfs/space/projects/Bolt/dataset/<drive-name> --video-type=prediction --model-path <path to pytorch model>
```

### Running visual backprop:
```bash
# Nvidia camera input with Pilotnet model
python -m viz.visual_backprop --model-name 20220215010503_aug-none --input-modality nvidia-camera --dataset-name 2022-01-28-14-47-23_e2e_rec_elva_forward --model-type pilotnet
# Ouster lidar input with old Pilotnet model
python -m viz.visual_backprop --model-name 20211124202413_lidar-v4 --input-modality ouster-lidar --dataset-name 2021-10-26-10-49-06_e2e_rec_ss20_elva --model-type pilotnet-old
```