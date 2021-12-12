# This packege containes code for visualisation scripts, like creating autonomous drives.

## Video creator

### Creating on-policy video from (autonomous) drive:
```bash
python video_creator.py --dataset-folder=/gpfs/space/projects/Bolt/dataset/<drive-name> --video-type=driving
```

### Creating off-policy video using trained model:
```bash
python video_creator.py --dataset-folder=/gpfs/space/projects/Bolt/dataset/<drive-name> --video-type=prediction --model-path <path to pytorch model>
```
