# LiDAR-as-Camera for End-to-End Driving

This repository contains the code to reproduce the experiments of the paper ["LiDAR-as-Camera for End-to-End Driving"](https://drive.google.com/file/d/1S-YzcJccHsM0LTmpnaC91I5yWi1FQqPk/view).

## Dataset

Before training a model, dataset (rosbags) must be downloaded and extracted. See [_dataloading_](./dataloading/README.md) manual for this. 

To get access to the dataset, please fill out [this form](https://forms.gle/nDkwcpzgBoYeJBE39).

## Models

[Pretrained models](../models/lidar-camera-paper) are included with the code. All training runs were also logged using 
[Weight & Biases](https://wandb.ai/rometaidla/lanefollowing-ut-camera-vs-lidar?workspace=user-). 

Models can be retrained by following these instructions:

### Camera models

```bash
python train.py --model-name camera-v1 --input-modality nvidia-camera --output-modality steering_angle  --model-type pilotnet --patience 10 --max-epochs 100
```

All camera based models (camera-v1, camera-v2, camera-v3) were trained using same arguments.

### LiDAR models

To train model using LiDAR input, just change `--input-modality` argument to `ouster-lidar`.

```bash
python train.py --model-name camera-v1 --input-modality ouster-lidar --output-modality steering_angle --model-type pilotnet --patience 10 --max-epochs 100 
```

All lidar based models (lidar-v1, lidar-v2, lidar-v3) were trained using same arguments.

To train lidar model with specific channel, argument `--lidar-channel` can be used to specify channel to use during training. 

```bash
python train.py --model-name lidar-intensity --input-modality ouster-lidar --lidar-channel intensity --output-modality steering_angle --model-type pilotnet --patience 10 --max-epochs 100 
python train.py --model-name lidar-ambience --input-modality ouster-lidar --lidar-channel ambient --output-modality steering_angle --model-type pilotnet --patience 10 --max-epochs 100 
python train.py --model-name lidar-range --input-modality ouster-lidar --lidar-channel range --output-modality steering_angle --model-type pilotnet --patience 10 --max-epochs 100 
```

## Results
### Off-policy metrics

Off-policy metrics can be calculated by running following script from root folder: 

```bash
python -m metrics.calculate_model_ol_metrics.py --root-path <path to extracted dataset>
```

Use `--root-path` parameter to defined path where Rally Estonia Dataset is downloaded and extracted.

### On-policy metrics

On policy metrics are calculated for each model using following command:

```bash
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-12-09-43_e2e_rec_elva-nvidia-v1-0.8 2021-11-25-12-21-17_e2e_rec_elva-nvidia-v1-0.8-forward
```

### Article tables

Once both off- and on-policy metrics are calculated following notebooks can be used to construct final tables:

- [article_main_table.ipynb](../notebooks/article_main_table.ipynb) for calculating main results table
- [article_main_table.ipynb](../notebooks/article_main_table.ipynb) for calculating whiteness anaysis tables

These notebooks already have precalculated metrics included, so can be re-run without needing to calculate off- and on-policy metrics.