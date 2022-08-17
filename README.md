# Rally Estonia End-to-End Driving

This repository contains a framework to train End-to-End driving models using the Rally Estonia dataset. It includes 
scripts to train models as well dataset of over 750 km of driving in all seasons.

## Articles and Theses

Following work has been produced using the framework:
 - [LiDAR-as-Camera for End-to-End Driving](./docs/lidar_as_camera.md) (ICRA2022 Workshop Article) 
 - Comparing Output Modalities in End-to-End Driving (Master Thesis)
 
Directory Structure
------
    .
    ├── data_extract            # Code for extracting dataset from drive recordings
    ├── dataloading             # Dataloaders for training models with PyTorch
    ├── docs                    # Documentation, instructions for reproducing articles and theses
    ├── metrics                 # Code used to calculate open and closed loop metrics.
    ├── models                  # Models trained using the framework
    └── notebooks               # Generic services
        ├── lidar-as-camera     # 'LiDAR-as-Camera for End-to-End Driving' article related notebooks
        ├── output-modalities   # 'Comparing Output Modalities in End-to-End Driving' master thesis related notebooks
    ├── tensorrt                # Inference with TensorRT library 
    ├── velocity_model          # Longitudinal model using predriven human trajectory to control vehicle's velocity.
    ├── viz                     # Scripts for interpret model's behavior using Visual BackProp, creating videos from drives with overlay etc.

## Dataset

To get access to the dataset, please fill out [this form](https://forms.gle/nDkwcpzgBoYeJBE39).

Before training a model, dataset needs to be downloaded and prepared . See [_dataloading_](./dataloading/README.md) manual for this. 

### Following input modalities are supported:
- Nvidia RGB cameras (only front wide camera used currently)
- Ouster LiDAR range, ambient, intensity image
![camera_crop](./media/camera-crops.png "Camera crop")

![camera_crop](./media/summer_lidar_crop.jpg "Summer lidar crop")

### Following output modalities are supported:
- Steering angle
- Trajectory waypoints (longitudinal part fixed)

Only lateral control (steering) is predicted, longitudinal control (throttle) is not predicted and must be controlled using other means.


## Training

### Environment setup

Environment can set up using conda by following commands:

```bash
# Set up Pytorch environment
conda create -n e2e pytorch torchvision cudatoolkit=11.1 jupyter pandas matplotlib tqdm scikit-learn scikit-image onnx seaborn -c pytorch -c nvidia
conda activate e2e

# Install TensorRT and pycuda
pip install nvidia-tensorrt==7.2.3.4
pip install 'pycuda<2021.1'

# Wandb
pip install wandb

# For visualising predictions
pip install opencv-contrib-python
# need to use specific version of moviepy as newer version did not work
pip install moviepy==1.0.0 
```

### Run training

Model can be trained using following command:

```bash
python train.py --input-modality nvidia-camera --output-modality steering_angle --patience 10 --max-epochs 100 --model-name steering-angle --model-type pilotnet-conditional --wandb-project summer-models-6 --dataset-folder <path to extracted dataset>
```

Use `--input-modality` parameter to train using camera or lidar images.

Use `--output-modality` parameter to use different prediction targets like steering angle or trajectory waypoints.

Use `--model-type` parameter to use different model architectures like `pilotnet-conditional` and `pilotnet-control`.

Use `--wandb-project` parameter to use log using W&B. To use without W&B, just omit this parameter.
