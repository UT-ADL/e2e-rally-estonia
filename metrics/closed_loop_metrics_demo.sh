#!/bin/bash

echo "Steering"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-demo --input-modality nvidia-camera --drive-datasets 2022-06-17-13-21-51_e2e_elva__steering

echo "Steering wide"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-demo --input-modality nvidia-camera --drive-datasets 2022-06-17-14-26-28_e2e_elva__steering_wide

echo "Trajectory"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-demo --input-modality nvidia-camera --drive-datasets 2022-06-17-13-42-44_e2e_elva__trajectory

echo "Trajectory balanced"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-demo --input-modality nvidia-camera --drive-datasets 2022-06-17-14-06-10_e2e_elva__trajectory_bal