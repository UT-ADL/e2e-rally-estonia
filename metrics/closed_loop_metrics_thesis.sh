#!/bin/bash

echo "Steering 1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-romet-thesis --input-modality nvidia-camera --drive-datasets 2022-06-29-15-09-08_e2e_elva__forward_steering1 2022-06-29-15-45-24_e2e_elva__backward_steering1-2

echo "Steering 2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-romet-thesis --input-modality nvidia-camera --drive-datasets 2022-06-29-10-46-40_e2e_elva__forward_steering2 2022-06-29-11-21-56_e2e_elva__backward_steering2

echo "Steering Overfit"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-romet-thesis --input-modality nvidia-camera --drive-datasets 2022-06-29-16-26-28_e2e_elva__forward_steering_overfit 2022-06-29-17-02-55_e2e_elva_backward_steering_overfit

echo "Waypoints 1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-romet-thesis --input-modality nvidia-camera --drive-datasets 2022-06-28-16-54-49_e2e_elva__forward_waypoints_bal 2022-06-28-17-29-21_e2e_elva__backwards_waypoints_bal

echo "Waypoints 2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-romet-thesis --input-modality nvidia-camera --drive-datasets 2022-06-29-12-18-31_e2e_elva__forward_trajectory_2 2022-06-29-12-57-41_e2e_elva__backward_trajectory_2

echo "Waypoints Overfit"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-romet-thesis --input-modality nvidia-camera --drive-datasets 2022-06-29-18-05-32_e2e_elva_forward_waypoinit_overfit 2022-06-29-18-21-39_e2e_elva_forward_waypoinit_overfit2 2022-06-29-18-42-10_e2e_elva_backward_waypoint_overfit
