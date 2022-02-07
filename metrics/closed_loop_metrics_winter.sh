#!/bin/bash

echo "Lidar Winter v1"
python metrics.py --input-modality ouster-lidar --drive-datasets 2022-02-02-11-32-37_e2e_rec_elva_winter_lidar-v3_forw_08 2022-02-02-11-45-34_e2e_rec_elva_winter_lidar-v3_backw_08
echo "Lidar Winter v2"
python metrics.py --input-modality ouster-lidar --drive-datasets 2022-02-02-10-39-23_e2e_rec_elva_winter_lidar_forward_08 2022-02-02-10-50-07_e2e_rec_elva_winter_lidar_forward_08 2022-02-02-10-53-16_e2e_rec_elva_winter_lidar_backw_08
echo "Lidar Winter v3"
python metrics.py --input-modality ouster-lidar --drive-datasets 2022-02-02-11-05-18_e2e_rec_elva_winter_lidar-v5_forw_08 2022-02-02-10-53-16_e2e_rec_elva_winter_lidar_backw_08
echo "Camera Winter v3"
python metrics.py --input-modality ouster-lidar --drive-datasets 2022-02-02-11-58-48_e2e_rec_elva_winter_camera-v3_forw_08'

