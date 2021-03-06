#!/bin/bash

echo "Camera v1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-12-09-43_e2e_rec_elva-nvidia-v1-0.8 2021-11-25-12-21-17_e2e_rec_elva-nvidia-v1-0.8-forward
echo "Camera v2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-14-01-46_e2e_rec_elva-licamera-v2-0.8-back 2021-11-25-14-13-59_e2e_rec_elva-licamera-v2-0.8-forward
echo "Camera v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-14-51-46_e2e_rec_elva-l-camera-v3-0.8-back 2021-11-25-15-04-26_e2e_rec_elva-l-camera-v3-0.8-forward
echo "Camera in train"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-13-11-40_e2e_rec_elva-licamera-inTrain-0.8-back 2021-11-25-13-24-00_e2e_rec_elva-licamera-inTrain-0.8-forward

echo "Lidar v1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-12-45-35_e2e_rec_elva-lidar-v1-0.8-back 2021-11-25-12-57-24_e2e_rec_elva-lidar-v1-0.8-forward
echo "Lidar v2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-14-27-56_e2e_rec_elva-lidar-v2-0.8-back 2021-11-25-14-39-43_e2e_rec_elva-lidar-v2-0.8-forward
echo "Lidar v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-15-16-31_e2e_rec_elva-l-lidar-v3-0.8-back 2021-11-25-15-27-38_e2e_rec_elva-l-lidar-v3-0.8-forward
echo "Lidar in train"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-13-37-42_e2e_rec_elva-lilidar-inTrain-0.8-back 2021-11-25-13-48-44_e2e_rec_elva-lilidar-inTrain-0.8-forward
echo "Lidar in train night"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-16-57-26_e2e_rec_elva-lidar-inTrain-0.8-forwardNight 2021-11-25-17-08-28_e2e_rec_elva-lidar-inTrain-0.8-backNight
echo "Lidar night"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-17-31-42_e2e_rec_elva-lidar-0.8-forwardNight 2021-11-25-17-43-47_e2e_rec_elva-lidar-0.8-backNight
echo "Lidar night #2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-17-56-16_e2e_rec_elva-lidar-0.8-forwardNight_attempt2 2021-11-25-18-07-28_e2e_rec_elva-lidar-0.8-backNight_attempt2

echo "Lidar all channels"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-11-19-15_e2e_rec_elva_i_allChannels_forward_0.8 2021-11-26-11-30-23_e2e_rec_elva_i_allChannels_back_0.8
echo "Lidar intensity"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-10-53-35_e2e_rec_elva_intensity_forward_0.8 2021-11-26-11-07-10_e2e_rec_elva_intensity_back_0.8
echo "Lidar range"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-11-42-02_e2e_rec_elva_i_range_forward_0.8
echo "Lidar ambience"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-11-53-18_e2e_rec_elva_i_ambience_forward_0.8

echo "Lidar Winter v1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-02-02-10-39-23_e2e_rec_elva_winter_lidar_forward_08 2022-02-02-10-50-07_e2e_rec_elva_winter_lidar_forward_08 2022-02-02-10-53-16_e2e_rec_elva_winter_lidar_backw_08
echo "Lidar Winter v2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-02-02-11-05-18_e2e_rec_elva_winter_lidar-v5_forw_08 2022-02-02-11-18-14_e2e_rec_elva_winter_lidar-v5_backw_08
echo "Lidar Winter v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-02-02-11-32-37_e2e_rec_elva_winter_lidar-v3_forw_08 2022-02-02-11-45-34_e2e_rec_elva_winter_lidar-v3_backw_08
echo "Camera Winter v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-02-02-11-58-48_e2e_rec_elva_winter_camera-v3_forw_08

echo "Camera in train spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-04-28-11-59-00_e2e_elva_forw_inTrain_camera_0.8 2022-04-28-12-10-19_e2e_elva_back_inTrain_camera_0.8

echo "Lidar in train spring x517 y45"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-04-28-13-02-25_e2e_elva_forw_inTrain_lidar_x517_y45_0.8 2022-04-28-13-14-00_e2e_elva_back_inTrain_lidar_x517_y45_0.8

echo "Camera spring in train BGR"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-04-28-13-37-48_e2e_elva_forw_BGR_inTrain_0.8 2022-04-28-13-48-48_e2e_elva_back_BGR_inTrain_0.8

echo "Camera spring RGB"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-04-28-15-23-45_e2e_elva_RGB_forw_0.8 2022-04-28-15-35-09_e2e_elva_RGB_back_0.8

echo "Camera spring BGR"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-04-28-15-46-30_e2e_elva_BGR_forw_0.8 2022-04-28-15-57-16_e2e_elva_BGR_back_0.8

echo "Lidar in train spring x45 y517"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-04-28-16-09-27_e2e_elva_lidar_forw_x45_y517_0.8 2022-04-28-16-20-09_e2e_elva_lidar_back_x45_y517_0.8

echo "Camera v1 spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-04-28-16-31-42_e2e_elva_camera_v1_forw_0.8 2022-04-28-16-42-14_e2e_elva_camera_v1_back_0.8

echo "Camera in train spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-05-04-11-11-34_e2e_elva_camera_inTrain_forw_0.8 2022-05-04-11-22-13_e2e_elva_camera_inTrain_back_0.8

echo "Camera v1 spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-05-04-11-33-45_e2e_elva_camera_v1_forw_0.8 2022-05-04-11-44-53_e2e_elva_camera_v1_back_0.8

echo "Camera v2 spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-05-04-11-55-47_e2e_elva_camera_v2_forw_0.8 2022-05-04-12-06-46_e2e_elva_camera_v2_back_0.8

echo "Camera v3 spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-05-04-12-18-31_e2e_elva_camera_v3_forw_0.8 2022-05-04-12-29-12_e2e_elva_camera_v3_back_0.8

echo "Lidar in train spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-05-04-13-39-42_e2e_elva_lidar_intrain_forw_0.8 2022-05-04-14-15-58_e2e_elva_lidar_intrain_back2_0.8

echo "Lidar intensity spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-05-04-14-22-26_e2e_elva_intensity_forw_0.8

echo "Lidar v2 spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-05-04-14-27-04_e2e_elva_lidar_v2_forw_0.8 2022-05-04-14-37-56_e2e_elva_lidar_v2_back_0.8

echo "Lidar v1 spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-05-04-14-48-47_e2e_elva_lidar_v1_forw_0.8 2022-05-04-14-59-26_e2e_elva_lidar_v1_back_0.8

echo "Lidar v3 spring"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-05-04-15-10-31_e2e_elva_lidar_v3_forw_0.8 2022-05-04-15-19-38_e2e_elva_lidar_v3_forw2_0.8 2022-05-04-15-27-12_e2e_elva_lidar_v3_back_0.8
