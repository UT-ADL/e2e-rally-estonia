import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

from skimage.util import random_noise

from dataloading.model import Camera


class NvidiaResizeAndCrop(object):
    def __call__(self, data):
        xmin = 186
        ymin = 600

        scale = 6.0
        width = 258
        height = 66
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        cropped = transforms.functional.resized_crop(data["image"], ymin, xmin, scaled_height, scaled_width,
                                                     (height, width))

        data["image"] = cropped
        return data


class NvidiaCropWide(object):
    def __init__(self, x_delta=0):
        self.x_delta = x_delta

    def __call__(self, data):
        xmin = 300
        xmax = 1620

        ymin = 570
        ymax = 914

        scale = 0.2

        height = ymax - ymin
        width = xmax - xmin
        cropped = F.resized_crop(data["image"], ymin, xmin + self.x_delta, height, width,
                                 (int(scale * height), int(scale * width)))

        data["image"] = cropped
        return data


class CropViT(object):
    def __call__(self, data):
        xmin = 540
        xmax = 1260

        ymin = 244
        ymax = 964

        scale = 0.312

        height = ymax - ymin
        width = xmax - xmin
        cropped = F.resized_crop(data["image"], ymin, xmin, height, width,
                                 (int(scale * height), int(scale * width)))
        data["image"] = cropped
        return data


class NvidiaSideCameraZoom(object):

    def __init__(self, zoom_ratio):
        self.zoom_ratio = zoom_ratio

    def __call__(self, data):
        width = 1920
        height = 1208

        xmin = int(self.zoom_ratio * width)
        ymin = int(self.zoom_ratio * height)

        scaled_width = width - (2 * xmin)
        scaled_height = height - (2 * ymin)

        cropped = F.resized_crop(data["image"], ymin, xmin, scaled_height, scaled_width,
                                 (height, width))

        data["image"] = cropped
        return data


class AugmentationConfig:
    def __init__(self, color_prob=0.0, noise_prob=0.0, blur_prob=0.0):
        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob


class AugmentImage:
    def __init__(self, augment_config):
        print(f"augmentation: color_prob={augment_config.color_prob}, "
              f"noise_prob={augment_config.noise_prob}, "
              f"blur_prob={augment_config.blur_prob}")
        self.augment_config = augment_config

    def __call__(self, data):
        if np.random.random() <= self.augment_config.color_prob:
            jitter = transforms.ColorJitter(contrast=0.5, saturation=0.5, brightness=0.5)
            data["image"] = jitter(data["image"])

        if np.random.random() <= self.augment_config.noise_prob:
            if np.random.random() > 0.5:
                data["image"] = torch.tensor(random_noise(data["image"], mode='gaussian', mean=0, var=0.005, clip=True),
                                             dtype=torch.float)
            else:
                data["image"] = torch.tensor(random_noise(data["image"], mode='salt', amount=0.005),
                                             dtype=torch.float)

        if np.random.random() <= self.augment_config.blur_prob:
            blurrer = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 1))
            data["image"] = blurrer(data['image'])

        return data


class Normalize(object):
    def __call__(self, data, transform=None):
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = data["image"]
        image = image / 255
        # data["image"] = normalize(image)
        data["image"] = image
        return data


class NvidiaDataset(Dataset):

    def __init__(self, dataset_paths, transform=None, camera="front_wide", name="Nvidia dataset",
                 filter_turns=False, output_modality="steering_angle", n_branches=1, n_waypoints=10,
                 metadata_file="nvidia_frames.csv", color_space="rgb", side_cameras_weight=0.33):
        self.name = name
        self.metadata_file = metadata_file
        self.color_space = color_space
        self.dataset_paths = dataset_paths
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([Normalize()])
        self.camera_name = camera
        self.output_modality = output_modality
        self.n_waypoints = n_waypoints
        self.side_cameras_weight = side_cameras_weight

        if self.output_modality == "waypoints":
            self.target_size = 2 * self.n_waypoints
        elif self.output_modality == "steering_angle":
            self.target_size = 1
        else:
            print(f"Unknown output modality {self.output_modality}")
            sys.exit()

        self.n_branches = n_branches

        if camera == 'all':
            datasets = [self.read_dataset(dataset_path, "left") for dataset_path in dataset_paths] + \
                       [self.read_dataset(dataset_path, "right") for dataset_path in dataset_paths] + \
                       [self.read_dataset(dataset_path, "front_wide") for dataset_path in dataset_paths]

        else:
            datasets = [self.read_dataset(dataset_path, camera) for dataset_path in dataset_paths]
        self.frames = pd.concat(datasets)

        if filter_turns:
            print("Filtering turns with blinker signal")
            self.frames = self.frames[self.frames.turn_signal == 1]

    def __getitem__(self, idx):
        frame = self.frames.iloc[idx]
        if self.color_space == "rgb":
            image = torchvision.io.read_image(frame["image_path"])
        elif self.color_space == "bgr":
            image = cv2.imread(frame["image_path"])
            image = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)
        else:
            print(f"Unknown color space: ", self.color_space)
            sys.exit()

        # TODO replace if-else with map
        if self.camera_name == Camera.LEFT.value:
            steering_angle = np.array(frame["steering_angle_left"])
        elif self.camera_name == Camera.RIGHT.value:
            steering_angle = np.array(frame["steering_angle_right"])
        else:
            steering_angle = np.array(frame["steering_angle"])

        data = {
            'image': image,
            'steering_angle': steering_angle,
            'vehicle_speed': np.array(frame["vehicle_speed"]),
            'autonomous': np.array(frame["autonomous"]),
            'position_x': np.array(frame["position_x"]),
            'position_y': np.array(frame["position_y"]),
            'yaw': np.array(frame["yaw"]),
            'turn_signal': np.array(frame["turn_signal"]),
            'row_id': np.array(frame["row_id"]),
        }

        turn_signal = int(frame["turn_signal"])

        if self.output_modality == "waypoints":
            waypoints = []
            for i in np.arange(1, self.n_waypoints + 1):
                waypoints.append(frame[f"wp{i}_{self.camera_name}_x"])
                waypoints.append(frame[f"wp{i}_{self.camera_name}_y"])

            data['waypoints'] = np.array(waypoints)
            target_values = waypoints
        else:
            target_values = frame["steering_angle"]

        if self.transform:
            data = self.transform(data)

        if self.n_branches > 1:
            target = np.zeros((self.n_branches, self.target_size))
            target[turn_signal, :] = target_values

            conditional_mask = np.zeros((self.n_branches, self.target_size))
            conditional_mask[turn_signal, :] = 1
        else:
            target = np.zeros((self.n_branches, self.target_size))
            target[0, :] = target_values
            conditional_mask = np.ones((self.n_branches, self.target_size))

        return data, target.reshape(-1), conditional_mask.reshape(-1)

    def __len__(self):
        return len(self.frames.index)

    def get_waypoints(self):
        wp_x_cols = [f"wp{i}_{self.camera_name}_x" for i in np.arange(1, self.n_waypoints + 1)]
        wp_y_cols = [f"wp{i}_{self.camera_name}_y" for i in np.arange(1, self.n_waypoints + 1)]
        waypoint_cols = np.column_stack((wp_x_cols, wp_y_cols)).reshape(-1)
        return self.frames[waypoint_cols].to_numpy()

    def read_dataset(self, dataset_path, camera):
        if type(dataset_path) is dict:
            frames_df = pd.read_csv(dataset_path['path'] / self.metadata_file)
            len_before_filtering = len(frames_df)
            frames_df = frames_df.iloc[dataset_path['start']:dataset_path['end']]
            dataset_path = dataset_path['path']
        else:
            frames_df = pd.read_csv(dataset_path / self.metadata_file)
            len_before_filtering = len(frames_df)

        frames_df["row_id"] = frames_df.index

        # temp hack
        if "autonomous" not in frames_df.columns:
            frames_df["autonomous"] = False
        # frames_df["autonomous"] = False

        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        if camera != Camera.FRONT_WIDE.value:
            frames_df = frames_df[frames_df['steering_angle_left'].notna()]
            frames_df = frames_df[frames_df['steering_angle_right'].notna()]
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]
        frames_df = frames_df[frames_df[f'{camera}_filename'].notna()]

        frames_df["turn_signal"].fillna(1, inplace=True)
        frames_df["turn_signal"] = frames_df["turn_signal"].astype(int)

        # Removed frames marked as skipped
        frames_df = frames_df[frames_df["turn_signal"] != -1]  # TODO: remove magic values.

        if self.output_modality == "waypoints":
            frames_df = frames_df[frames_df[f"position_x"].notna()]
            frames_df = frames_df[frames_df[f"position_y"].notna()]

            for i in np.arange(1, self.n_waypoints + 1):
                frames_df = frames_df[frames_df[f"wp{i}_{camera}_x"].notna()]
                frames_df = frames_df[frames_df[f"wp{i}_{camera}_y"].notna()]

        frames_df["yaw_delta"] = np.abs(frames_df["yaw"]) - np.abs(frames_df["yaw"]).shift(-1)
        frames_df = frames_df[np.abs(frames_df["yaw_delta"]) < 0.1]

        # if self.calculate_waypoints:
        #
        #     vehicle_x = frames_df["position_x"]
        #     vehicle_y = frames_df["position_y"]
        #
        #     for i in np.arange(1, self.N_WAYPOINTS + 1):
        #         wp_global_x = frames_df["position_x"].shift(-i * self.CAP_WAYPOINTS)
        #         wp_global_y = frames_df["position_y"].shift(-i * self.CAP_WAYPOINTS)
        #         frames_df[f"x_{i}"] = wp_global_x
        #         frames_df[f"y_{i}"] = wp_global_y
        #         yaw = frames_df["yaw"]
        #         #frames_df["yaw"] = yaw
        #
        #         wp_local_x = (wp_global_x - vehicle_x) * np.cos(yaw) + (wp_global_y - vehicle_y) * np.sin(yaw)
        #         wp_local_y = -(wp_global_x - vehicle_x) * np.sin(yaw) + (wp_global_y - vehicle_y) * np.cos(yaw)
        #         frames_df[f"x_{i}_offset"] = wp_local_x
        #         frames_df[f"y_{i}_offset"] = wp_local_y
        #
        #         # Remove rows without trajectory offsets, should be last N_WAYPOINTS rows
        #         frames_df = frames_df[frames_df[f"x_{i}_offset"].notna()]
        #
        #     # frames_df["yaw_delta"] = np.abs(frames_df["yaw"]) - np.abs(frames_df["yaw"]).shift(-1)
        #     # frames_df = frames_df[np.abs(frames_df["yaw_delta"]) < 0.1]
        #     #
        #     # frames_df["x_1_delta"] = frames_df["x_1_offset"] - frames_df["x_1_offset"].shift(-1)
        #     # frames_df = frames_df[np.abs(frames_df["x_1_delta"]) < 0.1]
        #     #
        #     # frames_df["y_1_delta"] = frames_df["y_1_offset"] - frames_df["y_1_offset"].shift(-1)
        #     # frames_df = frames_df[np.abs(frames_df["y_1_delta"]) < 0.1]
        #
        #     # frames_df = frames_df[np.abs(frames_df["steering_angle"]) < 2.0]

        len_after_filtering = len(frames_df)

        camera_images = frames_df[f"{camera}_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]
        if self.output_modality == "waypoints":
            for i in np.arange(1, self.n_waypoints + 1):
                frames_df[f"wp{i}_all_x"] = frames_df[f"wp{i}_{camera}_x"]
                frames_df[f"wp{i}_all_y"] = frames_df[f"wp{i}_{camera}_y"]

        frames_df["camera_type"] = camera

        print(f"{dataset_path}: lenght={len(frames_df)}, filtered={len_before_filtering - len_after_filtering}")
        frames_df.reset_index(inplace=True)
        return frames_df

    def steering_angles_degrees(self):
        return self.frames.steering_angle.to_numpy() / np.pi * 180


class NvidiaTrainDataset(NvidiaDataset):
    def __init__(self, root_path, output_modality="steering_angle", n_branches=3, n_waypoints=10,
                 camera="front_wide", augment_conf=AugmentationConfig(), metadata_file="nvidia_frames.csv"):
        self.dataset_paths = [
            # root_path / "2021-05-20-12-36-10_e2e_sulaoja_20_30",
            # root_path / "2021-05-20-12-43-17_e2e_sulaoja_20_30",
            # root_path / "2021-05-20-12-51-29_e2e_sulaoja_20_30",
            # root_path / "2021-05-20-13-44-06_e2e_sulaoja_10_10",
            # root_path / "2021-05-20-13-51-21_e2e_sulaoja_10_10",
            # root_path / "2021-05-20-13-59-00_e2e_sulaoja_10_10",
            root_path / "2021-05-28-15-07-56_e2e_sulaoja_20_30",
            root_path / "2021-05-28-15-17-19_e2e_sulaoja_20_30",
            {'path': root_path / "2021-06-09-13-14-51_e2e_rec_ss2", 'start': 125, 'end': 49725},
            {'path': root_path / "2021-06-09-13-55-03_e2e_rec_ss2_backwards", 'start': 150, 'end': 53625},
            {'path': root_path / "2021-06-09-14-58-11_e2e_rec_ss3", 'start': 175, 'end': 43775},
            {'path': root_path / "2021-06-09-15-42-05_e2e_rec_ss3_backwards", 'start': 100, 'end': 40625},
            root_path / "2021-06-09-16-24-59_e2e_rec_ss13",
            root_path / "2021-06-09-16-50-22_e2e_rec_ss13_backwards",
            root_path / "2021-06-10-12-59-59_e2e_ss4",
            root_path / "2021-06-10-13-19-22_e2e_ss4_backwards",
            root_path / "2021-06-10-13-51-34_e2e_ss12",
            root_path / "2021-06-10-14-02-24_e2e_ss12_backwards",
            root_path / "2021-06-10-14-44-24_e2e_ss3_backwards",
            root_path / "2021-06-10-15-03-16_e2e_ss3_backwards",
            root_path / "2021-06-14-11-08-19_e2e_rec_ss14",
            root_path / "2021-06-14-11-22-05_e2e_rec_ss14",
            root_path / "2021-06-14-11-43-48_e2e_rec_ss14_backwards",
            {'path': root_path / "2021-09-24-11-19-25_e2e_rec_ss10", 'start': 400, 'end': 34550},
            {'path': root_path / "2021-09-24-11-40-24_e2e_rec_ss10_2", 'start': 150, 'end': 16000},
            {'path': root_path / "2021-09-24-12-02-32_e2e_rec_ss10_3", 'start': 350, 'end': 8050},
            root_path / "2021-09-24-12-21-20_e2e_rec_ss10_backwards",
            root_path / "2021-09-24-13-39-38_e2e_rec_ss11",
            {'path': root_path / "2021-09-30-13-57-00_e2e_rec_ss14", 'start': 100, 'end': 3200},
            root_path / "2021-09-30-15-03-37_e2e_ss14_from_half_way",
            root_path / "2021-09-30-15-20-14_e2e_ss14_backwards",
            {'path': root_path / "2021-09-30-15-56-59_e2e_ss14_attempt_2", 'start': 80, 'end': 54600},
            root_path / "2021-10-07-11-05-13_e2e_rec_ss3",
            root_path / "2021-10-07-11-44-52_e2e_rec_ss3_backwards",
            root_path / "2021-10-07-12-54-17_e2e_rec_ss4",
            root_path / "2021-10-07-13-22-35_e2e_rec_ss4_backwards",
            root_path / "2021-10-11-16-06-44_e2e_rec_ss2",
            root_path / "2021-10-11-17-10-23_e2e_rec_last_part",
            root_path / "2021-10-11-17-14-40_e2e_rec_backwards",
            root_path / "2021-10-11-17-20-12_e2e_rec_backwards",
            root_path / "2021-10-20-14-55-47_e2e_rec_vastse_ss13_17",
            root_path / "2021-10-20-13-57-51_e2e_rec_neeruti_ss19_22",
            root_path / "2021-10-20-14-15-07_e2e_rec_neeruti_ss19_22_back",
            root_path / "2021-10-25-17-31-48_e2e_rec_ss2_arula",
            root_path / "2021-10-25-17-06-34_e2e_rec_ss2_arula_back",
        ]

        tr = transforms.Compose([AugmentImage(augment_config=augment_conf), Normalize()])
        super().__init__(self.dataset_paths, tr, camera=camera, output_modality=output_modality, n_branches=n_branches,
                         n_waypoints=n_waypoints, metadata_file=metadata_file)


class NvidiaValidationDataset(NvidiaDataset):
    # todo: remove default parameters
    def __init__(self, root_path, output_modality="steering_angle", n_branches=3, n_waypoints=10, camera="front_wide",
                 metadata_file="nvidia_frames.csv"):
        self.dataset_paths = [
            root_path / "2021-05-28-15-19-48_e2e_sulaoja_20_30",
            #root_path / "2021-06-07-14-20-07_e2e_rec_ss6",
            root_path / "2021-06-07-14-06-31_e2e_rec_ss6",
            root_path / "2021-06-07-14-09-18_e2e_rec_ss6",
            #root_path / "2021-06-07-14-36-16_e2e_rec_ss6",
            root_path / "2021-09-24-14-03-45_e2e_rec_ss11_backwards",
            #root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva",
            #root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back",
            root_path / "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back",
            {'path': root_path / "2021-10-11-14-50-59_e2e_rec_vahi", 'start': 100, 'end': 15000},
            {'path': root_path / "2021-10-14-13-08-51_e2e_rec_vahi_backwards", 'start': 80, 'end': 13420},
            #root_path / "2022-06-10-13-23-01_e2e_elva_forward",
            #root_path / "2022-06-10-13-03-20_e2e_elva_backward"
        ]

        tr = transforms.Compose([Normalize()])
        super().__init__(self.dataset_paths, tr, camera=camera, output_modality=output_modality, n_branches=n_branches,
                         n_waypoints=n_waypoints, metadata_file=metadata_file)


class NvidiaWinterTrainDataset(NvidiaDataset):
    def __init__(self, root_path, output_modality="steering_angle",
                 n_branches=3, n_waypoints=10, augment_conf=AugmentationConfig()):
        train_paths = [
            root_path / '2021-11-08-11-24-44_e2e_rec_ss12_raanitsa',
            root_path / '2021-11-08-12-08-40_e2e_rec_ss12_raanitsa_backward',
            root_path / "2022-01-28-10-21-14_e2e_rec_peipsiaare_forward",
            root_path / "2022-01-28-12-46-59_e2e_rec_peipsiaare_backward",
            root_path / "2022-01-14-10-05-16_e2e_rec_raanitsa_forward",
            root_path / "2022-01-14-10-50-05_e2e_rec_raanitsa_backward",
            root_path / "2022-01-14-11-54-33_e2e_rec_kambja_forward2",
            root_path / "2022-01-14-12-21-40_e2e_rec_kambja_forward2_continue",
            root_path / "2022-01-14-13-09-05_e2e_rec_kambja_backward",
            root_path / "2022-01-14-13-18-36_e2e_rec_kambja_backward_continue",
            root_path / "2022-01-14-12-35-13_e2e_rec_neeruti_forward",
            root_path / "2022-01-14-12-45-51_e2e_rec_neeruti_backward",
            root_path / "2022-01-18-13-03-03_e2e_rec_arula_backward",
            root_path / "2022-01-18-13-43-33_e2e_rec_otepaa_forward",
            root_path / "2022-01-18-13-52-35_e2e_rec_otepaa_forward",
            root_path / "2022-01-18-13-56-22_e2e_rec_otepaa_forward",
            root_path / "2022-01-18-14-12-14_e2e_rec_otepaa_backward",
            root_path / "2022-01-18-15-20-35_e2e_rec_kanepi_forward",
            root_path / "2022-01-18-15-49-26_e2e_rec_kanepi_backwards",
        ]

        tr = transforms.Compose([AugmentImage(augment_config=augment_conf), Normalize()])
        super().__init__(train_paths, tr, output_modality=output_modality, n_branches=n_branches, n_waypoints=n_waypoints)


class NvidiaWinterValidationDataset(NvidiaDataset):
    def __init__(self, root_path, output_modality="steering_angle", n_branches=3, n_waypoints=10):
        valid_paths = [
            root_path / "2022-01-18-12-37-01_e2e_rec_arula_forward",
            root_path / "2022-01-18-12-47-32_e2e_rec_arula_forward_continue",
            root_path / "2022-01-28-14-47-23_e2e_rec_elva_forward",
            root_path / "2022-01-28-15-09-01_e2e_rec_elva_backward",
            root_path / "2022-01-25-15-25-15_e2e_rec_vahi_forward",
            root_path / "2022-01-25-15-34-01_e2e_rec_vahi_backwards",
        ]

        tr = transforms.Compose([Normalize()])
        super().__init__(valid_paths, tr, output_modality=output_modality, n_branches=n_branches, n_waypoints=n_waypoints)
