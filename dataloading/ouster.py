import numpy as np
import pandas as pd

import torchvision
from torch.utils.data import Dataset


class OusterCrop(object):
    def __init__(self):
        self.fov = 256
        self.top = 54
        self.left = 256 - self.fov // 2
        self.height = 68
        self.width = 264

    def __call__(self, data):
        # TODO: clean up this mess
        data["image"] = data["image"][:, :, 256:-256]
        data["image"] = data["image"][..., self.top:self.top + self.height, self.left:self.left + self.width]
        return data


class OusterNormalize(object):
    def __call__(self, data, transform=None):
        data["image"] = data["image"] / 255
        return data


class OusterDataset(Dataset):
    N_WAYPOINTS = 5
    CAP_WAYPOINTS = 10

    def __init__(self, dataset_paths, transform=None):

        self.dataset_paths = dataset_paths
        self.transform = transform

        datasets = [self.read_dataset(dataset_path) for dataset_path in dataset_paths]
        self.frames = pd.concat(datasets)

    def __getitem__(self, idx):
        frame = self.frames.iloc[idx]
        image = torchvision.io.read_image(frame["image_path"])

        data = {
            'image': image,
            'steering_angle': np.array(frame["steering_angle"]),
            'vehicle_speed': np.array(frame["vehicle_speed"]),
            'waypoints': np.array([frame["x_1_offset"], frame["y_1_offset"],
                                   frame["x_2_offset"], frame["y_2_offset"],
                                   frame["x_3_offset"], frame["y_3_offset"],
                                   frame["x_4_offset"], frame["y_4_offset"],
                                   frame["x_5_offset"], frame["y_5_offset"]])
        }

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.frames.index)

    def read_dataset(self, dataset_path):
        frames_df = pd.read_csv(dataset_path / "lidar_frames.csv")

        # temp hack
        if "autonomous" not in frames_df.columns:
            frames_df["autonomous"] = False
        # frames_df["autonomous"] = False

        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]
        frames_df = frames_df[frames_df['lidar_filename'].notna()]

        vehicle_x = frames_df["position_x"]
        vehicle_y = frames_df["position_y"]
        for i in np.arange(1, self.N_WAYPOINTS + 1):
            wp_global_x = frames_df["position_x"].shift(-i * self.CAP_WAYPOINTS)
            wp_global_y = frames_df["position_y"].shift(-i * self.CAP_WAYPOINTS)
            yaw = frames_df["yaw"]

            wp_local_x = (wp_global_x - vehicle_x) * np.cos(yaw) + (wp_global_y - vehicle_y) * np.sin(yaw)
            wp_local_y = -(wp_global_x - vehicle_x) * np.sin(yaw) + (wp_global_y - vehicle_y) * np.cos(yaw)
            frames_df[f"x_{i}_offset"] = wp_local_x
            frames_df[f"y_{i}_offset"] = wp_local_y

            # Remove rows without trajectory offsets, should be last N_WAYPOINTS rows
            frames_df = frames_df[frames_df[f"x_{i}_offset"].notna()]

        camera_images = frames_df["lidar_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]

        return frames_df
