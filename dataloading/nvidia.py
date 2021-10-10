import numpy as np
import pandas as pd
from collections import defaultdict

from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

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

        ymin = 520
        ymax = 864

        scale = 0.2

        height = ymax - ymin
        width = xmax - xmin
        cropped = transforms.functional.resized_crop(data["image"], ymin, xmin+self.x_delta, height, width,
                                                     (int(scale*height), int(scale*width)))

        data["image"] = cropped
        return data

class NvidiaSideCameraZoom(object):

    def __init__(self, zoom_ratio):
        self.zoom_ratio = zoom_ratio

    def __call__(self, data):
        width = 1920
        height = 1208

        xmin = int(self.zoom_ratio*width)
        ymin = int(self.zoom_ratio*height)

        scaled_width = width - (2*xmin)
        scaled_height = height - (2*ymin)

        cropped = transforms.functional.resized_crop(data["image"], ymin, xmin, scaled_height, scaled_width,
                                                     (height, width))

        data["image"] = cropped
        return data

class Normalize(object):
    def __call__(self, data, transform=None):
        #normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = data["image"]
        image = image / 255
        #data["image"] = normalize(image)
        data["image"] = image
        return data

class NvidiaDataset(Dataset):

    N_WAYPOINTS = 5
    CAP_WAYPOINTS = 10

    def __init__(self, dataset_paths, transform=None, camera="front_wide"):

        self.dataset_paths = dataset_paths
        self.transform = transform
        self.camera_name = camera

        datasets = [self.read_dataset(dataset_path, camera) for dataset_path in dataset_paths]
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

    def read_dataset(self, dataset_path, camera):
        frames_df = pd.read_csv(dataset_path / "nvidia_frames.csv")

        # temp hack
        if "autonomous" not in frames_df.columns:
            frames_df["autonomous"] = False
        #frames_df["autonomous"] = False

        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]
        frames_df = frames_df[frames_df[f'{camera}_filename'].notna()]

        vehicle_x = frames_df["position_x"]
        vehicle_y = frames_df["position_y"]
        for i in np.arange(1, self.N_WAYPOINTS + 1):
            wp_global_x = frames_df["position_x"].shift(-i * self.CAP_WAYPOINTS)
            wp_global_y = frames_df["position_y"].shift(-i * self.CAP_WAYPOINTS)
            yaw = frames_df["yaw"]
            frames_df["yaw"] = yaw

            wp_local_x = (wp_global_x - vehicle_x) * np.cos(yaw) + (wp_global_y - vehicle_y) * np.sin(yaw)
            wp_local_y = -(wp_global_x - vehicle_x) * np.sin(yaw) + (wp_global_y - vehicle_y) * np.cos(yaw)
            frames_df[f"x_{i}_offset"] = wp_local_x
            frames_df[f"y_{i}_offset"] = wp_local_y

            # Remove rows without trajectory offsets, should be last N_WAYPOINTS rows
            frames_df = frames_df[frames_df[f"x_{i}_offset"].notna()]

        camera_images = frames_df[f"{camera}_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]

        return frames_df
