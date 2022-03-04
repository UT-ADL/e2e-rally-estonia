import numpy as np
import pandas as pd
import torch

import torchvision
from torchvision import transforms

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
    CHANNEL_MAP = {
        "ambience": 2,
        "intensity": 1,
        "range": 0
    }

    def __init__(self, dataset_paths, transform=None, filter_turns=False, channel=None):

        self.dataset_paths = dataset_paths
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([OusterCrop(), OusterNormalize()])

        print(f"Using only lidar channel {channel}")
        self.channel = channel

        datasets = [self.read_dataset(dataset_path) for dataset_path in dataset_paths]
        self.frames = pd.concat(datasets)

        if filter_turns:
            print("Filtering turns with blinker signal")
            self.frames = self.frames[self.frames.turn_signal == 1]

    def __getitem__(self, idx):
        frame = self.frames.iloc[idx]

        image = torchvision.io.read_image(frame["image_path"])
        if self.channel:
            channel_idx = self.CHANNEL_MAP[self.channel]
            image = torch.unsqueeze(image[channel_idx], dim=0)

        data = {
            'image': image,
            'steering_angle': np.array(frame["steering_angle"]),
            'vehicle_speed': np.array(frame["vehicle_speed"])
        }

        if self.transform:
            data = self.transform(data)

        return data, "dummy"

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

        frames_df["turn_signal"].fillna(1, inplace=True)
        frames_df["turn_signal"] = frames_df["turn_signal"].astype(int)

        camera_images = frames_df["lidar_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]

        print(f"{dataset_path}: {len(frames_df)}")
        return frames_df


class OusterTrainDataset(OusterDataset):
    def __init__(self, root_path, filter_turns=False, channel=None):
        train_paths = [
            root_path / "2021-05-20-12-36-10_e2e_sulaoja_20_30",
            root_path / "2021-05-20-12-43-17_e2e_sulaoja_20_30",
            root_path / "2021-05-20-12-51-29_e2e_sulaoja_20_30",
            root_path / "2021-05-20-13-44-06_e2e_sulaoja_10_10",
            root_path / "2021-05-20-13-51-21_e2e_sulaoja_10_10",
            root_path / "2021-05-20-13-59-00_e2e_sulaoja_10_10",
            root_path / "2021-05-28-15-07-56_e2e_sulaoja_20_30",
            root_path / "2021-05-28-15-17-19_e2e_sulaoja_20_30",
            root_path / "2021-06-09-13-14-51_e2e_rec_ss2",
            root_path / "2021-06-09-13-55-03_e2e_rec_ss2_backwards",
            root_path / "2021-06-09-14-58-11_e2e_rec_ss3",
            root_path / "2021-06-09-15-42-05_e2e_rec_ss3_backwards",
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
            root_path / "2021-09-24-11-19-25_e2e_rec_ss10",
            root_path / "2021-09-24-11-40-24_e2e_rec_ss10_2",
            root_path / "2021-09-24-12-02-32_e2e_rec_ss10_3",
            root_path / "2021-09-24-12-21-20_e2e_rec_ss10_backwards",
            root_path / "2021-09-24-13-39-38_e2e_rec_ss11",
            root_path / "2021-09-30-13-57-00_e2e_rec_ss14",
            root_path / "2021-09-30-15-03-37_e2e_ss14_from_half_way",
            root_path / "2021-09-30-15-20-14_e2e_ss14_backwards",
            root_path / "2021-09-30-15-56-59_e2e_ss14_attempt_2",
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
            root_path / "2021-10-25-17-06-34_e2e_rec_ss2_arula_back"
        ]

        tr = transforms.Compose([OusterCrop(), OusterNormalize()])

        super().__init__(train_paths, tr, filter_turns=filter_turns, channel=channel)

class OusterValidationDataset(OusterDataset):
    def __init__(self, root_path, filter_turns=False, channel=None):
        valid_paths = [
            root_path / "2021-05-28-15-19-48_e2e_sulaoja_20_30",
            root_path / "2021-06-07-14-20-07_e2e_rec_ss6",
            root_path / "2021-06-07-14-06-31_e2e_rec_ss6",
            root_path / "2021-06-07-14-09-18_e2e_rec_ss6",
            root_path / "2021-06-07-14-36-16_e2e_rec_ss6",
            root_path / "2021-09-24-14-03-45_e2e_rec_ss11_backwards",
            root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva",
            root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back",
            root_path / "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back",
            root_path / "2021-10-11-14-50-59_e2e_rec_vahi",
            root_path / "2021-10-14-13-08-51_e2e_rec_vahi_backwards"

        ]

        tr = transforms.Compose([OusterCrop(), OusterNormalize()])
        super().__init__(valid_paths, tr, filter_turns=filter_turns, channel=channel)
