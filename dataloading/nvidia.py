import numpy as np
import pandas as pd

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
        cropped = transforms.functional.resized_crop(data["image"], ymin, xmin + self.x_delta, height, width,
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
        cropped = transforms.functional.resized_crop(data["image"], ymin, xmin, height, width,
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

        cropped = transforms.functional.resized_crop(data["image"], ymin, xmin, scaled_height, scaled_width,
                                                     (height, width))

        data["image"] = cropped
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
    N_WAYPOINTS = 5
    CAP_WAYPOINTS = 30

    def __init__(self, dataset_paths, transform=None, camera="front_wide", name="Nvidia dataset",
                 filter_turns=False, output_modality="steering"):
        self.name = name
        self.dataset_paths = dataset_paths
        self.transform = transform
        self.camera_name = camera
        self.calculate_waypoints = output_modality == "waypoints"

        datasets = [self.read_dataset(dataset_path, camera) for dataset_path in dataset_paths]
        self.frames = pd.concat(datasets)

        if filter_turns:
            print("Filtering turns with blinker signal")
            self.frames = self.frames[self.frames.turn_signal == 1]

    def __getitem__(self, idx):
        frame = self.frames.iloc[idx]
        image = torchvision.io.read_image(frame["image_path"])

        data = {
            'image': image,
            'steering_angle': np.array(frame["steering_angle"]),
            'vehicle_speed': np.array(frame["vehicle_speed"]),
            'autonomous': np.array(frame["autonomous"]),
            'position_x': np.array(frame["position_x"]),
            'position_y': np.array(frame["position_y"]),
            'yaw': np.array(frame["yaw"]),
            'turn_signal': np.array(frame["turn_signal"])
        }

        if self.calculate_waypoints:
            waypoints = []
            for i in range(self.N_WAYPOINTS):
                waypoints.append(frame[f"x_{i + 1}_offset"])
                waypoints.append(frame[f"y_{i + 1}_offset"])

            data['waypoints'] = np.array(waypoints)

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
        # frames_df["autonomous"] = False

        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]
        frames_df = frames_df[frames_df[f'{camera}_filename'].notna()]

        frames_df["turn_signal"].fillna(1, inplace=True)
        frames_df["turn_signal"] = frames_df["turn_signal"].astype(int)

        vehicle_x = frames_df["position_x"]
        vehicle_y = frames_df["position_y"]

        len_before_filtering = len(frames_df)

        if self.calculate_waypoints:
            for i in np.arange(1, self.N_WAYPOINTS + 1):
                wp_global_x = frames_df["position_x"].shift(-i * self.CAP_WAYPOINTS)
                wp_global_y = frames_df["position_y"].shift(-i * self.CAP_WAYPOINTS)
                frames_df[f"x_{i}"] = wp_global_x
                frames_df[f"y_{i}"] = wp_global_y
                yaw = frames_df["yaw"]
                frames_df["yaw"] = yaw

                wp_local_x = (wp_global_x - vehicle_x) * np.cos(yaw) + (wp_global_y - vehicle_y) * np.sin(yaw)
                wp_local_y = -(wp_global_x - vehicle_x) * np.sin(yaw) + (wp_global_y - vehicle_y) * np.cos(yaw)
                frames_df[f"x_{i}_offset"] = wp_local_x
                frames_df[f"y_{i}_offset"] = wp_local_y

                # Remove rows without trajectory offsets, should be last N_WAYPOINTS rows
                frames_df = frames_df[frames_df[f"x_{i}_offset"].notna()]

            # frames_df["yaw_delta"] = np.abs(frames_df["yaw"]) - np.abs(frames_df["yaw"]).shift(-1)
            # frames_df = frames_df[np.abs(frames_df["yaw_delta"]) < 0.1]
            #
            # frames_df["x_1_delta"] = frames_df["x_1_offset"] - frames_df["x_1_offset"].shift(-1)
            # frames_df = frames_df[np.abs(frames_df["x_1_delta"]) < 0.1]
            #
            # frames_df["y_1_delta"] = frames_df["y_1_offset"] - frames_df["y_1_offset"].shift(-1)
            # frames_df = frames_df[np.abs(frames_df["y_1_delta"]) < 0.1]

            # frames_df = frames_df[np.abs(frames_df["steering_angle"]) < 2.0]

        len_after_filtering = len(frames_df)

        camera_images = frames_df[f"{camera}_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]

        print(f"{dataset_path}: lenght={len(frames_df)}, filtered={len_before_filtering-len_after_filtering}")
        return frames_df

    def steering_angles_degrees(self):
        return self.frames.steering_angle.to_numpy() / np.pi * 180


class NvidiaTrainDataset(NvidiaDataset):
    def __init__(self, root_path, filter_turns=False, output_modality="steering"):
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

        tr = transforms.Compose([Normalize()])

        super().__init__(train_paths, tr, filter_turns=filter_turns, output_modality=output_modality)


class NvidiaValidationDataset(NvidiaDataset):
    def __init__(self, root_path, filter_turns=False, output_modality="steering"):
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

        tr = transforms.Compose([Normalize()])
        super().__init__(valid_paths, tr, filter_turns=filter_turns, output_modality=output_modality)


class NvidiaTestDataset(NvidiaDataset):
    def __init__(self, root_path, filter_turns=False):
        valid_paths = [
            root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva",
            root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back"
        ]

        tr = transforms.Compose([Normalize()])
        super().__init__(valid_paths, tr, filter_turns=filter_turns)


class NvidiaSpringTrainDataset(NvidiaDataset):
    def __init__(self, root_path, filter_turns=False):
        train_paths = [
            root_path / "2021-05-20-12-36-10_e2e_sulaoja_20_30",
            root_path / "2021-05-20-12-43-17_e2e_sulaoja_20_30",
            root_path / "2021-05-20-12-51-29_e2e_sulaoja_20_30",
            root_path / "2021-05-20-13-44-06_e2e_sulaoja_10_10",
            root_path / "2021-05-20-13-51-21_e2e_sulaoja_10_10",
            root_path / "2021-05-20-13-59-00_e2e_sulaoja_10_10",
            root_path / "2021-05-28-15-07-56_e2e_sulaoja_20_30",
            root_path / "2021-05-28-15-17-19_e2e_sulaoja_20_30",
            root_path / "2021-06-07-14-06-31_e2e_rec_ss6",
            root_path / "2021-06-07-14-09-18_e2e_rec_ss6",
            root_path / "2021-06-07-14-36-16_e2e_rec_ss6",
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
            root_path / "2021-06-14-11-43-48_e2e_rec_ss14_backwards"
        ]

        tr = transforms.Compose([NvidiaCropWide(), Normalize()])

        super().__init__(train_paths, tr, filter_turns=filter_turns)

class NvidiaAutumnTrainDataset(NvidiaDataset):
    def __init__(self, root_path, filter_turns=False):
        train_paths = [
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
            root_path / "2021-10-25-17-06-34_e2e_rec_ss2_arula_back",
            root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva",
            root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back",
            root_path / "2021-09-24-14-03-45_e2e_rec_ss11_backwards",
            root_path / "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back",
        ]

        tr = transforms.Compose([NvidiaCropWide(), Normalize()])

        super().__init__(train_paths, tr, filter_turns=filter_turns)
