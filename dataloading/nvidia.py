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


class ImageDataset(Dataset):
    def __init__(self, image_paths, steering_angles, transform=None):
        self.image_paths = image_paths
        self.steering_angles = steering_angles
        self.transform = transform

    def __len__(self):
        return len(self.steering_angles)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx])
        data = {
            'image': image,
            'steering_angle': np.array(self.steering_angles[idx])
        }

        if self.transform:
            data = self.transform(data)

        return data


class NvidiaDataset(ImageDataset):
    def __init__(self, dataset_paths, transform=None, camera="front_wide", steering_correction=0.0):
        data = defaultdict(list)

        for dataset_path in dataset_paths:
            camera_images, steering_angles, autonomous = self.read_dataset(dataset_path, camera)
            data["images"].extend(camera_images)
            data["steering_angle"].extend(steering_angles + steering_correction)
            data["autonomous"].extend(autonomous)

        super().__init__(data["images"], data["steering_angle"], transform)
        self.autonomous = data["autonomous"]

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx])
        data = {
            'image': image,
            'steering_angle': np.array(self.steering_angles[idx]),
            'autonomous': np.array(self.autonomous[idx])
        }

        if self.transform:
            data = self.transform(data)

        return data

    def read_dataset(self, dataset_path, camera):
        frames_df = pd.read_csv(dataset_path / "frames.csv")
        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['left_filename'].notna()]
        frames_df = frames_df[frames_df['right_filename'].notna()]
        frames_df = frames_df[frames_df['front_wide_filename'].notna()]
        steering_angles = frames_df["steering_angle"].to_numpy()
        autonomous = frames_df["autonomous"].to_numpy()

        camera_images = frames_df[f"{camera}_filename"].to_numpy()
        camera_images = [str(dataset_path / image_path) for image_path in camera_images]

        return camera_images, steering_angles, autonomous
