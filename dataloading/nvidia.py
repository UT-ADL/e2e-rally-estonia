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


class Normalize(object):
    def __call__(self, data, transform=None):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = data["image"]
        image = image / 255
        data["image"] = normalize(image)
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
    def __init__(self, dataset_paths, transform=None):
        data = defaultdict(list)

        for dataset_path in dataset_paths:
            image_paths, steering_angles = self.read_dataset(dataset_path)
            data["images"].extend(image_paths)
            data["steering_angle"].extend(steering_angles)

        super().__init__(data["images"], data["steering_angle"], transform)

    def read_dataset(self, dataset_path):
        frames_df = pd.read_csv(dataset_path / "frames.csv")
        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        steering_angles = frames_df["steering_angle"].to_numpy()
        image_paths = frames_df["front_narrow_filename"].to_numpy()
        image_paths = [str(dataset_path / image_path) for image_path in image_paths]
        return image_paths, steering_angles
