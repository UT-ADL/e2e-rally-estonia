import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import trajectory
from dataloading.camera import Camera
from dataloading.nvidia import NvidiaValidationDataset, NvidiaTrainDataset


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--dataset-folder',
        default="/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021",
        help='Root path to the dataset.'
    )

    argparser.add_argument(
        '--dataset-name',
        required=False,
        help='Dataset name to preprocess. If not provided all datasets in given folder are preprocessed.'
    )

    return argparser.parse_args()


def preprocess_dataset(dataset_folder, dataset_name):
    root_path = Path(dataset_folder)
    if dataset_name:
        preprocess_steering_angle([root_path / dataset_name])
    else:
        dataset_paths = get_dataset_paths(root_path)
        preprocess_steering_angle(dataset_paths)


def preprocess_steering_angle(dataset_paths):
    progress = tqdm(dataset_paths)
    for dataset_path in progress:
        progress.set_description(dataset_path.name)
        frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
        create_steering_angles(frames_df, Camera.LEFT.value)
        create_steering_angles(frames_df, Camera.RIGHT.value)
        frames_df.to_csv(dataset_path / "nvidia_frames_ext2.csv", header=True)


def create_steering_angles(frames_df, camera):
    waypoints = get_waypoints(frames_df, camera)
    steering_angles = []
    for i in range(waypoints.shape[0]):
        calculated_steering_angle = trajectory.calculate_steering_angle(waypoints[i])
        steering_angles.append(calculated_steering_angle)
        #print(trajectory_waypoints, calculated_steering_angle)
    frames_df[f"steering_angle_{camera}"] = steering_angles


def get_waypoints(frames_df, camera_name):
    wp_x_cols = [f"wp{i}_{camera_name}_x" for i in np.arange(1, 11)]
    wp_y_cols = [f"wp{i}_{camera_name}_y" for i in np.arange(1, 11)]
    waypoint_cols = np.column_stack((wp_x_cols, wp_y_cols)).reshape(-1)
    return frames_df[waypoint_cols].to_numpy()


def get_dataset_paths(root_path):
    return NvidiaTrainDataset(root_path).dataset_paths + NvidiaValidationDataset(root_path).dataset_paths


if __name__ == "__main__":
    args = parse_arguments()
    preprocess_dataset(args.dataset_folder, args.dataset_name)