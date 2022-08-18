import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pytransform3d.urdf import UrdfTransformManager
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt

from dataloading.model import Camera

SKIP = -1


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--dataset-folder',
        default="/home/romet/data/datasets/rally-estonia/dataset-demo-small2",
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
        create_waypoints([root_path / dataset_name])
    else:
        dataset_paths = get_dataset_paths(root_path)
        create_waypoints(dataset_paths)


def get_dataset_paths(root_path):
    dataset_paths = [
        root_path / '2022-06-17-13-21-37_e2e_elva__steering',
        root_path / '2022-06-17-13-21-51_e2e_elva__steering',
        root_path / '2022-06-17-13-42-44_e2e_elva__trajectory',
        root_path / '2022-06-17-14-00-57_e2e_elva__trajectory_turn',
        root_path / '2022-06-17-14-06-10_e2e_elva__trajectory_bal',
        root_path / '2022-06-17-14-26-28_e2e_elva__steering_wide',
        root_path / '2022-06-17-14-41-00_e2e_elva__steering_wide',
        root_path / '2022-06-17-14-45-02_e2e_elva__steering_wide',
        root_path / '2022-06-17-14-51-25_e2e_elva__steering_wide_bal'
    ]
    return dataset_paths


def create_waypoints(dataset_paths):
    for dataset_path in dataset_paths:
        frames_df = pd.read_csv(dataset_path / "nvidia_frames.csv", index_col='index')
        frames_df = frames_df[frames_df[f"position_x"].notna()]

        # distance
        next_pos_df = frames_df.shift(-1)
        frames_df["distance"] = np.sqrt((next_pos_df.position_x - frames_df.position_x) ** 2 +
                                        (next_pos_df.position_y - frames_df.position_y) ** 2)

        N_WAYPOINTS = 10
        WAYPOINT_CAP = 5

        # initialize columns to NaN
        for wp_i in np.arange(1, N_WAYPOINTS + 1):
            frames_df[f"wp_steering_{wp_i}"] = np.nan

            frames_df[f"wp{wp_i}_x"] = np.nan
            frames_df[f"wp{wp_i}_y"] = np.nan
            frames_df[f"wp{wp_i}_z"] = np.nan

            frames_df[f"wp{wp_i}_{Camera.FRONT_WIDE.value}_x"] = np.nan
            frames_df[f"wp{wp_i}_{Camera.FRONT_WIDE.value}_y"] = np.nan
            frames_df[f"wp{wp_i}_{Camera.FRONT_WIDE.value}_z"] = np.nan

            frames_df[f"wp{wp_i}_{Camera.LEFT.value}_x"] = np.nan
            frames_df[f"wp{wp_i}_{Camera.LEFT.value}_y"] = np.nan
            frames_df[f"wp{wp_i}_{Camera.LEFT.value}_z"] = np.nan

            frames_df[f"wp{wp_i}_{Camera.RIGHT.value}_x"] = np.nan
            frames_df[f"wp{wp_i}_{Camera.RIGHT.value}_y"] = np.nan
            frames_df[f"wp{wp_i}_{Camera.RIGHT.value}_z"] = np.nan

        tm = get_transform_manager()

        progress_bar = tqdm(frames_df.iterrows(), total=frames_df.shape[0])
        for index, row in progress_bar:
            progress_bar.set_description(f"Processing {dataset_path.name}")

            window = frames_df[index:]
            window_cumsum = window['distance'].cumsum()

            base_transform = calculate_waypoint_transform(row)

            for wp_i in np.arange(1, N_WAYPOINTS + 1):  # TODO: vectorize
                window_index = window_cumsum.searchsorted(wp_i * WAYPOINT_CAP)
                next_wp = window.iloc[window_index]

                cumsum = window_cumsum[window_index]
                if math.isnan(cumsum) or math.ceil(cumsum) < wp_i * WAYPOINT_CAP:
                    break

                frames_df.loc[index, f"wp_steering_{wp_i}"] = next_wp["steering_angle"]

                wp_global = np.array([next_wp["position_x"], next_wp["position_y"], next_wp["position_z"], 1])
                wp_local = pt.transform(pt.invert_transform(base_transform), wp_global)
                frames_df.loc[index, f"wp{wp_i}_x"] = wp_local[0]
                frames_df.loc[index, f"wp{wp_i}_y"] = wp_local[1]
                frames_df.loc[index, f"wp{wp_i}_z"] = wp_local[2]

                center_cam_transform = tm.get_transform("base_link", "interfacea_link2")
                wp_center_cam = pt.transform(center_cam_transform, wp_local)
                # Camera frames are rotated compared to base_link frame (x = z, y = -x, z = -y)
                frames_df.loc[index, f"wp{wp_i}_{Camera.FRONT_WIDE.value}_x"] = wp_center_cam[2]
                frames_df.loc[index, f"wp{wp_i}_{Camera.FRONT_WIDE.value}_y"] = -wp_center_cam[0]
                frames_df.loc[index, f"wp{wp_i}_{Camera.FRONT_WIDE.value}_z"] = -wp_center_cam[1]

                left_cam_transform = tm.get_transform("base_link", "interfacea_link0")
                wp_left_cam = pt.transform(left_cam_transform, wp_local)
                frames_df.loc[index, f"wp{wp_i}_{Camera.LEFT.value}_x"] = wp_left_cam[2]
                frames_df.loc[index, f"wp{wp_i}_{Camera.LEFT.value}_y"] = -wp_left_cam[0]
                frames_df.loc[index, f"wp{wp_i}_{Camera.LEFT.value}_z"] = -wp_left_cam[1]

                right_cam_transform = tm.get_transform("base_link", "interfacea_link1")
                wp_right_cam = pt.transform(right_cam_transform, wp_local)
                frames_df.loc[index, f"wp{wp_i}_{Camera.RIGHT.value}_x"] = wp_right_cam[2]
                frames_df.loc[index, f"wp{wp_i}_{Camera.RIGHT.value}_y"] = -wp_right_cam[0]
                frames_df.loc[index, f"wp{wp_i}_{Camera.RIGHT.value}_z"] = -wp_right_cam[1]

        frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)


def calculate_waypoint_transform(row):
    rot_mat = pr.active_matrix_from_intrinsic_euler_xyz(np.array([row["roll"], row["pitch"], row["yaw"]]))
    translate_mat = np.array([row["position_x"], row["position_y"], row["position_z"]])
    wp_trans = pt.transform_from(rot_mat, translate_mat)
    return wp_trans


def get_transform_manager():
    tm = UrdfTransformManager()

    filename = "dataloading/platform.urdf"
    with open(filename, "r") as f:
        tm.load_urdf(f.read())

    return tm


if __name__ == "__main__":
    args = parse_arguments()
    preprocess_dataset(args.dataset_folder, args.dataset_name)
