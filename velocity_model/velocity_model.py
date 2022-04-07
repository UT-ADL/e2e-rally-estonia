import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from dataloading.nvidia import NvidiaDataset


class VelocityModel:
    def __init__(self, positions_parquet='positions.parquet', vector_velocity=30):
        self.vector_velocity = vector_velocity
        self.positions_df = pd.read_parquet(positions_parquet)
        self.tree = BallTree(self.positions_df[["position_x", "position_y", "position_x2", "position_y2"]])

    def find_speed_for_position(self, x, y, yaw):

        x2 = x + (self.vector_velocity * np.cos(yaw))
        y2 = y + (self.vector_velocity * np.sin(yaw))

        closest = self.tree.query([[x, y, x2, y2]])
        distance = closest[0][0][0]
        index = closest[1][0][0]
        return self.positions_df.iloc[index]["vehicle_speed"], distance


def create_velocity_model(output_filename, dataset_path):
    print(f"Creating velocity model from {dataset_path}")
    root_path = Path(dataset_path)

    paths = [
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
        root_path / "2021-09-24-14-03-45_e2e_rec_ss11_backwards",
        root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva",
        root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back",
        root_path / "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back",
        root_path / "2021-10-11-14-50-59_e2e_rec_vahi",
        root_path / "2021-10-14-13-08-51_e2e_rec_vahi_backwards",
    ]

    dataset = NvidiaDataset(paths, camera="front_wide", transform=None)
    positions_df = dataset.frames

    velocity = 30
    positions_df["position_x2"] = positions_df["position_x"] + (velocity * np.cos(positions_df["yaw"]))
    positions_df["position_y2"] = positions_df["position_y"] + (velocity * np.sin(positions_df["yaw"]))
    positions_df = positions_df[
        ["position_x", "position_y", "position_z", "position_x2", "position_y2", "yaw", "roll", "pitch",
         "vehicle_speed", "steering_angle", "image_path", "autonomous"]]

    positions_df = positions_df[positions_df['position_x'].notna()]
    print(f"Saving velocity model to {output_filename}")
    positions_df.to_parquet(output_filename, compression='GZIP')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--output-filename',
        required=True,
        help='Filename of the created velocity model.'
    )

    argparser.add_argument(
        '--dataset-path',
        required=False,
        default="/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021",
        help='Dataset used to create velocity model.'
    )

    args = argparser.parse_args()
    create_velocity_model(args.output_filename, args.dataset_path)
