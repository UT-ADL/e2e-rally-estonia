import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


LEFT = 2
STRAIGHT = 1
RIGHT = 0
SKIP = -1

def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--dataset-folder',
        default="/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021",
        help='Root path to the dataset.'
    )

    return argparser.parse_args()


def preprocess_dataset(dataset_folder):
    root_path = Path(dataset_folder)
    dataset_paths = get_dataset_paths(root_path)
    create_waypoints(dataset_paths)
    fix_frames(root_path)


def get_dataset_paths(root_path):
    dataset_paths = [
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
        root_path / "2021-10-25-17-06-34_e2e_rec_ss2_arula_back",
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
        root_path / "2021-10-14-13-08-51_e2e_rec_vahi_backwards",
        #         root_path / "2022-01-28-10-21-14_e2e_rec_peipsiaare_forward",
        #         root_path / "2022-01-28-12-46-59_e2e_rec_peipsiaare_backward",
        #         root_path / "2022-01-14-10-05-16_e2e_rec_raanitsa_forward",
        #         root_path / "2022-01-14-10-50-05_e2e_rec_raanitsa_backward",
        #         root_path / "2022-01-14-11-54-33_e2e_rec_kambja_forward2",
        #         root_path / "2022-01-14-12-21-40_e2e_rec_kambja_forward2_continue",
        #         root_path / "2022-01-14-13-09-05_e2e_rec_kambja_backward",
        #         root_path / "2022-01-14-13-18-36_e2e_rec_kambja_backward_continue",
        #         root_path / "2022-01-14-12-35-13_e2e_rec_neeruti_forward",
        #         root_path / "2022-01-14-12-45-51_e2e_rec_neeruti_backward",
        #         root_path / "2022-01-18-13-03-03_e2e_rec_arula_backward",
        #         root_path / "2022-01-18-13-43-33_e2e_rec_otepaa_forward",
        #         root_path / "2022-01-18-13-52-35_e2e_rec_otepaa_forward",
        #         root_path / "2022-01-18-13-56-22_e2e_rec_otepaa_forward",
        #         root_path / "2022-01-18-14-12-14_e2e_rec_otepaa_backward",
        #         root_path / "2022-01-18-15-20-35_e2e_rec_kanepi_forward",
        #         root_path / "2022-01-18-15-49-26_e2e_rec_kanepi_backwards",
        #         root_path / "2022-01-18-12-37-01_e2e_rec_arula_forward",
        #         root_path / "2022-01-18-12-47-32_e2e_rec_arula_forward_continue",
        #         root_path / "2022-01-28-14-47-23_e2e_rec_elva_forward",
        #         root_path / "2022-01-28-15-09-01_e2e_rec_elva_backward",
        #         root_path / "2022-01-25-15-25-15_e2e_rec_vahi_forward",
        #         root_path / "2022-01-25-15-34-01_e2e_rec_vahi_backwards",
    ]
    return dataset_paths


def create_waypoints(dataset_paths):
    for dataset_path in dataset_paths:
        frames_df = pd.read_csv(dataset_path / "nvidia_frames.csv", index_col='index')

        # distance
        next_pos_df = frames_df.shift(-1)
        frames_df["distance"] = np.sqrt((next_pos_df.position_x - frames_df.position_x) ** 2 +
                                        (next_pos_df.position_y - frames_df.position_y) ** 2)

        N_WAYPOINTS = 10
        WAYPOINT_CAP = 5

        # initialize columns to NaN
        for wp_i in np.arange(1, N_WAYPOINTS + 1):
            frames_df[f"wp_x_offset_{wp_i}"] = np.nan
            frames_df[f"wp_y_offset_{wp_i}"] = np.nan
            frames_df[f"wp_steering_{wp_i}"] = np.nan

        progress_bar = tqdm(frames_df.iterrows(), total=frames_df.shape[0])
        for index, row in progress_bar:
            progress_bar.set_description(f"Processing {dataset_path.name}")

            window = frames_df[index:]
            window_cumsum = window['distance'].cumsum()

            vehicle_x = row["position_x"]
            vehicle_y = row["position_y"]
            yaw = row["yaw"]

            for wp_i in np.arange(1, N_WAYPOINTS + 1):
                window_index = window_cumsum.searchsorted(wp_i * WAYPOINT_CAP)
                next_wp = window.iloc[window_index]

                cumsum = window_cumsum[window_index]
                if math.isnan(cumsum) or math.ceil(cumsum) < wp_i * WAYPOINT_CAP:
                    break

                wp_global_x = next_wp["position_x"]
                wp_global_y = next_wp["position_y"]

                wp_local_x = (wp_global_x - vehicle_x) * np.cos(yaw) + (wp_global_y - vehicle_y) * np.sin(yaw)
                wp_local_y = -(wp_global_x - vehicle_x) * np.sin(yaw) + (wp_global_y - vehicle_y) * np.cos(yaw)

                frames_df.loc[index, f"wp_x_offset_{wp_i}"] = wp_local_x
                frames_df.loc[index, f"wp_y_offset_{wp_i}"] = wp_local_y

                frames_df.loc[index, f"wp_steering_{wp_i}"] = next_wp["steering_angle"]

        frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)


def fix_frames(root_path):

    dataset_path = root_path / "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[5841:5937].index, 'turn_signal'] = STRAIGHT
    frames_df.loc[frames_df.iloc[7876:8444].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[10865:11160].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[19455:19670].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[25105:25335].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[17663:17880].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[17881:18303].index, 'turn_signal'] = STRAIGHT
    frames_df.loc[frames_df.iloc[31705:31950].index, 'turn_signal'] = STRAIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-07-14-36-16_e2e_rec_ss6"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[7395:7530].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[18900:19100].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-07-14-20-07_e2e_rec_ss6"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[22677:22824].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[9120:9280].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[14580:14843].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-09-14-58-11_e2e_rec_ss3"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[8635:8880].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[3775:4475].index, 'turn_signal'] = SKIP
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-09-16-24-59_e2e_rec_ss13"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[6802:7040].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[12459:12715].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[17275:17518].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[25335:25525].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[14650:15190].index, 'turn_signal'] = SKIP
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-10-13-19-22_e2e_ss4_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[2395:2645].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[6923:7190].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[14740:14970].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-09-15-42-05_e2e_rec_ss3_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[13618:13855].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[32989:33220].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[34566:34760].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[36935:37210].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[38989:39175].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-20-14-15-07_e2e_rec_neeruti_ss19_22_back"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[4800:5090].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[13885:14210].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-09-16-50-22_e2e_rec_ss13_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[13327:13522].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[18901:19215].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[19215:19431].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[21236:21554].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[21555:21675].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[21675:21787].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[24410:24649].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-14-11-22-05_e2e_rec_ss14"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[4230:4450].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[16939:17113].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-10-12-59-59_e2e_ss4"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[16159:16457].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[18257:18748].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-14-11-43-48_e2e_rec_ss14_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[25210:25434].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-10-14-44-24_e2e_ss3_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[7891:8000].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[29011:29216].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-11-14-50-59_e2e_rec_vahi"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[11365:11610].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-09-30-15-56-59_e2e_ss14_attempt_2"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[7785:8015].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[29232:29477].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[47877:47940].index, 'turn_signal'] = STRAIGHT
    frames_df.loc[frames_df.iloc[47940:48052].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-09-13-55-03_e2e_rec_ss2_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[19262:19484].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-07-13-22-35_e2e_rec_ss4_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[8471:8690].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-09-24-13-39-38_e2e_rec_ss11"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[7338:7564].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[8657:9043].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-06-10-15-03-16_e2e_ss3_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[1311:1470].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-07-11-05-13_e2e_rec_ss3"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[9820:10102].index, 'turn_signal'] = LEFT
    frames_df.loc[frames_df.iloc[35804:36017].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-25-17-06-34_e2e_rec_ss2_arula_back"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[18591:18870].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-09-30-15-03-37_e2e_ss14_from_half_way"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[8160:8534].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-09-24-12-02-32_e2e_rec_ss10_3"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[4793:4945].index, 'turn_signal'] = RIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-14-13-08-51_e2e_rec_vahi_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[2893:3105].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-10-20-14-55-47_e2e_rec_vastse_ss13_17"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[22109:22425].index, 'turn_signal'] = RIGHT
    frames_df.loc[frames_df.iloc[22426:22596].index, 'turn_signal'] = STRAIGHT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)

    dataset_path = root_path / "2021-09-24-14-03-45_e2e_rec_ss11_backwards"
    frames_df = pd.read_csv(dataset_path / "nvidia_frames_ext.csv", index_col='index')
    frames_df.loc[frames_df.iloc[1765:2153].index, 'turn_signal'] = LEFT
    frames_df.to_csv(dataset_path / "nvidia_frames_ext.csv", header=True)
    # Dirty/wet camera


if __name__ == "__main__":
    args = parse_arguments()
    preprocess_dataset(args.dataset_folder)
