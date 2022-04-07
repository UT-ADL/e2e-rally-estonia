import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

STEERING_ANGLE_RATIO = 14.7

def calculate_closed_loop_metrics(model_frames, expert_frames, fps=30, failure_rate_threshold=1.0):

    lat_errors = calculate_lateral_errors(model_frames, expert_frames, True)

    autonomous_frames = model_frames[model_frames.autonomous].reset_index(drop=True)
    model_steering = autonomous_frames.steering_angle.to_numpy() / np.pi * 180
    cmd_model_steering = autonomous_frames.cmd_steering_angle.to_numpy() / np.pi * 180
    cmd_model_steering = cmd_model_steering * STEERING_ANGLE_RATIO
    true_steering = expert_frames.steering_angle.to_numpy() / np.pi * 180
    whiteness = calculate_whiteness(model_steering, fps)
    cmd_whiteness = calculate_whiteness(cmd_model_steering, fps)
    expert_whiteness = calculate_whiteness(true_steering, fps)

    max = lat_errors.max()
    mae = lat_errors.mean()
    rmse = np.sqrt((lat_errors ** 2).mean())
    failure_rate = len(lat_errors[lat_errors > failure_rate_threshold]) / float(len(lat_errors)) * 100
    distance = calculate_distance(model_frames)
    interventions = calculate_interventions(model_frames)

    return {
        'mae': mae,
        'rmse': rmse,
        'max': max,
        'failure_rate': failure_rate,
        'distance': distance,
        'distance_per_intervention': distance / interventions,
        'interventions': interventions,
        'whiteness': whiteness,
        'cmd_whiteness': cmd_whiteness,
        'expert_whiteness': expert_whiteness,
    }


def calculate_open_loop_metrics(predicted_steering, true_steering, fps):
    predicted_degrees = predicted_steering / np.pi * 180
    true_degrees = true_steering / np.pi * 180
    errors = np.abs(true_degrees - predicted_degrees)
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    max = errors.max()

    whiteness = calculate_whiteness(predicted_degrees, fps)
    expert_whiteness = calculate_whiteness(true_degrees, fps)

    return {
        'mae': mae,
        'rmse': rmse,
        'max': max,
        'whiteness': whiteness,
        'expert_whiteness': expert_whiteness
    }


def calculate_trajectory_open_loop_metrics(predicted_waypoints, true_waypoints, fps):
    first_wp_error = np.hypot(predicted_waypoints[:, 0] - true_waypoints[:, 0],
                              predicted_waypoints[:, 1] - true_waypoints[:, 1])

    first_wp_whiteness = calculate_whiteness(predicted_waypoints[:, 1], fps=fps)
    first_wp_expert_whiteness = calculate_whiteness(true_waypoints[:, 1], fps=fps)

    sixth_wp_error = np.hypot(predicted_waypoints[:, 10] - true_waypoints[:, 10],
                              predicted_waypoints[:, 11] - true_waypoints[:, 11])

    sixth_wp_whiteness = calculate_whiteness(predicted_waypoints[:, 11], fps=fps)
    sixth_wp_expert_whiteness = calculate_whiteness(true_waypoints[:, 11], fps=fps)

    # number of predicted waypoints can be different, just take equal number of ground truth waypoints
    true_waypoints = true_waypoints[:, 0:predicted_waypoints.shape[1]]
    last_wp_error = np.hypot(predicted_waypoints[:, -2] - true_waypoints[:, -2],
                             predicted_waypoints[:, -1] - true_waypoints[:, -1])
    last_wp_whiteness = calculate_whiteness(predicted_waypoints[:, -1], fps=fps)
    last_wp_expert_whiteness = calculate_whiteness(true_waypoints[:, -1], fps=fps)

    #zipped_waypoints = tqdm(zip(predicted_waypoints, true_waypoints), total=len(true_waypoints))
    #zipped_waypoints.set_description("Calculating frechet distances")
    #zipped_waypoints = zip(predicted_waypoints, true_waypoints)
    #frechet_distances = np.array(
    #    [frdist(z[0].reshape(-1, 2), z[1].reshape(-1, 2)) for z in zipped_waypoints])

    return {
        'first_wp_mae': first_wp_error.mean(),
        'first_wp_rmse': np.sqrt((first_wp_error ** 2).mean()),
        'first_wp_max': first_wp_error.max(),
        'first_wp_whiteness': first_wp_whiteness,
        'first_wp_expert_whiteness': first_wp_expert_whiteness,
        'sixth_wp_mae': sixth_wp_error.mean(),
        'sixth_wp_whiteness': sixth_wp_whiteness,
        'sixth_wp_expert_whiteness': sixth_wp_expert_whiteness,
        'last_wp_mae': last_wp_error.mean(),
        'last_wp_rmse': np.sqrt((last_wp_error ** 2).mean()),
        'last_wp_max': last_wp_error.max(),
        'last_wp_whiteness': last_wp_whiteness,
        'last_wp_expert_whiteness': last_wp_expert_whiteness,
        #'frechet_distance': frechet_distances.mean()
    }


def calculate_whiteness(steering_angles, fps=30):
    current_angles = steering_angles[:-1]
    next_angles = steering_angles[1:]
    delta_angles = next_angles - current_angles
    whiteness = np.sqrt(((delta_angles * fps) ** 2).mean())
    return whiteness


def calculate_lateral_errors(model_frames, expert_frames, only_autonomous=True):
    model_trajectory_df = model_frames[["position_x", "position_y", "autonomous"]].rename(
        columns={"position_x": "X", "position_y": "Y"}).dropna()
    expert_trajectory_df = expert_frames[["position_x", "position_y", "autonomous"]].rename(
        columns={"position_x": "X", "position_y": "Y"}).dropna()

    if only_autonomous:
        model_trajectory_df = model_trajectory_df[model_trajectory_df.autonomous].reset_index(drop=True)

    tree = BallTree(expert_trajectory_df.values)
    inds, dists = tree.query_radius(model_trajectory_df.values, r=2, sort_results=True, return_distance=True)
    closest_l = []
    for i, ind in enumerate(inds):
        if len(ind) >= 2:
            closest = pd.DataFrame({
                'X1': [expert_trajectory_df.iloc[ind[0]].X],
                'Y1': [expert_trajectory_df.iloc[ind[0]].Y],
                'X2': [expert_trajectory_df.iloc[ind[1]].X],
                'Y2': [expert_trajectory_df.iloc[ind[1]].Y]},
                index=[i])
            closest_l.append(closest)
    closest_df = pd.concat(closest_l)
    f = model_trajectory_df.join(closest_df)
    lat_errors = abs((f.X2 - f.X1) * (f.Y1 - f.Y) - (f.X1 - f.X) * (f.Y2 - f.Y1)) / np.sqrt(
        (f.X2 - f.X1) ** 2 + (f.Y2 - f.Y1) ** 2)
    # lat_errors.dropna(inplace=True)  # Why na-s?

    return lat_errors


def calculate_interventions(frames):
    frames['autonomous_next'] = frames.shift(-1)['autonomous']
    return len(frames[frames.autonomous & (frames.autonomous_next == False)])


def calculate_distance(frames):
    x1 = frames['position_x']
    y1 = frames['position_y']
    x2 = frames.shift(-1)['position_x']
    y2 = frames.shift(-1)['position_y']
    frames['distance'] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return np.sum(frames[frames["autonomous"]]["distance"])


# Duplicated with read_frames_driving, should be removed
# when expert frames are re-extracted and have cmd_steering_angle column
def read_frames_expert(dataset_paths, filename):
    datasets = [pd.read_csv(dataset_path / filename) for dataset_path in dataset_paths]
    frames_df = pd.concat(datasets)
    frames_df = frames_df[['steering_angle', 'position_x', 'position_y', 'autonomous']].dropna()
    return frames_df


def read_frames_driving(dataset_paths, filename):
    datasets = [pd.read_csv(dataset_path / filename) for dataset_path in dataset_paths]
    frames_df = pd.concat(datasets)
    frames_df = frames_df[['steering_angle', 'cmd_steering_angle', 'position_x', 'position_y', 'autonomous']].dropna()

    return frames_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-path",
                        default="/gpfs/space/projects/Bolt/dataset",
                        help='Path to extracted datasets')
    parser.add_argument('--expert-datasets',
                        nargs='+',
                        default=['2021-10-26-10-49-06_e2e_rec_ss20_elva',
                                 '2021-10-26-11-08-59_e2e_rec_ss20_elva_back'],
                        help='Datasets used for ground truth tracjectories.')
    parser.add_argument('--drive-datasets',
                        nargs='+',
                        required=True,
                        default=[],
                        help='Datasets used to calculate metrics for.')
    parser.add_argument('--input-modality',
                        choices=['nvidia-camera', 'ouster-lidar'],
                        default='nvidia-camera',
                        help='Input modality used for driving')

    args = parser.parse_args()

    if args.input_modality == "nvidia-camera":
        frames_filename = "nvidia_frames.csv"
        fps = 30
    elif args.input_modality == "ouster-lidar":
        frames_filename = "lidar_frames.csv"
        fps = 10
    else:
        print("Uknown input modality")
        sys.exit()

    root_path = Path(args.root_path)

    expert_ds = [root_path / dataset_path for dataset_path in args.expert_datasets]
    expert_frames = read_frames_expert(expert_ds, frames_filename)

    drive_ds = [root_path / dataset_path for dataset_path in args.drive_datasets]
    model_frames = read_frames_driving(drive_ds, frames_filename)

    print(calculate_closed_loop_metrics(model_frames, expert_frames, fps=fps))
