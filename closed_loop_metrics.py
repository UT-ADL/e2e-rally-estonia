import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def calculate_closed_loop_metrics(model_frames, expert_frames, fps=30, failure_rate_threshold=1.0,
                                  only_autonomous=True):
    model_steering = model_frames.steering_angle.to_numpy() / np.pi * 180
    true_steering = expert_frames.steering_angle.to_numpy() / np.pi * 180

    lat_errors = calculate_lateral_errors(model_frames, expert_frames, only_autonomous)
    whiteness = calculate_whiteness(model_steering, fps)
    expert_whiteness = calculate_whiteness(true_steering, fps)

    max = lat_errors.max()
    mae = lat_errors.mean()
    rmse = np.sqrt((lat_errors ** 2).mean())
    failure_rate = len(lat_errors[lat_errors > failure_rate_threshold]) / float(len(lat_errors)) * 100
    interventions = calculate_interventions(model_frames)

    return {
        'mae': mae,
        'rmse': rmse,
        'max': max,
        'failure_rate': failure_rate,
        'interventions': interventions,
        'whiteness': whiteness,
        'expert_whiteness': expert_whiteness,
    }

def calculate_open_loop_metrics(predicted_steering, true_steering, fps):
    predicted_degrees = np.array(predicted_steering) / np.pi * 180
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


def calculate_whiteness(steering_angles, fps=30):
    current_angles = steering_angles[:-1]
    next_angles = steering_angles[1:]
    delta_angles = next_angles - current_angles
    whiteness = np.sqrt(((delta_angles * fps) ** 2).mean())
    return whiteness


def calculate_lateral_errors(model_frames, expert_frames, only_autonomous=True):
    print(model_frames)
    print(expert_frames)
    model_trajectory_df = model_frames[["position_x", "position_y", "autonomous"]].rename(
        columns={"position_x": "X", "position_y": "Y"})
    expert_trajectory_df = expert_frames[["position_x", "position_y", "autonomous"]].rename(
        columns={"position_x": "X", "position_y": "Y"})

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


def read_frames(dataset_paths, modality="nvidia"):
    datasets = [pd.read_csv(dataset_path / f"{modality}_frames.csv") for dataset_path in dataset_paths]
    return pd.concat(datasets)


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
                        default='nvidia',
                        help='Input modality used for driving')

    args = parser.parse_args()

    root_path = Path(args.root_path)

    expert_ds = [root_path / dataset_path for dataset_path in args.expert_datasets]
    expert_frames = read_frames(expert_ds, args.input_modality)

    drive_ds = [root_path / dataset_path for dataset_path in args.drive_datasets]
    model_frames = read_frames(drive_ds, args.input_modality)

    print(calculate_closed_loop_metrics(model_frames, expert_frames))
