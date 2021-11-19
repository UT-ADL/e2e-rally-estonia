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


def calculate_whiteness(steering_angles, fps=30):
    current_angles = steering_angles[:-1]
    next_angles = steering_angles[1:]
    delta_angles = next_angles - current_angles
    whiteness = np.sqrt(((delta_angles * fps) ** 2).mean())
    return whiteness


def calculate_lateral_errors(model_frames, expert_frames, only_autonomous=True):
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

def read_frames(dataset_paths, modality):
    datasets = [pd.read_csv(dataset_path / f"{modality}_frames.csv") for dataset_path in dataset_paths]
    return pd.concat(datasets)


if __name__ == "__main__":

    root_path = Path("/media/romet/data2/datasets/rally-estonia/dataset")
    expert_ds = [root_path / '2021-10-26-10-49-06_e2e_rec_ss20_elva',
                 root_path / '2021-10-26-11-08-59_e2e_rec_ss20_elva_back']
    expert_frames = read_frames(expert_ds, "nvidia")

    model_ds = [root_path / '2021-11-03-12-53-38_e2e_rec_elva_back_autumn-v3',
                root_path / '2021-11-03-12-35-19_e2e_rec_elva_autumn-v3']
    model_frames = read_frames(model_ds, "nvidia")

    print(calculate_closed_loop_metrics(model_frames, expert_frames))
