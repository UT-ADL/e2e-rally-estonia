from pathlib import Path

from matplotlib import pyplot as plt

from metrics import calculate_lateral_errors, read_frames


def draw_error_plot(ax, model_frames, expert_frames, title=None):
    lat_errors = calculate_lateral_errors(model_frames, expert_frames, only_autonomous=True)

    autonomous_df = model_frames[model_frames.autonomous].reset_index(drop=True)
    ax.scatter(autonomous_df["position_x"], autonomous_df["position_y"],
               s=5,
               c=lat_errors, cmap=plt.cm.coolwarm)

    interventions_df = model_frames[model_frames.autonomous == False].reset_index(drop=True)
    ax.scatter(interventions_df["position_x"], interventions_df["position_y"],
               s=5,
               c="#2BFA00")

    if title:
        ax.set_title(title)


if __name__ == "__main__":
    root_path = Path("/gpfs/space/projects/Bolt/dataset")

    datasets_backwards = {
        'Lidar in train night': read_frames([root_path / '2021-11-25-17-08-28_e2e_rec_elva-lidar-inTrain-0.8-backNight'], "lidar_frames.csv"),
        'Lidar night': read_frames([root_path / '2021-11-25-17-43-47_e2e_rec_elva-lidar-0.8-backNight'], "lidar_frames.csv"),
        'Lidar night #2': read_frames([root_path / '2021-11-25-18-07-28_e2e_rec_elva-lidar-0.8-backNight_attempt2'], "lidar_frames.csv"),
    }

    datasets = {
        'Lidar in train night': read_frames([root_path / '2021-11-25-16-57-26_e2e_rec_elva-lidar-inTrain-0.8-forwardNight'], "lidar_frames.csv"),
        'Lidar train night': read_frames([root_path / '2021-11-25-17-20-55_e2e_rec_elva-lidar-0.8-forwardNight', root_path / "2021-11-25-17-31-42_e2e_rec_elva-lidar-0.8-forwardNight"], "lidar_frames.csv"),
        'Lidar train night #2': read_frames([root_path / '2021-11-25-17-56-16_e2e_rec_elva-lidar-0.8-forwardNight_attempt2'], "lidar_frames.csv"),
    }
    expert_frames = read_frames([root_path / '2021-10-26-10-49-06_e2e_rec_ss20_elva'], "lidar_frames.csv")
    expert_frames_back = read_frames([root_path / '2021-10-26-11-08-59_e2e_rec_ss20_elva_back'], "lidar_frames.csv")

    fig, ax = plt.subplots(len(datasets), 2, figsize=(15, 30))

    for i, (name, model_frames) in enumerate(datasets.items()):
        print(i, name, "forward")
        draw_error_plot(ax[i][0], model_frames, expert_frames, f"{name} elva forward")

    for i, (name, model_frames) in enumerate(datasets_backwards.items()):
        print(i, name, "backward")
        draw_error_plot(ax[i][1], model_frames, expert_frames_back, f"{name} Elva backward")

    fig.savefig("elva-2021-11-25-lidar-night.png", facecolor="white")
