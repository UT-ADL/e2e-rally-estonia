from pathlib import Path

from matplotlib import pyplot as plt

from metrics import calculate_lateral_errors, read_frames_driving


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
        'Lidar v1': read_frames_driving([root_path / '2021-11-25-12-45-35_e2e_rec_elva-lidar-v1-0.8-back'], "lidar_frames.csv"),
        'Lidar v2': read_frames_driving([root_path / '2021-11-25-14-27-56_e2e_rec_elva-lidar-v2-0.8-back'], "lidar_frames.csv"),
        'Lidar v3': read_frames_driving([root_path / '2021-11-25-15-16-31_e2e_rec_elva-l-lidar-v3-0.8-back'], "lidar_frames.csv"),
        'Lidar in train': read_frames_driving([root_path / '2021-11-25-13-37-42_e2e_rec_elva-lilidar-inTrain-0.8-back'], "lidar_frames.csv"),

        'Lidar in train night': read_frames_driving([root_path / '2021-11-25-17-08-28_e2e_rec_elva-lidar-inTrain-0.8-backNight'], "lidar_frames.csv"),
        'Lidar night': read_frames_driving([root_path / '2021-11-25-17-43-47_e2e_rec_elva-lidar-0.8-backNight'], "lidar_frames.csv"),
        'Lidar night #2': read_frames_driving([root_path / '2021-11-25-18-07-28_e2e_rec_elva-lidar-0.8-backNight_attempt2'], "lidar_frames.csv"),

        'Lidar all channels': read_frames_driving([root_path / '2021-11-26-11-30-23_e2e_rec_elva_i_allChannels_back_0.8'], "lidar_frames.csv"),
        'Lidar intensity': read_frames_driving([root_path / '2021-11-26-11-07-10_e2e_rec_elva_intensity_back_0.8'], "lidar_frames.csv"),
    }

    datasets = {
        'Lidar v1': read_frames_driving([root_path / '2021-11-25-12-57-24_e2e_rec_elva-lidar-v1-0.8-forward'], "lidar_frames.csv"),
        'Lidar v2': read_frames_driving([root_path / '2021-11-25-14-39-43_e2e_rec_elva-lidar-v2-0.8-forward'], "lidar_frames.csv"),
        'Lidar v3': read_frames_driving([root_path / '2021-11-25-15-27-38_e2e_rec_elva-l-lidar-v3-0.8-forward'], "lidar_frames.csv"),
        'Lidar in train': read_frames_driving([root_path / '2021-11-25-13-48-44_e2e_rec_elva-lilidar-inTrain-0.8-forward'], "lidar_frames.csv"),

        'Lidar in train night': read_frames_driving([root_path / '2021-11-25-16-57-26_e2e_rec_elva-lidar-inTrain-0.8-forwardNight'], "lidar_frames.csv"),
        'Lidar train night': read_frames_driving([root_path / '2021-11-25-17-20-55_e2e_rec_elva-lidar-0.8-forwardNight', root_path / "2021-11-25-17-31-42_e2e_rec_elva-lidar-0.8-forwardNight"], "lidar_frames.csv"),
        'Lidar train night #2': read_frames_driving([root_path / '2021-11-25-17-56-16_e2e_rec_elva-lidar-0.8-forwardNight_attempt2'], "lidar_frames.csv"),

        'Lidar all channels': read_frames_driving([root_path / '2021-11-26-11-19-15_e2e_rec_elva_i_allChannels_forward_0.8'], "lidar_frames.csv"),
        'Lidar intensity': read_frames_driving([root_path / '2021-11-26-10-53-35_e2e_rec_elva_intensity_forward_0.8'], "lidar_frames.csv"),
        'Lidar range': read_frames_driving([root_path / '2021-11-26-11-42-02_e2e_rec_elva_i_range_forward_0.8'],
                                   "lidar_frames.csv"),
        'Lidar ambience': read_frames_driving([root_path / '2021-11-26-11-53-18_e2e_rec_elva_i_ambience_forward_0.8'],
                                      "lidar_frames.csv"),
    }

    expert_frames = read_frames_driving([root_path / '2021-10-26-10-49-06_e2e_rec_ss20_elva'], "lidar_frames.csv")
    expert_frames_back = read_frames_driving([root_path / '2021-10-26-11-08-59_e2e_rec_ss20_elva_back'], "lidar_frames.csv")

    fig, ax = plt.subplots(len(datasets), 2, figsize=(15, 30))

    for i, (name, model_frames) in enumerate(datasets.items()):
        print(i, name, "forward")
        draw_error_plot(ax[i][0], model_frames, expert_frames, f"{name} elva forward")

    for i, (name, model_frames) in enumerate(datasets_backwards.items()):
        print(i, name, "backward")
        draw_error_plot(ax[i][1], model_frames, expert_frames_back, f"{name} Elva backward")

    fig.savefig("elva-2021-11-25-lidar.png", facecolor="white")
