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
        'Lidar all channels': read_frames([root_path / '2021-11-26-11-30-23_e2e_rec_elva_i_allChannels_back_0.8'], "lidar_frames.csv"),
        'Lidar intensity': read_frames([root_path / '2021-11-26-11-07-10_e2e_rec_elva_intensity_back_0.8'], "lidar_frames.csv"),
    }

    datasets = {
        'Lidar all channels': read_frames([root_path / '2021-11-26-11-19-15_e2e_rec_elva_i_allChannels_forward_0.8'], "lidar_frames.csv"),
        'Lidar intensity': read_frames([root_path / '2021-11-26-10-53-35_e2e_rec_elva_intensity_forward_0.8'], "lidar_frames.csv"),
        'Lidar range': read_frames([root_path / '2021-11-26-11-42-02_e2e_rec_elva_i_range_forward_0.8'],
                                   "lidar_frames.csv"),
        'Lidar ambience': read_frames([root_path / '2021-11-26-11-53-18_e2e_rec_elva_i_ambience_forward_0.8'],
                                      "lidar_frames.csv"),
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

    fig.savefig("elva-2021-11-25-lidar-channels.png", facecolor="white")
