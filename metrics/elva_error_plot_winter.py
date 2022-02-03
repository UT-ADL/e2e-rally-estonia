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

    datasets = {
        'Lidar v1 winter': read_frames([root_path / '2022-02-02-11-32-37_e2e_rec_elva_winter_lidar-v3_forw_08'],
                                       "lidar_frames.csv"),
        'Lidar v2 winter': read_frames([root_path / '2022-02-02-10-39-23_e2e_rec_elva_winter_lidar_forward_08',
                                        root_path / '2022-02-02-10-50-07_e2e_rec_elva_winter_lidar_forward_08'],
                                       "lidar_frames.csv"),
        'Lidar v3 winter': read_frames([root_path / '2022-02-02-11-05-18_e2e_rec_elva_winter_lidar-v5_forw_08'],
                                       "lidar_frames.csv"),

        'Camera v3 winter': read_frames([root_path / '2022-02-02-11-58-48_e2e_rec_elva_winter_camera-v3_forw_08'],
                                        "nvidia_frames.csv"),
    }

    datasets_backwards = {
        'Lidar v1 winter': read_frames([root_path / '2022-02-02-11-45-34_e2e_rec_elva_winter_lidar-v3_backw_08'],
                                       "lidar_frames.csv"),
        'Lidar v2 winter': read_frames([root_path / '2022-02-02-10-53-16_e2e_rec_elva_winter_lidar_backw_08'],
                                       "lidar_frames.csv"),
        'Lidar v3 witer': read_frames([root_path / '2022-02-02-11-18-14_e2e_rec_elva_winter_lidar-v5_backw_08'],
                                      "lidar_frames.csv"),
    }

    expert_frames = read_frames([root_path / '2021-10-26-10-49-06_e2e_rec_ss20_elva'], "lidar_frames.csv")
    expert_frames_back = read_frames([root_path / '2021-10-26-11-08-59_e2e_rec_ss20_elva_back'], "lidar_frames.csv")

    fig, ax = plt.subplots(len(datasets), 2, figsize=(15, 30))

    for i, (name, model_frames) in enumerate(datasets.items()):
        print(i, name, "forward")
        draw_error_plot(ax[i][0], model_frames, expert_frames, f"{name} Elva forward")

    for i, (name, model_frames) in enumerate(datasets_backwards.items()):
        print(i, name, "backward")
        draw_error_plot(ax[i][1], model_frames, expert_frames_back, f"{name} Elva backward")

    fig.savefig("elva-2022-02-11-winter.png", facecolor="white")
