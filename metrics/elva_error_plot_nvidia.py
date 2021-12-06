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
        'Camera v1': read_frames([root_path / '2021-11-25-12-09-43_e2e_rec_elva-nvidia-v1-0.8'], "nvidia_frames.csv"),
        'Camera v2': read_frames([root_path / '2021-11-25-14-01-46_e2e_rec_elva-licamera-v2-0.8-back'], "nvidia_frames.csv"),
        'Camera v3': read_frames([root_path / '2021-11-25-14-51-46_e2e_rec_elva-l-camera-v3-0.8-back'], "nvidia_frames.csv"),
        'Camera in train': read_frames([root_path / '2021-11-25-13-11-40_e2e_rec_elva-licamera-inTrain-0.8-back'], "nvidia_frames.csv"),
    }

    datasets = {
        'Camera v1': read_frames([root_path / '2021-11-25-12-21-17_e2e_rec_elva-nvidia-v1-0.8-forward'], "nvidia_frames.csv"),
        'Camera v2': read_frames([root_path / '2021-11-25-14-13-59_e2e_rec_elva-licamera-v2-0.8-forward'], "nvidia_frames.csv"),
        'Camera v3': read_frames([root_path / '2021-11-25-15-04-26_e2e_rec_elva-l-camera-v3-0.8-forward'], "nvidia_frames.csv"),
        'Camera in train': read_frames([root_path / '2021-11-25-13-24-00_e2e_rec_elva-licamera-inTrain-0.8-forward'], "nvidia_frames.csv"),
    }

    expert_frames = read_frames([root_path / '2021-10-26-10-49-06_e2e_rec_ss20_elva'], "nvidia_frames.csv")
    expert_frames_back = read_frames([root_path / '2021-10-26-11-08-59_e2e_rec_ss20_elva_back'], "nvidia_frames.csv")

    fig, ax = plt.subplots(len(datasets), 2, figsize=(15, 30))

    for i, (name, model_frames) in enumerate(datasets.items()):
        print(i, name)
        draw_error_plot(ax[i][0], model_frames, expert_frames, f"{name} elva forward")

    for i, (name, model_frames) in enumerate(datasets_backwards.items()):
        print(i, name)
        draw_error_plot(ax[i][1], model_frames, expert_frames_back, f"{name} Elva backward")

    fig.savefig("elva-2021-11-25.png", facecolor="white")
