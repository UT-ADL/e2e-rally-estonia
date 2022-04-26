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
    root_path = Path("/media/romet/data2/datasets/rally-estonia/dataset")

    datasets = {
        'autumn-v3': read_frames_driving([root_path / '2021-11-03-12-35-19_e2e_rec_elva_autumn-v3']),
        'wide-v2': read_frames_driving([root_path / '2021-11-03-13-13-16_e2e_rec_elva_wide-v2']),
        'autumn-v1': read_frames_driving([root_path / '2021-11-03-13-51-53_e2e_rec_elva_autumn-v1',
                                  root_path / '2021-11-03-14-02-07_e2e_rec_elva_autumn-v1_continue'])
    }

    datasets_backwards = {
        'autumn-v3': read_frames_driving([root_path / '2021-11-03-12-53-38_e2e_rec_elva_back_autumn-v3']),
        'wide-v2': read_frames_driving([root_path / '2021-11-03-13-30-48_e2e_rec_elva_back_wide-v2']),
        'autumn-v1': read_frames_driving([root_path / '2021-11-03-14-12-10_e2e_rec_elva_back_autumn-v1'])
    }

    expert_frames = read_frames_driving([root_path / '2021-10-26-10-49-06_e2e_rec_ss20_elva'], "nvidia")
    expert_frames_back = read_frames_driving([root_path / '2021-10-26-11-08-59_e2e_rec_ss20_elva_back'])

    fig, ax = plt.subplots(3, 2, figsize=(15, 30))

    for i, (name, model_frames) in enumerate(datasets.items()):
        print(i, name)
        draw_error_plot(ax[i][0], model_frames, expert_frames, f"{name} elva forward")

    for i, (name, model_frames) in enumerate(datasets_backwards.items()):
        print(i, name)
        draw_error_plot(ax[i][1], model_frames, expert_frames_back, f"{name} Elva backward")

    fig.savefig("elva-2021-11-03.png", facecolor="white")
