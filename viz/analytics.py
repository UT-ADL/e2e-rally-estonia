import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch

from dataloading.nvidia import NvidiaDataset, Normalize

from trainer import Trainer
from torchvision import transforms

N_WAYPOINTS = 10


def create_waypoint_error_plot(model, dataset_name, n_branches=3, n_waypoints=6):
    root_path = Path("/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021")
    tr = transforms.Compose([Normalize()])
    dataset = NvidiaDataset([root_path / dataset_name], transform=tr, n_branches=n_branches,  n_waypoints=n_waypoints,
                            output_modality="waypoints")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)
    trainer = Trainer(n_conditional_branches=n_branches)
    predictions = trainer.predict(model, dataloader)

    wp_x_cols = [col for col in dataset.frames.columns if col.startswith('wp_x')]
    wp_y_cols = [col for col in dataset.frames.columns if col.startswith('wp_y')]
    waypoint_cols = np.column_stack((wp_x_cols, wp_y_cols)).reshape(-1)
    true_waypoints = dataset.frames[waypoint_cols].to_numpy()


    first_wp_error = np.hypot(predictions[:, 0] - true_waypoints[:, 0],
                              predictions[:, 1] - true_waypoints[:, 1])

    true_waypoints = true_waypoints[:, 0:predictions.shape[1]]
    last_wp_error = np.hypot(predictions[:, -2] - true_waypoints[:, -2],
                             predictions[:, -1] - true_waypoints[:, -1])

    # zipped_waypoints = tqdm(zip(predictions, true_waypoints), total=len(true_waypoints))
    # zipped_waypoints.set_description("Calculating frechet distances")
    # #zipped_waypoints = zip(predictions, true_waypoints)
    # frechet_distances = np.array(
    #     [frdist(z[0].reshape(-1, 2), z[1].reshape(-1, 2)) for z in zipped_waypoints])

    f, (ax) = plt.subplots(3, 1, figsize=(50, 25))
    ax[0].plot(dataset.frames.steering_angle, color="green")
    ax[0].plot(dataset.frames.turn_signal, linewidth=3, color="gold")
    ax[0].plot(dataset.frames.vehicle_speed, color="darkorange")
    ax[0].legend(['steering_angel', 'turn_signal', 'vehicle_speed'])
    ax[0].set_title(dataset_name + " | steering angle")

    # ax[1].plot(frechet_distances, color="darkblue")
    # ax[1].legend(['frechet distance'])
    # ax[1].set_title(dataset_name + " | frechet distances")

    ax[1].plot(first_wp_error, color="darkred")
    ax[1].set_title(dataset_name + " | first waypoint error")
    ax[1].legend(['first waypoint error'])

    ax[2].plot(last_wp_error, color="darkgreen")
    ax[2].set_title(dataset_name + " | last waypoint error")
    ax[2].legend(['last waypoint error'])

