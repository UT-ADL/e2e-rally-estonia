import math

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from tqdm.auto import tqdm

from camera_frame import CameraFrameTransformer
from dataloading.nvidia import NvidiaDataset, Normalize

from torchvision import transforms

from trajectory import calculate_steering_angle

N_WAYPOINTS = 2
REF_DISTANCE = 9.5


def create_waypoint_error_plot(model, trainer, dataset_name, n_branches=3, n_waypoints=6):
    root_path = Path("/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021")
    tr = transforms.Compose([Normalize()])
    dataset = NvidiaDataset([root_path / dataset_name], transform=tr, n_branches=n_branches,  n_waypoints=n_waypoints,
                            output_modality="waypoints")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)
    predictions = trainer.predict(model, dataloader)

    true_waypoints = dataset.get_waypoints()


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

    f, (ax) = plt.subplots(6, 1, figsize=(50, 25))
    ax[0].plot(dataset.frames.steering_angle, color="green")

    pred_steering_angles = []
    wp_progress_bar = tqdm(total=len(predictions), smoothing=0)
    wp_progress_bar.set_description("Calculating steering angles")

    transformer = CameraFrameTransformer()

    for wp in predictions:
        steering_angle_wp = [0.0, 0.0]
        steering_angle_wp.extend(wp[:2*N_WAYPOINTS])
        wp_baselink = transformer.transform_waypoints(steering_angle_wp, "interfacea_link2")
        pred_steering_angles.append(calculate_steering_angle(wp_baselink, ref_distance=REF_DISTANCE))
        wp_progress_bar.update(1)
    ax[0].plot(pred_steering_angles, color="red")

    #ax[0].plot(dataset.frames.turn_signal, linewidth=3, color="gold")
    #ax[0].plot(dataset.frames.vehicle_speed, color="darkorange")
    ax[0].legend(['true_steering_angle', 'pred_steering_angle', 'turn_signal', 'vehicle_speed'])
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

    first_wp_long_error = predictions[:, 0] - true_waypoints[:, 0]
    ax[3].plot(first_wp_long_error, color="darkred")
    ax[3].set_title(dataset_name + " | first waypoint longitudinal error")

    first_wp_lat_error = predictions[:, 1] - true_waypoints[:, 1]
    ax[4].plot(first_wp_lat_error, color="darkred")
    ax[4].set_title(dataset_name + " | first waypoint lateral error")

    ax[5].plot(true_waypoints[:, 0], color="darkred")
    ax[5].set_title(dataset_name + " | first waypoint lateral")


def create_steering_angle_error_plot(model, trainer, dataset_name, n_branches=3):
    root_path = Path("/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021")
    tr = transforms.Compose([Normalize()])
    dataset = NvidiaDataset([root_path / dataset_name], transform=tr, n_branches=n_branches, output_modality="steering_angle")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)
    pred_steering_angles = trainer.predict(model, dataloader)

    f, (ax) = plt.subplots(2, 1, figsize=(50, 25))
    true_steering_angle = dataset.frames.steering_angle
    ax[0].plot(true_steering_angle, color="green")
    ax[0].plot(pred_steering_angles, color="red")

    ax[0].plot(dataset.frames.turn_signal, linewidth=3, color="gold")
    ax[0].plot(dataset.frames.vehicle_speed, color="darkorange")
    ax[0].legend(['true_steering_angle', 'pred_steering_angle', 'turn_signal', 'vehicle_speed'])
    ax[0].set_title(dataset_name + " | steering angle")

    steering_error = np.abs(pred_steering_angles - true_steering_angle)
    ax[1].plot(steering_error, color="darkred")
    ax[1].set_title(dataset_name + " | steering error")
    ax[1].legend(['steering error'])


