import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from moviepy.editor import ImageSequenceClip
from skimage import io
from torchvision import transforms
from tqdm.auto import tqdm

from dataloading.nvidia import NvidiaDataset, Normalize, NvidiaCropWide
from pilotnet import PilotNetConditional, PilotnetControl
from trainer import Trainer, ConditionalTrainer, ControlTrainer
from velocity_model.velocity_model import VelocityModel


def create_driving_video(dataset_folder, output_modality):
    dataset_path = Path(dataset_folder)
    dataset = NvidiaDataset([dataset_path], output_modality=output_modality, n_branches=3,
                            metadata_file="nvidia_frames.csv")

    temp_frames_folder = dataset_path / 'temp'
    shutil.rmtree(temp_frames_folder, ignore_errors=True)
    temp_frames_folder.mkdir()

    draw_driving_frames(dataset, temp_frames_folder, output_modality)
    output_video_path = dataset_path / 'video.mp4'
    convert_frames_to_video(temp_frames_folder, output_video_path, fps=30)

    shutil.rmtree(temp_frames_folder, ignore_errors=True)

    print(f"{dataset.name}: output video {output_video_path} created.")


def create_prediction_video(dataset_folder, output_modality, model_path, model_type):
    dataset_path = Path(dataset_folder)
    dataset = NvidiaDataset([dataset_path], name=dataset_path.name, output_modality=output_modality,
                            n_branches=3, n_waypoints=10)

    #dataset.frames = dataset.frames[9160:9500]

    temp_frames_folder = dataset_path / 'temp'
    shutil.rmtree(temp_frames_folder, ignore_errors=True)
    temp_frames_folder.mkdir()

    if output_modality == "steering_angle":
        steering_predictions = get_predictions(dataset_path, model_path, "steering_angle")
        speed_predictions = get_speed_predictions(dataset)

        draw_prediction_frames(dataset, steering_predictions, speed_predictions, temp_frames_folder)

    if output_modality == "waypoints":
        trajectory = get_predictions(dataset_path, model_path, "waypoints")
        draw_prediction_frames_wp(dataset, trajectory, temp_frames_folder)

    output_video_path = dataset_path / f"{str(Path(model_path).parent.name)}.mp4"
    convert_frames_to_video(temp_frames_folder, output_video_path, fps=30)

    shutil.rmtree(temp_frames_folder, ignore_errors=True)

    print(f"{dataset.name}: output video {output_video_path} created.")


def get_predictions(dataset_path, model_path, output_modality):
    print(f"{dataset_path.name}: {output_modality} predictions")
    #trainer.force_cpu()  # not enough memory on GPU for parallel processing  # TODO: make input argument
    n_outputs = 1 if output_modality == "steering_angle" else 20
    if model_type == "pilotnet-conditional":
        model = PilotNetConditional(n_branches=3, n_outputs=n_outputs)
        trainer = ConditionalTrainer(None, target_name=output_modality, n_conditional_branches=3)
    elif model_type == "pilotnet-control":
        model = PilotnetControl(n_outputs=1)
        trainer = ControlTrainer(None, target_name=output_modality, n_conditional_branches=3)
    else:
        print(f"Unknown model type '{args.model_type}'")
        sys.exit()


    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dataloader = get_data_loader(dataset_path, output_modality)
    steering_predictions = trainer.predict(model, dataloader)
    return steering_predictions


def get_data_loader(dataset_path, output_modality):
    tr = transforms.Compose([NvidiaCropWide(), Normalize()])
    dataset = NvidiaDataset([Path(dataset_path)],
                            tr, name=dataset_path.name, output_modality=output_modality, n_branches=3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                                 num_workers=16, pin_memory=True, persistent_workers=True)
    return dataloader


def get_speed_predictions(dataset):
    print(f"{dataset.name}: speed predictions")
    velocity_model = VelocityModel(positions_parquet='velocity_model/summer2021-positions.parquet')

    frames = dataset.frames

    x = frames["position_x"]# + np.random.normal(0, 0.1, len(frames))
    y = frames["position_y"]# + np.random.normal(0, 0.1, len(frames))
    yaw = frames["yaw"]# + np.random.normal(0, 0.2, len(frames))

    result_df = pd.DataFrame(data={'x': x, 'y': y, 'yaw': yaw})
    result_df = result_df.fillna(0)  # TODO: correct NaN handling
    speed_predictions = result_df.apply(
        lambda df: velocity_model.find_speed_for_position(df['x'], df['y'], df['yaw'])[0], axis=1)
    return speed_predictions.to_numpy()


def draw_steering_angle(frame, steering_angle, steering_wheel_radius, steering_position, size, color):
    steering_angle_rad = math.radians(steering_angle)
    x = steering_wheel_radius * np.cos(np.pi / 2 + steering_angle_rad)
    y = steering_wheel_radius * np.sin(np.pi / 2 + steering_angle_rad)
    cv2.circle(frame, (steering_position[0] + int(x), steering_position[1] - int(y)), size, color, thickness=-1)


def draw_prediction_frames_wp(dataset, trajectory, temp_frames_folder):
    print("Creating video frames.")

    dataset.frames["turn_signal"].fillna(99, inplace=True)  # TODO correct NaN handling

    t = tqdm(enumerate(dataset), total=len(dataset))
    t.set_description(dataset.name)

    for frame_index, (data, target_values, condition_mask) in t:
        frame = data["image"].permute(1, 2, 0).cpu().numpy()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        true_angle = math.degrees(data["steering_angle"])
        true_speed = data["vehicle_speed"] * 3.6

        true_waypoints = data["waypoints"]
        predicted_waypoints = trajectory[frame_index]

        position_x = data["position_x"]
        position_y = data["position_y"]
        yaw = math.degrees(data["yaw"])
        turn_signal = int(data["turn_signal"])

        cv2.putText(frame, 'True: {:.2f} deg, {:.2f} km/h'.format(true_angle, true_speed), (10, 1150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        cv2.putText(frame, 'frame: {}'.format(frame_index), (10, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'x: {:.2f}'.format(position_x), (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'y: {:.2f}'.format(position_y), (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'yaw: {:.2f}'.format(yaw), (10, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        turn_signal_map = {
            1: "straight",
            2: "left",
            0: "right"
        }
        cv2.putText(frame, 'turn signal: {}'.format(turn_signal_map.get(turn_signal, "unknown")), (10, 1100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        draw_trajectory(frame, true_waypoints, (0, 255, 0))
        draw_trajectory(frame, predicted_waypoints, (255, 0, 0))

        io.imsave(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", frame)


def draw_prediction_frames(dataset, predicted_angles, predicted_speed, temp_frames_folder):
    print("Creating video frames.")

    #dataset.frames = dataset.frames[9160:23070]
    #dataset.frames = dataset.frames[9160:10070]

    dataset.frames["turn_signal"].fillna(99, inplace=True)  # TODO correct NaN handling

    t = tqdm(enumerate(dataset), total=len(dataset))
    t.set_description(dataset.name)

    for frame_index, (data, target_values, condition_mask) in t:
        frame = data["image"].permute(1, 2, 0).cpu().numpy()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        true_angle = math.degrees(data["steering_angle"])
        pred_angle = math.degrees(predicted_angles[frame_index])
        true_speed = data["vehicle_speed"] * 3.6
        pred_speed = predicted_speed[frame_index] * 3.6

        position_x = data["position_x"]
        position_y = data["position_y"]
        yaw = math.degrees(data["yaw"])
        turn_signal = int(data["turn_signal"])

        cv2.putText(frame, 'True: {:.2f} deg, {:.2f} km/h'.format(true_angle, true_speed), (10, 1150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Pred: {:.2f} deg, {:.2f} km/h'.format(pred_angle, pred_speed), (10, 1200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

        cv2.putText(frame, 'frame: {}'.format(frame_index), (10, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'x: {:.2f}'.format(position_x), (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'y: {:.2f}'.format(position_y), (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'yaw: {:.2f}'.format(yaw), (10, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        turn_signal_map = {
            1: "straight",
            2: "left",
            0: "right"
        }
        cv2.putText(frame, 'turn signal: {}'.format(turn_signal_map.get(turn_signal, "unknown")), (10, 1100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        radius = 200
        steering_pos = (960, 1200)
        cv2.circle(frame, steering_pos, radius, (255, 255, 255), 7)

        cv2.rectangle(frame, (905, 1200), (955, 1200 - int(3 * true_speed)), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (965, 1200), (1015, 1200 - int(3 * pred_speed)), (255, 0, 0), cv2.FILLED)

        draw_steering_angle(frame, true_angle, radius, steering_pos, 13, (0, 255, 0))
        draw_steering_angle(frame, pred_angle, radius, steering_pos, 9, (255, 0, 0))

        #frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        io.imsave(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", frame)


def draw_driving_frames(dataset, temp_frames_folder, output_modality):
    print(f"Drawing driving frames with {output_modality}")
    t = tqdm(enumerate(dataset), total=len(dataset))
    t.set_description(dataset.name)
    for frame_index, (data, target_values, condition_mask) in t:

        frame = data["image"].permute(1, 2, 0).cpu().numpy()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        true_angle = math.degrees(data["steering_angle"])
        true_speed = data["vehicle_speed"] * 3.6
        autonomous = data["autonomous"]

        if autonomous:
            color = (255, 0, 0)
            cv2.putText(frame, 'Mode:    AUTONOMOUS', (10, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        else:
            color = (0, 255, 0)
            cv2.putText(frame, 'Mode:    MANUAL'.format(true_angle), (10, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                        cv2.LINE_AA)

        cv2.putText(frame, 'Steering: {:.2f} deg'.format(true_angle), (10, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Speed:   {:.2f} km/h'.format(true_speed), (10, 1200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)

        if output_modality == "steering_angle":
            radius = 200
            steering_pos = (960, 1200)
            cv2.circle(frame, steering_pos, radius, (255, 255, 255), 7)

            draw_steering_angle(frame, true_angle, radius, steering_pos, 13, color)
            cv2.rectangle(frame, (935, 1200), (980, 1200 - int(3 * true_speed)), color, cv2.FILLED)

        if output_modality == "waypoints":
            draw_trajectory(frame, data["waypoints"], color)

        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        io.imsave(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", frame)


def draw_trajectory(frame, waypoints, color):
    scale = 5
    for (x, y) in zip(waypoints[0::2], waypoints[1::2]):
        cv2.circle(frame, (935 - int(scale * y), 1200 - int(scale * x)), 3, color, 5)


def convert_frames_to_video(frames_folder, output_video_path, fps=30):
    output_folder = Path(os.path.split(output_video_path)[:-1][0])
    output_folder.mkdir(parents=True, exist_ok=True)

    p = Path(frames_folder).glob('**/*.jpg')
    image_list = sorted([str(x) for x in p if x.is_file()])

    print("Creating video {}, FPS={}".format(frames_folder, fps))
    clip = ImageSequenceClip(image_list, fps=fps)
    clip.write_videofile(str(output_video_path))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--dataset-folder',
        required=True,
        help='Path to a dataset extracted from a bag file'
    )

    argparser.add_argument(
        '--video-type',
        required=True,
        choices=['driving', 'prediction'],
        help="Type of the video, 'driving' or 'prediction'."
    )

    argparser.add_argument(
        '--output-modality',
        default="steering_angle",
        choices=["steering_angle", "waypoints"],
        help="Choice of output modality to visualise."
    )

    argparser.add_argument(
        '--model-path',
        help="Path to pytorch model to use for creating steering predictions."
    )

    argparser.add_argument(
        '--model-type',
        required=False,
        default="pilotnet",
        choices=['pilotnet', 'pilotnet-conditional', 'pilotnet-control'],
    )

    args = argparser.parse_args()
    dataset_folder = args.dataset_folder
    video_type = args.video_type
    model_path = args.model_path
    model_type = args.model_type
    print("Creating video from: ", dataset_folder)

    if video_type == 'driving':
        create_driving_video(dataset_folder, args.output_modality)
    elif video_type == 'prediction':
        create_prediction_video(dataset_folder, args.output_modality, model_path=model_path, model_type=model_type)
