import argparse
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from skimage import io
from tqdm.auto import tqdm

from dataloading.nvidia import NvidiaDataset


def create_driving_video(dataset_folder):
    dataset_path = Path(dataset_folder)
    frames = NvidiaDataset([dataset_path], camera="front_wide")

    temp_frames_folder = dataset_path / 'temp'
    shutil.rmtree(temp_frames_folder, ignore_errors=True)
    temp_frames_folder.mkdir()

    draw_frames(frames, temp_frames_folder)
    output_video_path = dataset_path / 'video.mp4'
    convert_frames_to_video(temp_frames_folder, output_video_path, fps=30)

    shutil.rmtree(temp_frames_folder, ignore_errors=True)

    print(f"Output video {output_video_path} created.")


def draw_steering_angle(frame, steering_angle, steering_wheel_radius, steering_position, size, color):
    steering_angle_rad = math.radians(steering_angle)
    x = steering_wheel_radius * np.cos(np.pi / 2 + steering_angle_rad)
    y = steering_wheel_radius * np.sin(np.pi / 2 + steering_angle_rad)
    cv2.circle(frame, (steering_position[0] + int(x), steering_position[1] - int(y)), size, color, thickness=-1)


def draw_frames(dataset, temp_frames_folder):
    for frame_index, data in tqdm(enumerate(dataset), total=len(dataset)):

        frame = data["image"].permute(1, 2, 0).cpu().numpy()
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

        radius = 200
        steering_pos = (960, 1200)
        cv2.circle(frame, steering_pos, radius, (255, 255, 255), 7)

        draw_steering_angle(frame, true_angle, radius, steering_pos, 13, color)
        cv2.rectangle(frame, (935, 1200), (980, 1200 - int(3 * true_speed)), color, cv2.FILLED)

        io.imsave(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", frame)


def convert_frames_to_video(frames_folder, output_video_path, fps=25):
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

    args = argparser.parse_args()
    dataset_folder = args.dataset_folder
    print("Creating video from: ", dataset_folder)

    create_driving_video(dataset_folder)
