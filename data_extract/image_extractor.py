from collections import defaultdict
import rosbag
from cv_bridge import CvBridge
import cv2
import shutil
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from argparse import ArgumentParser
from tf.transformations import euler_from_quaternion


class NvidiaDriveImporter:

    def __init__(self, bag_files, extract_dir, resize_camera_images, extract_side_cameras, image_type):
        self.bag_files = bag_files
        self.extract_dir = extract_dir
        self.resize_camera_image = resize_camera_images
        self.extract_side_cameras = extract_side_cameras
        self.image_type = image_type

        self.steer_topic = '/pacmod/parsed_tx/steer_rpt'
        self.speed_topic = '/pacmod/parsed_tx/vehicle_speed_rpt'
        self.turn_topic = '/pacmod/parsed_tx/turn_rpt'
        self.autonomy_topic = '/pacmod/as_tx/enabled'
        self.current_pose = '/current_pose'
        self.general_topics = [self.steer_topic, self.speed_topic, self.turn_topic,
                               self.autonomy_topic, self.current_pose]

        # NVIDIA images
        # left 120'
        left_camera_topic_old = "/interfacea/link0/image_raw/compressed"
        left_camera_topic = "/interfacea/link0/image/compressed"
        # right 120'
        right_camera_topic_old = "/interfacea/link1/image_raw/compressed"
        right_camera_topic = "/interfacea/link1/image/compressed"
        # front 120'
        front_wide_camera_topic_old = "/interfacea/link2/image_raw/compressed"
        front_wide_camera_topic = "/interfacea/link2/image/compressed"
        # front 60'
        # self.front_narrow_camera_topic = "/interfacea/link3/image/compressed"
        self.nvidia_topics = [front_wide_camera_topic_old, front_wide_camera_topic]
        if extract_side_cameras:
            self.nvidia_topics = self.nvidia_topics + [left_camera_topic_old, left_camera_topic,
                                                       right_camera_topic_old, right_camera_topic]

        # OUSTER images
        self.lidar_amb_c = '/lidar_center/ambient_image'
        self.lidar_int_c = '/lidar_center/intensity_image'
        self.lidar_rng_c = '/lidar_center/range_image'
        self.lidar_topics = [self.lidar_amb_c, self.lidar_int_c, self.lidar_rng_c]

        self.topics = self.nvidia_topics + self.lidar_topics + self.general_topics

        self.topic_to_camera_name_map = {
            left_camera_topic_old: "left",
            left_camera_topic: "left",
            right_camera_topic_old: "right",
            right_camera_topic: "right",
            front_wide_camera_topic_old: "front_wide",
            front_wide_camera_topic: "front_wide",
        }

        # Camera image dimensions
        self.xmin = 300
        self.xmax = 1620
        self.ymin = 520
        self.ymax = 864
        self.scale = 0.2

        height = self.ymax - self.ymin
        width = self.xmax - self.xmin
        self.scaled_width = int(self.scale * width)
        self.scaled_height = int(self.scale * height)

    def import_bags(self):
        for bag_file in self.bag_files:
            print(f"Importing bag {bag_file}")
            self.import_bag(bag_file, self.image_type)

    def import_bag(self, bag_file, image_type):
        bag = rosbag.Bag(bag_file, "r")
        bridge = CvBridge()

        #  Create output folder (old data is deleted)
        bag_path = Path(bag_file)
        if self.extract_dir:  # extract to directory defined in argument
            root_folder = Path(self.extract_dir) / bag_path.stem
        else:  # extract to same directory where bag is
            root_folder = Path(bag_path.parent) / bag_path.stem
        shutil.rmtree(root_folder, ignore_errors=True)
        root_folder.mkdir(parents=True)

        for camera_topic in self.nvidia_topics:
            camera_name = self.topic_to_camera_name_map[camera_topic]
            camera_folder = root_folder / camera_name
            camera_folder.mkdir(exist_ok=True)
        lidar_folder = root_folder / "lidar"
        lidar_folder.mkdir(exist_ok=True)

        steering_dict = defaultdict(list)
        speed_dict = defaultdict(list)
        turn_dict = defaultdict(list)
        camera_dict = defaultdict(list)
        current_pose_dict = defaultdict(list)
        lidar_dict = defaultdict(list)

        autonomous = False
        autonomy_changed = False

        oi = OusterImage(0)
        first = True

        progress = tqdm(total=bag.get_message_count(self.nvidia_topics) + bag.get_message_count(self.lidar_topics))

        for topic, msg, ts in bag.read_messages(topics=self.topics):

            if topic == self.autonomy_topic:
                autonomy_changed = autonomous != msg.data
                autonomous = msg.data
                if autonomy_changed:
                    print("Autonomy changed to ", autonomous)

                if autonomy_changed and autonomous:
                    oi = OusterImage(0)
            else:
                msg_timestamp = msg.header.stamp.to_nsec()

                if topic == self.steer_topic:
                    steering_dict["timestamp"].append(msg_timestamp)
                    steering_dict["steering_angle"].append(msg.manual_input)

                elif topic == self.current_pose:
                    current_pose_dict["timestamp"].append(msg_timestamp)

                    current_pose_dict["position_x"].append(msg.pose.position.x)
                    current_pose_dict["position_y"].append(msg.pose.position.y)
                    current_pose_dict["position_z"].append(msg.pose.position.z)

                    quaternion = [
                        msg.pose.orientation.x, msg.pose.orientation.y,
                        msg.pose.orientation.z, msg.pose.orientation.w
                    ]
                    roll, pitch, yaw = euler_from_quaternion(quaternion)
                    current_pose_dict["roll"].append(roll)
                    current_pose_dict["pitch"].append(pitch)
                    current_pose_dict["yaw"].append(yaw)

                elif topic == self.speed_topic:
                    speed_dict["timestamp"].append(msg_timestamp)
                    speed_dict["vehicle_speed"].append(msg.vehicle_speed)

                elif topic == self.turn_topic:
                    turn_dict["timestamp"].append(msg_timestamp)
                    turn_dict["turn_signal"].append(int(msg.output))

                elif topic in self.nvidia_topics:
                    camera_name = self.topic_to_camera_name_map[topic]
                    output_folder = root_folder / camera_name
                    camera_dict["timestamp"].append(msg_timestamp)
                    camera_dict["autonomous"].append(autonomous)
                    camera_dict["camera"].append(camera_name)
                    image_name = f"{msg_timestamp}.{image_type}"
                    camera_dict["filename"].append(str(Path(output_folder.stem) / image_name))
                    cv_img = bridge.compressed_imgmsg_to_cv2(msg)
                    if self.resize_camera_image:
                        cv_img = self.crop(cv_img)
                        cv_img = self.resize(cv_img)
                    cv2.imwrite(str(output_folder / image_name), cv_img)
                    progress.update(1)
                elif topic in self.lidar_topics:
                    if msg_timestamp != oi.ts:
                        if not first:
                            lidar_image = oi.image()
                            if type(lidar_image) != type(None):
                                camera_name = "lidar"
                                output_folder = root_folder / camera_name
                                lidar_dict["timestamp"].append(oi.ts)
                                lidar_dict["autonomous"].append(autonomous)
                                image_name = f"{oi.ts}.{image_type}"
                                lidar_dict["lidar_filename"].append(str(Path(output_folder.stem) / image_name))
                                cv2.imwrite(str(output_folder / image_name), lidar_image)
                        oi = OusterImage(msg_timestamp)
                        first = False

                    cv_img = bridge.imgmsg_to_cv2(msg)
                    if topic == self.lidar_amb_c:
                        oi.set_amb(cv_img)
                    elif topic == self.lidar_int_c:
                        oi.set_inten(cv_img)
                    elif topic == self.lidar_rng_c:
                        oi.set_rng(cv_img)

                    progress.update(1)

        bag.close()

        camera_df = pd.DataFrame(data=camera_dict, columns=["timestamp", "camera", "filename", "autonomous"])
        self.create_timestamp_index(camera_df)

        front_wide_camera_df = self.create_camera_df(camera_df, "front_wide")

        steering_df = pd.DataFrame(data=steering_dict, columns=["timestamp", "steering_angle"])
        self.create_timestamp_index(steering_df)

        speed_df = pd.DataFrame(data=speed_dict, columns=["timestamp", "vehicle_speed"])
        self.create_timestamp_index(speed_df)

        turn_df = pd.DataFrame(data=turn_dict, columns=["timestamp", "turn_signal"])
        self.create_timestamp_index(turn_df)

        current_pose_df = pd.DataFrame(data=current_pose_dict, columns=["timestamp",
                                                                        "position_x", "position_y", "position_z",
                                                                        "roll", "pitch", "yaw"])
        self.create_timestamp_index(current_pose_df)

        dataframes = [front_wide_camera_df]
        if self.extract_side_cameras:
            left_camera_df = self.create_camera_df(camera_df, "left").drop(columns="autonomous")
            right_camera_df = self.create_camera_df(camera_df, "right").drop(columns="autonomous")
            dataframes += [left_camera_df, right_camera_df]
        dataframes += [steering_df, speed_df, turn_df, current_pose_df]

        merged = functools.reduce(lambda left, right:
                                  pd.merge(left, right, how='outer', left_index=True, right_index=True),
                                  dataframes)
        merged.interpolate(method='time', inplace=True)

        filtered_df = merged.loc[front_wide_camera_df.index]
        filtered_df.to_csv(root_folder / "nvidia_frames.csv", header=True)

        lidar_df = pd.DataFrame(data=lidar_dict)
        self.create_timestamp_index(lidar_df)

        merged_lidar = functools.reduce(lambda left, right:
                                        pd.merge(left, right, how='outer', left_index=True, right_index=True),
                                        [lidar_df, steering_df, speed_df, turn_df, current_pose_df])
        merged_lidar.interpolate(method='time', inplace=True)

        filtered_lidar_df = merged_lidar.loc[lidar_df.index]
        filtered_lidar_df.to_csv(root_folder / "lidar_frames.csv", header=True)

    def create_timestamp_index(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index(['timestamp'], inplace=True)
        df.index.rename('index', inplace=True)

    def create_camera_df(self, df, camera_name):
        camera_df = df[df["camera"] == camera_name]
        camera_df = camera_df.rename(columns={"filename": f"{camera_name}_filename"})
        camera_df.drop(["camera"], 1, inplace=True)
        return camera_df
    
    def resize(self, img):
        return cv2.resize(img, dsize=(self.scaled_width, self.scaled_height), interpolation=cv2.INTER_LINEAR)

    def crop(self, img):
        return img[self.ymin:self.ymax, self.xmin:self.xmax, :]



class OusterImage(object):
    def __init__(self, ts):
        self.ts = ts
        self.amb = None
        self.rng = None
        self.inten = None

    def set_amb(self, amb):
        self.amb = amb

    def set_inten(self, inten):
        self.inten = inten

    def set_rng(self, rng):
        self.rng = rng

    def image(self):
        if type(self.rng) != type(None) and type(self.amb) != type(None) and type(self.inten) != type(None):
            img = np.dstack((self.amb, self.inten, self.rng))
            return img
        else:
            print("failed")
            return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bag-file",
                        help="Path to bag file to extract")
    parser.add_argument("--extract-dir",
                        help="Directory where bag content is extracted to")

    parser.add_argument("--resize-camera-images",
                        default=False,
                        action='store_true',
                        help='Resize camera image for smaller size')

    parser.add_argument("--extract-side-cameras",
                        default=False,
                        action='store_true',
                        help='Extract left and right side camera images')

    parser.add_argument("--image-type",
                        default="png",
                        required=False,
                        choices=["png", "jpg"],
                        help="")

    args = parser.parse_args()

    bags = [
        args.bag_file
    ]
    importer = NvidiaDriveImporter(bags, args.extract_dir, args.resize_camera_images, args.extract_side_cameras, args.image_type)
    importer.import_bags()
