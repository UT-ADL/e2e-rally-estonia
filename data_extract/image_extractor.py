from collections import defaultdict
import rosbag
from cv_bridge import CvBridge
import cv2
import shutil
import functools
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class NvidiaDriveImporter:

    def __init__(self, bag_files):
        self.bag_files = bag_files

        # left 120'
        self.left_camera_topic = "/interfacea/link0/image/compressed"
        # right 120'
        self.right_camera_topic = "/interfacea/link1/image/compressed"
        # front 120'
        self.front_wide_camera_topic = "/interfacea/link2/image/compressed"
        # front 60'
        self.front_narrow_camera_topic = "/interfacea/link3/image/compressed"
        # steering topic
        self.steer_topic = '/pacmod/parsed_tx/steer_rpt'

        self.camera_topics = [self.left_camera_topic, self.right_camera_topic, self.front_wide_camera_topic, self.front_narrow_camera_topic]
        self.topics = self.camera_topics + [self.steer_topic]

        self.topic_to_camera_name_map = {
            self.left_camera_topic: "left",
            self.right_camera_topic: "right",
            self.front_wide_camera_topic: "front_wide",
            self.front_narrow_camera_topic: "front_narrow"
        }

    def import_bags(self):
        for bag_file in self.bag_files:
            print(f"Importing bag {bag_file}")
            self.import_bag(bag_file)

    def import_bag(self, bag_file):
        stats = {
            self.left_camera_topic: 0,
            self.right_camera_topic: 0,
            self.front_wide_camera_topic: 0,
            self.front_narrow_camera_topic: 0,
            self.steer_topic: 0
        }

        bag = rosbag.Bag(bag_file, "r")
        bridge = CvBridge()

        #  Create output folder (old data is deleted)
        bag_path = Path(bag_file)
        root_folder = Path(bag_path.parent) / bag_path.stem
        shutil.rmtree(root_folder, ignore_errors=True)
        root_folder.mkdir()

        for camera_topic in self.camera_topics:
            camera_name = self.topic_to_camera_name_map[camera_topic]
            camera_folder = root_folder / camera_name
            camera_folder.mkdir()

        steering_dict = defaultdict(list)
        camera_dict = defaultdict(list)

        progress = tqdm(total=bag.get_message_count(self.camera_topics))
        for topic, msg, ts in bag.read_messages(topics=self.topics):
            stats[topic] += 1

            msg_timestamp = msg.header.stamp.to_nsec()
            if topic == self.steer_topic:
                steering_dict["timestamp"].append(msg_timestamp)
                steering_dict["steering_angle"].append(msg.manual_input)
            elif topic in self.camera_topics:
                camera_name = self.topic_to_camera_name_map[topic]
                output_folder = root_folder / camera_name
                camera_dict["timestamp"].append(msg_timestamp)
                camera_dict["camera"].append(camera_name)
                image_name = f"{msg_timestamp}.jpg"
                camera_dict["filename"].append(str(Path(output_folder.stem) / image_name))
                cv_img = bridge.compressed_imgmsg_to_cv2(msg)
                cv2.imwrite(str(output_folder / image_name), cv_img)
                progress.update(1)

        bag.close()

        camera_df = pd.DataFrame(data=camera_dict, columns=["timestamp", "camera", "filename"])
        self.create_timestamp_index(camera_df)

        front_narrow_camera_df = self.create_camera_df(camera_df, "front_narrow")
        front_wide_camera_df = self.create_camera_df(camera_df, "front_wide")
        left_camera_df = self.create_camera_df(camera_df, "left")
        right_camera_df = self.create_camera_df(camera_df, "right")

        steering_df = pd.DataFrame(data=steering_dict, columns=["timestamp", "steering_angle"])
        self.create_timestamp_index(steering_df)

        merged = functools.reduce(lambda left, right:
                                  pd.merge(left, right, how='outer', left_index=True, right_index=True),
                                  [front_narrow_camera_df, front_wide_camera_df, left_camera_df, right_camera_df,
                                   steering_df])
        merged.interpolate(method='time', inplace=True)
        filtered_df = merged.loc[front_narrow_camera_df.index]
        filtered_df.to_csv(root_folder / "frames.csv", header=True)

    def create_timestamp_index(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index(['timestamp'], inplace=True)
        df.index.rename('index', inplace=True)

    def create_camera_df(self, df, camera_name):
        camera_df = df[df["camera"] == camera_name]
        camera_df = camera_df.rename(columns={"filename": f"{camera_name}_filename"})
        camera_df.drop(["camera"], 1, inplace=True)
        return camera_df

