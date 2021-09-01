from collections import defaultdict
import rosbag
from cv_bridge import CvBridge
import cv2
import shutil
import functools
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from argparse import ArgumentParser

class NvidiaDriveImporter:

    def __init__(self, bag_files, extract_dir):
        self.bag_files = bag_files
        self.extract_dir = extract_dir

        # left 120'
        self.left_camera_topic = "/interfacea/link0/image/compressed"
        # right 120'
        self.right_camera_topic = "/interfacea/link1/image/compressed"
        # front 120'
        self.front_wide_camera_topic = "/interfacea/link2/image/compressed"
        # front 60'
        #self.front_narrow_camera_topic = "/interfacea/link3/image/compressed"
        self.steer_topic = '/pacmod/parsed_tx/steer_rpt'
        self.speed_topic = '/pacmod/parsed_tx/vehicle_speed_rpt'
        self.turn_topic = '/pacmod/parsed_tx/turn_rpt'
        self.autonomy_topic = '/pacmod/as_tx/enabled'

        self.camera_topics = [self.left_camera_topic, self.right_camera_topic, self.front_wide_camera_topic]
        self.topics = self.camera_topics + [self.steer_topic, self.speed_topic, self.turn_topic, self.autonomy_topic]

        self.topic_to_camera_name_map = {
            self.left_camera_topic: "left",
            self.right_camera_topic: "right",
            self.front_wide_camera_topic: "front_wide"
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
            self.steer_topic: 0,
            self.speed_topic: 0,
            self.turn_topic: 0,
            self.autonomy_topic: 0
        }

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

        for camera_topic in self.camera_topics:
            camera_name = self.topic_to_camera_name_map[camera_topic]
            camera_folder = root_folder / camera_name
            camera_folder.mkdir()

        steering_dict = defaultdict(list)
        speed_dict = defaultdict(list)
        turn_dict = defaultdict(list)
        camera_dict = defaultdict(list)

        autonomous = False
        autonomy_changed = False

        progress = tqdm(total=bag.get_message_count(self.camera_topics))
        for topic, msg, ts in bag.read_messages(topics=self.topics):

            stats[topic] += 1

            if topic == self.autonomy_topic:
                autonomy_changed = autonomous != msg.data
                autonomous = msg.data
                if autonomy_changed:
                    print("Autonomy changed to ", autonomous)
            else:
                msg_timestamp = msg.header.stamp.to_nsec()

                if topic == self.steer_topic:
                    steering_dict["timestamp"].append(msg_timestamp)
                    steering_dict["steering_angle"].append(msg.manual_input)

                elif topic == self.speed_topic:
                    speed_dict["timestamp"].append(msg_timestamp)
                    speed_dict["vehicle_speed"].append(msg.vehicle_speed)

                elif topic == self.turn_topic:
                    turn_dict["timestamp"].append(msg_timestamp)
                    turn_dict["turn_signal"].append(msg)

                elif topic in self.camera_topics:
                    camera_name = self.topic_to_camera_name_map[topic]
                    output_folder = root_folder / camera_name
                    camera_dict["timestamp"].append(msg_timestamp)
                    camera_dict["autonomous"].append(autonomous)
                    camera_dict["camera"].append(camera_name)
                    image_name = f"{msg_timestamp}.jpg"
                    camera_dict["filename"].append(str(Path(output_folder.stem) / image_name))
                    cv_img = bridge.compressed_imgmsg_to_cv2(msg)
                    cv2.imwrite(str(output_folder / image_name), cv_img)
                    progress.update(1)

        bag.close()

        camera_df = pd.DataFrame(data=camera_dict, columns=["timestamp", "camera", "filename", "autonomous"])
        self.create_timestamp_index(camera_df)

        front_wide_camera_df = self.create_camera_df(camera_df, "front_wide")
        left_camera_df = self.create_camera_df(camera_df, "left")
        right_camera_df = self.create_camera_df(camera_df, "right")

        steering_df = pd.DataFrame(data=steering_dict, columns=["timestamp", "steering_angle"])
        self.create_timestamp_index(steering_df)

        speed_df = pd.DataFrame(data=speed_dict, columns=["timestamp", "vehicle_speed"])
        self.create_timestamp_index(speed_df)

        turn_df = pd.DataFrame(data=turn_dict, columns=["timestamp", "turn_signal"])
        self.create_timestamp_index(turn_df)

        merged = functools.reduce(lambda left, right:
                                  pd.merge(left, right, how='outer', left_index=True, right_index=True),
                                  [front_wide_camera_df, left_camera_df, right_camera_df, turn_df, speed_df, steering_df])
        merged.interpolate(method='time', inplace=True)

        print(merged)
        filtered_df = merged.loc[front_wide_camera_df.index]
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bag-file", type=str)
    parser.add_argument("--extract-dir", type=str)
    args = parser.parse_args()

    bags = [
        args.bag_file
    ]
    importer = NvidiaDriveImporter(bags, args.extract_dir)
    importer.import_bags()
