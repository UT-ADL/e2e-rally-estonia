import numpy as np
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d import transformations as pt


class CameraFrameTransformer:

    def __init__(self):
        self.transform_manager = self.create_transform_manager()

    def transform_waypoints(self, local_waypoints, camera_frame):

        center_cam_transform = self.transform_manager.get_transform("base_link", camera_frame)
        x_offset = np.array([local_waypoints[i] for i in np.arange(0, len(local_waypoints), 2)])
        y_offset = np.array([local_waypoints[i] for i in np.arange(1, len(local_waypoints), 2)])

        transformer_waypoints = []
        for x, y in zip(x_offset, y_offset):
            # Camera frames are rotated compared to base_link frame (x = z, y = -x, z = -y)
            wp_camera = np.array([-y, 0, x, 1])
            wp_base = pt.transform(pt.invert_transform(center_cam_transform), wp_camera)
            transformer_waypoints.append(wp_base)

        transformer_waypoints = np.array(transformer_waypoints)
        return transformer_waypoints[:, 0:2].flatten()  # select only x and y and flatten into 1D array

    @staticmethod
    def create_transform_manager():
        tm = UrdfTransformManager()

        filename = "dataloading/platform.urdf"
        with open(filename, "r") as f:
            tm.load_urdf(f.read())

        return tm
