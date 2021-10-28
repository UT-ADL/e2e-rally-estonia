import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


class VelocityModel:
    def __init__(self, positions_parquet='positions.parquet', vector_velocity=30):
        self.vector_velocity = vector_velocity
        self.positions_df = pd.read_parquet(positions_parquet)
        self.tree = BallTree(self.positions_df[["position_x", "position_y", "position_x2", "position_y2"]])

    def find_speed_for_position(self, x, y, yaw):

        x2 = x + (self.vector_velocity * np.cos(yaw))
        y2 = y + (self.vector_velocity * np.sin(yaw))

        closest = self.tree.query([[x, y, x2, y2]])
        distance = closest[0][0][0]
        index = closest[1][0][0]
        return self.positions_df.iloc[index]["vehicle_speed"], distance
