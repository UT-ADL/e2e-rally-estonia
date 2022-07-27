import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


def get_points(dataset, frame_idx, num_waypoints=10):
    wp_x = [f"wp_x_offset_{i}" for i in range(1, num_waypoints + 1)]
    wp_y = [f"wp_y_offset_{i}" for i in range(1, num_waypoints + 1)]

    row = dataset.frames.iloc[frame_idx]
    x = [0.0]
    x.extend(row[wp_x])
    y = [0.0]
    y.extend(row[wp_y])
    print("True angle: ", math.degrees(row["steering_angle"]))
    return x, y


# https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle.html
def fit_circle(x, y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = Ri_2.mean()
    return xc_2, yc_2, R_2


def calculate_steering_angle(waypoints, num_waypoints=3, ref_distance=6.5, use_vehicle_pos=True):

    # use specified number of waypoints
    waypoints = waypoints[:2*num_waypoints]

    # add current vehicle position to the trajectory
    if use_vehicle_pos:
        waypoints = np.hstack(([0.0, 0.0], waypoints))

    wp_x = waypoints[::2]
    wp_y = waypoints[1::2]
    x_center, y_center, radius = fit_circle(wp_x, wp_y)

    current_pos_theta = math.atan2(0 - y_center, 0 - x_center)
    circumference = 2 * np.pi * radius
    if current_pos_theta < 0:
        next_pos_theta = current_pos_theta + (2 * np.pi / circumference) * ref_distance
    else:
        next_pos_theta = current_pos_theta - (2 * np.pi / circumference) * ref_distance

    next_x = x_center + radius * np.cos(next_pos_theta)
    next_y = y_center + radius * np.sin(next_pos_theta)

    return np.arctan(next_y / next_x) * 14.7


def draw_circle(x, y, x_center, y_center, radius, ref_distance=10):
    theta_fit = np.linspace(-np.pi, np.pi, 180)

    x_fit2 = x_center + radius * np.cos(theta_fit)
    y_fit2 = y_center + radius * np.sin(theta_fit)
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(x_fit2, y_fit2, 'k--', lw=2)
    plt.scatter(x, y)
    plt.scatter(x_center, y_center, color="Green")

    current_pos_theta = math.atan2(0 - y_center, 0 - x_center)
    circumference = 2 * np.pi * radius
    if current_pos_theta < 0:
        next_pos_theta = current_pos_theta + (2 * np.pi / circumference) * ref_distance
    else:
        next_pos_theta = current_pos_theta - (2 * np.pi / circumference) * ref_distance

    next_x = x_center + radius * np.cos(next_pos_theta)
    next_y = y_center + radius * np.sin(next_pos_theta)

    print("Trajectory angle: ", math.degrees(np.arctan(next_y / next_x)))

    plt.scatter(next_x, next_y, color="Red")
    plt.show()


def draw_trajectory_circle(dataset, frame_idx, num_waypoints):
    x, y = get_points(dataset, frame_idx, num_waypoints)
    center_x, center_y, radius = fit_circle(x, y)
    draw_circle(x, y, center_x, center_y, radius)
