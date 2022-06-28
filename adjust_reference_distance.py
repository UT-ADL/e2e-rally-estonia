import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from torch.utils.data import DataLoader
from tqdm import tqdm

import trajectory as tr
from dataloading.nvidia import NvidiaValidationDataset
from metrics.metrics import calculate_open_loop_metrics
from pilotnet import PilotNetConditional
from trainer import ConditionalTrainer

"""
Script for finding best reference distance and number of waypoints for trajectory controller. 
Uses given model and dataset to find reference distance with lowest MAE.

Usage:
    ./adjust_reference_distance.py --dataset-path <root-path-to-dataset> --model-path <path-to-pytorch-model>
"""


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--dataset-path',
        default='/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021',
        required=False,
        help='Path to dataset used for optimizing'
    )

    argparser.add_argument(
        '--waypoints-source',
        choices=["model", "ground-truth"],
        default="model",
        required=False,
        help="Method used for calculuting trajectory waypoints."
    )

    argparser.add_argument(
        '--model-path',
        default='models/20220511212923_waypoints-conditional/best.pt',
        required=False,
        help='Path to PyTorch model used for optimizing'
    )

    argparser.add_argument(
        '--max-evals',
        default=100,
        type=int,
        required=False,
        help='Number of optimizations evaluation'
    )

    return argparser.parse_args()


def optimize_lidar_crop(args):
    dataset = NvidiaValidationDataset(Path(args.dataset_path), "waypoints", n_branches=3, n_waypoints=10,
                                      metadata_file="nvidia_frames_ext.csv")

    space = {
        'reference_distance': hp.uniform('reference_distance', 1.0, 20.0),
        'num_waypoints': hp.uniformint('num_waypoints', 2, 10),
        'use_vehicle_pos': hp.choice('use_vehicle_pos', [True, False]),
        'waypoints_source': args.waypoints_source,
        'dataset_path': args.dataset_path,
        'model_path': args.model_path,
        'dataset': dataset
    }

    trials = Trials()
    best = fmin(optimize_fun, space, trials=trials, algo=tpe.suggest, max_evals=args.max_evals, show_progressbar=False)

    print(best)
    print(trials.results)


def optimize_fun(opt_args):
    num_waypoints = opt_args['num_waypoints']
    reference_distance = opt_args['reference_distance']
    use_vehicle_pos = opt_args['use_vehicle_pos']
    dataset = opt_args['dataset']

    waypoints_source = opt_args['waypoints_source']
    if waypoints_source == 'model':
        model = load_model(opt_args['model_path'])
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8,
                                pin_memory=True, persistent_workers=True)
        trainer = ConditionalTrainer()
        trajectories = trainer.predict(model, dataloader)

    elif waypoints_source == 'ground-truth':
        trajectories = dataset.get_waypoints()

    else:
        print(f"Uknown waypoints source {waypoints_source}")
        sys.exit()

    calculated_steering = []
    for trajectory in tqdm(trajectories, desc="Calculating steering angles"):
        calculated_steering.append(tr.calculate_steering_angle(trajectory, num_waypoints,
                                                               reference_distance, use_vehicle_pos))

    true_steering = dataset.frames.steering_angle.to_numpy()
    open_loop_metrics = calculate_open_loop_metrics(np.array(calculated_steering), true_steering, fps=30)
    print(f"reference distance: {reference_distance}, num_waypoints: {num_waypoints}, "
          f"use_vehicle_pos: {use_vehicle_pos}, loss: {open_loop_metrics['mae']}")

    return {
        'loss': open_loop_metrics['mae'],
        'status': STATUS_OK,
        'params': {'num_waypoints': num_waypoints, 'reference_distance': reference_distance, 'use_vehicle_pos': use_vehicle_pos}
    }


def load_model(model_path):
    model = PilotNetConditional(n_input_channels=3, n_outputs=20, n_branches=3)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    args = parse_arguments()
    optimize_lidar_crop(args)
