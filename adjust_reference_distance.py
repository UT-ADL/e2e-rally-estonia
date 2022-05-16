import argparse
from pathlib import Path

import numpy as np
import torch
from hyperopt import hp, fmin, tpe
from torch.utils.data import DataLoader
from tqdm import tqdm

import trajectory
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
    space = {
        'reference_distance': hp.uniform('reference_distance', 1.0, 20.0),
        'num_waypoints': hp.uniformint('num_waypoints', 2, 10),
        'dataset_path': args.dataset_path,
        'model_path': args.model_path
    }

    best = fmin(optimize_fun, space, algo=tpe.suggest, max_evals=args.max_evals, show_progressbar=False)

    print(best)


def optimize_fun(opt_args):
    model = load_model(opt_args['model_path'])
    num_waypoints = opt_args['num_waypoints']
    reference_distance = opt_args['reference_distance']

    dataset = NvidiaValidationDataset(Path(opt_args['dataset_path']), "waypoints", n_branches=3, n_waypoints=10)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8,
                            pin_memory=True, persistent_workers=True)

    trainer = ConditionalTrainer()
    predictions = trainer.predict(model, dataloader)
    predicted_steering = []
    for waypoints in tqdm(predictions, desc="Calculating steering angles"):
        predicted_steering.append(trajectory.calculate_steering_angle(waypoints[:num_waypoints*2], reference_distance))

    true_steering = dataset.frames.steering_angle.to_numpy()
    open_loop_metrics = calculate_open_loop_metrics(np.array(predicted_steering), true_steering, fps=10)
    print(f"reference distance: {reference_distance}, num_waypoints: {num_waypoints} loss: {open_loop_metrics['mae']}")
    return open_loop_metrics['mae']


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
