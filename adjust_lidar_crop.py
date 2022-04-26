import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloading.ouster import OusterNormalize, OusterCrop, OusterDataset
from metrics.metrics import calculate_open_loop_metrics
from pilotnet import PilotNet
from trainer import PilotNetTrainer

from hyperopt import hp, fmin, tpe

"""
Script for finding best crop location for lidar images. Uses given model and dataset to find crop with lowest MAE.

Usage:
    ./adjust_lidar_crop.py --dataset-path <path-to-dataset> --model-path <path-to-pytorch-model>
"""


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--dataset-path',
        default='/home/romet/data/rally_estonia-temp/2022-04-25-16-59-02',
        required=False,
        help='Path to dataset used for optimizing lidar crop'
    )

    argparser.add_argument(
        '--model-path',
        default='models/lidar-camera-paper/lidar-with-test-track.pt',
        required=False,
        help='Path to PyTorch model used for optimizing lidar crop'
    )

    return argparser.parse_args()


def optimize_lidar_crop(args):
    space = {
        'xmin': hp.uniformint('xmin', 450, 550),
        'ymin': hp.uniformint('ymin', 34, 60),
        'dataset_path': args.dataset_path,
        'model_path': args.model_path
    }

    best = fmin(optimize_fun, space, algo=tpe.suggest, max_evals=100)

    print(best)


def optimize_fun(opt_args):
    model = load_model(opt_args['model_path'])
    dataset = OusterDataset([Path(opt_args['dataset_path'])],
                            transform=transforms.Compose([OusterCrop(xmin=opt_args['xmin'], ymin=opt_args['ymin']),
                                                          OusterNormalize()]))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    trainer = PilotNetTrainer()

    predicted_steering = trainer.predict(model, dataloader)
    true_steering = dataset.frames.steering_angle.to_numpy()

    open_loop_metrics = calculate_open_loop_metrics(predicted_steering, true_steering, fps=10)
    return open_loop_metrics['mae']


def load_model(model_path):
    model = PilotNet(3)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    args = parse_arguments()
    optimize_lidar_crop(args)
