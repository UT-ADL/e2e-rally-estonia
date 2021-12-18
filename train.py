import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from dataloading.nvidia import NvidiaTrainDataset, NvidiaValidationDataset
from dataloading.ouster import OusterTrainDataset, OusterValidationDataset
from network import PilotNet
from trainer import Trainer


def train_model(model_name, dataset_folder, input_modality, lidar_channel, target_name,
                wandb_project, max_epochs, patience,
                learning_rate, weight_decay, filter_blinker_turns, batch_size, num_workers):
    train_loader, valid_loader = load_data(dataset_folder, input_modality, lidar_channel, filter_blinker_turns, batch_size, num_workers)

    print(f"Training model {model_name}, wandb_project={wandb_project}")
    if lidar_channel:
        model = PilotNet(n_input_channels=1)
    else:
        model = PilotNet(n_input_channels=3)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=weight_decay, amsgrad=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    if input_modality == "ouster-lidar":
        fps = 10
    else:
        fps = 30

    print(f"Training for target={target_name}")
    trainer = Trainer(model_name, wandb_project=wandb_project, target_name=target_name)
    trainer.train(model, train_loader, valid_loader, optimizer, criterion, max_epochs, patience, fps)


def load_data(dataset_folder, input_modality, lidar_channel, filter_blinker_turns, batch_size, num_workers):
    print(f"Reading {input_modality} data from {dataset_folder}, lidar_channel={lidar_channel}, filter_blinker_turns={filter_blinker_turns}")
    dataset_path = Path(dataset_folder)
    if input_modality == "nvidia-camera":
        trainset = NvidiaTrainDataset(dataset_path, filter_turns=filter_blinker_turns)
        validset = NvidiaValidationDataset(dataset_path, filter_turns=filter_blinker_turns)
    elif input_modality == "ouster-lidar":
        trainset = OusterTrainDataset(dataset_path, filter_turns=filter_blinker_turns, channel=lidar_channel)
        validset = OusterValidationDataset(dataset_path, filter_turns=filter_blinker_turns, channel=lidar_channel)
    else:
        print("Uknown input modality")
        sys.exit()

    print(f"Training data has {len(trainset.frames)} frames")
    print(f"Validation data has {len(validset.frames)} frames")
    print(f"Creating {num_workers} workers with batch size {batch_size}")

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, valid_loader


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model used for saving model and logging in W&B.'
    )

    argparser.add_argument(
        '--input-modality',
        required=True,
        choices=['nvidia-camera', 'ouster-lidar'],
    )

    argparser.add_argument(
        '--lidar-channel',
        required=False,
        choices=['ambience', 'intensity', 'range'],
        help="Lidar channels to use for training. Combined image is used if not provided. "
             "Only applies to 'ouster-lidar' modality."
    )

    argparser.add_argument(
        '--target-name',
        required=False,
        default="steering_angle",
        choices=["steering_angle", "x_1_offset", "y_1_offset", "waypoints"],
    )

    argparser.add_argument(
        '--dataset-folder',
        default="/home/romet/data2/datasets/rally-estonia/dataset-small",
        help='Root path to the dataset.'
    )

    argparser.add_argument(
        '--wandb-project',
        required=False,
        help='W&B project name to use for metrics. Wandb logging is disabled when no project name is provided.'
    )

    argparser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help="Maximium number of epochs to train"
    )

    argparser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Number of epochs to train without improvement in validation loss. Used for early stopping."
    )

    argparser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help="Learning rate used in training."
    )

    argparser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-02,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--filter-blinker-turns',
        default=False,
        action='store_true',
        help='When true, turns with blinker (left or right) on will be removed from training and validation data.'
    )

    argparser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Weight decay used in training.'
    )

    args = argparser.parse_args()
    train_model(args.model_name,
                args.dataset_folder,
                args.input_modality,
                args.lidar_channel,
                args.target_name,
                args.wandb_project,
                args.max_epochs,
                args.patience,
                args.learning_rate,
                args.weight_decay,
                args.filter_blinker_turns,
                args.batch_size,
                args.num_workers)
