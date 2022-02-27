import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from dataloading.nvidia import NvidiaTrainDataset, NvidiaValidationDataset, NvidiaWinterTrainDataset, \
    NvidiaWinterValidationDataset, NvidiaAllTrainDataset, NvidiaAllValidationDataset, AugmentationConfig
from dataloading.ouster import OusterTrainDataset, OusterValidationDataset
from network import PilotNet
from trainer import Trainer


class TrainingConfig:
    def __init__(self, dataset_folder, input_modality, lidar_channel, output_modality, conditional_learning,
                 learning_rate, weight_decay, patience, max_epochs, batch_size, batch_sampler, num_workers,
                 wandb_project):
        self.dataset_folder = dataset_folder
        self.input_modality = input_modality
        self.lidar_channel = lidar_channel
        self.output_modality = output_modality
        self.conditional_learning = conditional_learning
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.wandb_project = wandb_project

        self.n_input_channels = 1 if self.lidar_channel else 3
        self.n_outputs = 10 if self.output_modality == "waypoints" else 1
        self.n_branches = 3 if self.conditional_learning else 1
        self.fps = 10 if self.input_modality == "ouster-lidar" else 30

def train_model(model_name, train_conf, augment_conf):

    print(f"Training model {model_name}, wandb_project={train_conf.wandb_project}")

    train_loader, valid_loader = load_data(train_conf, augment_conf)

    model = PilotNet(train_conf.n_input_channels, train_conf.n_outputs, train_conf.n_branches)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_conf.learning_rate, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=train_conf.weight_decay, amsgrad=False)

    # todo: move this to trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    trainer = Trainer(model_name, train_conf.output_modality, train_conf.n_branches, train_conf.wandb_project)
    trainer.train(model, train_loader, valid_loader, optimizer, criterion,
                  train_conf.max_epochs, train_conf.patience, train_conf.fps)


def load_data(train_conf, augment_conf):
    print(f"Reading {train_conf.input_modality} data from {train_conf.dataset_folder}, "
          f"lidar_channel={train_conf.lidar_channel}, output_modality={train_conf.output_modality}")

    dataset_path = Path(train_conf.dataset_folder)
    if train_conf.input_modality == "nvidia-camera":
        trainset = NvidiaTrainDataset(dataset_path, train_conf.output_modality, train_conf.n_branches)
        validset = NvidiaValidationDataset(dataset_path, train_conf.output_modality, train_conf.n_branches)
    elif train_conf.input_modality == "nvidia-camera-winter":
        trainset = NvidiaWinterTrainDataset(dataset_path, train_conf.output_modality,
                                            train_conf.n_branches, augment_conf)
        validset = NvidiaWinterValidationDataset(dataset_path, train_conf.output_modality, train_conf.n_branches)
    elif train_conf.input_modality == "nvidia-camera-all":
        trainset = NvidiaAllTrainDataset(dataset_path, train_conf.output_modality, train_conf.n_branches)
        validset = NvidiaAllValidationDataset(dataset_path, train_conf.output_modality, train_conf.n_branches)
    elif train_conf.input_modality == "ouster-lidar":
        trainset = OusterTrainDataset(dataset_path, train_conf.output_modality)
        validset = OusterValidationDataset(dataset_path, train_conf.output_modality)
    else:
        print(f"Uknown input modality {train_conf.input_modality}")
        sys.exit()

    print(f"Training data has {len(trainset.frames)} frames")
    print(f"Validation data has {len(validset.frames)} frames")
    print(f"Creating {train_conf.num_workers} workers with batch size {train_conf.batch_size} using {train_conf.batch_sampler} sampler.")

    if train_conf.batch_sampler == 'weighted':
        weights = calculate_weights(trainset.frames)
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=False,
                                                   sampler=sampler, num_workers=train_conf.num_workers,
                                                   pin_memory=True, persistent_workers=True)
    elif train_conf.batch_sampler == 'random':
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=True,
                                                   num_workers=train_conf.num_workers, pin_memory=True,
                                             persistent_workers=True)
    else:
        print(f"Unknown batch sampler {train_conf.batch_sampler}")
        sys.exit()

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=train_conf.batch_size, shuffle=False,
                                               num_workers=train_conf.num_workers, pin_memory=True,
                                               persistent_workers=True)

    return train_loader, valid_loader


def calculate_weights(df):
    optimized_bins = np.array([-8.81234893e+00, -2.78245811e+00, -1.02905812e+00, -4.43559368e-01,
                               -1.64549582e-01, 6.90239861e-03, 1.69872354e-01, 4.35963640e-01,
                               9.63913148e-01, 2.70831896e+00, 8.57767303e+00])

    bin_ranges = pd.cut(df["steering_angle"], optimized_bins, labels=np.arange(1, 11))
    df["bins"] = bin_ranges
    counts = bin_ranges.value_counts(sort=False)
    widths = np.diff(optimized_bins)
    weights = (widths / counts) * sum(counts) / sum(widths)

    weights_df = pd.DataFrame(data=weights)
    weights_df.reset_index(inplace=True)
    weights_df.columns = ["bins", "weight"]
    weights_df.set_index('bins', inplace=True)
    df = df.join(weights_df, on="bins")
    return df["weight"].to_numpy()


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
        choices=['nvidia-camera', 'nvidia-camera-winter', 'nvidia-camera-all', 'ouster-lidar'],
    )

    argparser.add_argument(
        '--lidar-channel',
        required=False,
        choices=['ambience', 'intensity', 'range'],
        help="Lidar channels to use for training. Combined image is used if not provided. "
             "Only applies to 'ouster-lidar' modality."
    )

    argparser.add_argument(
        '--output-modality',
        required=False,
        default="steering_angle",
        choices=["steering_angle", "waypoints"],
        help="Choice of output modalities to train model with."
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
        '--batch-sampler',
        required=False,
        choices=['weighted', 'random'],
        default='weighted',
        help='Sampler used for creating batches for training.'
    )

    argparser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--conditional-learning',
        default=False,
        action='store_true',
        help="When true, network is trained with conditional branches using turn blinkers."
    )

    argparser.add_argument(
        '--aug-color-prob',
        type=float,
        default=0.0,
        help='Probability of augmenting input image color by changing brightness, saturation and contrast.'
    )

    argparser.add_argument(
        '--aug-noise-prob',
        type=float,
        default=0.0,
        help='Probability of augmenting input image with noise.'
    )

    argparser.add_argument(
        '--aug-blur-prob',
        type=float,
        default=0.0,
        help='Probability of augmenting input image by blurring it.'
    )

    args = argparser.parse_args()
    train_config = TrainingConfig(args.dataset_folder, args.input_modality, args.lidar_channel,
                                  args.output_modality, args.conditional_learning, args.learning_rate,
                                  args.weight_decay, args.patience, args.max_epochs,
                                  args.batch_size, args.batch_sampler, args.num_workers, args.wandb_project)
    aug_config = AugmentationConfig(args.aug_color_prob, args.aug_noise_prob, args.aug_blur_prob)
    train_model(args.model_name, train_config, aug_config)
