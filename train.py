import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch import Tensor
from torch.nn import L1Loss, MSELoss
from torch.utils.data import ConcatDataset, WeightedRandomSampler
#from torchsummary import summary
from dataloading.camera import Camera, TurnSignal
from dataloading.nvidia import NvidiaTrainDataset, NvidiaValidationDataset, NvidiaWinterTrainDataset, \
    NvidiaWinterValidationDataset, AugmentationConfig
from dataloading.ouster import OusterTrainDataset, OusterValidationDataset
from efficient_net import effnetv2_s
from pilotnet import PilotNetConditional, PilotnetControl
from trainer import ControlTrainer, ConditionalTrainer, PilotNetTrainer


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model used for saving model and logging in W&B.'
    )

    argparser.add_argument(
        '--model-type',
        required=True,
        choices=['pilotnet', 'pilotnet-conditional', 'pilotnet-control', 'efficientnet'],
        help='Defines which model will be trained.'
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
        '--camera-name',
        required=False,
        default="front_wide",
        choices=['front_wide', 'left', 'right', 'all'],
        help="Camera to use for training. Only applies to 'nvidia-camera' modality."
    )

    argparser.add_argument(
        '--output-modality',
        required=False,
        default="steering_angle",
        choices=["steering_angle", "waypoints"],
        help="Choice of output modalities to train model with."
    )

    argparser.add_argument(
        '--num-waypoints',
        type=int,
        default=10,
        help="Number of waypoints used for trajectory."
    )

    argparser.add_argument(
        '--dataset-folder',
        default="/home/romet/data2/datasets/rally-estonia/dataset-new-small/summer2021",
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
        '--learning-rate-patience',
        type=int,
        default=10,
        help="Num of epochs after with learning rate will be retuced by factor of 0.1."
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
        '--epoch-size',
        type=int,
        default=262_144,  # default batch-size^9
        help='Number of training samples in one epochs.'
    )

    argparser.add_argument(
        '--batch-sampler',
        required=False,
        choices=['random', 'weighted', 'camera-weighted', 'turn-weighted'],
        default='random',
        help='Sampler used for creating batches for training.'
    )

    argparser.add_argument(
        '--side-camera-weight',
        required=False,
        type=float,
        default=0.33,
        help="Ratio of side camera images used for augmentation during training. "
             "Default uses same ratio of all camera images."
    )

    argparser.add_argument(
        '--turn-sampling-weight',
        required=False,
        type=float,
        default=0.33,
        help="Weight for frames with turn signal (left or right) when sampling batch during training. "
             "Default uses same ratio as in dataset."
    )

    argparser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of workers used for data loading.'
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

    argparser.add_argument(
        '--loss',
        required=False,
        choices=['mse', 'mae', 'mse-weighted', 'mae-weighted'],
        default='mae',
        help='Loss function used for training.'
    )

    argparser.add_argument(
        "--loss-discount-rate",
        required=False,
        type=float,
        default=0.8,
        help="Used to discount waypoints in trajectory as nearer waypoints are more important. "
             "Only used with weighted loss."
    )

    argparser.add_argument(
        '--pretrained-model',
        required=False,
        help='Pretrained model used to initialize weights.'
    )

    argparser.add_argument(
        '--metadata-file',
        required=False,
        default="nvidia_frames.csv",
        help='Dataset metadata file used for reading metadata like steering angles etc.'
    )

    return argparser.parse_args()


class WeighedL1Loss(L1Loss):
    def __init__(self, weights):
        super().__init__(reduction='none')
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super().forward(input, target)
        return (loss * self.weights).mean()


class WeighedMSELoss(MSELoss):
    def __init__(self, weights):
        super().__init__(reduction='none')
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super().forward(input, target)
        return (loss * self.weights).mean()


class TrainingConfig:
    def __init__(self, args):
        self.model_type = args.model_type
        self.dataset_folder = args.dataset_folder
        self.input_modality = args.input_modality
        self.lidar_channel = args.lidar_channel
        self.camera_name = args.camera_name
        self.output_modality = args.output_modality
        self.n_waypoints = args.num_waypoints
        self.learning_rate = args.learning_rate
        self.learning_rate_patience = args.learning_rate_patience
        self.weight_decay = args.weight_decay
        self.patience = args.patience
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.epoch_size = args.epoch_size
        self.batch_sampler = args.batch_sampler
        self.side_camera_weight = args.side_camera_weight
        self.turn_sampling_weight = args.turn_sampling_weight
        self.num_workers = args.num_workers
        self.wandb_project = args.wandb_project
        self.loss = args.loss
        self.loss_discount_rate = args.loss_discount_rate
        self.metadata_file = args.metadata_file

        self.n_input_channels = 1 if self.lidar_channel else 3
        if self.output_modality == "waypoints":
            self.n_outputs = 2 * self.n_waypoints
        elif self.output_modality == "steering_angle":
            self.n_outputs = 1
        else:
            print(f"Uknown output modality {self.output_modality}")
            sys.exit()

        self.n_branches = 3 if self.model_type == "pilotnet-conditional" else 1
        self.fps = 10 if self.input_modality == "ouster-lidar" else 30
        self.pretrained_model = args.pretrained_model


def train_model(model_name, train_conf, augment_conf):

    print(f"Training model {model_name}, wandb_project={train_conf.wandb_project}")
    if train_conf.wandb_project:
        wandb.init(project=train_conf.wandb_project)
    print('train_conf: ', train_conf.__dict__)
    print('augment_conf: ', augment_conf.__dict__)

    train_loader, valid_loader = load_data(train_conf, augment_conf)

    # TODO: model and trainer should be combined
    if train_conf.model_type == "pilotnet-control":
        model = PilotnetControl(train_conf.n_input_channels, train_conf.n_outputs)
        trainer = ControlTrainer(model_name, train_conf.output_modality, train_conf.n_branches,
                                 train_conf.wandb_project)
    elif train_conf.model_type == "pilotnet-conditional":
        model = PilotNetConditional(train_conf.n_input_channels, train_conf.n_outputs, train_conf.n_branches)
        trainer = ConditionalTrainer(model_name, train_conf.output_modality, train_conf.n_branches,
                                     train_conf.wandb_project)
    elif train_conf.model_type == "efficientnet":
        model = effnetv2_s()
        trainer = PilotNetTrainer(model_name, target_name="steering_angle")
    else:
        model = PilotNetConditional(train_conf.n_input_channels, train_conf.n_outputs, train_conf.n_branches)
        trainer = ConditionalTrainer(model_name, train_conf.output_modality, train_conf.n_branches,
                                     train_conf.wandb_project)

    #summary(model, input_size=(3, 660, 172), device="cpu")
    #summary(model, input_size=(3, 264, 68), device="cpu")

    if train_conf.pretrained_model:
        print(f"Initializing weights with pretrained model: {train_conf.pretrained_model}")
        pretrained_model = load_model(train_conf.pretrained_model,
                                      train_conf.n_input_channels,
                                      train_conf.n_outputs,
                                      n_branches=1)
        model.features.load_state_dict(pretrained_model.features.state_dict())
        for i in range(train_conf.n_branches):
            model.conditional_branches[i].load_state_dict(pretrained_model.conditional_branches[0].state_dict())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.FloatTensor([(train_conf.loss_discount_rate ** i, train_conf.loss_discount_rate ** i)
                                 for i in range(train_conf.n_waypoints)]).to(device)
    weights = weights.flatten()
    if train_conf.n_branches > 1:  # todo: this is conditional learning specific and should be handled there
        weights = torch.cat(tuple(weights for i in range(train_conf.n_branches)), 0)

    if train_conf.loss == "mse":
        criterion = MSELoss()
    elif train_conf.loss == "mae":
        criterion = L1Loss()
    elif train_conf.loss == "mse-weighted":
        criterion = WeighedMSELoss(weights)
    elif train_conf.loss == "mae-weighted":
        criterion = WeighedL1Loss(weights)
    else:
        print(f"Uknown loss function {train_conf.loss}")
        sys.exit()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_conf.learning_rate, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=train_conf.weight_decay, amsgrad=False)

    # todo: move this to trainer
    model = model.to(device)
    criterion = criterion.to(device)

    trainer.train(model, train_loader, valid_loader, optimizer, criterion,
                  train_conf.max_epochs, train_conf.patience, train_conf.learning_rate_patience, train_conf.fps)


def load_model(model_name, n_input_channels=3, n_outputs=1, n_branches=1):
    model = PilotNetConditional(n_input_channels=n_input_channels, n_outputs=n_outputs, n_branches=n_branches)
    model.load_state_dict(torch.load(f"models/{model_name}/best.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model


def load_data(train_conf, augment_conf):
    print(f"Reading {train_conf.input_modality} data from {train_conf.dataset_folder}, "
          f"camera name={train_conf.camera_name}, lidar_channel={train_conf.lidar_channel}, "
          f"output_modality={train_conf.output_modality}")

    dataset_path = Path(train_conf.dataset_folder)
    if train_conf.input_modality == "nvidia-camera":
        trainset = NvidiaTrainDataset(dataset_path, train_conf.output_modality,
                                      train_conf.n_branches, n_waypoints=train_conf.n_waypoints,
                                      camera=train_conf.camera_name, augment_conf=augment_conf,
                                      metadata_file=train_conf.metadata_file)
        validset = NvidiaValidationDataset(dataset_path, train_conf.output_modality, train_conf.n_branches,
                                           n_waypoints=train_conf.n_waypoints,
                                           metadata_file=train_conf.metadata_file)
    elif train_conf.input_modality == "nvidia-camera-winter":
        trainset = NvidiaWinterTrainDataset(dataset_path, train_conf.output_modality,
                                            train_conf.n_branches, n_waypoints=train_conf.n_waypoints,
                                            augment_conf=augment_conf)
        validset = NvidiaWinterValidationDataset(dataset_path, train_conf.output_modality,
                                                 train_conf.n_branches, n_waypoints=train_conf.n_waypoints)
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
        sampler = WeightedRandomSampler(weights, num_samples=train_conf.epoch_size, replacement=True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=False,
                                                   sampler=sampler, num_workers=train_conf.num_workers,
                                                   pin_memory=True, persistent_workers=True)
    elif train_conf.batch_sampler == 'random':
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=True,
                                                   num_workers=train_conf.num_workers, pin_memory=True,
                                                   persistent_workers=True)

    elif train_conf.batch_sampler == 'camera-weighted':
        center_camera_weight = (1-2*train_conf.side_camera_weight)
        weights = [center_camera_weight if camera_type == Camera.FRONT_WIDE.value
                   else train_conf.side_camera_weight
                   for camera_type in trainset.frames["camera_type"].to_numpy()]
        sampler = WeightedRandomSampler(weights, num_samples=train_conf.epoch_size, replacement=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=False,
                                                  sampler=sampler, num_workers=train_conf.num_workers,
                                                  pin_memory=True, persistent_workers=True)

    elif train_conf.batch_sampler == 'turn-weighted':
        without_turn_weight = (1-2*train_conf.turn_sampling_weight)
        weights = [without_turn_weight if turn_signal == TurnSignal.STRAIGHT.value
                   else train_conf.turn_sampling_weight
                   for turn_signal in trainset.frames["turn_signal"].to_numpy()]
        sampler = WeightedRandomSampler(weights, num_samples=train_conf.epoch_size, replacement=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=False,
                                                  sampler=sampler, num_workers=train_conf.num_workers,
                                                  pin_memory=True, persistent_workers=True)

    else:
        print(f"Unknown batch sampler {train_conf.batch_sampler}")
        sys.exit()

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=train_conf.batch_size, shuffle=False,
                                               num_workers=train_conf.num_workers, pin_memory=True,
                                               persistent_workers=True)

    return train_loader, valid_loader


def calculate_weights(df):
    # optimized_bins = np.array([df["steering_angle"].min() - 0.00001, -2.78245811e+00, -1.02905812e+00, -4.43559368e-01,
    #                            -1.64549582e-01, 6.90239861e-03, 1.69872354e-01, 4.35963640e-01,
    #                            9.63913148e-01, 2.70831896e+00, df["steering_angle"].max() + 0.00001])

    optimized_bins = np.array([df["steering_angle"].min() - 0.00001,
                               -1.121, -0.176, 0.206, 1.088,
                               df["steering_angle"].max() + 0.00001])

    bin_ranges = pd.cut(df["steering_angle"], optimized_bins, labels=np.arange(1, 6))
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
    args = parse_arguments()
    train_config = TrainingConfig(args)
    aug_config = AugmentationConfig(args.aug_color_prob, args.aug_noise_prob, args.aug_blur_prob)
    train_model(args.model_name, train_config, aug_config)
