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


def train_model(model_name, dataset_folder, input_modality, wandb_logging, max_epochs, patience, learning_rate,
                weight_decay,
                filter_blinker_turns):
    train_loader, valid_loader = load_data(dataset_folder, input_modality, filter_blinker_turns)

    print(f"Training model {model_name}")
    model = PilotNet()
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=weight_decay, amsgrad=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    trainer = Trainer(model_name, wandb_logging=wandb_logging)
    trainer.train(model, train_loader, valid_loader, optimizer, criterion, max_epochs, patience)


def load_data(dataset_folder, input_modality, filter_blinker_turns):
    print(f"Reading {input_modality} data from {dataset_folder}")
    dataset_path = Path(dataset_folder)
    if input_modality == "nvidia-camera":
        trainset = NvidiaTrainDataset(dataset_path, filter_turns=filter_blinker_turns)
        validset = NvidiaValidationDataset(dataset_path, filter_turns=filter_blinker_turns)
    elif input_modality == "ouster-lidar":
        trainset = OusterTrainDataset(dataset_path, filter_turns=filter_blinker_turns)
        validset = OusterValidationDataset(dataset_path, filter_turns=filter_blinker_turns)
    else:
        print("Uknown input modality")
        sys.exit()

    print(f"Training data has {len(trainset.frames)} frames")
    print(f"Validation data has {len(validset.frames)} frames")

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,
                                               num_workers=60, pin_memory=True, persistent_workers=True)

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False,
                                               num_workers=32, pin_memory=True, persistent_workers=True)

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
        '--dataset_folder',
        default="/media/romet/data2/datasets/rally-estonia/dataset",
        help='Root path to the dataset.'
    )

    argparser.add_argument(
        '--wandb-logging',
        default=False,
        help='Log training information using W&B.'
    )

    argparser.add_argument(
        '--max-epochs',
        default=100,
        help="Maximium number of epochs to train"
    )

    argparser.add_argument(
        '--patience',
        default=10,
        help="Number of epochs to train without improvement in validation loss. Used for early stopping."
    )

    argparser.add_argument(
        '--learning_rate',
        default=1e-3,
        help="Learning rate used in training."
    )

    argparser.add_argument(
        '--weight-decay',
        default=1e-02,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--filter-blinker-turns',
        default=False,
        help='When true, turns with blinker (left or right) on will be removed from training and validation data.'
    )

    args = argparser.parse_args()
    model_name = args.model_name
    input_modality = args.input_modality
    dataset_folder = args.dataset_folder
    wandb_logging = args.wandb_logging
    max_epochs = args.max_epochs
    patience = args.patience
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    filter_blinker_turns = args.filter_blinker_turns

    train_model(model_name, dataset_folder, input_modality, wandb_logging, max_epochs, patience, learning_rate,
                weight_decay,
                filter_blinker_turns)
