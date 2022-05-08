import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloading.nvidia import NvidiaDataset, Normalize, NvidiaCropWide
from dataloading.ouster import OusterCrop, OusterNormalize, OusterDataset
from metrics.metrics import calculate_open_loop_metrics
from pilotnet import PilotNet
from trainer import PilotNetTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path",
                        default="/home/romet/data/datasets/rally-estonia/dataset",
                        help='Path to extracted datasets')
    args = parser.parse_args()
    root_path = Path(args.root_path)

    results = {}
    nvidia_spring_ds = load_dataset(root_path, input_modality="nvidia", season="spring")
    results["nvidia-v1-spring"] = calculate_metrics(load_model("nvidia-v1"), nvidia_spring_ds, fps=30)
    results["nvidia-v2-spring"] = calculate_metrics(load_model("nvidia-v2"), nvidia_spring_ds, fps=30)
    results["nvidia-v3-spring"] = calculate_metrics(load_model("nvidia-v3"), nvidia_spring_ds, fps=30)
    results["nvidia-in-train-spring"] = calculate_metrics(load_model("nvidia-with-test-track"), nvidia_spring_ds,
                                                          fps=30)

    nvidia_autumn_ds = load_dataset(root_path, input_modality="nvidia", season="autumn", color_space="bgr")
    results["nvidia-v1-autumn"] = calculate_metrics(load_model("nvidia-v1"), nvidia_autumn_ds, fps=30)
    results["nvidia-v2-autumn"] = calculate_metrics(load_model("nvidia-v2"), nvidia_autumn_ds, fps=30)
    results["nvidia-v3-autumn"] = calculate_metrics(load_model("nvidia-v3"), nvidia_autumn_ds, fps=30)
    results["nvidia-in-train-autumn"] = calculate_metrics(load_model("nvidia-with-test-track"), nvidia_autumn_ds,
                                                          fps=30)

    ouster_spring_ds = load_dataset(root_path, input_modality="ouster", season="spring")
    results["lidar-v3-spring"] = calculate_metrics(load_model("lidar-v3"), ouster_spring_ds, fps=10)
    results["lidar-v4-spring"] = calculate_metrics(load_model("lidar-v4"), ouster_spring_ds, fps=10)
    results["lidar-v5-spring"] = calculate_metrics(load_model("lidar-v5"), ouster_spring_ds, fps=10)
    results["lidar-in-train-spring"] = calculate_metrics(load_model("lidar-with-test-track"), ouster_spring_ds,
                                                         fps=10)

    # intensity_autumn_ds = load_dataset(root_path, "ouster", channel="intensity", season="autumn")
    # results["lidar-intensity-autumn"] = calculate_metrics(load_model("lidar-intensity", n_input_channels=1), intensity_autumn_ds, 10)
    #
    # intensity_spring_ds = load_dataset(root_path, "ouster", channel="intensity", season="spring")
    # results["lidar-intensity-spring"] = calculate_metrics(load_model("lidar-intensity", n_input_channels=1), intensity_spring_ds, 10)

    print(results)


def calculate_metrics(model, dataloader, fps):
    trainer = PilotNetTrainer()
    steering_predictions = trainer.predict(model, dataloader)
    true_steering_angles = dataloader.dataset.frames.steering_angle.to_numpy()
    metrics = calculate_open_loop_metrics(steering_predictions, true_steering_angles, fps)
    return metrics


def load_model(model_name, n_input_channels=3):
    model = PilotNet(n_input_channels)
    model.load_state_dict(torch.load(f"models/lidar-camera-paper/{model_name}.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def load_dataset(root_path, input_modality, channel=None, season='autumn', color_space='rgb'):
    if season == 'autumn':
        ouster_xmin = 384
        ouster_ymin = 54
        valid_paths = [
            #root_path / "2021-09-24-14-03-45_e2e_rec_ss11_backwards",
            root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva",
            root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back",
            #root_path / "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back",
            #root_path / "2021-10-11-14-50-59_e2e_rec_vahi",
            #root_path / "2021-10-14-13-08-51_e2e_rec_vahi_backwards"
        ]
    elif season == 'winter':
        ouster_xmin = 384
        ouster_ymin = 54
        valid_paths = [
            #root_path / "2022-01-18-12-37-01_e2e_rec_arula_forward",
            #root_path / "2022-01-18-12-47-32_e2e_rec_arula_forward_continue",
            root_path / "2022-01-28-14-47-23_e2e_rec_elva_forward",
            root_path / "2022-01-28-15-09-01_e2e_rec_elva_backward",
            #root_path / "2022-01-25-15-25-15_e2e_rec_vahi_forward",
            #root_path / "2022-01-25-15-34-01_e2e_rec_vahi_backwards",
        ]
    elif season == 'spring':
        ouster_xmin = 516
        ouster_ymin = 46
        valid_paths = [
            root_path / "2022-05-04-10-54-24_e2e_elva_seasonal_val_set_forw",
            root_path / "2022-05-04-11-01-40_e2e_elva_seasonal_val_set_back"
        ]
    else:
        print(f"Unknown season: {season}")
        sys.exit()
    if input_modality == 'nvidia':
        tr = transforms.Compose([NvidiaCropWide(), Normalize()])
        validset = NvidiaDataset(valid_paths, transform=tr, metadata_file="nvidia_frames.csv", color_space=color_space)
    elif input_modality == 'ouster':
        tr = transforms.Compose([OusterCrop(xmin=ouster_xmin, ymin=ouster_ymin), OusterNormalize()])
        validset = OusterDataset(valid_paths, transform=tr, channel=channel)
    else:
        print(f"Unknown modality: {input_modality}")
        sys.exit()

    validloader = DataLoader(validset, batch_size=64, shuffle=False,
                             num_workers=16, pin_memory=True,
                             persistent_workers=True)

    return validloader

if __name__ == "__main__":
    main()

