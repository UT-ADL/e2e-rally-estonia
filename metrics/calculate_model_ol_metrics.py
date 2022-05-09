import argparse
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
    nvidia_spring_ds = NvidiaSpringDataset(root_path)
    results["nvidia-v1-spring"] = calculate_metrics(load_model("nvidia-v1"), nvidia_spring_ds, fps=30)
    results["nvidia-v2-spring"] = calculate_metrics(load_model("nvidia-v2"), nvidia_spring_ds, fps=30)
    results["nvidia-v3-spring"] = calculate_metrics(load_model("nvidia-v3"), nvidia_spring_ds, fps=30)
    results["nvidia-in-train-spring"] = calculate_metrics(load_model("nvidia-with-test-track"), nvidia_spring_ds,
                                                          fps=30)

    nvidia_autumn_ds = NvidiaAutumnDataset(root_path)
    results["nvidia-v1-autumn"] = calculate_metrics(load_model("nvidia-v1"), nvidia_autumn_ds, fps=30)
    results["nvidia-v2-autumn"] = calculate_metrics(load_model("nvidia-v2"), nvidia_autumn_ds, fps=30)
    results["nvidia-v3-autumn"] = calculate_metrics(load_model("nvidia-v3"), nvidia_autumn_ds, fps=30)
    results["nvidia-in-train-autumn"] = calculate_metrics(load_model("nvidia-with-test-track"), nvidia_autumn_ds,
                                                          fps=30)

    ouster_spring_ds = OusterSpringDataset(root_path)
    results["lidar-v3-spring"] = calculate_metrics(load_model("lidar-v3"), ouster_spring_ds, fps=10)
    results["lidar-v4-spring"] = calculate_metrics(load_model("lidar-v4"), ouster_spring_ds, fps=10)
    results["lidar-v5-spring"] = calculate_metrics(load_model("lidar-v5"), ouster_spring_ds, fps=10)
    results["lidar-in-train-spring"] = calculate_metrics(load_model("lidar-with-test-track"), ouster_spring_ds,
                                                         fps=10)
    intensity_spring_ds = OusterSpringDataset(root_path, channel="intensity")
    results["lidar-intensity-spring"] = calculate_metrics(load_model("lidar-intensity", n_input_channels=1),
                                                          intensity_spring_ds, fps=10)

    ouster_winter_ds = OusterWinterDataset(root_path)
    results["lidar-v3-winter"] = calculate_metrics(load_model("lidar-v3"), ouster_winter_ds, fps=10)
    results["lidar-v4-winter"] = calculate_metrics(load_model("lidar-v4"), ouster_winter_ds, fps=10)
    results["lidar-v5-winter"] = calculate_metrics(load_model("lidar-v5"), ouster_winter_ds, fps=10)
    results["lidar-in-train-winter"] = calculate_metrics(load_model("lidar-with-test-track"), ouster_winter_ds,
                                                         fps=10)
    intensity_winter_ds = OusterWinterDataset(root_path, channel="intensity")
    results["lidar-intensity-winter"] = calculate_metrics(load_model("lidar-intensity", n_input_channels=1),
                                                          intensity_winter_ds, fps=10)

    ouster_autumn_ds = OusterAutumnDataset(root_path)
    results["lidar-v3-autumn"] = calculate_metrics(load_model("lidar-v3"), ouster_autumn_ds, fps=10)
    results["lidar-v4-autumn"] = calculate_metrics(load_model("lidar-v4"), ouster_autumn_ds, fps=10)
    results["lidar-v5-autumn"] = calculate_metrics(load_model("lidar-v5"), ouster_autumn_ds, fps=10)
    results["lidar-in-train-autumn"] = calculate_metrics(load_model("lidar-with-test-track"), ouster_autumn_ds,
                                                         fps=10)
    intensity_autumn_ds = OusterAutumnDataset(root_path, channel="intensity")
    results["lidar-intensity-autumn"] = calculate_metrics(load_model("lidar-intensity", n_input_channels=1),
                                                          intensity_autumn_ds, fps=10)

    print(results)


def calculate_metrics(model, dataset, fps):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True,
                            persistent_workers=True)

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


class OusterSpringDataset(OusterDataset):
    def __init__(self, root_path, channel=None):
        data_paths = [
            root_path / "2022-05-04-10-54-24_e2e_elva_seasonal_val_set_forw",
            root_path / "2022-05-04-11-01-40_e2e_elva_seasonal_val_set_back"
        ]

        tr = transforms.Compose([OusterCrop(xmin=516, ymin=46), OusterNormalize()])
        super().__init__(data_paths, tr, channel=channel)


class OusterAutumnDataset(OusterDataset):
    def __init__(self, root_path, channel=None):
        data_paths = [
            {'path': root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva", 'start': 3080, 'end': 7708},
            {'path': root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back", 'start': 3173, 'end': 7900}
        ]

        tr = transforms.Compose([OusterCrop(), OusterNormalize()])
        super().__init__(data_paths, tr, channel=channel)


class OusterWinterDataset(OusterDataset):
    def __init__(self, root_path, channel=None):
        data_paths = [
            {'path': root_path / "2022-01-28-14-47-23_e2e_rec_elva_forward", 'start': 2360, 'end': 6940},
            {'path': root_path / "2022-01-28-15-09-01_e2e_rec_elva_backward", 'start': 3420, 'end': 8360}
        ]

        tr = transforms.Compose([OusterCrop(), OusterNormalize()])
        super().__init__(data_paths, tr, channel=channel)


class NvidiaSpringDataset(NvidiaDataset):
    def __init__(self, root_path):
        data_paths = [
            root_path / "2022-05-04-10-54-24_e2e_elva_seasonal_val_set_forw",
            root_path / "2022-05-04-11-01-40_e2e_elva_seasonal_val_set_back"
        ]

        tr = transforms.Compose([NvidiaCropWide(), Normalize()])
        super().__init__(data_paths, tr, metadata_file="nvidia_frames.csv", color_space="rgb")


class NvidiaAutumnDataset(NvidiaDataset):
    def __init__(self, root_path):
        data_paths = [
            {'path': root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva", 'start': 9240, 'end': 23125},
            {'path': root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back", 'start': 9520, 'end': 23700}
        ]

        tr = transforms.Compose([NvidiaCropWide(), Normalize()])
        super().__init__(data_paths, tr, metadata_file="nvidia_frames.csv", color_space="bgr")


if __name__ == "__main__":
    main()

