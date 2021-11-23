from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

import wandb
from network import PilotNet


class Trainer:

    def __init__(self, model_name=None, target_name="steering_angle", wandb_logging=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_name = target_name
        self.wandb_logging = wandb_logging

        if model_name:
            datetime_prefix = datetime.today().strftime('%Y%m%d%H%M%S')
            self.save_dir = Path("models") / f"{datetime_prefix}_{model_name}"
            self.save_dir.mkdir(parents=True, exist_ok=False)

    def force_cpu(self):
        self.device = 'cpu'

    def train(self, model, train_loader, valid_loader, optimizer, criterion, n_epoch, patience=10):
        if self.wandb_logging:
            wandb.init(project="lanefollowing-ut-vahi")
            wandb.watch(model, criterion)

        best_valid_loss = float('inf')
        epochs_of_no_improve = 0

        for epoch in range(n_epoch):

            progress_bar = tqdm(total=len(train_loader), smoothing=0)
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, progress_bar)
            valid_loss = self.evaluate(model, valid_loader, criterion)
            progress_bar.set_description(
                f'epoch: {epoch + 1} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}')

            if valid_loss < best_valid_loss:
                print("Saving best model.")
                best_valid_loss = valid_loss

                torch.save(model.state_dict(), self.save_dir / "best.pt")
                epochs_of_no_improve = 0
            else:
                epochs_of_no_improve += 1

            if self.wandb_logging:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_los": valid_loss})

            if epochs_of_no_improve == patience:
                print(f'Early stopping, on epoch: {epoch + 1}.')
                break

        self.save_models(model, valid_loader)

        model.load_state_dict(torch.load(f"{self.save_dir}/best.pt"))
        model.to(self.device)
        metrics = self.calculate_open_loop_metrics(model, valid_loader, fps=30)
        print(metrics)
        if self.wandb_logging:
            wandb.log(metrics)

        return best_valid_loss

    def save_models(self, model, valid_loader):
        torch.save(model.state_dict(), self.save_dir / "last.pt")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/last.pt")
            wandb.save(f"{self.save_dir}/best.pt")

        self.save_onnx(model, valid_loader)

    def save_onnx(self, model, valid_loader):
        model.load_state_dict(torch.load(f"{self.save_dir}/best.pt"))
        model.to(self.device)

        data = iter(valid_loader).next()
        sample_inputs = data['image'].to(self.device)

        torch.onnx.export(model, sample_inputs, f"{self.save_dir}/best.onnx")
        onnx.checker.check_model(f"{self.save_dir}/best.onnx")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/best.onnx")

        model.load_state_dict(torch.load(f"{self.save_dir}/last.pt"))
        model.to(self.device)

        torch.onnx.export(model, sample_inputs, f"{self.save_dir}/last.onnx")
        onnx.checker.check_model(f"{self.save_dir}/last.onnx")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/last.onnx")

    def train_epoch(self, model, loader, optimizer, criterion, progress_bar):
        running_loss = 0.0

        model.train()

        for i, data in enumerate(loader):
            inputs = data['image'].to(self.device)
            target_values = data[self.target_name].to(self.device)

            optimizer.zero_grad()

            predictions = model(inputs).squeeze(1)
            loss = criterion(predictions, target_values)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'train loss: {(running_loss / (i + 1)):.4f}')

        return running_loss / len(loader)

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            for data in dataloader:
                inputs = data['image'].to(self.device)
                predictions = model(inputs).squeeze(1)
                all_predictions.extend(predictions.cpu().numpy())
                progress_bar.update(1)

        return all_predictions

    def evaluate(self, model, iterator, criterion):
        epoch_loss = 0.0

        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(iterator), smoothing=0)
            for data in iterator:
                inputs = data['image'].to(self.device)
                target_values = data[self.target_name].to(self.device)

                predictions = model(inputs).squeeze(1)
                loss = criterion(predictions, target_values)

                epoch_loss += loss.item()
                progress_bar.update(1)

        total_loss = epoch_loss / len(iterator)
        return total_loss

    def calculate_whiteness(self, steering_angles, fps=30):
        current_angles = steering_angles[:-1]
        next_angles = steering_angles[1:]
        delta_angles = next_angles - current_angles
        whiteness = np.sqrt(((delta_angles * fps) ** 2).mean())
        return whiteness

    def calculate_open_loop_metrics(self, model, dataloader, fps):
        predictions = self.predict(model, dataloader)
        predicted_degrees = np.array(predictions) / np.pi * 180
        true_degrees = dataloader.dataset.frames.steering_angle.to_numpy() / np.pi * 180
        errors = np.abs(true_degrees - predicted_degrees)
        mae = errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        max = errors.max()

        whiteness = self.calculate_whiteness(predicted_degrees, fps)
        expert_whiteness = self.calculate_whiteness(true_degrees, fps)

        return {
            'mae': mae,
            'rmse': rmse,
            'max': max,
            'whiteness': whiteness,
            'expert_whiteness': expert_whiteness
        }

    def calculate_closed_loop_metrics(self, model_dataset, expert_dataset, fps=30, failure_rate_threshold=1.0,
                                      only_autonomous=True):
        model_steering = model_dataset.steering_angles_degrees()
        true_steering = expert_dataset.steering_angles_degrees()

        lat_errors = self.calculate_lateral_errors(model_dataset, expert_dataset, only_autonomous)
        whiteness = self.calculate_whiteness(model_steering, fps)
        expert_whiteness = self.calculate_whiteness(true_steering, fps)

        max = lat_errors.max()
        mae = lat_errors.mean()
        rmse = np.sqrt((lat_errors ** 2).mean())
        failure_rate = len(lat_errors[lat_errors > failure_rate_threshold]) / float(len(lat_errors)) * 100
        interventions = self.calculate_interventions(model_dataset)

        return {
            'mae': mae,
            'rmse': rmse,
            'max': max,
            'failure_rate': failure_rate,
            'interventions': interventions,
            'whiteness': whiteness,
            'expert_whiteness': expert_whiteness,
        }

    def calculate_lateral_errors(self, model_dataset, expert_dataset, only_autonomous):
        model_trajectory_df = model_dataset.frames[["position_x", "position_y", "autonomous"]].rename(
            columns={"position_x": "X", "position_y": "Y"})
        expert_trajectory_df = expert_dataset.frames[["position_x", "position_y", "autonomous"]].rename(
            columns={"position_x": "X", "position_y": "Y"})

        if only_autonomous:
            model_trajectory_df = model_trajectory_df[model_trajectory_df.autonomous].reset_index(drop=True)

        tree = BallTree(expert_trajectory_df.values)
        inds, dists = tree.query_radius(model_trajectory_df.values, r=2, sort_results=True, return_distance=True)
        closest_l = []
        for i, ind in enumerate(inds):
            if len(ind) >= 2:
                closest = pd.DataFrame({
                    'X1': [expert_trajectory_df.iloc[ind[0]].X],
                    'Y1': [expert_trajectory_df.iloc[ind[0]].Y],
                    'X2': [expert_trajectory_df.iloc[ind[1]].X],
                    'Y2': [expert_trajectory_df.iloc[ind[1]].Y]},
                    index=[i])
                closest_l.append(closest)
        closest_df = pd.concat(closest_l)
        f = model_trajectory_df.join(closest_df)
        lat_errors = abs((f.X2 - f.X1) * (f.Y1 - f.Y) - (f.X1 - f.X) * (f.Y2 - f.Y1)) / np.sqrt(
            (f.X2 - f.X1) ** 2 + (f.Y2 - f.Y1) ** 2)
        # lat_errors.dropna(inplace=True)  # Why na-s?

        return lat_errors

    def calculate_interventions(self, driving_dataset):
        frames = driving_dataset.frames
        frames['autonomous_next'] = frames.shift(-1)['autonomous']
        return len(frames[frames.autonomous & (frames.autonomous_next == False)])
