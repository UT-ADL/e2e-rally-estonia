from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from tqdm.auto import tqdm

import wandb
from network import PilotNet


class Trainer:

    def __init__(self, model_name, target_name="steering_angle", wandb_logging=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_name = target_name
        datetime_prefix = datetime.today().strftime('%Y%m%d%H%M%S')
        self.save_dir = Path("models") / f"{datetime_prefix}_{model_name}"
        self.save_dir.mkdir(parents=True, exist_ok=False)
        self.wandb_logging = wandb_logging

    def load_model(self, model_path):
        model = PilotNet()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        return model

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

        best_model = self.load_model(self.save_dir / "best.pt")
        metrics = self.calculate_open_loop_metrics(best_model, valid_loader, fps=30)
        print(metrics)
        if self.wandb_logging:
            wandb.log(metrics)

        return best_valid_loss

    def save_models(self, model, valid_loader):
        torch.save(model.state_dict(), self.save_dir / "last.pt")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/last.pt")
            wandb.save(f"{self.save_dir}/best.pt")

        self.save_onnx(valid_loader)

    def save_onnx(self, valid_loader):
        best_model = PilotNet()
        best_model.load_state_dict(torch.load(f"{self.save_dir}/best.pt"))
        best_model.to(self.device)

        data = iter(valid_loader).next()
        sample_inputs = data['image'].to(self.device)
        torch.onnx.export(best_model, sample_inputs, f"{self.save_dir}/best.onnx")

        onnx.checker.check_model(f"{self.save_dir}/best.onnx")

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
            progress_bar.set_description(f'train loss: {(running_loss / ( i +1)):.4f}')

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
            'MAE': mae,
            'RMSE': rmse,
            'Max': max,
            'Whiteness': whiteness,
            'Expert whiteness': expert_whiteness
        }
