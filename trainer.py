from datetime import datetime
from pathlib import Path

import onnx
import torch
import wandb
from tqdm.auto import tqdm

from metrics.metrics import calculate_open_loop_metrics


class Trainer:

    def __init__(self, model_name=None, target_name="steering_angle", n_conditional_branches=1, wandb_project=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_name = target_name
        self.n_conditional_branches = n_conditional_branches
        self.wandb_logging = False

        if wandb_project:
            self.wandb_logging = True
            wandb.init(project="nvidia-e2e-tests")

        if model_name:
            datetime_prefix = datetime.today().strftime('%Y%m%d%H%M%S')
            self.save_dir = Path("models") / f"{datetime_prefix}_{model_name}"
            self.save_dir.mkdir(parents=True, exist_ok=False)

    def force_cpu(self):
        self.device = 'cpu'

    def train(self, model, train_loader, valid_loader, optimizer, criterion, n_epoch, patience=10, fps=30):
        if self.wandb_logging:
            wandb.watch(model, criterion)

        best_valid_loss = float('inf')
        epochs_of_no_improve = 0

        for epoch in range(n_epoch):

            progress_bar = tqdm(total=len(train_loader), smoothing=0)
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, progress_bar, epoch)

            progress_bar.reset(total=len(valid_loader))
            valid_loss = self.evaluate(model, valid_loader, criterion, progress_bar, epoch, train_loss)

            if valid_loss < best_valid_loss:
                progress_bar.set_description(f'*epoch {epoch + 1} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}')
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

        if self.target_name == "steering":
            model.load_state_dict(torch.load(f"{self.save_dir}/best.pt"))
            model.to(self.device)
            self.calculate_metrics(fps, model, valid_loader)
        # todo: calculate metrics for waypoints

        return best_valid_loss

    def calculate_metrics(self, fps, model, valid_loader):
        predictions = self.predict(model, valid_loader)
        true_steering_angles = valid_loader.dataset.frames.steering_angle.to_numpy()
        metrics = calculate_open_loop_metrics(predictions, true_steering_angles, fps=fps)
        print(metrics)
        if self.wandb_logging:
            wandb.log(metrics)

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

    def train_epoch(self, model, loader, optimizer, criterion, progress_bar, epoch):
        running_loss = 0.0

        model.train()

        for i, (data, target_values, condition_mask) in enumerate(loader):
            inputs = data['image'].to(self.device)
            target_values = target_values.to(self.device)
            condition_mask = condition_mask.to(self.device)

            optimizer.zero_grad()

            predictions = model(inputs)
            loss = criterion(predictions*condition_mask, target_values) * self.n_conditional_branches

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'epoch {epoch+1} | train loss: {(running_loss / (i + 1)):.4f}')

        return running_loss / len(loader)

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = data['image'].to(self.device)
                condition_mask = condition_mask.to(self.device)
                predictions = model(inputs)

                masked_predictions = predictions[condition_mask == 1]
                masked_predictions = masked_predictions.reshape(predictions.shape[0], -1)

                all_predictions.extend(masked_predictions.cpu().numpy())
                progress_bar.update(1)

        return all_predictions

    def evaluate(self, model, iterator, criterion, progress_bar, epoch, train_loss):
        epoch_loss = 0.0

        model.eval()

        with torch.no_grad():
            for i, (data, target_values, condition_mask) in enumerate(iterator):
                inputs = data['image'].to(self.device)
                target_values = target_values.to(self.device)
                condition_mask = condition_mask.to(self.device)

                predictions = model(inputs)
                loss = criterion(predictions*condition_mask, target_values) * self.n_conditional_branches

                epoch_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_description(f'epoch {epoch + 1} | train loss: {train_loss:.4f} | valid loss: {(epoch_loss / (i + 1)):.4f}')

        total_loss = epoch_loss / len(iterator)
        return total_loss