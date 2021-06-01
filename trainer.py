from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from network import PilotNet
import wandb


class Trainer:

    def __init__(self, save_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model_path):
        model = PilotNet()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        return model

    def train(self, model, train_loader, valid_loader, optimizer, criterion, n_epoch, patience=10):
        wandb.init(project="lanefollowing-ut")
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

            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_los": valid_loss})

            if epochs_of_no_improve == patience:
                print(f'Early stopping, on epoch: {epoch + 1}.')
                break

    def train_epoch(self, model, loader, optimizer, criterion, progress_bar):
        running_loss = 0.0

        model.train()

        for i, data in enumerate(loader):
            inputs = data['image'].to(self.device)
            steering_angles = data['steering_angle'].to(self.device)

            optimizer.zero_grad()

            predictions = model(inputs).squeeze(1)
            loss = criterion(predictions, steering_angles)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'train loss: {(running_loss / ( i +1)):.4f}')

        return running_loss / len(loader)

    def predict(self, model, dataset):
        all_predictions = []
        model.eval()

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

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
                steering_angles = data['steering_angle'].to(self.device)

                predictions = model(inputs).squeeze(1)
                loss = criterion(predictions, steering_angles)

                epoch_loss += loss.item()
                progress_bar.update(1)

        total_loss = epoch_loss / len(iterator)
        return total_loss

