import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl


class PilotNetConditional(nn.Module):

    def __init__(self, n_input_channels=3, n_outputs=1, n_branches=1):
        super(PilotNetConditional, self).__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(n_input_channels, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.conditional_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1664, 100),
                nn.BatchNorm1d(100),
                nn.LeakyReLU(),
                nn.Linear(100, 50),
                nn.BatchNorm1d(50),
                nn.LeakyReLU(),
                nn.Linear(50, 10),
                nn.LeakyReLU(),
                nn.Linear(10, n_outputs),
            ) for i in range(n_branches)
        ])

    def forward(self, x):
        x = self.features(x)
        x = torch.cat([out(x) for out in self.conditional_branches], dim=1)
        return x


class PilotnetControl(nn.Module):

    def __init__(self, n_input_channels=3, n_outputs=1):
        super(PilotnetControl, self).__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(n_input_channels, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.regressor1 = nn.Sequential(
            nn.Linear(1667, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
        )

        self.regressor2 = nn.Sequential(
            nn.Linear(103, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
        )

        self.regressor3 = nn.Sequential(
            nn.Linear(53, 10),
            nn.LeakyReLU(),
            nn.Linear(10, n_outputs),
        )

    def forward(self, x, control):
        x = self.features(x)
        x = self.regressor1(torch.cat([x, control], dim=1))
        x = self.regressor2(torch.cat([x, control], dim=1))
        x = self.regressor3(torch.cat([x, control], dim=1))
        return x

class PilotNetOld(nn.Module):

    def __init__(self, n_input_channels=3):
        super(PilotNetOld, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(1664, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class SteeringNet(pl.LightningModule):
    def __init__(self):
        super(SteeringNet, self).__init__()

        input_size = 20

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=1e-4, amsgrad=False)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.net(x).flatten()
        loss = F.l1_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.net(x).flatten()
        loss = F.l1_loss(y_pred, y)
        self.log('val_loss', loss)

