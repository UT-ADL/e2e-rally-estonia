import torch
import torch.nn as nn


class PilotNet(nn.Module):

    def __init__(self, n_input_channels=3, n_outputs=1, n_branches=1):
        super(PilotNet, self).__init__()

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

