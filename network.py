import torch.nn as nn


class PilotNet(nn.Module):

    def __init__(self):
        super(PilotNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
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

