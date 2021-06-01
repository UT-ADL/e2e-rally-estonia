import torch.nn as nn


class PilotNet(nn.Module):

    def __init__(self):
        super(PilotNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.2)

        #self.lin1 = nn.Linear(1152, 100)
        self.lin1 = nn.Linear(1600, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 10)
        self.lin4 = nn.Linear(10, 1)

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(self.conv1(x))
        x = self.lr(self.conv2(x))
        x = self.lr(self.conv3(x))
        x = self.lr(self.conv4(x))
        x = self.lr(self.conv5(x))
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.lr(self.lin1(x))
        x = self.dropout(x)
        x = self.lr(self.lin2(x))
        x = self.dropout(x)
        x = self.lr(self.lin3(x))
        x = self.lin4(x)

        return x

