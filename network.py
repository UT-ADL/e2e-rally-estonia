import torch.nn as nn

class Crop(nn.Module):

    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        img = img / 255
        return img[..., self.top:self.top + self.height, self.left:self.left + self.width]

class PilotNet(nn.Module):

    def __init__(self):
        super(PilotNet, self).__init__()

        self.crop = Crop(600, 186, 369, 1734)
        self.conv1 = nn.Conv2d(3, 24, 10, stride=4)
        self.conv2 = nn.Conv2d(24, 24, 5, stride=3)
        self.conv3 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv4 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv5 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(3)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.2)

        self.lin1 = nn.Linear(1856, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 10)
        self.lin4 = nn.Linear(10, 1)

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.crop(x)
        x = self.bn1(x)
        x = self.lr(self.conv1(x))
        x = self.lr(self.conv2(x))
        x = self.lr(self.conv3(x))
        x = self.lr(self.conv4(x))
        x = self.lr(self.conv5(x))
        x = self.lr(self.conv6(x))
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.lr(self.lin1(x))
        x = self.dropout(x)
        x = self.lr(self.lin2(x))
        x = self.dropout(x)
        x = self.lr(self.lin3(x))
        x = self.lin4(x)

        return x
