import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

"""
Network for predicting steering angle from waypoints. Quick scratch implementation and not optimized. 
"""
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

