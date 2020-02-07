import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(510, 512),
            nn.ReLU(True),
            nn.Linear(512, 384),
            nn.ReLU(True),
            nn.Linear(384, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.seq(x.squeeze())
        return x
