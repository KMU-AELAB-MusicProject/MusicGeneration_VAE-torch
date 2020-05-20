import torch
import torch.nn as nn

from graph.weights_initializer import weights_init


class PhraseZDiscriminator(nn.Module):
    def __init__(self, z_dim=1152):
        super(PhraseZDiscriminator, self).__init__()

        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim, 384),
            nn.ReLU(True),
            nn.Linear(384, 384),
            nn.ReLU(True),
            nn.Linear(384, 384),
            nn.ReLU(True),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


class BarZDiscriminator(nn.Module):
    def __init__(self, z_dim=1152):
        super(BarZDiscriminator, self).__init__()

        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim, 384),
            nn.ReLU(True),
            nn.Linear(384, 384),
            nn.ReLU(True),
            nn.Linear(384, 384),
            nn.ReLU(True),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)
