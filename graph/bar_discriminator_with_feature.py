import torch
import torch.nn as nn

from graph.weights_initializer import weights_init

class BarFeatureDiscriminator(nn.Module):
    def __init__(self):
        super(BarFeatureDiscriminator, self).__init__()

        self.linear1 = nn.Linear(1152, 512, bias=False)
        self.linear2 = nn.Linear(96, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 1152)

        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        outputs = self.sigmoid(x2)

        return outputs