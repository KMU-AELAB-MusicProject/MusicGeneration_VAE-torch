import torch
import torch.nn as nn

from graph.weights_initializer import weights_init
from graph.encodingBlock import TimePitchModule, PitchTimeModule, PoolingModule, ResidualModule

class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()

        self.time_pitch = TimePitchModule()
        self.pitch_time = PitchTimeModule()

        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(ResidualModule(layers[i - 1]))
            self.layers.append(PoolingModule(layers[i - 1], layers[i]))
        self.layers = nn.ModuleList(self.layers)

        self.avg = nn.AvgPool2d(kernel_size=(3, 2))

        self.linear = nn.Linear(1024, 1152)

        self.apply(weights_init)

    def forward(self, x):
        time = self.time_pitch(x)
        pitch = self.pitch_time(x)

        out = torch.cat((pitch, time), dim=1)

        for layer in self.layers:
            out = layer(out)

        out = self.avg(out)

        z = self.linear(out)

        return z
