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

        self.avg = nn.AvgPool2d(kernel_size=(3, 3))

        self.mean = nn.Linear(1024, 1152, bias=False)
        self.var = nn.Linear(1024, 1152, bias=False)

        self.apply(weights_init)

    def reparameterize(self, mean, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        time = self.time_pitch(x)
        pitch = self.pitch_time(x)

        out = torch.cat((pitch, time), dim=1)

        for layer in self.layers:
            out = layer(out)

        out = self.avg(out)

        mean = self.mean(out.view(-1, 1024))
        var = self.var(out.view(-1, 1024))

        z = self.reparameterize(mean, var)

        return z, mean, var
