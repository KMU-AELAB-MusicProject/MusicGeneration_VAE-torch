import torch
import torch.nn as nn

from graph.weights_initializer import weights_init
from graph.encodingBlock import TimePitchModule, PitchTimeModule, PoolingModule, ResidualModule


class PhraseEncoder(nn.Module):
    def __init__(self, layers):     # [64, 128, 256, 512, 1024]
        super(PhraseEncoder, self).__init__()

        self.time_pitch = TimePitchModule()
        self.pitch_time = PitchTimeModule()

        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(ResidualModule(layers[i - 1]))
            self.layers.append(PoolingModule(layers[i - 1], layers[i]))
        self.layers = nn.ModuleList(self.layers)

        self.avg = nn.AvgPool2d(kernel_size=(12, 2))

        self.linear = nn.Linear(1024, 1152, bias=False)

        self.apply(weights_init)

    def forward(self, x):
        pitch = self.pitch_time(x)
        time = self.time_pitch(x)

        out = torch.cat((pitch, time), dim=1)
        
        for layer in self.layers:
            out = layer(out)

        out = self.avg(out)

        z = self.linear(out)

        return z


class PhraseModel(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.phrase_encoder = PhraseEncoder(layers)

        self.position_embedding = nn.Embedding(332, 1152)

        self.apply(weights_init)

    def forward(self, phrase, position):        
        z_phrase = self.phrase_encoder(phrase)
        phrase_feature = z_phrase + self.position_embedding(position)

        return phrase_feature