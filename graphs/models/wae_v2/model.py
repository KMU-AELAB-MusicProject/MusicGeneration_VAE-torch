import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from graphs.weights_initializer import weights_init


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.position_embedding = nn.Embedding(332, 510)

        self.apply(weights_init)

    def forward(self, note, pre_note, position, is_train=True):
        if is_train:
            z, mean, var = self.encoder(note)
            pre_z, pre_mean, pre_var = self.encoder(pre_note)

            return self.decoder(z + pre_z + self.position_embedding(position)), z
        else:
            pre_z, pre_mean, pre_var = self.encoder(pre_note)

            return self.decoder(note + pre_z + self.position_embedding(position))
