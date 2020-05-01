import torch
import torch.nn as nn

from .phrase_encoder import PhraseEncoder
from .decoder import Decoder
from .encoder import Encoder
from graphs.weights_initializer import weights_init


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.apply(weights_init)

    def forward(self, note, pre_note, phrase_feature, is_train=True):
        if is_train:
            z, mean, var = self.encoder(note)
            pre_z, pre_mean, pre_var = self.encoder(pre_note)

            bar_feature = z + pre_z
            feature = torch.cat((bar_feature, phrase_feature), dim=1)

            gen_note = self.decoder(feature)

            z_gen, _, _ = self.encoder(torch.gt(gen_note, 0.35).type('torch.cuda.FloatTensor'))

            return gen_note, mean, var, pre_mean, pre_var, z, z_gen
        else:
            pre_z, pre_mean, pre_var = self.encoder(pre_note)

            bar_feature = note + pre_z
            feature = torch.cat((bar_feature, phrase_feature), dim=1)

            return self.decoder(feature)
