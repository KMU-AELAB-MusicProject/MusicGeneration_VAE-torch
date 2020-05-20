import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .phrase_encoder import PhraseModel
from graph.weights_initializer import weights_init


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder([64, 128, 256, 512, 1024])
        self.decoder = Decoder([1024, 512, 256, 128, 64])
        self.phrase_encoder = PhraseModel([64, 128, 256, 512, 1024])

        self.apply(weights_init)

    def forward(self, note, pre_note, phrase, position, is_train=True):
        if is_train:
            phrase_feature = self.phrase_encoder(phrase, position)

            z = self.encoder(note)
            pre_z = self.encoder(pre_note)
            bar_feature = z + pre_z

            feature = torch.cat((bar_feature, phrase_feature), dim=1)

            gen_note = self.decoder(feature)
            
            return gen_note, z, pre_z
        else:
            phrase_feature = self.phrase_encoder(phrase, position)

            pre_z, pre_mean, pre_var = self.encoder(pre_note)
            bar_feature = note + pre_z

            feature = torch.cat((bar_feature, phrase_feature), dim=1)

            return self.decoder(feature)
