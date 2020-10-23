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

    def forward(self, note, pre_note, phrase, position, is_note=True):
        if is_note:
            phrase_feature = self.phrase_encoder(phrase)

            z = self.encoder(note)
            pre_z = self.encoder(pre_note)

            gen_note = self.decoder(z, pre_z, phrase_feature, position)
            fake_note = torch.gt(gen_note, 0.35).type('torch.cuda.FloatTensor')
            return gen_note, z, pre_z, phrase_feature, self.encoder(fake_note)
        else:
            phrase_feature = self.phrase_encoder(phrase)

            pre_z = self.encoder(pre_note)

            gen_note = self.decoder(note, pre_z, phrase_feature, position)
            fake_note = torch.gt(gen_note, 0.35).type('torch.cuda.FloatTensor')

            return gen_note, self.encoder(fake_note)
