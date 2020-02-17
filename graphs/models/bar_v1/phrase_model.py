import torch
import torch.nn as nn

from .phrase_encoder import PhraseEncoder
from graphs.weights_initializer import weights_init


class PhraseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.phrase_encoder = PhraseEncoder()

        self.position_embedding = nn.Embedding(332, 1152)

        self.apply(weights_init)

    def forward(self, phrase, position):
        z_phrase, mean_phrase, var_phrase = self.phrase_encoder(phrase)
        phrase_feature = z_phrase + self.position_embedding(position)

        return phrase_feature, mean_phrase, var_phrase
