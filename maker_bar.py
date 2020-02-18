import torch
import argparse
import pypianoroll
import numpy as np

from config import Config
from graphs.models.bar_v1.model import Model
from graphs.models.bar_v1.phrase_model import PhraseModel


##### set args #####
parser = argparse.ArgumentParser()
parser.add_argument('--load_best', help="Train Classifier model before train vae.", action='store_false')
parser.add_argument('--music_length', type=int, default=10, help="Music length that want to make.")
args = parser.parse_args()

##### set model & device #####
config = Config()
model = Model()
phrase_model = PhraseModel()
device = torch.device("cuda")

##### load model #####
if args.load_best:
    filename = 'model/model_best.pth.tar'
else:
    filename = 'model/checkpoint.pth.tar'

checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['model_state_dict'])
phrase_model.load_state_dict(checkpoint['phrase_model_state_dict'])
model.to(device)
phrase_model.to(device)

##### make music #####
outputs = []
pre_phrase = torch.zeros(1, 1, 384, 96, dtype=torch.float32)
pre_bar = torch.zeros(1, 1, 96, 96, dtype=torch.float32)
phrase_idx = [330] + [i for i in range(args.music_length - 2, -1, -1)]
for idx in range(args.music_length):
    bar_set = []
    for _ in range(4):
        phrase_feature, _, _ = phrase_model(pre_phrase.cuda(), torch.tensor([phrase_idx[idx]], dtype=torch.long).cuda())
        pre_bar = model(torch.randn(1, 1152, dtype=torch.float32).cuda(), pre_bar.cuda(), phrase_feature, False)

        pre_bar = torch.gt(pre_bar, 0.35).type('torch.FloatTensor') # 1, 1, 96, 96
        bar_set.append(np.reshape(pre_bar.numpy(), [96, 96]))

    pre_phrase = np.concatenate(bar_set, axis=0)
    outputs.append(pre_phrase)
    pre_phrase = torch.from_numpy(np.reshape(pre_phrase, [1, 1, 96*4, 96])).float().cuda()

##### set note size #####
note = np.concatenate(outputs, axis=0) * 127
note = np.pad(note, [[0, 0], [25, 7]], mode='constant', constant_values=0.)

##### save to midi #####
track = pypianoroll.Track(note, name='piano')
pianoroll = pypianoroll.Multitrack(tracks=[track], beat_resolution=24, name='test')
pianoroll.write('./test.mid')

