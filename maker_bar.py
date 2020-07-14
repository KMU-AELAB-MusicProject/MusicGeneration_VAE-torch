import torch
import argparse
import pypianoroll
import numpy as np
from torch import nn

from config import Config
from graph.model import Model


##### set args #####
parser = argparse.ArgumentParser()
parser.add_argument('--model_number', type=int, default=10, help="Music length that want to make.")
parser.add_argument('--music_length', type=int, default=10, help="Music length that want to make.")
args = parser.parse_args()

##### set model & device #####
config = Config()
model = Model()

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

##### load model #####
filename = 'model/checkpoint_{}.pth.tar'.format(args.model_number)

checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['generator_state_dict'])

##### make music #####
outputs = []
pre_phrase = torch.zeros(1, 1, 384, 60, dtype=torch.float32)
pre_bar = torch.zeros(1, 1, 96, 60, dtype=torch.float32)
phrase_idx = [330] + [i for i in range(args.music_length - 2, -1, -1)]
for idx in range(args.music_length):
    bar_set = []
    for _ in range(4):
        pre_bar = model(torch.randn(1, 1152, dtype=torch.float32).cuda(), pre_bar.cuda(), pre_phrase, torch.from_numpy(np.array([phrase_idx[idx]])), False)
        pre_bar = torch.gt(pre_bar, 0.35).type('torch.FloatTensor') # 1, 1, 96, 96
        bar_set.append(np.reshape(pre_bar.numpy(), [96, 60]))

    pre_phrase = np.concatenate(bar_set, axis=0)
    outputs.append(pre_phrase)
    pre_phrase = torch.from_numpy(np.reshape(pre_phrase, [1, 1, 96*4, 60])).float().cuda()

##### set note size #####
note = np.concatenate(outputs, axis=0) * 127
print(note.shape)
note = np.pad(note, [[0, 0], [27, 41]], mode='constant', constant_values=0.)
print(note.shape)
##### save to midi #####
track = pypianoroll.Track(note, name='piano')
pianoroll = pypianoroll.Multitrack(tracks=[track], beat_resolution=24, name='test')
pianoroll.write('./test.mid')

