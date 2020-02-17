import torch
import argparse
import pypianoroll
import numpy as np

from config import Config
from graphs.models.v1.model import Model


##### set args #####
parser = argparse.ArgumentParser()
parser.add_argument('--load_best', help="Train Classifier model before train vae.", action='store_false')
parser.add_argument('--music_length', type=int, default=10, help="Music length that want to make.")
args = parser.parse_args()

##### set model & device #####
config = Config()
model = Model()
device = torch.device("cuda")

##### load model #####
if args.load_best:
    filename = 'modelmodel_best.pth.tar'
else:
    filename = 'modelcheckpoint.pth.tar'

checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

##### make music #####
outputs = []
pre_phrase = torch.zeros(1, 1, 384, 96, dtype=torch.float32)
phrase_idx = [330] + [i for i in range(args.music_length - 2, -1, -1)]

for idx in range(args.music_length):
    pre_phrase = model(torch.randn(1, 510, dtype=torch.float32).cuda(), pre_phrase.cuda(),
                       torch.tensor([phrase_idx[idx]], dtype=torch.long).cuda(), False)
    outputs.append(np.reshape(pre_phrase.cpu().detach().numpy(), [96 * 4, 96, 1]))

##### set note size #####
note = np.array(outputs)
note = note.reshape(96 * 4 * args.seq_size, 96) * 127
note = np.pad(note, [[0, 0], [25, 7]], mode='constant', constant_values=0.)

##### save to midi #####
track = pypianoroll.Track(note, name='piano')
pianoroll = pypianoroll.Multitrack(tracks=[track], beat_resolution=24, name='test')
pianoroll.write('./test.mid')

