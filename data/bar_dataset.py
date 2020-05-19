import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class NoteDataset(Dataset):
    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.file_list = os.listdir(os.path.join(self.root_dir, config.data_path))
        self.config = config

        self.num_iterations = (len(self.file_list) + config.batch_size - 1) // config.batch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.config.data_path, self.file_list[idx])

        with np.load(file_name) as data:
            return {'note': data['note'], 'pre_note': data['pre_note'], 'pre_phrase': data['pre_phrase'],
                    'position': data['position']}

class TestDataset(Dataset):
    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.file_list = os.listdir(os.path.join(self.root_dir, config.data_path))[100:200]

        self.config = config
        self.num_iterations = (len(self.file_list) + config.batch_size - 1) // config.batch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.config.data_path, self.file_list[idx])

        with np.load(file_name) as data:
            return {'note': data['note'], 'pre_note': data['pre_note'], 'pre_phrase': data['pre_phrase'],
                    'position': data['position']}
