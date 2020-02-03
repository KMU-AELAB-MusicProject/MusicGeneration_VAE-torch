import os
import torch
import numpy as np
from torch.utils.data import Dataset


class NoteDataset(Dataset):
    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.file_list = os.listdir(self.root_dir)

        self.num_iterations = (len(self.file_list) + config.batch_size - 1) // config.batch_size

    def __len__(self):
        return os.listdir(self.root_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir, self.file_list[idx])

        with np.load(file_name) as data:
            sample = {'note': data['note'], 'pre_note': data['pre_note'], 'position': data['position']}

        return sample
