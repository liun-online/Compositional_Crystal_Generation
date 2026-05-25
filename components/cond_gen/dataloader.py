import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CrystalDataset(Dataset):
    def __init__(self, dataset, val=False):
        if val:
            self.material_id = np.load(os.path.join("./data/cond_gen/", dataset, "val_ids.npy"))
            self.num_atoms = torch.load(os.path.join("./data/cond_gen/", dataset, "val_nums.pt"))
            self.quants = torch.load(os.path.join("./data/cond_gen/", dataset, "val_before_quants.pt"))
        else:
            self.material_id = np.load(os.path.join("./data/cond_gen/", dataset, "train_ids.npy"))
            self.num_atoms = torch.load(os.path.join("./data/cond_gen/", dataset, "train_nums.pt"))
            self.quants = torch.load(os.path.join("./data/cond_gen/", dataset, "train_before_quants.pt"))
        self.offset = torch.cat([torch.tensor([0]), torch.cumsum(self.num_atoms, -1)], 0).long()
        self.scaler = None
    
    def __len__(self) -> int:
        return len(self.material_id)

    def __getitem__(self, index):
        start, end = self.offset[index], self.offset[index+1]
        scaler_quants = self.scaler.transform(self.quants[start: end])
        num_atoms = self.num_atoms[index]
        return scaler_quants, num_atoms
