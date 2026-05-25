from torch.utils.data import Dataset
import torch

class CrystalDataset(Dataset):
    def __init__(self, dataset):
        self.num_atoms = torch.load("./data/cond_gen/"+dataset+"/ft_num_atoms.pt")
        self.quants = torch.load("./data/cond_gen/"+dataset+"/ft_before_quant.pt")
        self.offset = torch.cat([torch.tensor([0]), torch.cumsum(self.num_atoms, -1)], 0).long()
        self.scaler = None
    
    def __len__(self) -> int:
        return len(self.num_atoms)

    def __getitem__(self, index):
        start, end = self.offset[index], self.offset[index+1]
        scaler_quants = self.scaler.transform(self.quants[start: end])
        num_atoms = self.num_atoms[index]
        return scaler_quants, num_atoms
