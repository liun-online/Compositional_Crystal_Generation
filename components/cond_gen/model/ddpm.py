import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from tqdm import tqdm
from components.cond_gen.model.quant import quant_noise
from components.cond_gen.model.transformer import Transformer

class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.config = config
        self.h, self.w = config.dataset.max_atoms, config.dataset.max_types
        self.decoder = Transformer(config)
        self.T = config.diffusion.timesteps
        self.device = config.device
        self.quant_noise = quant_noise(self.T)
        
    def to_device(self, data):
        return [i.to(self.device) for i in data]
    
    def forward(self, data):
        num_atoms, quants = self.to_device(data)
        b, device = num_atoms.shape[0], num_atoms.device
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)

        batch_quants, mask = to_dense_batch(quants, batch)

        t = torch.randint(0, self.T, (b,), device=self.device).long()
        t_reshape = t.reshape(b, 1, 1)
        noisy_quant, quant_noise = self.quant_noise(t_reshape, batch_quants)

        x_t = noisy_quant

        pred_noise = self.decoder(x_t, mask, t)

        loss_quant = self.cal_loss(pred_noise, quant_noise, mask)
        return loss_quant

    def cal_loss(self, pred_noise, quant_noise, mask):
        loss_quant = self.quant_noise.cal_loss(pred_noise[mask], quant_noise[mask])
        return loss_quant
    
    def reverse(self, b, num_atoms):
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        _, mask = to_dense_batch(batch, batch)
        max_atom = num_atoms.max()
        noisy_quant = self.quant_noise.gaussian(torch.Size([b, max_atom, 8]), self.device)
        x = noisy_quant

        for t in tqdm(reversed(range(1, self.T)), desc='Sampling ...', total=self.T):
            curr_t = (torch.ones(b) * t).long().to(self.device)
            curr_t_reshape = curr_t.reshape(b, 1, 1)
            pred_noise = self.decoder(x, mask, curr_t)
            
            x = self.one_reverse(pred_noise, x, curr_t_reshape)
        return x[mask], batch

    def one_reverse(self, pred_noise, x, t):
        new_x = self.quant_noise.reverse(t, x, pred_noise)
        return new_x   
