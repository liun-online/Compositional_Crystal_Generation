import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch

from tqdm import tqdm
from components.base.model.type import type_noise
from components.base.model.matrix import matrix_noise
from components.base.model.cart import cart_noise
from components.base.model.transformer import Transformer
from components.base.val_eval import CrystalReconstructionEvaluator
from utils.crys_utils import vector2matrix


class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.config = config
        self.h, self.w = config.dataset.max_atoms, config.dataset.max_types
        self.decoder = Transformer(config)
        self.T = config.diffusion.timesteps
        self.device = config.device
        self.type_noise, self.matrix_noise, self.cart_noise = type_noise(self.T), matrix_noise(self.T), cart_noise(self.T)
        self.val_reconstruction_evaluators = CrystalReconstructionEvaluator()
        self.cond_map = nn.Sequential(
            nn.Linear(config.model.cond_dim, config.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim)
            )

    def to_device(self, data):
        return [i.to(self.device) for i in data]
    
    def forward(self, data):
        matrix, cart_coords, atomic_numbers, num_atoms, cond = self.to_device(data)
        b, device = matrix.shape[0], matrix.device
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        cond_emb = self.cond_map(cond)

        batch_cart_coords, _ = to_dense_batch(cart_coords, batch)
        batch_atomic_numbers, _ = to_dense_batch(F.one_hot(atomic_numbers, 100), batch)
        batch_cond_emb, mask = to_dense_batch(cond_emb, batch)

        t = torch.randint(0, self.T, (b,), device=self.device).long()
        t_reshape = t.reshape(b, 1, 1)

        noisy_matrix, matrix_noise = self.matrix_noise(t_reshape, matrix.unsqueeze(1).repeat(1, batch_cart_coords.shape[1], 1))
        noisy_cart, cart_noise = self.cart_noise(t_reshape, batch_cart_coords)
        noisy_type, type_noise = self.type_noise(t_reshape, batch_atomic_numbers)

        x_t = self.compose(noisy_matrix, noisy_type, noisy_cart)

        if self.config.model.cfg.use:
            prob = torch.rand(b).to(device)
            final_mask = prob < self.config.model.cfg.p_cond
            batch_cond_emb[final_mask] = 0.
            
        pred_noise = self.decoder(x_t, batch_cond_emb, mask, t)

        loss_matrix, loss_type, loss_cart = self.cal_loss(pred_noise, matrix_noise, type_noise, cart_noise, mask)
        return loss_matrix, loss_type, loss_cart, self.config.model.loss.m * loss_matrix + self.config.model.loss.t * loss_type + self.config.model.loss.c * loss_cart

    def compose(self, matrix, type, cart):
        whole = torch.cat([cart, type, matrix], -1)
        return whole

    def cal_loss(self, pred_noise, matrix_noise, type_noise, cart_noise, mask):
        pred_matrix_noise, pred_type_noise, pred_cart_noise = pred_noise[:, :, -6:], pred_noise[:, :, 3:-6], pred_noise[:, :, :3]
        
        loss_matrix = self.matrix_noise.cal_loss(pred_matrix_noise[mask], matrix_noise[mask])
        loss_cart = self.cart_noise.cal_loss(pred_cart_noise[mask], cart_noise[mask])
        loss_type = self.type_noise.cal_loss(pred_type_noise[mask], type_noise[mask])
        return loss_matrix, loss_type, loss_cart

    def eval_val(self, val_loader, matrix_scaler, get_fps=False):
        self.val_reconstruction_evaluators.clear()
        self.val_reconstruction_evaluators.device = self.device

        for i, data in tqdm(enumerate(val_loader)):
            matrix, cart_coords, atomic_numbers, num_atoms, cond = self.to_device(data)
            b = matrix.shape[0]

            x, batch = self.reverse(b, cond, num_atoms)
            atomic_numbers_pred = x[:, 3:-6].argmax(-1)
            atomic_numbers_pred[atomic_numbers_pred==0] = 1
            cart_coords_pred = x[:, :3]
            cart_coords_pred = matrix_scaler.inverse_transform(cart_coords_pred) * num_atoms[batch].reshape(-1, 1).float()**(1/3)
            matrix_pred = matrix_scaler.inverse_transform(scatter_mean(x[:, -6:], batch, dim=0)) * num_atoms.reshape(-1, 1).float()**(1/3)

            cart_coords = matrix_scaler.inverse_transform(cart_coords) * num_atoms[batch].reshape(-1, 1).float()**(1/3)
            matrix = matrix_scaler.inverse_transform(matrix) * num_atoms.reshape(-1, 1).float()**(1/3)

            start_id = 0
            for idx_in_batch, num in enumerate(num_atoms):
                _atom_types = atomic_numbers[start_id: start_id + num]
                _cart_coords = cart_coords[start_id: start_id + num]
                _matrix = matrix[idx_in_batch]
                
                self.val_reconstruction_evaluators.append_gt_array(
                    {
                        "atom_types": _atom_types.detach().cpu().numpy(),
                        "cart_coords": _cart_coords.detach().cpu().numpy(),
                        "matrices": vector2matrix(_matrix.detach().cpu()).numpy()
                    }
                )
                start_id = start_id + num
            
            start_id = 0
            for idx_in_batch, num in enumerate(num_atoms):
                _atom_types = atomic_numbers_pred[start_id: start_id + num]
                _cart_coords = cart_coords_pred[start_id: start_id + num]
                _matrix = matrix_pred[idx_in_batch]

                self.val_reconstruction_evaluators.append_pred_array(
                    {
                        "atom_types": _atom_types.detach().cpu().numpy(),
                        "cart_coords": _cart_coords.detach().cpu().numpy(),
                        "matrices": vector2matrix(_matrix.detach().cpu()).numpy()
                    }
                )
                start_id = start_id + num
        results = self.val_reconstruction_evaluators.get_metrics(cart=True)
        return results["match_rate"]
    
    def reverse(self, b, cond, num_atoms):
        cond_emb = self.cond_map(cond)
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        max_atom = num_atoms.max()
        batch_cond_emb, mask = to_dense_batch(cond_emb, batch)

        noisy_matrix = self.matrix_noise.gaussian(torch.Size([b, max_atom, 6]), self.device)
        noisy_cart = self.cart_noise.gaussian(torch.Size([b, max_atom, 3]), self.device)
        noisy_type = self.type_noise.gaussian(torch.Size([b, max_atom, self.w]), self.device)
        x = self.compose(noisy_matrix, noisy_type, noisy_cart)

        for t in tqdm(reversed(range(1, self.T)), desc='Sampling ...', total=self.T):
            curr_t = (torch.ones(b) * t).long().to(self.device)
            curr_t_reshape = curr_t.reshape(b, 1, 1)
            if self.config.model.cfg.use:
                guide_strength = self.config.model.cfg.guide_strength
                zero_cond = torch.zeros_like(batch_cond_emb).to(self.device)
                pred_noise_uncond = self.decoder(x, zero_cond, mask, curr_t)
                pred_noise_cond = self.decoder(x, batch_cond_emb, mask, curr_t)
                pred_noise = guide_strength * pred_noise_cond + (1-guide_strength) * pred_noise_uncond
            else:
                pred_noise = self.decoder(x, batch_cond_emb, mask, curr_t)
            
            x = self.one_reverse(pred_noise, x, curr_t_reshape)
        return x[mask], batch

    def one_reverse(self, pred_noise, x, t):
        pred_matrix_noise, pred_type_noise, pred_cart_noise = pred_noise[:, :, -6:], pred_noise[:, :, 3:-6], pred_noise[:, :, :3]
        noisy_matrix, noisy_type, noisy_cart = x[:, :, -6:], x[:, :, 3:-6], x[:, :, :3]

        new_matrix = self.matrix_noise.reverse(t, noisy_matrix, pred_matrix_noise)
        new_type = self.type_noise.reverse(t, noisy_type, pred_type_noise)
        new_cart = self.cart_noise.reverse(t, noisy_cart, pred_cart_noise)
        new_x = self.compose(new_matrix, new_type, new_cart)
        return new_x   
