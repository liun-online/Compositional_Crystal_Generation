import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from components.vae.model.decoder import Decoder
from components.vae.model.encoder import Encoder
from components.vae.val_eval import CrystalReconstructionEvaluator
from utils.crys_utils import vector2matrix

from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config
        self.device = config.device

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Linear(config.model.hidden_dim, 2 * config.model.latent_dim)
        self.post_quant_conv = nn.Linear(config.model.latent_dim, config.model.hidden_dim)
        self.val_reconstruction_evaluators = CrystalReconstructionEvaluator()
    
    def to_device(self, data):
        return [i.to(self.device) for i in data[1:]]
    
    def encode(self, x, neigh_mask):
        h = self.encoder(x, neigh_mask)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior
    
    def decode(self, quant, batch):
        quant = self.post_quant_conv(quant)
        h, mask = to_dense_batch(quant, batch)
        pred_atom, pred_pos, pred_matrix = self.decoder(h, mask, batch)
        return pred_atom, pred_pos, pred_matrix

    def forward(self, data):
        matrix, cart_coords, atomic_numbers, num_atoms, neigh_coords, neigh_types = self.to_device(data)
        b = matrix.shape[0]
        neigh_mask = neigh_types!=99

        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        neigh_input = torch.cat([neigh_coords, F.one_hot(neigh_types, 100).float(), matrix[batch].unsqueeze(1).repeat(1, neigh_types.shape[1] , 1)], -1) # B 12 103

        posterior = self.encode(neigh_input, neigh_mask)
        quant = posterior.sample()
        kl_loss = posterior.kl().mean()

        pred_atom, pred_pos, pred_matrix = self.decode(quant, batch)

        ## loss
        atom_loss = F.cross_entropy(pred_atom, atomic_numbers)
        pos_loss = F.mse_loss(pred_pos, cart_coords)
        matrix_loss = F.mse_loss(pred_matrix, matrix)

        return atom_loss, pos_loss, matrix_loss, kl_loss, (atom_loss + pos_loss + matrix_loss + kl_loss * 0.00001) / 4

    def eval_val(self, val_loader, matrix_scaler):
        self.val_reconstruction_evaluators.clear()
        self.val_reconstruction_evaluators.device = self.device

        for i, data in tqdm(enumerate(val_loader)):
            matrix, cart_coords, atomic_numbers, num_atoms, neigh_coords, neigh_types = self.to_device(data)
            b = matrix.shape[0]
            batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)

            neigh_mask = neigh_types!=99
            neigh_input = torch.cat([neigh_coords, F.one_hot(neigh_types, 100).float(), matrix[batch].unsqueeze(1).repeat(1, neigh_types.shape[1] , 1)], -1) # B 12 103
            
            posterior = self.encode(neigh_input, neigh_mask)
            quant = posterior.sample()
            pred_atom, pred_pos, pred_matrix = self.decode(quant, batch)

            pred_atom = pred_atom.argmax(-1)
            pred_atom[pred_atom==0] = 1
            pred_matrix = matrix_scaler.inverse_transform(pred_matrix) * num_atoms.reshape(-1, 1).float()**(1/3)
            pred_pos = matrix_scaler.inverse_transform(pred_pos) * num_atoms[batch].reshape(-1, 1).float()**(1/3)

            matrix = matrix_scaler.inverse_transform(matrix) * num_atoms.reshape(-1, 1).float()**(1/3)
            cart_coords = matrix_scaler.inverse_transform(cart_coords) * num_atoms[batch].reshape(-1, 1).float()**(1/3)

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
                _atom_types = pred_atom[start_id: start_id + num]
                _cart_coords = pred_pos[start_id: start_id + num]
                _matrix = pred_matrix[idx_in_batch]

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

    def get_emb(self, loader):
        all_embs = []
        all_nums = []
        for i, data in tqdm(enumerate(loader)):
            matrix, cart_coords, atomic_numbers, num_atoms, neigh_coords, neigh_types = self.to_device(data)
            b = matrix.shape[0]
            batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)

            neigh_mask = neigh_types!=99
            neigh_input = torch.cat([neigh_coords, F.one_hot(neigh_types, 100).float(), matrix[batch].unsqueeze(1).repeat(1, neigh_types.shape[1] , 1)], -1) # B 12 103
            
            posterior = self.encode(neigh_input, neigh_mask)
            quant = posterior.sample()
            
            all_embs.append(quant.detach().cpu())
            all_nums.append(num_atoms.detach().cpu())
        all_embs = torch.cat(all_embs, 0)
        all_nums = torch.cat(all_nums, 0)
        return all_embs, all_nums


class DiagonalGaussianDistribution:
    """
    https://github.com/facebookresearch/all-atom-diffusion-transformer
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=-1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=-1,
                )

    def mode(self):
        return self.mean

    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"
