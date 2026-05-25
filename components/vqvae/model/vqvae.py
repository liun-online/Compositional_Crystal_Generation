import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from components.vae.model.decoder import Decoder
from components.vae.model.encoder import Encoder
from components.vae.val_eval import CrystalReconstructionEvaluator
from components.vqvae.model.quantize import VectorQuantizer
from utils.crys_utils import vector2matrix

from tqdm import tqdm

class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.config = config
        self.device = config.device

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quantize = VectorQuantizer(config)
        self.quant_conv = nn.Linear(config.model.hidden_dim, config.model.latent_dim)
        self.post_quant_conv = nn.Linear(config.model.latent_dim, config.model.hidden_dim)
        self.val_reconstruction_evaluators = CrystalReconstructionEvaluator()
    
    def to_device(self, data):
        return [i.to(self.device) for i in data[1:]]
    
    def encode(self, x, neigh_mask):
        h = self.encoder(x, neigh_mask)
        h = self.quant_conv(h)
        quant, codebook_loss = self.quantize(h)
        return quant, codebook_loss
    
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

        quant, codebook_loss = self.encode(neigh_input, neigh_mask)

        pred_atom, pred_pos, pred_matrix = self.decode(quant, batch)

        ## loss
        atom_loss = F.cross_entropy(pred_atom, atomic_numbers)
        pos_loss = F.mse_loss(pred_pos, cart_coords)
        matrix_loss = F.mse_loss(pred_matrix, matrix)

        return atom_loss, pos_loss, matrix_loss, codebook_loss, (atom_loss + pos_loss + matrix_loss + codebook_loss * self.config.model.code_weight) / 4

    def eval_val(self, val_loader, matrix_scaler):
        self.val_reconstruction_evaluators.clear()
        self.val_reconstruction_evaluators.device = self.device

        for i, data in tqdm(enumerate(val_loader)):
            matrix, cart_coords, atomic_numbers, num_atoms, neigh_coords, neigh_types = self.to_device(data)
            b = matrix.shape[0]
            batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)

            neigh_mask = neigh_types!=99
            neigh_input = torch.cat([neigh_coords, F.one_hot(neigh_types, 100).float(), matrix[batch].unsqueeze(1).repeat(1, neigh_types.shape[1] , 1)], -1) # B 12 103
            
            quant, _ = self.encode(neigh_input, neigh_mask)
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
            
            quant, _ = self.encode(neigh_input, neigh_mask)
            
            all_embs.append(quant.detach().cpu())
            all_nums.append(num_atoms.detach().cpu())
        all_embs = torch.cat(all_embs, 0)
        all_nums = torch.cat(all_nums, 0)
        return all_embs, all_nums

