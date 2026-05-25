import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from I_train_vae import collate
from utils.utils import get_scaler_min_max, dict2namespace, last_ckpt
from utils.config import DictAction, get_params
from components.vae.dataloader import CrystalDataset
from components.vqvae.model.vqvae import VQVAE

import argparse
from einops import rearrange
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mp_20')
parser.add_argument('--conf_new', nargs='+', action=DictAction)
args = parser.parse_args()
args.task = "vqvae"
args.log = os.path.join('logs', args.task, "mp_20")
args.cfg = os.path.join(args.log, 'backup-config.yaml')
config = get_params(args, down=False)
config = dict2namespace(config)
device = torch.device('cuda')
config.device = device

model = VQVAE(config).cuda()
ckpt = last_ckpt("vqvae", "mp_20")
checkpoint = torch.load(ckpt)
new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
model.load_state_dict(new_state_dict)
model.eval()

emb = model.quantize.embed.weight.cpu()

train = CrystalDataset('vae', args.dataset, config=config)
val = CrystalDataset('vae', args.dataset, config=config, val=True)
datasets = [train, val]
flags = ["train", "val"]
saved_path = os.path.join("./data/cond_gen/", args.dataset)
os.makedirs(saved_path, exist_ok=True)

for data, fl in zip(datasets, flags):
    data.matrix_scaler = get_scaler_min_max('vae', "mp_20")
    loader = DataLoader(data, shuffle=False, batch_size=50, collate_fn=collate)

    indices = []
    before_quants = []
    quants = []
    ids = []
    nums = []
    with torch.no_grad():
        for data in tqdm(loader):
            matrix, cart_coords, atomic_numbers, num_atoms, neigh_coords, neigh_types = model.to_device(data)
            material_id = data[0]
            b = matrix.shape[0]
            batch = torch.repeat_interleave(torch.arange(b).to(model.device), num_atoms)

            neigh_mask = neigh_types!=99
            neigh_input = torch.cat([neigh_coords, F.one_hot(neigh_types, 100).float(), matrix[batch].unsqueeze(1).repeat(1, neigh_types.shape[1] , 1)], -1)
            
            h = model.encoder(neigh_input, neigh_mask)
            h = model.quant_conv(h)

            d = torch.sum(h ** 2, dim=1, keepdim=True) + \
                torch.sum(model.quantize.embed.weight**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', h, rearrange(model.quantize.embed.weight, 'n d -> d n'))

            quant, codebook_loss = model.quantize(h)

            min_encoding_indices = torch.argmin(d, dim=1)
            indices.append(min_encoding_indices)
            quants.append(quant)
            ids += material_id
            before_quants.append(h)
            nums.append(num_atoms.cpu())
    indices = torch.cat(indices, dim=-1).cpu().data
    quants = torch.cat(quants, 0).cpu().data
    before_quants = torch.cat(before_quants, 0).cpu().data
    ids = np.array(ids)
    nums = torch.cat(nums, 0).cpu().data

    unique_indices, counts = torch.unique(indices, return_counts=True)
    torch.save(before_quants, saved_path+"/"+fl+"_before_quants.pt")
    np.save(saved_path+"/"+fl+"_ids.npy", ids)
    torch.save(nums, saved_path+"/"+fl+"_nums.pt")
    torch.save(quants, saved_path+"/"+fl+"_quants.pt")
