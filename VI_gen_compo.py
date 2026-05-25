import os
import torch

from components.cond_gen.dataloader import CrystalDataset
from components.cond_gen.model.ddpm import DDPM
from components.vqvae.model.vqvae import VQVAE
from utils.utils import get_scaler_mean_std, get_scaler_min_max, dict2namespace, last_ckpt
from utils.crys_utils import  vector2matrix
from utils.config import DictAction, get_params

import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mp_20')
parser.add_argument('--conf_new', nargs='+', action=DictAction)
args = parser.parse_args()
args.task = "cond_gen"
args.log = os.path.join('logs', args.task, args.dataset)
args.cfg = os.path.join(args.log, 'backup-config.yaml')
config = get_params(args, down=False)
config = dict2namespace(config)
device = torch.device('cuda')
config.device = device

diffusion = DDPM(config).cuda()
ckpt = last_ckpt(args.task, args.dataset)
checkpoint = torch.load(ckpt)
new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
diffusion.load_state_dict(new_state_dict)
diffusion.eval()

dataset = CrystalDataset(args.dataset)
dataset.scaler = get_scaler_mean_std("cond_gen", args.dataset)

num_prob = torch.zeros(21)
for i in range(1, 21):
    num_prob[i] = (dataset.num_atoms == i).sum()
num_prob /= num_prob.sum()

gen_num = 100_000
b = 1000
gen_turn = int(gen_num / b)
gen_before_quants = []
all_sample_num = []
with torch.no_grad():
    for i in tqdm(range(gen_turn)):
        sample_num = torch.multinomial(num_prob, num_samples=b, replacement=True).cuda()
        all_sample_num.append(sample_num)
        before_quants, _ = diffusion.reverse(b, sample_num)
        before_quants = dataset.scaler.inverse_transform(before_quants.cpu().data)
        gen_before_quants.append(before_quants)

gen_before_quants = torch.cat(gen_before_quants, 0).cpu().data
num_atoms = torch.cat(all_sample_num, 0).cpu().data

save_path = os.path.join(args.log, 'gen')
os.makedirs(save_path, exist_ok=True)
torch.save(num_atoms, os.path.join(save_path, "num_atoms.pt"))
torch.save(gen_before_quants, os.path.join(save_path, "gen_before_quants.pt"))

# quantize
args.log = 'logs/vqvae/mp_20'
args.cfg = os.path.join(args.log, 'backup-config.yaml')
config = get_params(args, down=False)
config = dict2namespace(config)
config.device = device

vqvae = VQVAE(config).cuda()
ckpt = last_ckpt("vqvae", "mp_20")
checkpoint = torch.load(ckpt)
new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
vqvae.load_state_dict(new_state_dict)
vqvae.eval()
matrix_scaler = get_scaler_min_max("vae", "mp_20")

offset = torch.cat([torch.zeros(1), torch.cumsum(num_atoms, -1)], -1).long()

start = 0
bs = 100
gen_structs = []
quants = []
while True:
    end = min(start+bs, len(num_atoms))
    curr_num_atoms = num_atoms[start: end].cuda()
    curr_before_quants = gen_before_quants[offset[start]: offset[end]].cuda()
    batch = torch.repeat_interleave(torch.arange(len(curr_num_atoms)).cuda(), curr_num_atoms)
    curr_quants, _ = vqvae.quantize(curr_before_quants)
    quants.append(curr_quants)
    pred_atom, pred_pos, pred_matrix = vqvae.decode(curr_quants, batch)

    pred_atom = pred_atom.argmax(-1)
    pred_atom[pred_atom==0] = 1
    pred_matrix = matrix_scaler.inverse_transform(pred_matrix) * curr_num_atoms.reshape(-1, 1).float()**(1/3)
    pred_pos = matrix_scaler.inverse_transform(pred_pos) * curr_num_atoms[batch].reshape(-1, 1).float()**(1/3)

    start_id = 0
    for idx_in_batch, num in enumerate(curr_num_atoms):
        _atom_types = pred_atom[start_id: start_id + num]
        _cart_coords = pred_pos[start_id: start_id + num]
        _matrix = pred_matrix[idx_in_batch]

        gen_structs.append(
            {
                "atom_types": _atom_types.detach().cpu().numpy(),
                "cart_coords": _cart_coords.detach().cpu().numpy(),
                "matrices": vector2matrix(_matrix.detach().cpu()).numpy()
            }
        )
        start_id = start_id + num
    
    start = end
    if end == len(num_atoms):
        break
    else:
        print("Finish ", end / len(num_atoms))
gen_quants = torch.cat(quants, 0)
torch.save(gen_quants, os.path.join("logs/cond_gen", args.dataset, 'gen', "gen_quants.pt"))
torch.save(gen_structs, os.path.join("logs/cond_gen", args.dataset, 'gen', "gen_structs.pt"))
