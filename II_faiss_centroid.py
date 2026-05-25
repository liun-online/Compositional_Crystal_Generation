import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import faiss

from I_train_vae import collate
from components.vae.model.vae import VAE
from components.vae.dataloader import CrystalDataset
from utils.utils import get_scaler_min_max, dict2namespace, last_ckpt
from utils.config import DictAction, get_params

import argparse
from tqdm import tqdm


def faiss_kmeans(X, k, n_iter=20):
    N, D = X.shape
    kmeans = faiss.Kmeans(d=D, k=k, niter=n_iter, verbose=False, min_points_per_centroid = 1)

    kmeans.train(X)
    Dists, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)

    avg_dist = np.sqrt(Dists).mean()
    return avg_dist, I, kmeans

def evaluate_k_range_avg_dist(X, k_list, niter=20):
    avg_dists = []
    models = {}

    for k in k_list:
        avg_dist, labels, model = faiss_kmeans(X, k, n_iter=niter)
        avg_dists.append(avg_dist)
        models[k] = {"avg_dist": avg_dist, "labels": labels, "model": model}
    return avg_dists, models

parser = argparse.ArgumentParser()
parser.add_argument('--conf_new', nargs='+', action=DictAction)
args = parser.parse_args()
args.task = "vae"
args.log = os.path.join('logs', args.task, "mp_20")
args.cfg = os.path.join(args.log, 'backup-config.yaml')
config = get_params(args, down=False)
config = dict2namespace(config)
device = torch.device('cuda')
config.device = device

model = VAE(config).to(device)
ckpt = last_ckpt("vae", "mp_20")
checkpoint = torch.load(ckpt, map_location=device)
new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
model.load_state_dict(new_state_dict)
model.eval()

train = CrystalDataset(args.task, "mp_20", config=config)
train.matrix_scaler = get_scaler_min_max("vae", "mp_20")
train_loader = DataLoader(train, shuffle=False, batch_size=100, collate_fn=collate)

all_train_emb = []
with torch.no_grad():
    for data in tqdm(train_loader):
        matrix, cart_coords, atomic_numbers, num_atoms, neigh_coords, neigh_types = model.to_device(data)
        b = matrix.shape[0]
        batch = torch.repeat_interleave(torch.arange(b).to(model.device), num_atoms)

        neigh_mask = neigh_types!=99
        neigh_input = torch.cat([neigh_coords, F.one_hot(neigh_types, 100).float(), matrix[batch].unsqueeze(1).repeat(1, neigh_types.shape[1] , 1)], -1) # B 12 103
        
        posterior = model.encode(neigh_input, neigh_mask)
        quant = posterior.sample()
        all_train_emb.append(quant)
all_train_emb = torch.cat(all_train_emb, dim=0).cpu().data


k_list = [10_000]
avg_dists, models = evaluate_k_range_avg_dist(all_train_emb, k_list, niter=30)

for i in k_list:
    torch.save(torch.from_numpy(models[i]["model"].centroids).float(), "./data/10000_kmeans_centroids.pt")