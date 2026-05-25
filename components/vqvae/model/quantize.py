import torch
import torch.nn as nn
from einops import rearrange


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super(VectorQuantizer, self).__init__()
        pre_emb = torch.load("./data/10000_kmeans_centroids.pt")
        self.n_e = pre_emb.shape[0]
        self.e_dim = config.model.latent_dim
        self.beta = 0.25
        self.embed = nn.Embedding(self.n_e, self.e_dim)
        self.embed.weight.data.copy_(pre_emb)
    
    def forward(self, z):
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embed.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z, rearrange(self.embed.weight, 'n d -> d n'))
    
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embed(min_encoding_indices)
        codebook_loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        return z_q, codebook_loss
    