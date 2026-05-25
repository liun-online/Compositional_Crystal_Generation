import torch.nn as nn
from components.vae.model.transformer import Transformer

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        assert n_layers >= 2
        super(MLP, self).__init__()
        self.map = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(0, n_layers-2):
            self.map.append(nn.Linear(hidden_dim, hidden_dim))
        self.map.append(nn.Linear(hidden_dim, out_dim))
        self.act = nn.ReLU()
        self.n_layers = n_layers
    
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.act(self.map[i](x))
        x = self.map[-1](x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.neigh_emb = MLP(109, config.model.hidden_dim, config.model.hidden_dim)
        self.neigh_trans = Transformer(config.model.hidden_dim, config.model.neigh_layer, config.model.attn_head, config.model.dropout)
        
    def forward(self, x, neigh_mask):
        neigh_emb = self.neigh_emb(x)
        neigh_emb = self.neigh_trans(neigh_emb, neigh_mask)[:, 0]
        return neigh_emb