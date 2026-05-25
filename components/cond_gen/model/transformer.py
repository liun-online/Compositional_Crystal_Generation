import torch
import torch.nn as nn
import math

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

class MHA(nn.Module):
    def __init__(self, attn_head, dim, dropout):
        super(MHA, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.attn_head = attn_head
        self.softmax = nn.Softmax(dim=-1)
        self.WO = nn.Linear(dim, dim)
    
    def forward(self, q, k, v, batch):
        b, lq, _ = q.shape
        b, kq, _ = k.shape
        b, vq, _ = v.shape

        dim_head = self.dim // self.attn_head
        q = q.reshape(b, lq, self.attn_head, dim_head).transpose(1, 2)
        k = k.reshape(b, kq, self.attn_head, dim_head).transpose(1, 2)
        v = v.reshape(b, vq, self.attn_head, dim_head).transpose(1, 2)

        mask = batch.unsqueeze(1).unsqueeze(-1).float()
        mask = mask @ mask.transpose(-1, -2)

        attn_scores = q @ k.transpose(2, 3) / math.sqrt(dim_head)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-1e3'))
        attn = self.softmax(attn_scores)
        attn = self.dropout(attn)
        
        attn_out = (attn @ v).transpose(1, 2).reshape(b, lq, self.dim)
        attn_out = self.dropout(attn_out)
        attn_out = self.WO(attn_out)
        return attn_out

class Decoder_layer(nn.Module):
    def __init__(self, dim, attn_head, dropout):
        assert dim % attn_head == 0
        super(Decoder_layer, self).__init__()
        self.LN_MHA_SA = nn.LayerNorm(dim)
        self.LN_MLP = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, 3*dim)
        self.mha_sa = MHA(attn_head, dim, dropout)
        self.MLP = nn.Sequential(nn.Linear(dim, 4*dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
    
    def forward(self, h, batch):
        ## self_attention
        h[~batch] = 0.
        h_ini = h
        h = self.LN_MHA_SA(h)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        sa_attn_out = self.mha_sa(q, k, v, batch)
        sa_attn_out[~batch] = 0.
        h = h_ini + sa_attn_out

        ## add & norm
        h = h + self.MLP(self.LN_MLP(h))
        h[~batch] = 0.
        return h

class Sine_PE_t(nn.Module):
    def __init__(self, time_dim):
        super(Sine_PE_t, self).__init__()
        self.div_term = torch.exp(torch.arange(0, time_dim, 2) * -(math.log(10000.0) / time_dim))
    
    def forward(self, t):
        ## t: b
        pos = torch.einsum("b,l -> bl", t, self.div_term.to(t.device))
        sin_pe, cos_pe = torch.sin(pos), torch.cos(pos)
        return torch.cat([sin_pe, cos_pe], -1)
    
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        input_dim = 8
        decoder_layers = config.model.decoder_layers
        dropout = config.model.dropout
        hidden_dim = config.model.hidden_dim
        attn_head = config.model.attn_head

        self.in_mlp = MLP(input_dim, hidden_dim, hidden_dim)
        self.out_mlp = MLP(hidden_dim, hidden_dim, input_dim)

        self.pe_emb = torch.nn.Embedding(50, hidden_dim)
        self.time_emb = nn.Sequential(Sine_PE_t(hidden_dim), MLP(hidden_dim, 2*hidden_dim, hidden_dim))
        
        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_layers):
            self.decoder_layers.append(Decoder_layer(hidden_dim, attn_head, dropout))

    def forward(self, x, batch, t):
        b, l, _ = x.shape
        h = self.in_mlp(x)
        t_emb = self.time_emb(t).unsqueeze(1)
        p_emb = self.pe_emb.weight[:l].unsqueeze(0).repeat(b, 1, 1)
        h = h + p_emb

        for i in range(len(self.decoder_layers)):
            h = h + t_emb
            h[~batch] = 0.
            h = self.decoder_layers[i](h, batch)
        h = self.out_mlp(h)
        h[~batch] = 0.
        return h
