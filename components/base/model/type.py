import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class type_noise(nn.Module):
    def __init__(self, T):
        super(type_noise, self).__init__()
        self.cosine_schedule(T)
    
    def forward(self, t, type):
        device = type.device
        noise = self.gaussian(type.shape, device)
        type_t = self.sqrt_alphas_cumprod.to(device)[t] * type + self.sqrt_one_minus_alphas_cumprod.to(device)[t] * noise
        return type_t, noise
    
    def reverse(self, t, noisy_type, pred_noise):
        device = noisy_type.device
        mu_t = self.mu_coe_1.to(device)[t] * (noisy_type - self.mu_coe_2.to(device)[t] * pred_noise)
        log_var_t = self.log_beta_coe.to(device)[t]
        noisy_type = mu_t + (0.5 * log_var_t).exp() * self.gaussian(noisy_type.shape, device)
        return noisy_type
    
    def cal_loss(self, pred, true):
        return F.mse_loss(pred, true)
    
    def gaussian(self, shape, device):
        noise = torch.randn(shape).to(device)
        return noise

    def cosine_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.mu_coe_1 = 1. / torch.sqrt(alphas)
        self.mu_coe_2 = (1. - alphas) / torch.sqrt(1 - alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        beta_coe = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.log_beta_coe = torch.log(beta_coe.clamp(min =1e-20))