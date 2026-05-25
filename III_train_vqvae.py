
import os
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from I_train_vae import collate
from components.vae.dataloader import CrystalDataset
from components.vqvae.model.vqvae import VQVAE

from utils.utils import parse_args_and_config, get_scaler_min_max, last_ckpt, check_save_num

import wandb
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def train(rank, world_size, args, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    start_epoch = 0

    ## initialize
    train = CrystalDataset('vae', "mp_20", config=config)
    all_scaled_matrix = train.data['scaled_matrix']
    train.matrix_scaler = get_scaler_min_max('vae', "mp_20", all_scaled_matrix)
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train, sampler=train_sampler, batch_size=config.training.batch_size, collate_fn=collate)
    
    val = CrystalDataset('vae', "mp_20", config=config, val=True)
    val.matrix_scaler = get_scaler_min_max('vae', "mp_20")
    val_sampler = DistributedSampler(val, num_replicas=world_size, rank=rank, shuffle=True)
    val_loader = DataLoader(val, sampler=val_sampler, batch_size=config.training.batch_size, collate_fn=collate)

    device = torch.device(f"cuda:{rank}")
    model = VQVAE(config).to(device)
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    ckpt = last_ckpt("vae")
    checkpoint = torch.load(ckpt, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        module_name = k.replace('module.', '', 1)
        if module_name == "quant_conv.weight" or module_name == "quant_conv.bias":
            new_state_dict[module_name] = v[: int(v.shape[0] / 2)]
        else:
            new_state_dict[module_name] = v

    model.load_state_dict(new_state_dict, strict=False)

    model = DDP(model, device_ids=[rank], output_device=rank)

    if rank==0:
            wandb.init(
                project=config.wandb.project, 
                name=args.task+"_"+args.dataset,
                config=config,
                resume="allow"
            )
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.training.lr, weight_decay=0.0, betas=(0.9, 0.99), amsgrad=False)
    if config.use_gradscalar:
        scaler = GradScaler(enabled = True)
    if config.use_schedule:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.6, patience=70, min_lr=1e-6)
    else:
        scheduler = None

    best_val_match_rate = 0.
    print("start at ", start_epoch)
    ## Training
    for epoch in tqdm(range(start_epoch+1, start_epoch+config.training.epoch), desc="Training..."):
        curr_loss = []
        curr_a_loss, curr_p_loss, curr_m_loss, curr_c_loss = [], [], [], []
        train_sampler.set_epoch(epoch)
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            atom_loss_, pos_loss_, matrix_loss_, codebook_loss_, loss_ = model(data)
            loss, atom_loss, pos_loss, matrix_loss, codebook_loss = loss_.mean(), atom_loss_.mean(), pos_loss_.mean(), matrix_loss_.mean(), codebook_loss_.mean()
            if config.use_gradscalar:
                with autocast(device_type="cuda"):
                    scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            curr_loss.append(loss.detach().item())
            curr_a_loss.append(atom_loss.detach().item())
            curr_p_loss.append(pos_loss.detach().item())
            curr_m_loss.append(matrix_loss.detach().item())
            curr_c_loss.append(codebook_loss.detach().item())

        curr_loss = sum(curr_loss) / len(curr_loss)
        curr_a_loss = sum(curr_a_loss) / len(curr_a_loss)
        curr_p_loss = sum(curr_p_loss) / len(curr_p_loss)
        curr_m_loss = sum(curr_m_loss) / len(curr_m_loss)
        curr_c_loss = sum(curr_c_loss) / len(curr_c_loss)

        if epoch % config.training.check == 0:
            model.eval()
            with torch.no_grad():
                val_match_rate = model.module.eval_val(val_loader, train.matrix_scaler).mean().cpu().item()

        if rank==0:
            wandb.log({"epoch": epoch, "epoch_loss": curr_loss, "epoch_a_loss": curr_a_loss, "epoch_p_loss": curr_p_loss, "epoch_m_loss": curr_m_loss, "epoch_c_loss": curr_c_loss}, step=epoch)
            if config.use_schedule:
                scheduler.step(curr_loss)
                wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)

            if epoch % config.training.check == 0:
                wandb.log({"val_match_rate": val_match_rate}, step=epoch)

                if best_val_match_rate < val_match_rate:
                    best_val_match_rate = val_match_rate
                    save_path = os.path.join(args.log, 'saved_model')
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_match_rate": val_match_rate,
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    }, os.path.join(save_path , f'model_{epoch}.pt'))
                    
                    check_save_num(save_path)
    args.logger.info('training completed')
    dist.destroy_process_group()
    if rank==0:
        wandb.finish()


if __name__ == '__main__':
    args, config = parse_args_and_config("vqvae")
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args, config), nprocs=world_size, join=True)