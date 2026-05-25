
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from components.base.model.ddpm import DDPM
from components.base.dataloader import CrystalDataset
from utils.utils import parse_args_and_config, get_scaler_min_max, check_save_num

import wandb
import warnings
from tqdm import tqdm
from datetime import timedelta

warnings.filterwarnings('ignore')


def collate(batch):
    material_id, scaler_scaled_matrix, scalar_cart_coords, atomic_numbers, num_atoms, cond = zip(*batch)
    
    matrix = torch.stack(scaler_scaled_matrix, 0)
    cart_coords = torch.cat(scalar_cart_coords, 0)
    cond = torch.cat(cond, 0)
    atomic_numbers = torch.cat(atomic_numbers, 0)
    num_atoms = torch.LongTensor(list(num_atoms))
    return matrix.float(), cart_coords.float(), atomic_numbers.long(), num_atoms, cond.float()

def train(rank, world_size, args, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timedelta(hours=2))
    torch.cuda.set_device(rank)

    start_epoch = 0

    ## initialize
    train = CrystalDataset(args.task, args.dataset, config=config)
    all_scaled_matrix = train.data['scaled_matrix']
    train.matrix_scaler = get_scaler_min_max(args.task, args.dataset, all_scaled_matrix)
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train, sampler=train_sampler, batch_size=config.training.batch_size, collate_fn=collate)

    val = CrystalDataset(args.task, args.dataset, config=config, val=True)
    val.matrix_scaler = get_scaler_min_max(args.task, args.dataset)
    val_sampler = DistributedSampler(val, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val, sampler=val_sampler, batch_size=config.training.batch_size_eval, collate_fn=collate)

    device = torch.device(f"cuda:{rank}")
    diffusion = DDPM(config).to(device)
    print("Model parameters: {}".format(sum(p.numel() for p in diffusion.parameters())))

    diffusion = DDP(diffusion, device_ids=[rank], output_device=rank)

    if rank==0:
            wandb.init(
                project=config.wandb.project, 
                name=args.task+"_"+args.dataset,
                config=config,
                resume="allow"
            )
        
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=config.training.lr, weight_decay=0.0, betas=(0.9, 0.99), amsgrad=False)
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
        curr_loss_l, curr_loss_a, curr_loss_x = [], [], []
        train_sampler.set_epoch(epoch)
        diffusion.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss_l_ , loss_a_, loss_x_, loss_ = diffusion(data)
            loss_l, loss_a, loss_x, loss = loss_l_.mean(), loss_a_.mean(), loss_x_.mean(), loss_.mean()
            if config.use_gradscalar:
                with autocast(device_type="cuda"):
                    scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            curr_loss.append(loss.detach().item())
            curr_loss_l.append(loss_l.detach().item())
            curr_loss_a.append(loss_a.detach().item())
            curr_loss_x.append(loss_x.detach().item())

        curr_loss = sum(curr_loss) / len(curr_loss)
        curr_loss_l = sum(curr_loss_l) / len(curr_loss_l)
        curr_loss_a = sum(curr_loss_a) / len(curr_loss_a)
        curr_loss_x = sum(curr_loss_x) / len(curr_loss_x)

        if epoch % config.training.check == 0:
            diffusion.eval()
            with torch.no_grad():
                val_match_rate = diffusion.module.eval_val(val_loader, train.matrix_scaler).mean().cpu().item()

        if rank==0:
            wandb.log({"epoch": epoch, "epoch_loss": curr_loss, \
                       "epoch_l": curr_loss_l, "epoch_a": curr_loss_a, "epoch_x": curr_loss_x}, step=epoch)
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
                        "model_state_dict": diffusion.state_dict(),
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
    args, config = parse_args_and_config("base")
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args, config), nprocs=world_size, join=True)