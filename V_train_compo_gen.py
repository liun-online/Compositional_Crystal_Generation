
import os
import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from components.cond_gen.model.ddpm import DDPM
from components.cond_gen.dataloader import CrystalDataset
from utils.utils import parse_args_and_config, get_scaler_mean_std, last_ckpt, check_save_num

import wandb
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
def collate(batch):
    scaler_quants, num_atoms = zip(*batch)  # unzip list of tuples
    quants = torch.cat(scaler_quants, 0)
    num_atoms = torch.LongTensor(list(num_atoms))
    return num_atoms, quants.float()

def train(rank, world_size, args, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    start_epoch = 0
    train = CrystalDataset(args.dataset)
    all_quants = train.quants
    train.scaler = get_scaler_mean_std("cond_gen", args.dataset, all_quants)
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train, sampler=train_sampler, batch_size=config.training.batch_size, collate_fn=collate)
    

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
    best_loss = 1e9
    
    print("start at ", start_epoch)
    ## Training
    for epoch in tqdm(range(start_epoch+1, start_epoch+config.training.epoch), desc="Training..."):
        curr_loss = []
        train_sampler.set_epoch(epoch)
        diffusion.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss = diffusion(data).mean()
            if config.use_gradscalar:
                with autocast(device_type="cuda"):
                    scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            curr_loss.append(loss.detach().item())

        curr_loss = sum(curr_loss) / len(curr_loss)

        if rank==0:
            wandb.log({"epoch": epoch, "epoch_loss": curr_loss}, step=epoch)
            if config.use_schedule:
                scheduler.step(curr_loss)
                wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
            if curr_loss < best_loss:
                best_loss = curr_loss
                save_path = os.path.join(args.log, 'saved_model')
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                }, os.path.join(save_path , f'model_{epoch}.pt'))
                    
                check_save_num(save_path)

    args.logger.info('training completed')
    dist.destroy_process_group()
    if rank==0:
        wandb.finish()


if __name__ == '__main__':
    args, config = parse_args_and_config("cond_gen")
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args, config), nprocs=world_size, join=True)