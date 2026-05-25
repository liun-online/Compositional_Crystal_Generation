import os
import numpy as np
import torch
import gc
import argparse
import subprocess
import sys
import os
from utils.set_logger import get_logger
from utils.config import DictAction, get_params


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config(task):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mp_20')
    parser.add_argument('--port', type=str, default='11111')
    parser.add_argument('--conf_new', nargs='+', action=DictAction)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_seed', action='store_true')
    args = parser.parse_args()

    args.task = task
    args.log = os.path.join('logs', args.task, args.dataset)
    os.makedirs(args.log, exist_ok=True)

    args.cfg = os.path.join('configs', args.task, args.dataset+'.yml')
    config = get_params(args)
    new_config = dict2namespace(config)

    #### setup logger #### 
    logger = get_logger(path=os.path.join(args.log, 'running.log'))
    args.logger = logger

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    if args.use_seed == True:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

class Scaler_min_max(object):
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def fit(self, X):
        X = X.float()
        self.min = torch.min(X)
        self.max = torch.max(X)

    def transform(self, X):
        X = X.float()
        new_X = (X - self.min.to(X.device)) / (self.max.to(X.device) - self.min.to(X.device))
        return new_X

    def inverse_transform(self, X):
        X = X.float()
        new_X = X * (self.max.to(X.device) - self.min.to(X.device)) + self.min.to(X.device)
        return new_X

    def pack(self):
        return torch.tensor([self.min, self.max])
    
    def load(self, re):
        self.min, self.max = re


class Scaler_mean_std(object):
    def __init__(self, means=None, stds=None, replace_nan_token=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        self.means = torch.mean(X, dim=0, keepdim=True)
        self.stds = torch.std(X, dim=0, keepdim=True)
        return self

    def transform(self, X):
        X = X.float()
        new_X = (X - self.means) / self.stds
        return new_X

    def inverse_transform(self, X):
        X = X.float()
        new_X = X * self.stds + self.means
        return new_X

    def pack(self):
        return torch.cat([self.means, self.stds], 0)
    
    def load(self, re):
        self.means, self.stds = re

def get_scaler_min_max(task, dataset, scaled_matrix=None):
    matrix_scaler_path = os.path.join('data', task, dataset, 'min_max_scaler.pt')
    matrix_scaler = Scaler_min_max()
    if not os.path.exists(matrix_scaler_path):
        matrix_scaler.fit(scaled_matrix)
        torch.save(matrix_scaler.pack(), matrix_scaler_path)
    else:
        matrix_scaler.load(torch.load(matrix_scaler_path))
    return matrix_scaler

def get_scaler_mean_std(task, dataset, scaled_matrix=None):
    matrix_scaler_path = os.path.join('data', task, dataset, 'mean_std_scaler.pt')
    matrix_scaler = Scaler_mean_std()
    if not os.path.exists(matrix_scaler_path):
        matrix_scaler.fit(scaled_matrix)
        torch.save(matrix_scaler.pack(), matrix_scaler_path)
    else:
        matrix_scaler.load(torch.load(matrix_scaler_path))
    return matrix_scaler

def last_ckpt(task, dataset="mp_20"):
    saved_model_path = os.path.join('logs', task, dataset, 'saved_model')
    all_files = os.listdir(saved_model_path)
    all_saved_epochs = []
    for file in all_files:
        if os.path.splitext(file)[1]==".pt":
            all_saved_epochs.append(int(file.split('.')[0].split('_')[1]))
    all_saved_epochs.sort()
    ckpt = os.path.join(saved_model_path, 'model_'+str(all_saved_epochs[-1])+'.pt')
    return ckpt

def check_save_num(save_path):
    all_files = os.listdir(save_path)
    all_saved_epochs, all_saved_files = [], []
    for file in all_files:
        if os.path.splitext(file)[1]==".pt":
            all_saved_files.append(str(file))
            all_saved_epochs.append(int(os.path.splitext(file)[0].split("_")[1]))
    if len(all_saved_epochs) > 10:
        all_saved_files, all_saved_epochs = np.array(all_saved_files), np.array(all_saved_epochs)
        idx = np.argsort(all_saved_epochs)
        all_saved_files, all_saved_epochs = all_saved_files[idx], all_saved_epochs[idx]
        remove_file = all_saved_files[0]
        os.remove(os.path.join(save_path, remove_file))

def run_relax_subprocess(turn, args):
    cmd = [
        sys.executable,
        "relax_worker.py",
        "--gen_path", os.path.join(args.gen_path, f"gen_{turn}.pt"),
        "--relax_path", os.path.join(args.eval_path, str(turn), f"relaxed_{args.mlff}.json"),
        "--eval_path", os.path.join(args.eval_path, str(turn)),
        "--mlff", args.mlff,
        "--steps", str(args.max_relax_steps),
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Relax subprocess failed for turn={turn}")
    
    v_indicator = np.load(os.path.join(args.eval_path, str(turn), "v_indicator.npy"))
    return len(v_indicator), v_indicator
