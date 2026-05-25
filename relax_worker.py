import argparse
import torch
import numpy as np
import os

from utils.crys_utils import dict_to_struct
from eval.vsun.relax import relax_structures


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, required=True)
    parser.add_argument("--relax_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--mlff", type=str, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gen = torch.load(os.path.join(args.gen_path), weights_only=False)
    num_samples = len(gen)

    ########## convert to Structures ##########
    print("Converting generated tensors into Structures...")
    structures, v_indicator = dict_to_struct(gen)
    assert len(v_indicator) == num_samples
    np.save(os.path.join(args.eval_path, "v_indicator.npy"), v_indicator)


    ########## relax structure ##########
    print("Relaxing the generated structures, using "+args.mlff)
    
    if not os.path.exists(args.relax_path):
        df = relax_structures(structures, device=args.device, steps=args.steps, mlff=args.mlff)
        df.to_json(args.relax_path)



if __name__ == "__main__":
    main()