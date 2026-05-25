import argparse
import torch
from copy import deepcopy
import os
import pandas as pd

from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

from eval.vsun.relax import relax_structures
from eval.vsun.structure_summary import get_metrics_structure_summaries
from eval.vsun.reference.reference_dataset import ReferenceDataset
from eval.vsun.reference.presets import MP20_train, Alex_MP_train
from eval.vsun.evaluator import Evaluator

from utils.set_logger import get_logger
from utils.crys_utils import dict_to_valid_struct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mp_20')
    args = parser.parse_args()
    args.task = "cond_gen"
    gen = torch.load(os.path.join('logs', args.task, args.dataset, 'gen', "gen_structs.pt"), weights_only=False)

    args.num_samples = len(gen)
    args.eval_path = os.path.join('logs', args.task, args.dataset, 'eval', 'vsun')
    os.makedirs(args.eval_path, exist_ok=True)
    
    logger = get_logger(path=os.path.join(args.eval_path, 'eval_vsun.log'))
    args.logger = logger
    args.mlff = 'mattersim'
    args.max_relax_steps = 500
    device = torch.device('cuda')

    print("Loading Reference")
    if args.dataset == 'mp_20':
        args.ehull_ref = "mp20_train"
        reference = MP20_train()
        args.stable_delta = 0.1
    elif args.dataset == 'alex_mp_20':
        args.ehull_ref = "alex_mp_train"
        reference = Alex_MP_train()
        args.stable_delta = 0.05
    
    relax_path = os.path.join(args.eval_path, "relaxed_"+args.mlff+".json")
    print("Relax")
    if not os.path.exists(relax_path):
        structures, valid_idx = dict_to_valid_struct(gen)
        valid_idx = torch.LongTensor(valid_idx)
        torch.save(valid_idx, os.path.join(args.eval_path, "v_idx.pt"))
        df = relax_structures(structures, device=device, steps=args.max_relax_steps, mlff=args.mlff, logger=logger)
        df.to_json(relax_path)
    else:
        valid_idx = torch.load(os.path.join(args.eval_path, "v_idx.pt"))
        df = pd.read_json(relax_path)
    print("Relax Finish")
    
    relaxed_structures, relaxed_energies = [Structure.from_dict(s) for s in df["structure"]], df["e_relax"].tolist()
    structure_summaries = get_metrics_structure_summaries(
            structures=relaxed_structures,
            energies=relaxed_energies,
            energy_correction_scheme=MaterialsProject2020Compatibility(),
        )
    data_entries = [deepcopy(s.entry) for s in structure_summaries]
    for i, e in enumerate(data_entries):
        e.entry_id = i
    gen_ref = ReferenceDataset.from_entries("data_entries", data_entries)

    evaluator = Evaluator(args, gen_ref, reference)

    e_hull_path = os.path.join(args.eval_path, "e_hull_"+args.mlff+"_"+args.ehull_ref+"_"+str(args.stable_delta)+".json")
    s_indicator = evaluator.get_stability(df, e_hull_path)
    u_indicator = evaluator.get_uniqueness()
    n_indicator = evaluator.get_novelty()
    sun_indicators = torch.BoolTensor(s_indicator * u_indicator * n_indicator)

    vsun_idx = valid_idx[sun_indicators]
    torch.save(vsun_idx, os.path.join(args.eval_path, "vsun_idx.pt"))

    before_quant= torch.load(os.path.join("logs", args.task, args.dataset, 'gen', "gen_before_quants.pt"))
    num_atoms = torch.load(os.path.join("logs", args.task, args.dataset, 'gen', "num_atoms.pt"))
    offset = torch.cat([torch.zeros(1), torch.cumsum(num_atoms, -1)], -1).long()
    
    vsun_num_atoms = [num_atoms[i] for i in vsun_idx]
    vsun_before_quant = [before_quant[offset[i]: offset[i+1]] for i in vsun_idx]
    
    vsun_num_atoms = torch.LongTensor(vsun_num_atoms)
    vsun_before_quant = torch.cat(vsun_before_quant, 0)
    torch.save(vsun_num_atoms, os.path.join("./data/cond_gen/", args.dataset, "ft_num_atoms.pt"))
    torch.save(vsun_before_quant, os.path.join("./data/cond_gen/", args.dataset, "ft_before_quant.pt"))

