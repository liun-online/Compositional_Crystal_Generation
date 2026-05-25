import os
import json
import pandas as pd
from copy import deepcopy
import argparse
import torch

from eval.vsun.utils import maybe_get_missing_columns, COLUMNS_COMPUTATIONS, filter_prerelaxed
from eval.vsun.structure_summary import get_metrics_structure_summaries
from eval.vsun.reference.reference_dataset import ReferenceDataset
from eval.vsun.reference.presets import MP2023, ReferenceMP2020Correction
from eval.vsun.evaluator import Evaluator

from utils.utils import run_relax_subprocess
from utils.set_logger import get_logger
from utils.crys_utils import generate_compo_one_batch, generate_crys_based_on_compo_one_batch

from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mp_20')
    args = parser.parse_args()
    args.task = "base"
    args.mlff = 'mattersim'
    args.stable_delta = 0.1
    args.max_relax_steps = 500
    args.gen_path = os.path.join('logs', args.task, args.dataset, "gen")
    args.eval_path = os.path.join('logs', args.task, args.dataset, 'eval')
    os.makedirs(args.eval_path, exist_ok=True)

    logger = get_logger(path=os.path.join(args.eval_path, 'eval_vsun.log'))
    args.logger = logger
    args.logger.info(args)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'mp_20':
        args.ehull_ref = "mp"
        reference = MP2023()
    elif args.dataset == 'alex_mp_20':
        args.ehull_ref = "alex_mp"
        reference = ReferenceMP2020Correction()
    
    # Note: The evaluation pipeline may crash if MatterSim receives a material that cannot be transformed into a graph. 
    # Currently, we handle this by ignoring the batch and generating a new one  -- a procedure applied to all evaluated models. 
    # Future work will patch MatterSim to simply flag these materials as invalid.
    for turn in range(10):
        while True:
            try:
                relax_path = os.path.join(args.eval_path, str(turn), "relaxed_"+args.mlff+".json")
                os.makedirs(os.path.join(args.eval_path, str(turn)), exist_ok=True)
        
                args.num_samples, v_indicator = run_relax_subprocess(turn, args)
                ########## change to Reference dataset (MatterGen) ##########
                df = pd.read_json(relax_path)
                df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
                df = filter_prerelaxed(df)

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

                ########## compute e_hull ##########
                e_hull_path = os.path.join(args.eval_path, str(turn), "e_hull_"+args.mlff+"_"+args.ehull_ref+"_"+str(args.stable_delta)+".json")
                s_indicator = evaluator.get_stability(df, e_hull_path)
                u_indicator = evaluator.get_uniqueness()
                n_indicator = evaluator.get_novelty()
                assert len(v_indicator) == len(s_indicator) == len(u_indicator) == len(n_indicator)

                ########## compute VSUN ##########
                args.logger.info("Computing V.S.U.N ...")
                indicators = {
                    "v_indicator": v_indicator,
                    "s_indicator": s_indicator,
                    "u_indicator": u_indicator,
                    "n_indicator": n_indicator,
                }
                metrics = {
                    "v": (v_indicator).sum().item() / args.num_samples,
                    "s": (s_indicator).sum().item() / args.num_samples,
                    "u": (u_indicator).sum().item() / args.num_samples,
                    "n": (n_indicator).sum().item() / args.num_samples,
                    "sun": (s_indicator * u_indicator * n_indicator).sum().item() / args.num_samples,
                    "vsun": (v_indicator * s_indicator * u_indicator * n_indicator).sum().item() / args.num_samples,
                }

                torch.save(indicators, os.path.join(args.eval_path, str(turn), "indicators_"+args.mlff+"_"+args.ehull_ref+"_"+str(args.stable_delta)+".pt"))
                with open(os.path.join(args.eval_path, str(turn), "vsun_"+args.mlff+"_"+args.ehull_ref+"_"+str(args.stable_delta)+".json"), "a") as file:
                    json.dump(metrics, file, indent=4)
                break
            except RuntimeError:
                args.logger.info("Batch "+str(turn)+" : MatterSim crashed, so regenerate this batch.")
                generate_compo_one_batch(dataname=args.dataset, turn_id=turn)
                generate_crys_based_on_compo_one_batch(dataname=args.dataset, turn_id=turn)

    ########## Collect and Summarize ##########
    import numpy as np
    results = {}
    for turn in range(10):
        try:
            with open(args.eval_path+"/"+str(turn)+"/vsun_"+args.mlff+"_"+args.ehull_ref+"_"+str(args.stable_delta)+".json", "r") as file:
                metrics = json.load(file)
            for k, v in metrics.items():
                if k not in results:
                    results[k] = []
                results[k].append(v)
        except:
            pass
    string = '\n'
    for k, v in results.items():
        v = np.array(v)
        string += str(k)+": "+str(v.mean()*100)[:4]+" ± "+str(v.std()*100)[:3]+"\n"
    with open(args.eval_path+"/"+"metrics_summary.log", "a") as file:
        file.write(string)
        file.close()
