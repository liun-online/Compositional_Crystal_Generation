
import argparse
from tqdm import tqdm

from utils.crys_utils import generate_crys_based_on_compo_one_batch
from utils.config import DictAction

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mp_20')
parser.add_argument('--conf_new', nargs='+', action=DictAction)
args = parser.parse_args()

for turn_id in tqdm(range(10)):
    generate_crys_based_on_compo_one_batch(args.dataset, turn_id)
