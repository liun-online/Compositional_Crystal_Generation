import numpy as np
import torch
import smact
import itertools
from collections import Counter
from smact.screening import pauling_test
from tqdm import tqdm

from pymatgen.core import Structure, Element
from pymatgen.core.lattice import Lattice


class Crystal(object):
    def __init__(self, crys_array_dict, gt=False, cart=False, la=False, test_validity=True):
        self.cart, self.la = cart, la
        if self.cart:
            self.card_coords = crys_array_dict['cart_coords']
        else:
            self.frac_coords = crys_array_dict['frac_coords']

        if self.la:
            self.length = crys_array_dict['length']
            self.angle = crys_array_dict['angle']
        else:
            self.matrix = crys_array_dict['matrices']
        self.atom_types = crys_array_dict['atom_types']
        self.structure = None
        self.get_structure()
        if self.structure is None:
            self.valid = False
        else:
            try:
                self.get_composition()
                if gt:
                    self.valid = True
                else:
                    if test_validity:
                        self.get_validity()
                    else:
                        self.valid = True
            except:
                self.valid = False

    def get_structure(self):
        try:
            if self.la:
                lattice = Lattice.from_parameters(*(self.length.tolist() + self.angle.tolist()))
            else:
                lattice = self.matrix
            if self.cart:
                self.structure = Structure(
                    lattice=lattice, species=self.atom_types, coords=self.card_coords, coords_are_cartesian=True)
            else:
                self.structure = Structure(
                    lattice=lattice, species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
            self.constructed = True
        except Exception:
            self.constructed = False
    
    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = self.smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = self.structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid
    
    def structure_validity(self, structure: Structure, cutoff: float = 0.5) -> bool:
        dist_mat = structure.distance_matrix
        # Pad diagonal with a large number
        dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
        # Note: the threshold 0.1 comes from the CDVAE code
        # https://github.com/txie-93/cdvae/blob/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/scripts/eval_utils.py#L170
        if dist_mat.min() < cutoff or structure.volume < 0.1:
            return False
        else:
            return True

    def smact_validity(
        self,
        comp: tuple[int, ...] | tuple[str, ...],
        count: tuple[int, ...],
        use_pauling_test: bool = True,
        include_alloys: bool = True,
        include_cutoff: bool = False,
        use_element_symbol: bool = False,
    ) -> bool:
        """Computes SMACT validity.

        Args:
            comp: Tuple of atomic number or element names of elements in a crystal.
            count: Tuple of counts of elements in a crystal.
            use_pauling_test: Whether to use electronegativity test. That is, at least in one
                combination of oxidation states, the more positive the oxidation state of a site,
                the lower the electronegativity of the element for all pairs of sites.
            include_alloys: if True, returns True without checking charge balance or electronegativity
                if the crystal is an alloy (consisting only of metals) (default: True).
            include_cutoff: assumes valid crystal if the combination of oxidation states is more
                than 10^6 (default: False).

        Returns:
            True if the crystal is valid, False otherwise.
        """
        try:
            assert len(comp) == len(count)
            if use_element_symbol:
                elem_symbols = comp
            else:
                elem_symbols = tuple([str(Element.from_Z(Z=elem)) for elem in comp])  # type:ignore
            space = smact.element_dictionary(elem_symbols)
            smact_elems = [e[1] for e in space.items()]
            electronegs = [e.pauling_eneg for e in smact_elems]
            ox_combos = [e.oxidation_states for e in smact_elems]
            if len(set(elem_symbols)) == 1:
                return True
            if include_alloys:
                is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
                if all(is_metal_list):
                    return True

            threshold = np.max(count)
            compositions = []
            n_comb = np.prod([len(ls) for ls in ox_combos])
            # If the number of possible combinations is big, it'd take too much time to run the smact checker
            # In this case, we assume that at least one of the combinations is valid
            if n_comb > 1e6 and include_cutoff:
                return True
            for ox_states in itertools.product(*ox_combos):
                stoichs = [(c,) for c in count]
                # Test for charge balance
                cn = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
                if len(cn) == 2:
                    cn_e, cn_r = cn
                elif len(cn) == 1:
                    cn_r = cn
                    cn_e = len(cn_r) > 0
                # Electronegativity test
                if cn_e:
                    if use_pauling_test:
                        try:
                            electroneg_OK = pauling_test(ox_states, electronegs)
                        except TypeError:
                            # if no electronegativity data, assume it is okay
                            electroneg_OK = True
                    else:
                        electroneg_OK = True
                    if electroneg_OK:
                        for ratio in cn_r:
                            compositions.append(tuple([elem_symbols, ox_states, ratio]))
            compositions = [(i[0], i[2]) for i in compositions]
            compositions = list(set(compositions))
            if len(compositions) > 0:
                return True
            else:
                return False
        except:
            return False

def vector2matrix(vector):
    A = torch.zeros(3, 3, dtype=torch.float32, device=vector.device)
    A[0, 0] = vector[0]
    A[0, 1] = A[1, 0] = vector[1]
    A[0, 2] = A[2, 0] = vector[2]
    A[1, 1] = vector[3]
    A[1, 2] = A[2, 1] = vector[4]
    A[2, 2] = vector[5]
    return A

def replace_elements(structure: Structure) -> Structure:
    ## This def is to replace elements > 94 with 94
    ## MatterSim only accepts elements with index <=94
    for i, site in enumerate(structure):
        elem = site.specie.element if hasattr(site.specie, "element") else site.specie
        if elem.Z > 94:
            structure.replace(i, Element.from_Z(94))
    return structure

def dict_to_struct(gen):
    structures = []
    valid_index = []
    k = 0
    for one in tqdm(gen):
        cry = Crystal(one, cart=True)
        if cry is not None:
            structure = cry.structure
            if structure is not None:
                structures.append(replace_elements(structure))
                if cry.valid:
                    valid_index.append(k)
                k += 1
    v_indicator = torch.zeros(len(structures))
    v_indicator[torch.LongTensor(valid_index)] = 1.
    return structures, v_indicator.bool().numpy()

def dict_to_valid_struct(gen):
    structures = []
    valid_idx = []
    for i, one in tqdm(enumerate(gen)):
        cry = Crystal(one, cart=True)
        if cry is not None:
            structure = cry.structure
            if structure is not None and cry.valid:
                structures.append(replace_elements(structure))
                valid_idx.append(i)
    return structures, valid_idx

def generate_compo_one_batch(dataname, turn_id):
    import os
    from components.vqvae.model.vqvae import VQVAE
    from components.cond_gen.model.ddpm import DDPM
    from components.cond_gen.dataloader import CrystalDataset
    from utils.utils import get_scaler_mean_std, dict2namespace, last_ckpt
    from utils.config import parse_cfg

    task = "cond_gen_ft"
    device = torch.device('cuda')
    config = parse_cfg('logs/'+task+'/'+dataname+'/backup-config.yaml')
    config = dict2namespace(config)
    config.device = device
    diffusion = DDPM(config)
    ckpt = last_ckpt(task, dataname)
    checkpoint = torch.load(ckpt, map_location="cpu")
    new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
    diffusion.load_state_dict(new_state_dict)
    diffusion = diffusion.cuda()
    diffusion.eval()

    dataset = CrystalDataset(dataname)
    generator_scaler = get_scaler_mean_std("cond_gen", dataname)

    num_prob = torch.zeros(21)
    for i in range(1, 21):
        num_prob[i] = (dataset.num_atoms == i).sum()
    num_prob /= num_prob.sum()

    save_path = "./logs/cond_gen_ft/"+dataname+"/gen"
    os.makedirs(save_path, exist_ok=True)
    b = 1000

    ## generate new compositions
    num_atoms = torch.multinomial(num_prob, num_samples=b, replacement=True).cuda()
    with torch.no_grad():
        before_quants, _ = diffusion.reverse(b, num_atoms)
        before_quants = generator_scaler.inverse_transform(before_quants.cpu().data)

    ## quantize the compositions
    config = parse_cfg('logs/vqvae/mp_20/backup-config.yaml')
    config = dict2namespace(config)
    config.device = device

    vqvae = VQVAE(config).cuda()
    ckpt = last_ckpt("vqvae", "mp_20")
    checkpoint = torch.load(ckpt)
    new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
    vqvae.load_state_dict(new_state_dict)
    vqvae.eval()

    with torch.no_grad():
        gen_quants, _ = vqvae.quantize(before_quants.cuda())
    torch.save(gen_quants.cpu().data, os.path.join(save_path, "gen_quants_"+str(turn_id)+".pt"))
    torch.save(num_atoms.cpu().data, os.path.join(save_path, "num_atoms_"+str(turn_id)+".pt"))

def generate_crys_based_on_compo_one_batch(dataname, turn_id):
    import os
    from torch_scatter import scatter_mean
    from components.base.model.ddpm import DDPM
    from utils.utils import get_scaler_min_max, dict2namespace, last_ckpt
    from utils.config import parse_cfg

    task = "base"
    device = torch.device('cuda')
    config = parse_cfg('logs/'+task+'/'+dataname+'/backup-config.yaml')
    config = dict2namespace(config)
    config.device = device

    diffusion = DDPM(config).cuda()
    ckpt = last_ckpt("base", dataname)
    checkpoint = torch.load(ckpt)
    new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
    diffusion.load_state_dict(new_state_dict)
    diffusion.eval()
    base_scaler = get_scaler_min_max("base", dataname)

    gen_path = os.path.join('logs', task, dataname, "gen")
    os.makedirs(gen_path, exist_ok=True)

    gen_quants= torch.load(os.path.join("logs/cond_gen_ft", dataname, 'gen', "gen_quants_"+str(turn_id)+".pt")).cuda()
    num_atoms = torch.load(os.path.join("logs/cond_gen_ft", dataname, 'gen', "num_atoms_"+str(turn_id)+".pt")).cuda()

    b = len(num_atoms)
    gen_crys = []
    with torch.no_grad():
        x, batch = diffusion.reverse(b, gen_quants, num_atoms)
        atomic_numbers_pred = x[:, 3:-6].argmax(-1)
        atomic_numbers_pred[atomic_numbers_pred==0] = 1
        cart_coords_pred = x[:, :3]
        cart_coords_pred = base_scaler.inverse_transform(cart_coords_pred) * num_atoms[batch].reshape(-1, 1).float()**(1/3)
        matrix_pred = base_scaler.inverse_transform(scatter_mean(x[:, -6:], batch, dim=0)) * num_atoms.reshape(-1, 1).float()**(1/3)

        start_id = 0
        for idx_in_batch, num in enumerate(num_atoms):
            _atom_types = atomic_numbers_pred[start_id: start_id + num]
            _cart_coords = cart_coords_pred[start_id: start_id + num]
            _matrix = matrix_pred[idx_in_batch]

            gen_crys.append(
                {
                    "atom_types": _atom_types.detach().cpu().numpy(),
                    "cart_coords": _cart_coords.detach().cpu().numpy(),
                    "matrices": vector2matrix(_matrix.detach().cpu()).numpy()
                }
            )
            start_id = start_id + num
    torch.save(gen_crys, os.path.join(gen_path, "gen_"+str(turn_id)+".pt"))