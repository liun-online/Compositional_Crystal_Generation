import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pymatgen.core.structure import Structure


def type_coords(frac_coords, atomic_numbers):
    atomic_tensor = atomic_numbers.reshape(-1, 1)
    all_tensor = torch.cat([atomic_tensor, frac_coords], -1)
    coe = 1000*all_tensor[:, 0] + 100*all_tensor[:, 1] + 10*all_tensor[:, 2] + 1*all_tensor[:, 3]
    index_order = torch.argsort(coe)
    return index_order

class CrystalDataset(Dataset):
    def __init__(self, task, datset, config, val=False):
        data_path = os.path.join('data', task, datset)
        os.makedirs(data_path, exist_ok=True)
        if val:
            path = os.path.join('./data/vae/', datset, 'val.csv')
            self.path_prepare = os.path.join(data_path, "prepared_val_data.pt")
            self.cond = torch.load(os.path.join("./data/cond_gen/", datset, "val_quants.pt"))
            self.cond_num = torch.load(os.path.join("./data/cond_gen/", datset, "val_nums.pt"))
            self.cond_id = np.load(os.path.join("./data/cond_gen/", datset, "val_ids.npy"))
        else:
            path = os.path.join('./data/vae/', datset, 'train.csv')
            self.path_prepare = os.path.join(data_path, "prepared_train_data.pt")
            self.cond = torch.load(os.path.join("./data/cond_gen/", datset, "train_quants.pt"))
            self.cond_num = torch.load(os.path.join("./data/cond_gen/", datset, "train_nums.pt"))
            self.cond_id = np.load(os.path.join("./data/cond_gen/", datset, "train_ids.npy"))
            
        self.config = config
        self.max_atoms = config.dataset.max_atoms
        self.max_types = config.dataset.max_types
        self.num_workers = config.dataset.num_workers
        self.primitive = config.dataset.primitive
        self.read_from_cif(path)
        self.matrix_scaler = None
        self.offset = torch.cat([torch.tensor([0]), torch.cumsum(self.data["num_atoms"], -1)], 0).long()
        self.offset_cond = torch.cat([torch.tensor([0]), torch.cumsum(self.cond_num, -1)], 0).long()
    
    def __len__(self) -> int:
        return len(self.data['material_id'])

    def __getitem__(self, index):
        start, end = self.offset[index], self.offset[index+1]
        scaled_matrix, num_atoms = self.data['scaled_matrix'][index], self.data['num_atoms'][index]
        cart_coords = self.data['cart_coords'][start: end]
        atomic_numbers = self.data['atomic_numbers'][start: end]
        
        scaled_cart_coords = cart_coords / num_atoms**(1/3)
        
        scalar_cart_coords = self.matrix_scaler.transform(scaled_cart_coords)
        scaler_scaled_matrix = self.matrix_scaler.transform(scaled_matrix)
        material_id = self.data["material_id"][index]

        cond_index = np.argwhere(self.cond_id==material_id)[0]
        cond_start, cond_end = self.offset_cond[cond_index], self.offset_cond[cond_index+1]
        cond = self.cond[cond_start: cond_end]
        return material_id, scaler_scaled_matrix, scalar_cart_coords, atomic_numbers, num_atoms, cond
    
    def add_scaled_matrix(self, data, scale_len=True):
        matrix = data['matrix']
        num_atoms = data['num_atoms'].reshape(-1, 1)
        if scale_len:
            matrix = matrix / num_atoms.float()**(1/3)
        data['scaled_matrix'] = matrix

    def read_from_cif(self, path):
        if not os.path.exists(self.path_prepare):
            df = pd.read_csv(path)

            unordered_results = []
            from tqdm import tqdm
            for idx in tqdm(range(len(df))):
                unordered_results.append(self.cif_info(df.iloc[idx]))
            unordered_results = np.array(unordered_results)

            # unordered_results = np.array(p_umap(self.cif_info, [df.iloc[idx] for idx in range(len(df))], num_cpus=self.num_workers)) # 
            order = np.array([result['material_id'] for result in unordered_results]).argsort()
            order_results = unordered_results[order]
            data = self.unpack(order_results)
            self.add_scaled_matrix(data)
            self.data = {
                'material_id': data['material_id'], 
                'cart_coords': data['cart_coords'], 
                'atomic_numbers': data['atomic_numbers'], 
                'scaled_matrix': data['scaled_matrix'],
                'num_atoms': data['num_atoms']
                }
            torch.save(self.data, self.path_prepare)
        else:
            self.data = torch.load(self.path_prepare, weights_only=False)
    
    def unpack(self, results):
        material_id, cart_coords, atomic_numbers, matrix, num_atoms = [], [], [], [], []
        for re in results:
            material_id.append(re['material_id'])
            num_atoms.append(torch.LongTensor([re['num_atoms']]))
            matrix.append(re['matrix'])
            cart_coords.append(re['cart_coords'])
            atomic_numbers.append(re['atomic_numbers'].long())

        matrix, material_id, num_atoms, cart_coords, atomic_numbers = \
            torch.stack(matrix, 0), np.array(material_id), torch.cat(num_atoms, 0), torch.cat(cart_coords, 0), torch.cat(atomic_numbers, 0)
        
        return {'material_id': material_id, 'cart_coords':cart_coords, 'atomic_numbers':atomic_numbers, 'matrix':matrix, 'num_atoms': num_atoms}

    def cif_info(self, row):
        cif, material_id = row['cif'], row['material_id']
        structure = Structure.from_str(cif, fmt='cif')
        if self.primitive:
            structure = structure.get_primitive_structure()
        structure = structure.get_reduced_structure()  ## niggli
        
        matrix = torch.tensor(structure.lattice.matrix)
        matrix = self.compute_lattice_polar_decomposition(matrix)
        canonical_structure = Structure(
            lattice=matrix, 
            species=structure.species,
            coords=structure.frac_coords,
            coords_are_cartesian=False,
        )

        cart_coords = torch.FloatTensor(canonical_structure.cart_coords)
        atomic_numbers = torch.FloatTensor(canonical_structure.atomic_numbers)
        matrix = torch.tensor(canonical_structure.lattice.matrix)
        tri_indices = torch.triu_indices(3, 3)
        matrix = matrix[tri_indices[0], tri_indices[1]]

        index_order = type_coords(cart_coords, atomic_numbers)
        cart_coords = cart_coords[index_order]
        atomic_numbers = atomic_numbers[index_order]

        num_atom = len(atomic_numbers)
        return {'material_id': material_id, 'cart_coords': cart_coords, 'atomic_numbers':atomic_numbers, 'matrix': matrix, 'num_atoms': num_atom,}

    def compute_lattice_polar_decomposition(self, lattice_matrix: torch.Tensor) -> torch.Tensor:
        W, S, V_transp = torch.linalg.svd(lattice_matrix)
        S_square = torch.diag_embed(S)
        V = V_transp.transpose(0, 1)
        U = W @ V_transp
        P = V @ S_square @ V_transp
        P_prime = U @ P @ U.transpose(0, 1)
        # symmetrized lattice matrix
        symm_lattice_matrix = P_prime
        symm_lattice_matrix[torch.abs(symm_lattice_matrix) < 1e-5] = 0.

        # tri_indices = torch.triu_indices(3, 3)
        # symm_lattice_matrix = symm_lattice_matrix[tri_indices[0], tri_indices[1]]
        return symm_lattice_matrix
