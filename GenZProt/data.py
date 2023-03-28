"""
functions adapted and modified from CGVAE (Wang et al., ICML2022) 
https://github.com/wwang2/CoarseGrainingVAE/data.py
"""
import sys
import copy
from copy import deepcopy
from tqdm import tqdm 
import numbers

import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from ase import Atoms
from ase.neighborlist import neighbor_list

import torch
from torch.utils.data import Dataset as TorchDataset


def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch
    
def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    
def get_higher_order_adj_matrix(adj, order):
    """
    from https://github.com/MinkaiXu/ConfVAE-ICML21/blob/main/utils/transforms.py
    Args:
        adj:        (N, N)
    """
    adj_mats = [torch.eye(adj.size(0)).long(), binarize(adj + torch.eye(adj.size(0)).long())]
    for i in range(2, order+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    # print(adj_mats)

    order_mat = torch.zeros_like(adj)
    for i in range(1, order+1):
        order_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return order_mat


def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = list(set((nbr for n in nbrs for nbr in G[n])))
    return nbrs


def get_k_hop_graph(g, k=2):
    twonbrs = []
    nodelist = list(g.nodes)
    for n in nodelist:
        twonbrs.append([n.index, [nbr.index for nbr in knbrs(g, n, k)]])
    _twonbrs = []
    for n in twonbrs:
        for n2 in n[1]:
            if n[0] != n2 and n[0] < n2: 
                _twonbrs.append([n[0], n2])
    k_hop_edge_pair = torch.LongTensor(_twonbrs)
    
    return k_hop_edge_pair


def get_neighbor_list(xyz, device='cpu', cutoff=5, undirected=True):

    xyz = torch.Tensor(xyz).to(device)
    n = xyz.size(0)

    # calculating distances
    dist = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

    # neighbor list
    mask = (dist <= cutoff)
    mask[np.diag_indices(n)] = 0
    nbr_list = torch.nonzero(mask)

    if undirected:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    return nbr_list


class CGDataset(TorchDataset):
    
    def __init__(self,
                 props,
                 check_props=True):
        self.props = props

    def __len__(self):
        return len(self.props['CG_nxyz'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.props.items()}

    def generate_aux_edges(self, auxcutoff, device='cpu', undirected=True):
        edge_list = []
        for nxyz in tqdm(self.props['CG_nxyz'], desc='building aux edge list', file=sys.stdout):
            edge_list.append(get_neighbor_list(nxyz[:, 1:4], device, auxcutoff, undirected).to("cpu"))

        self.props['bond_edge_list'] = edge_list

    def generate_neighbor_list(self, atom_cutoff, cg_cutoff, device='cpu', undirected=True, use_bond=False):

        cg_nbr_list = []
        for nxyz in tqdm(self.props['CG_nxyz'], desc='building CG nbr list', file=sys.stdout):
            cg_nbr_list.append(get_neighbor_list(nxyz[:, 1:4], device, cg_cutoff, undirected).to("cpu"))

        self.props['CG_nbr_list'] = cg_nbr_list


def CG_collate(dicts):
    # new indices for the batch: the first one is zero and the
    # last does not matter

    cumulative_atoms = np.cumsum([0] + [d['num_atoms'] for d in dicts])[:-1]
    cumulative_CGs = np.cumsum([0] + [d['num_CGs'] for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        d['nbr_list'] = d['nbr_list'] + int(n)
        d['bond_edge_list'] = d['bond_edge_list'] + int(n)
        d['mask_xyz_list'] = d['mask_xyz_list'] + int(n)
        d['bb_NO_list'] = d['bb_NO_list'] + int(n)
        d['interaction_list'] = d['interaction_list'] + int(n)
        d['pi_pi_list'] = d['pi_pi_list'] + int(n)
            
    for n, d in zip(cumulative_CGs, dicts):
        d['CG_mapping'] = d['CG_mapping'] + int(n)
        d['CG_nbr_list'] = d['CG_nbr_list'] + int(n)

    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        if hasattr(val, 'shape') and len(val.shape) > 0:
            batch[key] = torch.cat([
                data[key]
                for data in dicts
            ], dim=0)

        elif type(val) == str: 
            batch[key] = [data[key] for data in dicts]
        else:
            batch[key] = torch.stack(
                [data[key] for data in dicts],
                dim=0
            )

    return batch


def CG_collate_inf(dicts):
    cumulative_atoms = np.cumsum([0] + [d['num_atoms'] for d in dicts])[:-1]
    cumulative_CGs = np.cumsum([0] + [d['num_CGs'] for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        d['mask_xyz_list'] = d['mask_xyz_list'] + int(n)
            
    for n, d in zip(cumulative_CGs, dicts):
        d['CG_mapping'] = d['CG_mapping'] + int(n)
        d['CG_nbr_list'] = d['CG_nbr_list'] + int(n)

    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        if hasattr(val, 'shape') and len(val.shape) > 0:
            batch[key] = torch.cat([
                data[key]
                for data in dicts
            ], dim=0)

        elif type(val) == str: 
            batch[key] = [data[key] for data in dicts]
        else:
            batch[key] = torch.stack(
                [data[key] for data in dicts],
                dim=0
            )

    return batch


def split_train_test(dataset,
                     test_size=0.2):

    idx = list(range(len(dataset)))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, shuffle=True)

    train = get_subset_by_indices(idx_train, dataset)
    test = get_subset_by_indices(idx_test, dataset)

    return train, test


def get_subset_by_indices(indices, dataset):

    if isinstance(dataset, CGDataset):
        subset = CGDataset(
            props={key: [val[i] for i in indices]
                   for key, val in dataset.props.items()},
        )
    elif isinstance(dataset, SCNCGDataset):
        subset = SCNCGDataset(
            props={key: [val[i] for i in indices]
                   for key, val in dataset.props.items()},
        )
    elif isinstance(dataset, DiffPoolDataset):
        subset = DiffPoolDataset(
            props={key: [val[i] for i in indices]
                   for key, val in dataset.props.items()},
        )
    else:
        raise ValueError("dataset type {} not recognized".format(dataset.__name__))

    return subset 


def split_train_validation_test(dataset,
                                val_size=0.2,
                                test_size=0.2,
                                **kwargs):

    train, validation = split_train_test(dataset,
                                         test_size=val_size,
                                         **kwargs)
    train, test = split_train_test(train,
                                   test_size=test_size / (1 - val_size),
                                   **kwargs)

    return train, validation, test
