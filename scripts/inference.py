import os 
import glob
import sys
import argparse 
import copy
import json
import time
from datetime import timedelta
import random

from tqdm import tqdm 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from sklearn.model_selection import KFold


import torch
from torch import nn
from torch.nn import Sequential 
from torch_scatter import scatter_mean
from torch.utils.data import DataLoader

# sys.path.append("../GenZProt/")
from GenZProt.data import CGDataset, CG_collate_inf
from GenZProt.cgvae import *
from GenZProt.genzprot import *
from GenZProt.e3nn_enc import e3nnEncoder, e3nnPrior
from GenZProt.conv import * 
from GenZProt.datasets import *
from utils import * 
from utils_ic import *
from sampling import sample_ic_backmap

import warnings
warnings.filterwarnings("ignore")


optim_dict = {'adam':  torch.optim.Adam, 'sgd':  torch.optim.SGD}

 
def build_dataset(cg_traj, aa_top, prot_idx=None):
    atomic_nums = [atom.element.number for atom in aa_top.atoms] 
    protein_index = aa_top.select("protein")
    protein_top = aa_top.subset(protein_index)
    atomic_nums = np.array([atom.element.number for atom in protein_top.atoms])

    mapping = torch.LongTensor(get_alpha_mapping(aa_top))
    cg_traj.xyz *= 10
    dataset = build_cg_dataset(mapping, 
                                cg_traj, aa_top, 
                                params['atom_cutoff'], 
                                params['cg_cutoff'],
                                atomic_nums,
                                prot_idx=prot_idx)

    return dataset, mapping


def run_cv(params):
    working_dir = params['logdir']
    device  = params['device']
    
    ndata = params['ndata']
    batch_size  = params['batch_size']
    sampling_batch_size = params['sampling_batch_size']
    n_ensemble = params['n_ensemble']
    nepochs = params['nepochs']
    lr = params['lr']
    optim = optim_dict[params['optimizer']]
    factor = params['factor']
    patience = params['patience']
    threshold = params['threshold']

    beta  = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    eta = params['eta']
    zeta = params['zeta']

    n_basis  = params['n_basis']
    n_rbf  = params['n_rbf']
    atom_cutoff = params['atom_cutoff']
    cg_cutoff = params['cg_cutoff']

    # for model
    enc_nconv  = params['enc_nconv']
    dec_nconv  = params['dec_nconv']
    enc_type = params['enc_type']
    dec_type = params['dec_type']
    
    # unused
    activation = params['activation']
    dataset_label = params['dataset']
    shuffle_flag = params['shuffle']
    nevals = params['nevals']
    tqdm_flag = params['tqdm_flag']
    det = params['det']
    mapshuffle = params['mapshuffle']
    savemodel = params['savemodel']
    invariantdec = params['invariantdec']
    n_cgs = params['n_cgs']

    # set random seed 
    seed = params['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    min_lr = 1e-8
    if det:
        beta = 0.0
        print("Recontruction Task")
    else:
        print("Sampling Task")

    # requires one all-atom pdb (for mapping generation)
    aa_traj = md.load_pdb(params['topology_path'])
    # throw ValueError if any hydrogens in the all-atom file.
    if any([atom.element.symbol == 'H' for atom in aa_traj.top.atoms]):
        raise ValueError('All-atom topology file contains hydrogen atoms. Please remove them.')

    info, n_cgs = traj_to_info(aa_traj)
    info_dict = {0: info}

    cg_traj = md.load_pdb(params['ca_trace_path'])

    # create subdirectory 
    create_dir(working_dir)     

    # start timing 
    start =  time.time()

    testset_list = []
    atomic_nums, protein_index = get_atomNum(aa_traj)
    table, _ = aa_traj.top.to_dataframe()

    # multiple chain
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']

    nfirst = len(table.loc[table.newSeq==table.newSeq.min()])
    nlast = len(table.loc[table.newSeq==table.newSeq.max()])

    n_atoms = atomic_nums.shape[0]
    n_atoms = n_atoms - (nfirst+nlast)
    atomic_nums = atomic_nums[nfirst:-nlast]
    _top = aa_traj.top.subset(np.arange(aa_traj.top.n_atoms)[nfirst:-nlast])

    all_idx = np.arange(len(cg_traj))
    ndata = len(all_idx)-len(all_idx)%sampling_batch_size
    all_idx = all_idx[:ndata]

    testset, mapping = build_dataset(cg_traj, aa_traj.top, prot_idx=0)
    testset = torch.utils.data.ConcatDataset([testset])
    testloader = DataLoader(testset, batch_size=sampling_batch_size, collate_fn=CG_collate_inf, shuffle=shuffle_flag, pin_memory=True)

    decoder = ZmatInternalDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation)
    encoder = e3nnEncoder(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cross_max_distance=cg_cutoff+5, atom_max_radius=atom_cutoff+5, cg_max_radius=cg_cutoff+5)
    cgPrior = e3nnPrior(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cg_max_radius=cg_cutoff+5)

    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))

    model = GenZProt(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior, det=det, equivariant= not invariantdec).to(device)
    model.train()

    # load model
    load_model_path = params['load_model_path']
    epoch = params['test_epoch']
    model.load_state_dict(torch.load(os.path.join(params['load_model_path'], 'model.pt'), map_location=torch.device('cpu')))
    model.to(device)

    print("model loaded successfully")

    print("Sampling geometries")
    gen_xyzs = sample_ic_backmap(testloader, device, model, atomic_nums, n_cgs, n_ensemble=n_ensemble, info_dict=info_dict, tqdm_flag=True)
    gen_xyzs /= 10
    save_runtime(time.time() - start, working_dir)
    
    print("Saving geometries in npy, xtc, and pdb format")
    np.save(os.path.join(working_dir, f'sample_xyz.npy'), gen_xyzs)
    gen_xyzs = gen_xyzs.transpose(1, 0, 2, 3).reshape(-1,gen_xyzs.shape[-2],3)
    gen_traj = md.Trajectory(gen_xyzs, topology=_top)
    # also save as xtc for easier loading?
    gen_traj.save_xtc(os.path.join(working_dir, f'sample_traj.xtc'))
    gen_traj.save_pdb(os.path.join(working_dir, f'sample_traj.pdb'))
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # paths and environment
    parser.add_argument("-load_json", type=str, default=None)
    parser.add_argument("-load_model_path", type=str, default=None)
    parser.add_argument("-test_epoch", type=int, default=None)
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)

    # dataset
    parser.add_argument("-ca_trace_path", type=str, default=None)
    parser.add_argument("-topology_path", type=str, default=None)
    parser.add_argument("-dataset", type=str, default='dipeptide')
    parser.add_argument("-cg_method", type=str, default='minimal')

    # training + sampling
    parser.add_argument("-seed", type=int, default=12345)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-sampling_batch_size", type=int, default=64)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-nepochs", type=int, default=2)
    parser.add_argument("-n_ensemble", type=int, default=16)

    # learning rate
    parser.add_argument("-lr", type=float, default=2e-4)
    parser.add_argument("-threshold", type=float, default=1e-3)
    parser.add_argument("-patience", type=int, default=5)
    parser.add_argument("-factor", type=float, default=0.6)

    # loss
    parser.add_argument("-beta", type=float, default=0.001)
    parser.add_argument("-gamma", type=float, default=0.01)
    parser.add_argument("-delta", type=float, default=0.01)
    parser.add_argument("-eta", type=float, default=0.01)

    # model
    parser.add_argument("-enc_type", type=str, default='equiv_enc')
    parser.add_argument("-dec_type", type=str, default='ic_dec')

    parser.add_argument("-n_basis", type=int, default=512)
    parser.add_argument("-n_rbf", type=int, default=10)
    parser.add_argument("-atom_cutoff", type=float, default=4.0)
    parser.add_argument("-cg_cutoff", type=float, default=4.0)
    parser.add_argument("-edgeorder", type=int, default=2)

    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-enc_nconv", type=int, default=4)
    parser.add_argument("-dec_nconv", type=int, default=4)

    # always use default
    parser.add_argument("-n_cgs", type=int)
    parser.add_argument("-mapshuffle", type=float, default=0.0)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)
    parser.add_argument("--savemodel", action='store_true', default=True)

    # not used
    parser.add_argument("-ndata", type=int, default=200)
    parser.add_argument("-nsamples", type=int, default=200)
    parser.add_argument("-nevals", type=int, default=36)
    parser.add_argument("-auxcutoff", type=float, default=0.0)
    parser.add_argument("-kappa", type=float, default=0.01)    
    parser.add_argument("-nsplits", type=int, default=5)
    parser.add_argument("-cgae_reg_weight", type=float, default=0.25)
    parser.add_argument("--cross", action='store_true', default=False)
    parser.add_argument("--graph_eval", action='store_true', default=False)
    parser.add_argument("--cg_mp", action='store_true', default=False)
    parser.add_argument("--cg_radius_graph", action='store_true', default=False)
    parser.add_argument("--invariantdec", action='store_true', default=False)
    parser.add_argument("--reflectiontest", action='store_true', default=False)


    params = vars(parser.parse_args())
    params['savemodel'] = True
    params['load_json'] = os.path.join(params['load_model_path'], 'modelparams.json')
    with open(params['load_json'], 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        params = vars(parser.parse_args(namespace=t_args))

    epoch = params['test_epoch']
    model_name = params['load_model_path'].split('/')[-1]
    data_name = params['ca_trace_path'].split('/')[-1].split('.')[0]
    params['logdir'] = f'./result_{model_name}_{data_name}'
    run_cv(params)