import os 
import glob
import sys
import argparse 
import copy
import json
import time
from datetime import timedelta
import random
# sys.path.append("../scripts/")
# sys.path.append("../src/")

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

sys.path.append("../Peptide_backmap/")
# import CoarseGrainingVAE
from data import CGDataset, CG_collate
from cgvae import *
from e3nn_enc import e3nnEncoder, e3nnPrior
from conv import * 
from datasets import load_protein_traj, get_atomNum, get_cg_and_xyz, build_ic_peptide_dataset, create_info_dict  
from utils import * 
from utils_ic import *
from sampling import sample_ic as sample

import warnings
warnings.filterwarnings("ignore")


optim_dict = {'adam':  torch.optim.Adam, 'sgd':  torch.optim.SGD}

 
def build_split_dataset(traj, params, mapping=None, n_cgs=None, prot_idx=None):

    if n_cgs == None:
        n_cgs = params['n_cgs']

    atomic_nums, protein_index = get_atomNum(traj)
    new_mapping, frames, cg_coord = get_cg_and_xyz(traj, params=params, cg_method=params['cg_method'], n_cgs=n_cgs,
                                                     mapshuffle=params['mapshuffle'], mapping=mapping)

    if mapping is None:
        mapping = new_mapping

    dataset = build_ic_peptide_dataset(mapping,
                                        frames, 
                                        params['atom_cutoff'], 
                                        params['cg_cutoff'],
                                        atomic_nums,
                                        traj.top,
                                        order=params['edgeorder'] ,
                                        cg_traj=cg_coord, prot_idx=prot_idx)

    return dataset, mapping


def run_cv(params):
    working_dir = params['logdir']
    device  = params['device']
    
    ndata = params['ndata']
    batch_size  = params['batch_size']
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

    n_basis  = params['n_basis']
    n_rbf  = params['n_rbf']
    atom_cutoff = params['atom_cutoff']
    cg_cutoff = params['cg_cutoff']

    # for CGVAE
    enc_nconv  = params['enc_nconv']
    dec_nconv  = params['dec_nconv']
    
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

    device=3
    batch_size=4

    # set random seed 
    seed = 42
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

    test_PED_PDBs = []
    dataset_label_list = []
    test_prefixs = [params['test_data']]
    for prefix in test_prefixs:
        test_PED_PDBs += glob.glob(f'/home/gridsan/sjyang/backmap_exp/data/use_files/PED{prefix}*.pdb')
        test_PED_PDBs += glob.glob(f'/home/soojungy/backmap_exp/data/use_files/PED{prefix}*.pdb')        
           
    for PDBfile in test_PED_PDBs:
        ID = PDBfile.split('/')[-1].split('.')[0][3:]
        dataset_label_list.append(ID)
    print(dataset_label_list)
    n_cg_list, traj_list, info_dict = create_info_dict(dataset_label_list)

    # create subdirectory 
    create_dir(working_dir)     
     
    cv_stats_pd = pd.DataFrame( { 'test_all_recon': [],
                    'test_KL': [], 
                    'test_graph': [],
                    'test_nbr': [],
                    'test_inter': [],
                    'test_all_valid_ratio': [],
                    'test_all_ged': []}  )

    # start timing 
    start =  time.time()

    testset_list = []
    for i, traj in enumerate(traj_list):
        atomic_nums, protein_index = get_atomNum(traj)
        table, _ = traj.top.to_dataframe()
        # multiple chain
        table['newSeq'] = table['resSeq'] + 5000*table['chainID']

        nfirst = len(table.loc[table.newSeq==table.newSeq.min()])
        nlast = len(table.loc[table.newSeq==table.newSeq.max()])

        n_atoms = atomic_nums.shape[0]
        n_atoms = n_atoms - (nfirst+nlast)
        atomic_nums = atomic_nums[nfirst:-nlast]

        all_idx = np.arange(len(traj))
        random.shuffle(all_idx)

        ndata = len(all_idx)-len(all_idx)%batch_size
        all_idx = all_idx[:ndata]

        n_cgs = n_cg_list[i]
        testset, mapping = build_split_dataset(traj[all_idx], params, mapping=None, prot_idx=i)
        print("created dataset-------", dataset_label_list[i])
        testset_list.append(testset)

    testset = torch.utils.data.ConcatDataset(testset_list)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
    
    if n_cgs == 3:
        breaksym= True 
    else:
        breaksym = False

    # initialize model 
    decoder = InternalDecoder56(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation)

    encoder = e3nnEncoder(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cross_max_distance=cg_cutoff+5, atom_max_radius=atom_cutoff+5, cg_max_radius=cg_cutoff+5)
    cgPrior = e3nnPrior(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cg_max_radius=cg_cutoff+5)

    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    model = peptideCGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior, det=det, equivariant= not invariantdec)
    
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, 
                                                            factor=factor, verbose=True, 
                                                            threshold=threshold,  min_lr=min_lr)
    early_stopping = EarlyStopping(patience=patience)
    
    model.train()

    # load model
    load_model_path = params['load_model_path']
    epoch = params['test_epoch']
    model.load_state_dict(torch.load(os.path.join(load_model_path, f'model_{epoch}.pt'), map_location=torch.device('cpu')))
    model.to(device)

    print("model loaded successfully")

    print("Starting testing")
    test_true_xyzs, test_recon_xyzs, test_cg_xyzs, test_all_valid_ratio, test_heavy_valid_ratio, test_all_ged, test_heavy_ged = \
                                            get_all_true_reconstructed_structures(testloader, 
                                                                                device,
                                                                                model,
                                                                                atomic_nums,
                                                                                n_cg=n_cgs,
                                                                                tqdm_flag=tqdm_flag, reflection=params['reflectiontest'],
                                                                                ic_flag=True, top_table=None, info_dict=info_dict)

    epoch = 1
    # this is just to get KL loss 
    test_loss, mean_kl_test, mean_recon_test, mean_graph_test, mean_nbr_test, mean_inter_test, mean_xyz_test = loop(testloader, optimizer, device,
                                                model, beta, gamma, delta, eta, epoch, 
                                                train=False,
                                                looptext='epoch {} test'.format(epoch),
                                                tqdm_flag=tqdm_flag,
                                                ic_flag=True,
                                                info_dict=info_dict
                                                )

    # sample geometries 
    true_xyzs, recon_xyzs, recon_ics = sample(testloader, device, model, atomic_nums, n_cgs, info_dict)
    with open(os.path.join(working_dir, f'sample_recon_xyz.pkl'), 'wb') as filehandler:
        pickle.dump(recon_xyzs, filehandler)
    with open(os.path.join(working_dir, f'sample_recon_ic.pkl'), 'wb') as filehandler:
        pickle.dump(recon_ics, filehandler)

    # compute test rmsds  
    test_recon_xyzs = test_recon_xyzs.reshape(-1,n_atoms,3) 
    test_true_xyzs = test_true_xyzs.reshape(-1,n_atoms,3)

    test_all_dxyz = (test_recon_xyzs - test_true_xyzs)
    test_all_rmsd = np.sqrt(np.power(test_all_dxyz, 2).sum(-1).mean(-1)) 
    unaligned_test_all_rmsd = test_all_rmsd.mean() 

    # dump test rmsd 
    np.savetxt(os.path.join(working_dir, 'test_all_rmsd{:.4f}.txt'.format(unaligned_test_all_rmsd)), np.array([unaligned_test_all_rmsd]))

    # dump result files
    with open(os.path.join(working_dir, f'rmsd.pkl'), 'wb') as filehandler:
        pickle.dump(test_all_rmsd, filehandler)
    with open(os.path.join(working_dir, f'recon_xyz.pkl'), 'wb') as filehandler:
        pickle.dump(test_recon_xyzs, filehandler)
    with open(os.path.join(working_dir, f'true_xyz.pkl'), "wb") as filehandler:
        pickle.dump(test_true_xyzs, filehandler)

    test_stats = {
            'test_all_recon': unaligned_test_all_rmsd,
            'test_KL': mean_kl_test, 
            'test_graph': mean_graph_test,
            'test_nbr': mean_nbr_test,
            'test_inter': mean_inter_test,
            'test_all_valid_ratio': test_all_valid_ratio,
            'test_all_ged': test_all_ged} 

    for key in test_stats:
        print(key, test_stats[key])

    cv_stats_pd = cv_stats_pd.append(test_stats, ignore_index=True)
    cv_stats_pd.to_csv(os.path.join(working_dir, 'cv_stats.csv'),  index=False, float_format='%.4f')

    save_runtime(time.time() - start, working_dir)

    return cv_stats_pd['test_all_recon'].mean(), cv_stats_pd['test_all_recon'].std()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # paths and environment
    parser.add_argument("-load_json", type=str, default=None)
    parser.add_argument("-load_model_path", type=str, default=None)
    parser.add_argument("-test_epoch", type=int, default=None)
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)

    # dataset
    parser.add_argument("-test_data", type=str, default=None)
    parser.add_argument("-dataset", type=str, default='dipeptide')
    parser.add_argument("-cg_method", type=str, default='minimal')

    # training
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-nepochs", type=int, default=2)

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
    parser.add_argument("-dec_type", type=str, default='InternalDecoder2')
    parser.add_argument("-enc_type", type=str, default='e3nnEncoder')

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
    parser.add_argument("-n_ensemble", type=int, default=16)
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

    # add more info about this job 
    if params['det']:
        task = 'recon'
    else:
        task = 'sample'
    epoch = params['test_epoch']
    params['logdir'] += f'_epoch_{epoch}'
    params['logdir'] += '_test_'
    params['logdir'] += params['test_data']

    run_cv(params)