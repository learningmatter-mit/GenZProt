# sys.path.append("../scripts/")
# sys.path.append("../src/")

import os 
import glob
import sys
import argparse 
import random
import copy
import json
import time
from datetime import timedelta

import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import KFold
import pandas as pd
import statsmodels.api as sm

sys.path.append("../Peptide_backmap/")
# import CoarseGrainingVAE
from data import CGDataset, CG_collate
from cgvae import MLPDecoder, internalCGequiVAE
from e3nn_enc import e3nnEncoder, e3nnPrior
from conv import * 
from datasets import load_protein_traj, get_atomNum, get_cg_and_xyz, build_ic_multiprotein_dataset, create_info_dict  
from visualization import xyz_grid_view, rotate_grid, save_rotate_frames
from utils import * 
from utils_ic import *
from sampling import * 


import torch
from torch import nn
from torch.nn import Sequential 
from torch_scatter import scatter_mean
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

# set random seed 
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

optim_dict = {'adam':  torch.optim.Adam, 'sgd':  torch.optim.SGD}

def build_split_dataset(traj, params, mapping=None, n_cgs=None, prot_idx=None):

    if n_cgs == None:
        n_cgs = params['n_cgs']

    atomic_nums, protein_index = get_atomNum(traj)
    new_mapping, frames, cg_coord = get_cg_and_xyz(traj, params=params, cg_method=params['cg_method'], n_cgs=n_cgs,
                                                     mapshuffle=params['mapshuffle'], mapping=mapping)

    if mapping is None:
        mapping = new_mapping

    dataset = build_ic_multiprotein_dataset(mapping,
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
    n_cgs  = params['n_cgs']
    n_basis  = params['n_basis']
    n_rbf  = params['n_rbf']
    atom_cutoff = params['atom_cutoff']
    cg_cutoff = params['cg_cutoff']
    enc_nconv  = params['enc_nconv']
    dec_nconv  = params['dec_nconv']
    batch_size  = params['batch_size']
    beta  = params['beta']
    nsplits = params['nsplits']
    ndata = params['ndata']
    nsamples = params['nsamples']
    nepochs = params['nepochs']
    lr = params['lr']
    activation = params['activation']
    optim = optim_dict[params['optimizer']]
    dataset_label = params['dataset']
    shuffle_flag = params['shuffle']
    cg_mp_flag = params['cg_mp']
    nevals = params['nevals']
    graph_eval = params['graph_eval']
    tqdm_flag = params['tqdm_flag']
    n_ensemble = params['n_ensemble']
    det = params['det']
    gamma = params['gamma']
    factor = params['factor']
    patience = params['patience']
    eta = params['eta']
    kappa = params['kappa']
    mapshuffle = params['mapshuffle']
    threshold = params['threshold']
    savemodel = params['savemodel']
    auxcutoff = params['auxcutoff']
    invariantdec = params['invariantdec']

    failed = False
    # min_lr = 5e-8
    min_lr = 1e-8

    if det: 
        beta = 0.0
        print("Recontruction Task")
    else:
        print("Sampling Task")

    print('cuda: ', torch.cuda.is_available())

    prefix = 'PED'
    dataset_label_list = []
    PED_PDBs = glob.glob(f'../data/processed/{prefix}*.pdb')
    for PDBfile in PED_PDBs:
        ID = PDBfile.split('/')[-1].split('.')[0][len(prefix):]
        dataset_label_list.append(ID)
    print("train data list: ", dataset_label_list)
    dataset_label_list = dataset_label_list[1:]

    n_cg_list, traj_list, info_dict = create_info_dict(dataset_label_list)

    # create subdirectory 
    create_dir(working_dir)     
    kf = KFold(n_splits=nsplits, shuffle=True)
    split_iter = kf.split(list(range(ndata)))
    cv_stats_pd = pd.DataFrame( { 'train_recon': [],
                    'test_all_recon': [],
                    'test_heavy_recon': [],
                    'train_KL': [], 'test_KL': [], 
                    'train_graph': [],  'test_graph': [],
                    'recon_all_ged': [], 'recon_heavy_ged': [], 
                    'recon_all_valid_ratio': [], 
                    'recon_heavy_valid_ratio': [],
                    'sample_all_ged': [], 'sample_heavy_ged': [], 
                    'sample_all_valid_ratio': [], 
                    'sample_heavy_valid_ratio': [],
                    'sample_all_rmsd': [], 'sample_heavy_rmsd':[]}  )

    for _, (train_index, test_index) in enumerate(split_iter):

        # start timing 
        start =  time.time()

        split_dir = os.path.join(working_dir, 'fold{}'.format(0)) 
        create_dir(split_dir)

        trainset_list, valset_list, testset_list = [], [], []
        for i, traj in enumerate(traj_list):
            try:
                n_train, n_val, n_test = int(len(traj)*0.8), int(len(traj)*0.1), int(len(traj)*0.1)
                all_idx = np.arange(len(traj))
                random.shuffle(all_idx)
                train_index, val_index, test_index = all_idx[:n_train], all_idx[n_train:n_train+n_val], all_idx[n_train+n_val:]

                n_cgs = n_cg_list[i]
                trainset, mapping = build_split_dataset(traj[train_index], params, mapping=None, n_cgs=n_cgs, prot_idx=i)
                true_n_cgs = len(list(set(mapping.tolist())))

                valset, mapping = build_split_dataset(traj[val_index], params, mapping, prot_idx=i)
                testset, mapping = build_split_dataset(traj[test_index], params, mapping, prot_idx=i)
                print("created dataset-------", dataset_label_list[i])

                trainset_list.append(trainset)
                valset_list.append(valset)
                testset_list.append(testset)
            except:
                print("failed to create dataset--", dataset_label_list[i])

        trainset = torch.utils.data.ConcatDataset(trainset_list)
        valset = torch.utils.data.ConcatDataset(valset_list)
        testset = torch.utils.data.ConcatDataset(testset_list)

        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
        valloader = DataLoader(valset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
        
        # initialize model 
        atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
        atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))

        if n_cgs == 3:
            breaksym= True 
        else:
            breaksym = False

        # decoder = EquivariantDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation)
        decoder = MLPDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation)

        encoder = e3nnEncoder(device=device, n_atom_basis=n_basis)
        cgPrior = e3nnPrior(device=device, n_atom_basis=n_basis)
        
        model = internalCGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior,
                            det=det, equivariant= not invariantdec).to(device)
        
        optimizer = optim(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, 
                                                                factor=factor, verbose=True, 
                                                                threshold=threshold,  min_lr=min_lr)
        early_stopping = EarlyStopping(patience=patience)
        
        model.train()
        print('using cuda: ', torch.cuda.current_device())

        with open(os.path.join(split_dir, 'modelparams.json'), "w") as outfile: 
            json.dump(params, outfile, indent=4)

        # intialize training log 
        train_log = pd.DataFrame({'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [],
                   'train_KL': [], 'val_KL': [], 'train_graph': [], 'val_graph': [], 'train_nbr': [], 'val_nbr': [], 'train_inter': [], 'val_inter': []})


        for epoch in range(nepochs):
            # train
            train_loss, mean_kl_train, mean_recon_train, mean_graph_train, mean_nbr_train, mean_inter_train, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                                                       model, beta, epoch, 
                                                       train=True,
                                                        gamma=gamma,
                                                        eta=eta,
                                                        kappa=kappa,
                                                        looptext='Ncg {} Fold {} train'.format(n_cgs, i),
                                                        tqdm_flag=tqdm_flag,
                                                        ic_flag=True, info_dict=info_dict)


            val_loss, mean_kl_val, mean_recon_val, mean_graph_val, mean_nbr_val, mean_inter_val, xyz_val, xyz_val_recon = loop(valloader, optimizer, device,
                                                       model, beta, epoch, 
                                                       train=False,
                                                        gamma=gamma,
                                                        eta=eta,
                                                        kappa=kappa,
                                                        looptext='Ncg {} Fold {} test'.format(n_cgs, i),
                                                        tqdm_flag=tqdm_flag,
                                                        ic_flag=True, info_dict=info_dict)

            stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                    'train_loss': train_loss, 'val_loss': val_loss, 
                    'train_recon': mean_recon_train, 'val_recon': mean_recon_val,
                   'train_KL': mean_kl_train, 'val_KL': mean_kl_val, 
                   'train_graph': mean_graph_train, 'val_graph': mean_graph_val,
                   'train_nbr': mean_nbr_train, 'val_nbr': mean_nbr_val,
                   'train_inter': mean_inter_train, 'val_inter': mean_inter_val}

            train_log = train_log.append(stats, ignore_index=True)

            # smoothen the validation curve 
            smooth = sm.nonparametric.lowess(train_log['val_loss'].values,  # y
                                            train_log['epoch'].values, # x
                                            frac=0.2)
            smoothed_valloss = smooth[-1, 1]

            scheduler.step(smoothed_valloss)

            if optimizer.param_groups[0]['lr'] <= min_lr * 1.5:
                print('converged')
                break

            early_stopping(smoothed_valloss)
            if early_stopping.early_stop:
                break

            # check NaN
            if np.isnan(mean_recon_val):
                print("NaN encoutered, exiting...")
                failed = True
                break 

            # dump training curve 
            train_log.to_csv(os.path.join(split_dir, 'train_log.csv'),  index=False)

            if savemodel and epoch%5==0:
                torch.save(model.state_dict(), os.path.join(split_dir, 'model.pt'))
        
        print("finished training")
        exit(-1)
    return cv_stats_pd['test_all_recon'].mean(), cv_stats_pd['test_all_recon'].std(), cv_stats_pd['recon_all_ged'].mean(), cv_stats_pd['recon_all_ged'].std(), failed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-load_json", type=str, default=None)
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)
    parser.add_argument("-n_cgs", type=int)
    parser.add_argument("-lr", type=float, default=2e-4)
    parser.add_argument("-dataset", type=str, default='dipeptide')
    parser.add_argument("-n_basis", type=int, default=512)
    parser.add_argument("-n_rbf", type=int, default=10)
    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-cg_method", type=str, default='minimal')
    parser.add_argument("-atom_cutoff", type=float, default=4.0)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-cg_cutoff", type=float, default=4.0)
    parser.add_argument("-enc_nconv", type=int, default=4)
    parser.add_argument("-dec_nconv", type=int, default=4)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-nepochs", type=int, default=2)
    parser.add_argument("-ndata", type=int, default=200)
    parser.add_argument("-nsamples", type=int, default=200)
    parser.add_argument("-n_ensemble", type=int, default=16)
    parser.add_argument("-nevals", type=int, default=36)
    parser.add_argument("-edgeorder", type=int, default=2)
    parser.add_argument("-auxcutoff", type=float, default=0.0)
    parser.add_argument("-beta", type=float, default=0.001)
    parser.add_argument("-gamma", type=float, default=0.01)
    parser.add_argument("-eta", type=float, default=0.01)
    parser.add_argument("-kappa", type=float, default=0.01)
    parser.add_argument("-threshold", type=float, default=1e-3)
    parser.add_argument("-nsplits", type=int, default=5)
    parser.add_argument("-patience", type=int, default=5)
    parser.add_argument("-factor", type=float, default=0.6)
    parser.add_argument("-mapshuffle", type=float, default=0.0)
    parser.add_argument("-cgae_reg_weight", type=float, default=0.25)
    parser.add_argument("--dec_type", type=str, default='EquivariantDecoder')
    parser.add_argument("--cross", action='store_true', default=False)
    parser.add_argument("--graph_eval", action='store_true', default=False)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--cg_mp", action='store_true', default=False)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)
    parser.add_argument("--cg_radius_graph", action='store_true', default=False)
    parser.add_argument("--invariantdec", action='store_true', default=False)
    parser.add_argument("--reflectiontest", action='store_true', default=False)

    params = vars(parser.parse_args())
    params['savemodel'] = True

    if params['load_json']:
        with open(params['load_json'], 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            params = vars(parser.parse_args(namespace=t_args))

    # add more info about this job 
    if params['det']:
        task = 'recon'
    else:
        task = 'sample'

    if params['invariantdec']:
        params['logdir'] = annotate_job(params['cg_method'] + '_invariantdec_' + task + '_ndata{}'.format(params['ndata']), params['logdir'], params['n_cgs'])
    else:
        params['logdir'] = annotate_job(params['cg_method'] + '_' + task + '_ndata{}'.format(params['ndata']), params['logdir'], params['n_cgs'])

    if params['cross']:
        params['logdir']  += '_cross'

    if params['reflectiontest']:
        params['logdir']  += '_reflectiontest'

    run_cv(params)
