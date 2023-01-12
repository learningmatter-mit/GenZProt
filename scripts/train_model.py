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
import pandas as pd
import statsmodels.api as sm

sys.path.append("../Peptide_backmap/")
# import CoarseGrainingVAE
from data import CGDataset, CG_collate
from cgvae import *
from e3nn_enc import e3nnEncoder, e3nnPrior
from conv import * 
from datasets import load_protein_traj, get_atomNum, get_cg_and_xyz, build_ic_peptide_dataset, create_info_dict  
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
    n_cgs  = params['n_cgs']

    # set random seed 
    seed = 42
    os.environ['PYTHONHASHSEED']=str(seed)
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

    print('cuda: ', torch.cuda.is_available())
    print('using cuda: ', torch.cuda.current_device())
    device = torch.cuda.current_device()
    device = 1

    # Load PED files
    train_prefixs = ['PED00158', 'PED00104', 'PED00094', 'PED00141',
       'PED00043', 'PED00052', 'PED00160', 'PED00161', 'PED00072',
       'PED00098', 'PED00100', 
       'PED00121', 'PED00054', 'PED00119', 'PED00143', 'PED00097',
       'PED00155', 'PED00092', 'PED00180', 'PED00050',
       'PED00125', 'PED00004', 'PED00111',
       'PED00024', 'PED00145',
       'PED00101', 'PED00095', 'PED00109', 'PED00115',
       'PED00053', 'PED00175', 'PED00041',
       'PED00062', 'PED00113', 'PED00085',
       'PED00150', 'PED00193', 'PED00077', 'PED00044',
       'PED00217', 'PED00114', 'PED00181', 'PED00185', 'PED00003',
       'PED00159', 'PED00156', 'PED00123', 'PED00120',
       'PED00074', 'PED00088',
       'PED00099', 'PED00022', 'PED00040', 'PED00148',
       'PED00045', 'PED00011', 'PED00118', 'PED00112',
       'PED00225', 'PED00032',
       'PED00086', 'PED00157', 'PED00013', 'PED00117', 'PED00034',
       'PED00006', 'PED00033', 'PED00227', 'PED00220',
       'PED00056', 'PED00078', 'PED00107',
       'PED00051', 'PED00102', 'PED00132',
       'PED00093', 'PED00135', 
       'PED00073', 'PED00192', 'PED00025', 'PED00087', 'PED00023',
       'PED00046', 'PED00036', 'PED00124',
       'PED00190', 'PED00126', 'PED00080'
       ]

    # 'PED00174', 'PED00191', 'PED00187', 'PED00162', 'PED00019', 'PED00215', 'PED00214', 'PED00213', 'PED00212', 'PED00194', 'PED00076'
    # val_prefixs = ['PED00048', 'PED00151', 'PED00082', 'PED00090']
    # # val_prefixs = ['PED00080', 'PED00036', 'PED00174', 'PED00022', 'PED00159']
    val_prefixs = ['PED00151', 'PED00090', 'PED00055', 'PED00218']
    
    # train_prefixs = ['PED00150ecut0', 'PED00150ecut1']
    # val_prefixs = ['PED00150ecut2']
    
    train_prefixs = list(set(train_prefixs))
    print("num training data entries", len(train_prefixs))
    train_label_list, val_label_list = [], []
    train_PED_PDBs = []
    for prefix in train_prefixs:
        train_PED_PDBs += glob.glob(f'/home/gridsan/sjyang/backmap_exp/data/use_files/{prefix}*.pdb')
        train_PED_PDBs += glob.glob(f'/home/soojungy/backmap_exp/data/use_files/{prefix}*.pdb')        
           
    for PDBfile in train_PED_PDBs:
        ID = PDBfile.split('/')[-1].split('.')[0][3:]
        train_label_list.append(ID)

    val_PED_PDBs = []
    for prefix in val_prefixs:
        val_PED_PDBs += glob.glob(f'/home/gridsan/sjyang/backmap_exp/data/use_files/{prefix}*.pdb')
        val_PED_PDBs += glob.glob(f'/home/soojungy/backmap_exp/data/use_files/{prefix}*.pdb')        
    for PDBfile in val_PED_PDBs:
        ID = PDBfile.split('/')[-1].split('.')[0][3:]
        val_label_list.append(ID)

    # train_label_list = ['train_chignolin']
    # val_label_list = ['val_chignolin']
    
    train_n_cg_list, train_traj_list, info_dict = create_info_dict(train_label_list)
    val_n_cg_list, val_traj_list, val_info_dict = create_info_dict(val_label_list)

    val_info_dict = {k+len(train_label_list): val_info_dict[k] for k in val_info_dict.keys()}
    info_dict.update(val_info_dict)

    # create subdirectory 
    create_dir(working_dir)     

    # start timing 
    start =  time.time()

    trainset_list, valset_list = [], []
    success_list = []
    print("TRAINING DATA")
    for i, traj in enumerate(train_traj_list):
        print("start generating dataset-------", train_label_list[i])
        n_cgs = train_n_cg_list[i]
        trainset, mapping = build_split_dataset(traj, params, mapping=None, n_cgs=n_cgs, prot_idx=i)
        print("created dataset-------", train_label_list[i])
        success_list.append(train_label_list[i])
        trainset_list.append(trainset)
    
    print("TEST DATA")
    for i, traj in enumerate(val_traj_list):
        print("start generating dataset-------", val_label_list[i])
        n_cgs = val_n_cg_list[i]
        valset, mapping = build_split_dataset(traj, params, mapping=None, n_cgs=n_cgs, prot_idx=i+len(train_label_list))
        print("created dataset-------", val_label_list[i])
        success_list.append(val_label_list[i])
        valset_list.append(valset)

    print('success: ', success_list)
    
    trainset = torch.utils.data.ConcatDataset(trainset_list)
    valset = torch.utils.data.ConcatDataset(valset_list)

    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
    
    # initialize model 
    if n_cgs == 3:
        breaksym= True 
    else:
        breaksym = False

    # decoder = EquivariantDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation)
    # encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=atom_cutoff)
    # cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=cg_cutoff)

    decoder = InternalDecoder56(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation)

    encoder = e3nnEncoder(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cross_max_distance=cg_cutoff+5, atom_max_radius=atom_cutoff+5, cg_max_radius=cg_cutoff+5)
    cgPrior = e3nnPrior(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cg_max_radius=cg_cutoff+5)
    
    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    model = peptideCGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior, det=det, equivariant= not invariantdec).to(device)

    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, 
                                                            factor=factor, verbose=True, 
                                                            threshold=threshold,  min_lr=min_lr, cooldown=1)
    # early_stopping = EarlyStopping(patience=patience)
    early_stopping = EarlyStopping(patience=20)
    
    model.train()

    with open(os.path.join(working_dir, 'modelparams.json'), "w") as outfile: 
        json.dump(params, outfile, indent=4)

    # intialize training log 
    train_log = pd.DataFrame({'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [], 'train_xyz': [], 'val_xyz': [],
                'train_KL': [], 'val_KL': [], 'train_graph': [], 'val_graph': [], 'train_nbr': [], 'val_nbr': [], 'train_inter': [], 'val_inter': []})

    for epoch in range(nepochs):
        # train
        train_loss, mean_kl_train, mean_recon_train, mean_graph_train, mean_nbr_train, mean_inter_train, mean_xyz_train = loop(trainloader, optimizer, device,
                                                    model, beta, gamma, delta, eta, epoch, 
                                                    train=True,
                                                    looptext='epoch {} train'.format(epoch),
                                                    tqdm_flag=tqdm_flag,
                                                    ic_flag=True, info_dict=info_dict)


        val_loss, mean_kl_val, mean_recon_val, mean_graph_val, mean_nbr_val, mean_inter_val, mean_xyz_val = loop(valloader, optimizer, device,
                                                    model, beta, gamma, delta, eta, epoch, 
                                                    train=False,
                                                    looptext='epoch {} train'.format(epoch),
                                                    tqdm_flag=tqdm_flag,
                                                    ic_flag=True, info_dict=info_dict)

        stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                'train_loss': train_loss, 'val_loss': val_loss, 
                'train_recon': mean_recon_train, 'val_recon': mean_recon_val,
                'train_xyz': mean_xyz_train, 'val_xyz': mean_xyz_val,
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
            break 

        # dump training curve 
        train_log.to_csv(os.path.join(working_dir, 'train_log.csv'),  index=False, float_format='%.5f')

        if savemodel and epoch%3==0:
            torch.save(model.state_dict(), os.path.join(working_dir, f'model_{epoch}.pt'))
    
    print("finished training")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # paths and environment
    parser.add_argument("-load_json", type=str, default=None)
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)

    # dataset
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

    params['logdir'] = annotate_job(task, params['logdir'])
    print(params['logdir'])
    run_cv(params)
