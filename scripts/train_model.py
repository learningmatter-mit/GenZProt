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
    seed = 123
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
    # Load PED files
    prefixs = [
    'PED00001', 'PED00002', 'PED00004', 'PED00005', 'PED00008',
    'PED00012', 'PED00013', 'PED00019', 'PED00021', 'PED00023',
    'PED00032', 'PED00033', 'PED00034', 'PED00036', 'PED00040',
    'PED00041', 'PED00043', 'PED00044', 'PED00045', 'PED00046',
    'PED00048', 'PED00050', 'PED00051', 'PED00052', 'PED00053',
    'PED00054', 'PED00055', 'PED00056', 'PED00062', 'PED00070',
    'PED00071', 'PED00072', 'PED00073', 'PED00074', 'PED00076',
    'PED00077', 'PED00078', 'PED00080', 'PED00082', 'PED00083',
    'PED00084', 'PED00085', 'PED00086', 'PED00087', 'PED00088',
    'PED00090', 'PED00092', 'PED00093', 'PED00094', 'PED00095',
    'PED00097', 'PED00098', 'PED00099', 'PED00100', 'PED00101',
    ]
    dataset_label_list = []
    # PED_PDBs = glob.glob(f'../data/processed/{prefix}*.pdb')
    # PED_PDBs = glob.glob(f'/home/soojungy/eofe8_mnt/use_files/{prefix}*.pdb')
    PED_PDBs = []
    for prefix in prefixs:
        PED_PDBs += glob.glob(f'/home/soojungy/eofe8_mnt/use_files/{prefix}*.pdb')
    for PDBfile in PED_PDBs:
        ID = PDBfile.split('/')[-1].split('.')[0][3:]
        dataset_label_list.append(ID)
    # print("train data list: ", dataset_label_list)
    print(len(dataset_label_list))
    # dataset_label_list = [dataset_label_list[14]]
    print(dataset_label_list)
    # dataset_label_list = ['00022e005', '00024e001', '00080e000']
    # dataset_label_list = ['00036e000']#, '00024e001', '00080e000']
    n_cg_list, traj_list, info_dict = create_info_dict(dataset_label_list)

    # create subdirectory 
    create_dir(working_dir)     

    # start timing 
    start =  time.time()

    trainset_list, valset_list, testset_list = [], [], []
    success_list, fail_list = [], []
    for i, traj in enumerate(traj_list):
        # try:
        print("start generating dataset-------", dataset_label_list[i])
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
        success_list.append(dataset_label_list[i])

        trainset_list.append(trainset)
        valset_list.append(valset)
        testset_list.append(testset)
        # except:
        #     print("failed to create dataset--", dataset_label_list[i])
        #     fail_list.append(dataset_label_list[i])
    print('success: ', success_list)
    print('fail   : ', fail_list)
    
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
    decoder = InternalDecoder2(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation)
    # decoder = InternalDecoder1(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation)

    encoder = e3nnEncoder(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=3)
    cgPrior = e3nnPrior(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=3)
    # encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=atom_cutoff)
    # cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=atom_cutoff)

    model = peptideCGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior,
                        det=det, equivariant= not invariantdec).to(device)
    
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, 
                                                            factor=factor, verbose=True, 
                                                            threshold=threshold,  min_lr=min_lr)
    early_stopping = EarlyStopping(patience=patience)
    
    model.train()
    print('using cuda: ', torch.cuda.current_device())

    with open(os.path.join(working_dir, 'modelparams.json'), "w") as outfile: 
        json.dump(params, outfile, indent=4)

    # intialize training log 
    train_log = pd.DataFrame({'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [],
                'train_KL': [], 'val_KL': [], 'train_graph': [], 'val_graph': [], 'train_nbr': [], 'val_nbr': [], 'train_inter': [], 'val_inter': []})

    for epoch in range(nepochs):
        # train
        train_loss, mean_kl_train, mean_recon_train, mean_graph_train, mean_nbr_train, mean_inter_train, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                                                    model, beta, gamma, delta, eta, epoch, 
                                                    train=True,
                                                    looptext='epoch {} train'.format(epoch),
                                                    tqdm_flag=tqdm_flag,
                                                    ic_flag=True, info_dict=info_dict)


        val_loss, mean_kl_val, mean_recon_val, mean_graph_val, mean_nbr_val, mean_inter_val, xyz_val, xyz_val_recon = loop(valloader, optimizer, device,
                                                    model, beta, gamma, delta, eta, epoch, 
                                                    train=False,
                                                    looptext='epoch {} train'.format(epoch),
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
            break 

        # dump training curve 
        train_log.to_csv(os.path.join(working_dir, 'train_log.csv'),  index=False, float_format='%.5f')

        if savemodel and epoch%5==0:
            torch.save(model.state_dict(), os.path.join(working_dir, 'model.pt'))
    
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
