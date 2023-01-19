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
    n_cgs  = params['n_cgs']

    # set random seed 
    seed = params['seed']
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

    # Load PED files
    train_label_list = \
        ['00033e000', '00053e000', '00050e000', '00073e000', '00078e000',
         '00132e000', '00113e000', '00034e000', '00044e000', '00032e000',
         '00160e002', '00094e000', '00124e000', '00013e001', '00054e000',
         '00181ecut', '00056e000', '00003e001', '00175e024', '00175e028',
         '00175e025', '00175e029', '00175e022', '00175e023', '00175e026',
         '00175e027', '00175e021', '00175e020', '00051e000', '00114e000',
         '00004e001', '00150ecut0', '00150ecut2', '00150ecut1', '00080e000',
         '00156e002', '00193e003', '00145ecut2', '00145ecut1', '00145ecut0',
         '00022e012', '00022e011', '00022e013', '00022e001', '00022e009',
         '00022e002', '00022e005', '00022e007', '00022e008', '00022e010',
         '00022e003', '00022e004', '00022e006', '00141e001', '00159e002',
         '00095e000', '00087e000', '00180ecut', '00120e000', '00006e001',
         '00074e000', '00225e000', '00011ecut', '00088e000', '00107e000',
         '00085e000', '00112e000', '00023e001', '00023e002', '00023e003',
         '00157e002', '00192e002', '00046e000', '00220e000', '00115e000',
         '00190e000', '00100e000', '00148ecut1', '00148ecut2', '00072e000',
         '00217e000','00125e000', '00158e006', '00077e000', '00043e000',
         '00104e000', '00123e000', '00117e000', '00098e000', '00185e000',
         '00126e000', '00143ecut', '00099e000', '00036e000', '00024e001',
         '00161e002', '00118e000', '00119e000', '00041e000', '00045e000',
         '00121e000', '00092e000', '00109e000', '00155ecut', '00111e000',
         '00227e000', '00097e000', '00093e000', '00025ecut', '00062e000',
         '00052e000', '00101e000', '00102e000', '00086e000', '00040e000',
         '00135e000']

    val_label_list = ['00151ecut0', '00151ecut2', '00151ecut1', '00090e000', '00055e000', '00218e000']

    # For single chemistry training    
    # train_label_list = ['00151ecut1', '00151ecut2']
    # val_label_list = ['00151ecut0']
    
    print("num training data entries", len(train_label_list))
    
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

    # Z-matrix generation or xyz generation
    if dec_type == 'ic_dec': ic_flag = True
    else: ic_flag = False

    if ic_flag:
        decoder = InternalDecoder56(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation)
    else:
        decoder = EquivariantPsuedoDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation, breaksym=breaksym)

    if enc_type == 'equiv_enc':
        encoder = e3nnEncoder(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
        cross_max_distance=cg_cutoff+5, atom_max_radius=atom_cutoff+5, cg_max_radius=cg_cutoff+5)
        cgPrior = e3nnPrior(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
        cg_max_radius=cg_cutoff+5)
    else:
        encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=atom_cutoff)
        cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=cg_cutoff)

    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))

    if ic_flag:
        model = peptideCGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior, det=det, equivariant= not invariantdec).to(device)
    else:
        model = CGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior, det=det, equivariant= not invariantdec).to(device)

    # load_model_path='./test37_dec56_notorsion_beta1e-1_max1e-2_01-11_sample'
    # model.load_state_dict(torch.load(os.path.join(load_model_path, f'model_132.pt'), map_location=torch.device('cpu')))
    # model.to(device)

    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, 
                                                            factor=factor, verbose=True, 
                                                            threshold=threshold,  min_lr=min_lr, cooldown=1)
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
                                                    ic_flag=ic_flag, info_dict=info_dict)


        val_loss, mean_kl_val, mean_recon_val, mean_graph_val, mean_nbr_val, mean_inter_val, mean_xyz_val = loop(valloader, optimizer, device,
                                                    model, beta, gamma, delta, eta, epoch, 
                                                    train=False,
                                                    looptext='epoch {} train'.format(epoch),
                                                    tqdm_flag=tqdm_flag,
                                                    ic_flag=ic_flag, info_dict=info_dict)

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
    parser.add_argument("-seed", type=int, default=12345)
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

    params['logdir'] = annotate_job(params['seed'], params['logdir'])
    print(params['logdir'])
    run_cv(params)
