import os 
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

import CoarseGrainingVAE
from CoarseGrainingVAE.data import CGDataset, CG_collate
from CoarseGrainingVAE.cgvae import SequentialDecoder, MLPDecoder, internalEquivariantPsuedoDecoder, EquiEncoder, CGprior, internalCGequiVAE 
from CoarseGrainingVAE.e3nn_enc import e3nnEncoder, e3nnPrior
from CoarseGrainingVAE.conv import * 
from CoarseGrainingVAE.datasets import load_protein_traj, get_atomNum, get_cg_and_xyz, build_ic_multiprotein_dataset  
from utils import * 
from CoarseGrainingVAE.visualization import xyz_grid_view, rotate_grid, save_rotate_frames
from sampling import sample_ic as sample



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
    min_lr = 5e-8

    if det:
        beta = 0.0
        print("Recontruction Task")
    else:
        print("Sampling Task")

    device = 3
    batch_size = 8
    dataset_label_list = [params['test_data']]

    traj_list, n_cg_list, info_dict = [], [], {}
    for idx, label in enumerate(dataset_label_list):
        traj = shuffle_traj(load_protein_traj(label))
        table, _ = traj.top.to_dataframe()
        reslist = list(set(list(table.resSeq)))
        reslist.sort()

        n_cg = len(reslit)
        n_cg_list.append(n_cg)

        atomic_nums, protein_index = get_atomNum(traj)
        traj_list.append(traj)

        atomn = [list(table.loc[table.resSeq==res].name) for res in reslist][1:-1]
        resn = list(table.loc[table.name=='CA'].resName)[1:-1]
        info_dict[idx] = (atomn, resn)

    nfirst = len(table.loc[table.resSeq==table.resSeq.min()])
    nlast = len(table.loc[table.resSeq==table.resSeq.max()])
    n_atoms = atomic_nums.shape[0]
    n_atoms = n_atoms - (nfirst+nlast)
    atomic_nums = atomic_nums[nfirst:-nlast]
    """"""

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

    for i, (train_index, test_index) in enumerate(split_iter):

        # start timing 
        start =  time.time()

        split_dir = os.path.join(working_dir, 'fold{}'.format(i)) 
        create_dir(split_dir)

        testset_list = []
        for i, traj in enumerate(traj_list):
            all_idx = np.arange(len(traj))
            random.shuffle(all_idx)
            all_idx = all_idx[:ndata]

            n_cgs = n_cg_list[i]
            testset, mapping = build_split_dataset(traj[all_idx], params, mapping=None, prot_idx=i)
            print("created dataset-------", dataset_label_list[i])
            testset_list.append(testset)

        testset = torch.utils.data.ConcatDataset(testset_list)
        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
        
        # initialize model 
        atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
        atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))

        if n_cgs == 3:
            breaksym= True 
        else:
            breaksym = False
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

        # load model
        load_model_path = params['load_model_path']
        model.load_state_dict(torch.load(os.path.join(load_model_path, f'model.pt')))
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

        epoch = 0
        # this is just to get KL loss 
        test_loss, mean_kl_test, mean_recon_test, mean_graph_test, mean_nbr_test, mean_inter_test, xyz_test, xyz_test_recon = loop(testloader, optimizer, device,
                                                    model, beta, epoch, 
                                                    train=False,
                                                    gamma=gamma,
                                                    eta=eta,
                                                    kappa=kappa,
                                                    looptext='Ncg {} Fold {} test'.format(n_cgs, i),
                                                    tqdm_flag=tqdm_flag,
                                                    ic_flag=True,
                                                    info_dict=info_dict
                                                    )

        # sample geometries 
        true_xyzs, recon_xyzs, recon_ics = sample(testloader, device, model, atomic_nums, n_cgs, info_dict)
        with open(os.path.join(split_dir, f'sample_recon_xyz.pkl'), 'wb') as filehandler:
            pickle.dump(recon_xyzs, filehandler)
        with open(os.path.join(split_dir, f'sample_recon_ic.pkl'), 'wb') as filehandler:
            pickle.dump(recon_ics, filehandler)

        # compute test rmsds  
        heavy_filter = atomic_nums != 1.
        test_recon_xyzs = test_recon_xyzs.reshape(-1,n_atoms,3) 
        test_true_xyzs = test_true_xyzs.reshape(-1,n_atoms,3)

        test_all_dxyz = (test_recon_xyzs - test_true_xyzs)#.reshape(-1)
        test_heavy_dxyz = test_all_dxyz[:, heavy_filter, :]
        test_all_rmsd = np.sqrt(np.power(test_all_dxyz, 2).sum(-1).mean(-1)) 
        unaligned_test_all_rmsd = test_all_rmsd.mean() 
        unaligned_test_heavy_rmsd = np.sqrt(np.power(test_heavy_dxyz, 2).sum(-1).mean()).mean()

        # dump test rmsd 
        np.savetxt(os.path.join(split_dir, 'test_all_rmsd{:.4f}.txt'.format(unaligned_test_all_rmsd)), np.array([unaligned_test_all_rmsd]))
        np.savetxt(os.path.join(split_dir, 'test_heavy_rmsd{:.4f}.txt'.format(unaligned_test_heavy_rmsd)), np.array([unaligned_test_heavy_rmsd]))

        # dump result files
        with open(os.path.join(split_dir, f'rmsd.pkl'), 'wb') as filehandler:
            pickle.dump(test_all_rmsd, filehandler)
        with open(os.path.join(split_dir, f'recon_xyz.pkl'), 'wb') as filehandler:
            pickle.dump(test_recon_xyzs, filehandler)
        with open(os.path.join(split_dir, f'true_xyz.pkl'), "wb") as filehandler:
            pickle.dump(test_true_xyzs, filehandler)

        ##### generate rotating movies for visualization #####

        test_stats = {
                'test_all_recon': unaligned_test_all_rmsd,
                'test_heavy_recon': unaligned_test_heavy_rmsd,
                'test_KL': mean_kl_test, 
                'test_graph': mean_graph_test,
                'test_nbr': mean_nbr_test,
                'test_inter': mean_inter_test,
                'recon_all_ged': test_all_ged, 'recon_heavy_ged': test_heavy_ged, 
                'recon_all_valid_ratio': test_all_valid_ratio, 
                'recon_heavy_valid_ratio': test_heavy_valid_ratio,} 

        for key in test_stats:
            print(key, test_stats[key])

        cv_stats_pd = cv_stats_pd.append(test_stats, ignore_index=True)
        cv_stats_pd.to_csv(os.path.join(working_dir, 'cv_stats.csv'),  index=False)

        # save_rotate_frames(sample_xyzs, data_xyzs, cg_xyzs, recon_xyzs, n_cgs, n_ensemble, atomic_nums, split_dir)
        save_runtime(time.time() - start, split_dir)

    return cv_stats_pd['test_all_recon'].mean(), cv_stats_pd['test_all_recon'].std(), cv_stats_pd['recon_all_ged'].mean(), cv_stats_pd['recon_all_ged'].std(), failed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-test_data", type=str, default=None)
    parser.add_argument("-load_json", type=str, default=None)
    parser.add_argument("-load_model_path", type=str, default=None)
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

    params['logdir'] += '_test_'
    params['logdir'] += params['test_data']

    run_cv(params)