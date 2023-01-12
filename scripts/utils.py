import os 
import sys
import time
from tqdm import tqdm 
from datetime import date

import numpy as np
from ase import Atoms, io 
import networkx as nx
from sklearn.utils import shuffle

import torch
import torch.autograd.profiler as profiler

from sampling import *
from utils_ic import * 

from pynvml import *
nvmlInit()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

EPS = 1e-7

    
def shuffle_traj(traj):
    full_idx = list(range(len(traj)))
    full_idx = shuffle(full_idx)
    return traj[full_idx]

# def annotate_job(task, job_name, N_cg):
#     today = date.today().strftime('%m-%d')
#     return "{}_{}_{}_N{}".format(job_name, today, task, N_cg)

def annotate_job(task, job_name):
    today = date.today().strftime('%m-%d')
    return "{}_{}_{}".format(job_name, today, task)

def create_dir(name):
    if not os.path.isdir(name):
        os.mkdir(name)   

def save_runtime(dtime, dir):
    hours = dtime//3600
    dtime = dtime - 3600*hours
    minutes = dtime//60
    seconds = dtime - 60*minutes
    format_time = '%d:%d:%d' %(hours,minutes,seconds)
    np.savetxt(os.path.join(dir, '{}.txt'.format(format_time)), np.ones(10))
    print("time elapsed: {}".format(format_time))
    return format_time

def check_CGgraph(dataset):
    frame_idx = np.random.randint(0, len(dataset), 20)

    for idx in frame_idx:
        a = dataset.props['CG_nbr_list'][idx]
        adj = [ tuple(pair.tolist()) for pair in a ]
        G = nx.Graph()
        G.add_edges_from(adj)
        connected = nx.is_connected(G)
        if not connected:
            print("One of the sampled CG graphs is not connected, training failed")
            return connected
        return True

class EarlyStopping():
    '''from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/'''
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0    
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def KL(mu1, std1, mu2, std2):
    if mu2 == None:
        return -0.5 * torch.sum(1 + torch.log(std1.pow(2)) - mu1.pow(2) - std1.pow(2), dim=-1).mean()
    else:
        return 0.5 * ( (std1.pow(2) / std2.pow(2)).sum(-1) + ((mu1 - mu2).pow(2) / std2).sum(-1) + \
            torch.log(std2.pow(2)).sum(-1) - torch.log(std1.pow(2)).sum(-1) - std1.shape[-1] ).mean()


import math
def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L  


def real_number_batch_to_one_hot_vector_bins(real_numbers, bins, device):
    """Converts a batch of real numbers to a batch of one hot vectors for the bins the real numbers fall in."""
    _, indexes = (real_numbers.view(-1, 1) - bins.to(device).view(1, -1)).abs().min(dim=1)
    # return indexes_to_one_hot(indexes, n_dims=bins.shape[0], device=device)
    return indexes

def indexes_to_one_hot(indexes, n_dims=None, device=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size()[0], n_dims).to(device)
    one_hots = one_hots.scatter_(1, indexes, 1)
    # one_hots = one_hots.view(*indexes.shape, -1)
    return one_hots

from torch import nn
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from scipy.ndimage import gaussian_filter

def loop(loader, optimizer, device, model, beta, gamma, delta, eta, epoch, 
         train=True, looptext='', tqdm_flag=True, ic_flag=False, info_dict=None, prefix=''):
    
    torsion_loss = nn.CrossEntropyLoss(reduction='none') #, label_smoothing=0.1)
    
    total_loss = []
    recon_loss = []
    orient_loss = []
    norm_loss = []
    graph_loss = []
    nbr_loss = []
    inter_loss = []
    kl_loss = []
    xyz_loss = []
    
    if train:
        model.train()
        mode = '{} train'.format(looptext)
    else:
        model.train() # yes, still set to train when reconstructing
        mode = '{} valid'.format(looptext)

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='({} epoch #{})'.format(mode, epoch))
  
    # beta = torch.clamp(torch.exp(torch.Tensor([epoch])*0.8) * 1e-5, max=0.01).item() # v4

    # beta_np_cyc = [0.0, 0.0, 0.0, 1e-4, 1e-3, 1e-2, 0.5] #vv4
    # beta = beta_np_cyc[int(epoch%len(beta_np_cyc))]

    # beta = 1e-1
    # beta = 1
    maxkl = 0.01

    # v1 beta
    if epoch < 15:
        beta = 0.1
    else:
        beta = 1.5

    if epoch > 0 or not train:
        # gamma, delta, eta, zeta = 1, 1, 0.1, 1
        gamma, delta, eta, zeta = 3, 3, 0.1, 1
    else:
        gamma, delta, eta, zeta = 0.0, 0.0, 0.0, 0.0
    
    for i, batch in enumerate(loader):
        batch = batch_to(batch, device)
        st = time.time()
        if ic_flag:
            S_mu, S_sigma, H_prior_mu, H_prior_sigma, ic, ic_recon = model(batch)
            xyz, xyz_recon = None, None
        else:
            S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)
   
        if S_mu is not None:
            loss_kl = KL(S_mu, S_sigma, H_prior_mu, H_prior_sigma) 
            loss_kl = torch.maximum(loss_kl-maxkl, torch.tensor(0.0).to(device))

            print("beta     :", "{:.5f}".format(beta))
            print("kl       : ", "{:.5f}".format(loss_kl.item()))
            kl_loss.append(loss_kl.item())
        else:
            loss_kl = 0.0
            kl_loss.append(0.0)

        print(f"-{i}th batch---------------")
        if ic_flag:    
            # n_bins = 36
            # ic_bond, ic_angle, ic_torsion = ic_recon
            
            mask_batch = torch.cat([batch['mask']])
            natom_batch = mask_batch.sum()

            loss_bond = ((ic_recon[:,:,0] - ic[:,:,0]).reshape(-1)) * mask_batch
            # loss_bond = ((ic_bond - ic[:,:,0]).reshape(-1)) * mask_batch
            loss_angle = (2*(1 - torch.cos(ic[:,:,1] - ic_recon[:,:,1])) + EPS).sqrt().reshape(-1) * mask_batch 
            # loss_angle = (2*(1 - torch.cos(ic[:,:,1] - ic_angle)) + EPS).sqrt().reshape(-1) * mask_batch 
            loss_torsion = (2*(1 - torch.cos(ic[:,:,2] - ic_recon[:,:,2])) + EPS).sqrt().reshape(-1) * mask_batch
            
            # n_bins = 36
            # bins = torch.arange(0,n_bins) * 2 * math.pi/n_bins + math.pi/n_bins
            # ic_torsion_true = gaussian_filter(real_number_batch_to_one_hot_vector_bins(ic[:,:,2], bins, device), sigma=2)
            
            # ic_torsion_true = batch['ic_torsion'].reshape(-1,36)
            # loss_torsion = torsion_loss(ic_torsion, ic_torsion_true) * mask_batch

            print(loss_bond[:25])
            print(loss_angle[:25])
            print(loss_torsion[:25])
            
            loss_bond = loss_bond.pow(2).sum()/natom_batch
            loss_angle = loss_angle.sum()/natom_batch
            loss_torsion = loss_torsion.sum()/natom_batch
            # loss_torsion = torch.abs(loss_torsion.sum()/natom_batch - 2.72)
            # loss_torsion = torch.abs(loss_torsion.sum()/natom_batch - 1.12) * 5

            print("bond     : ", "{:.5f}".format(loss_bond.item()))
            print("angle    : ", "{:.5f}".format(loss_angle.item()))
            print("torsion  : ", "{:.5f}".format(loss_torsion.item()))

            # loss_recon = (loss_bond + loss_angle + loss_torsion)
            # loss_recon = loss_bond + loss_angle
            loss_recon = loss_bond * 5 + loss_angle

            print("ic       : ", "{:.5f}".format(loss_recon.item()))
            recon_loss.append(loss_recon.item())
 
            # ic_torsion = torch.multinomial(ic_torsion, 1) * 2 * math.pi/n_bins + math.pi/n_bins
            # ic_recon = torch.cat([ic_bond.unsqueeze(-1), ic_angle.unsqueeze(-1), ic_torsion.reshape(-1,13,1)],axis=-1)

        else:
            loss_recon = (xyz_recon - xyz).pow(2).mean()
            print("xyz      : ", "{:.5f}".format(loss_recon.item()))
            recon_loss.append(loss_recon.item())

        # if not ic_flag:
        if batch['prot_idx'][0] == batch['prot_idx'][-1]:
            if ic_flag:
                nres = batch['num_CGs'][0]+2
                xyz = batch['nxyz'][:, 1:]

                OG_CG_nxyz = batch['OG_CG_nxyz'].reshape(-1, nres, 4)
                ic_recon = ic_recon.reshape(-1, nres-2, 13, 3)

                info = info_dict[int(batch['prot_idx'][0])]
                xyz_recon = ic_to_xyz(OG_CG_nxyz, ic_recon, info).reshape(-1,3)
                
                mask_xyz = batch['mask_xyz_list']
                xyz[mask_xyz] *= 0
                xyz_recon[mask_xyz] *= 0
                
                loss_xyz = (xyz_recon - xyz).pow(2).sum(-1).mean()
                print("xyz      : ", "{:.5f}".format(loss_xyz.item()))
                loss_recon += loss_xyz * zeta

            # add graph loss 
            if gamma != 0.0:
                edge_list = batch['bond_edge_list']
                print("n edges  : ", edge_list.shape[0])
                gen_dist = ((xyz_recon[edge_list[:, 0]] - xyz_recon[edge_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
                data_dist = ((xyz[edge_list[:, 0 ]] - xyz[edge_list[:, 1 ]]).pow(2).sum(-1) + EPS).sqrt()
                loss_graph = (gen_dist - data_dist).pow(2).mean()
                print("graph    : ", "{:.5f}".format(loss_graph.item()))
                loss_recon += loss_graph * gamma
            else:
                loss_graph = torch.tensor(0.0).to(device)

            if delta != 0.0:
                nbr_list = batch['nbr_list']
                combined = torch.cat((edge_list, nbr_list))
                uniques, counts = combined.unique(dim=0, return_counts=True)
                difference = uniques[counts == 1]
                print("n nbrs   : ", difference.shape[0])
                nbr_dist = ((xyz_recon[difference[:, 0]] - xyz_recon[difference[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
                loss_nbr = torch.maximum(2.0 - nbr_dist,torch.tensor(0.0).to(device)).mean()

                bb_NO_list = batch['bb_NO_list']
                bb_NO_dist = ((xyz_recon[bb_NO_list[:, 0]] - xyz_recon[bb_NO_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
                loss_bb_NO = torch.maximum(2.2 - bb_NO_dist, torch.tensor(0.0).to(device)).mean()
                print("bb_NO    : ", "{:.5f}".format(loss_bb_NO.item())) 
                loss_nbr += loss_bb_NO
                loss_recon += loss_nbr * delta
                print("nbr      : ", "{:.5f}".format(loss_nbr.item()))
            else:
                loss_nbr = torch.tensor(0.0).to(device)
            
            # add interaction loss
            if eta != 0.0:                
                interaction_list = batch['interaction_list']
                n_inter = interaction_list.shape[0]
                pi_pi_list = batch['pi_pi_list']
                n_pi_pi = pi_pi_list.shape[0]
                # pi_ion_list = batch['pi_ion_list']
                # n_pi_ion = pi_ion_list.shape[0]
                # n_inter_total = n_inter + n_pi_pi + n_pi_ion
                n_inter_total = n_inter + n_pi_pi 

                print("n inter  : ", n_inter)
                if n_inter > 0:
                    inter_dist = ((xyz_recon[interaction_list[:, 0]] - xyz_recon[interaction_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
                    loss_inter = torch.maximum(inter_dist - 4.0, torch.tensor(0.0).to(device)).mean()
                    print("inter    : ", "{:.5f}".format(loss_inter.item())) 
                    loss_inter *= n_inter/n_inter_total
                else:
                    loss_inter = torch.tensor(0.0).to(device)

                print("n pi-pi  : ", n_pi_pi)
                if n_pi_pi > 0:
                    pi_center_0 = (xyz_recon[pi_pi_list[:,0]] + xyz_recon[pi_pi_list[:,1]])/2
                    pi_center_1 = (xyz_recon[pi_pi_list[:,2]] + xyz_recon[pi_pi_list[:,3]])/2
                    pi_pi_dist = ((pi_center_0 - pi_center_1).pow(2).sum(-1) + EPS).sqrt()
                    loss_pi_pi = torch.maximum(pi_pi_dist - 6.0, torch.tensor(0.0).to(device)).mean()
                    print("pi-pi    : ", "{:.5f}".format(loss_pi_pi.item())) 
                    loss_inter += loss_pi_pi * n_pi_pi/n_inter_total
                else:
                    loss_pi_pi = torch.tensor(0.0).to(device)

                # print("n pi-ion  : ", n_pi_ion)
                # if n_pi_ion > 0:
                #     pi_ion_dist = ((xyz_recon[pi_ion_list[:, 0]] - xyz_recon[pi_ion_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
                #     loss_pi_ion = torch.maximum(pi_ion_dist - 6.5, torch.Tensor([0.0]).to(xyz.device)).mean()
                #     print("pi-ion   : ", "{:.5f}".format(loss_pi_ion.item())) 
                #     loss_inter += loss_pi_ion * n_pi_ion/n_inter_total
                # else:
                #     loss_pi_ion = torch.Tensor([0.0]).to(device)

                if n_inter_total > 0:
                    loss_recon += loss_inter * eta
            else:
                loss_inter = torch.tensor(0.0).to(device)
                
        else:
            loss_graph = torch.tensor(0.0).to(device)
            loss_nbr = torch.tensor(0.0).to(device)
            loss_inter = torch.tensor(0.0).to(device)
        
        loss =  loss_recon + loss_kl * beta

        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'memory usage : ', "{:.5f} %".format(info.used*100/info.total))
        end = time.time()
        print('time     : ', end-st)

        if loss.item() >= (gamma+3)*2000.0 or torch.isnan(loss) :
            print("weird batch ", loss.item())
            continue 

        # optimize 
        if train:
            optimizer.zero_grad()
            loss.backward()

            # perfrom gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

        graph_loss.append(loss_graph.item())
        nbr_loss.append(loss_nbr.item())
        inter_loss.append(loss_inter.item())
        xyz_loss.append(loss_xyz.item())
        total_loss.append(loss.item())

        mean_kl = np.array(kl_loss).mean()
        mean_recon = np.array(recon_loss).mean()
        mean_graph = np.array(graph_loss).mean()
        mean_nbr = np.array(nbr_loss).mean()
        mean_inter = np.array(inter_loss).mean()
        mean_total_loss = np.array(total_loss).mean()
        mean_xyz = np.array(xyz_loss).mean()

        postfix = ['total={:.3f}'.format(mean_total_loss),
                    'KL={:.4f}'.format(mean_kl) , 
                   'recon={:.4f}'.format(mean_recon),
                   'graph={:.4f}'.format(mean_graph) , 
                   'nbr={:.4f}'.format(mean_nbr) ,
                   'inter={:.4f}'.format(mean_inter) ,
                   'memory ={:.4f} Mb'.format(memory) 
                   ]
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

        del loss, loss_graph, loss_kl, loss_recon#, S_mu, S_sigma, H_prior_mu, H_prior_sigma
        # torch.cuda.empty_cache()
        
    for result in postfix:
        print(result)
    
    # return mean_total_loss, mean_kl, mean_recon, mean_graph, mean_nbr, mean_inter, xyz, xyz_recon 
    return mean_total_loss, mean_kl, mean_recon, mean_graph, mean_nbr, mean_inter, mean_xyz


def get_all_true_reconstructed_structures(loader, device, model, atomic_nums=None, n_cg=10, atomwise_z=False, tqdm_flag=True, reflection=False, ic_flag=False, top_table=None, info_dict=None):
    model = model.to(device)
    model.eval()

    true_xyzs = []
    recon_xyzs = []
    cg_xyzs = []

    heavy_ged = []
    all_ged = []

    all_valid_ratios = []
    heavy_valid_ratios = []

    if tqdm_flag:
        loader = tqdm(loader, position=0, leave=True) 

    for batch in loader:
        batch = batch_to(batch, device)

        atomic_nums = batch['nxyz'][:, 0].detach().cpu()

        if reflection: 
            xyz = batch['nxyz'][:,1:]
            xyz[:, 1] *= -1 # reflect around x-z plane
            cgxyz = batch['CG_nxyz'][:,1:]
            cgxyz[:, 1] *= -1 

        if ic_flag:
            S_mu, S_sigma, H_prior_mu, H_prior_sigma, ic, ic_recon = model(batch)
        else:
            S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)

        if ic_flag:
            nres = batch['num_CGs'][0]+2
            xyz = batch['nxyz'][:, 1:]

            OG_CG_nxyz = batch['OG_CG_nxyz'].reshape(-1, nres, 4)
            ic_recon = ic_recon.reshape(-1, nres-2, 13, 3)

            info = info_dict[int(batch['prot_idx'][0])]
            xyz_recon = ic_to_xyz(OG_CG_nxyz, ic_recon, info).reshape(-1,3)
            
            mask_xyz = batch['mask_xyz_list']

            # ring toggle
            xyz[mask_xyz] *= 0
            xyz_recon[mask_xyz] *= 0

        true_xyzs.append(xyz.detach().cpu()) 
        recon_xyzs.append(xyz_recon.detach().cpu())
        cg_xyzs.append(batch['CG_nxyz'][:, 1:].detach().cpu())

        recon = xyz_recon.detach().cpu()
        data = xyz.detach().cpu()

        recon_x_split =  torch.split(recon, batch['num_atoms'].tolist())
        data_x_split =  torch.split(data, batch['num_atoms'].tolist())
        atomic_nums_split = torch.split(atomic_nums, batch['num_atoms'].tolist())

        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        postfix = ['memory ={:.4f} Mb'.format(memory)]

        for i, x in enumerate(data_x_split):

            z = atomic_nums_split[i].numpy()

            ref_atoms = Atoms(numbers=z, positions=x.numpy())
            recon_atoms = Atoms(numbers=z, positions=recon_x_split[i].numpy())

            # compute ged diff 
            all_rmsds, heavy_rmsds, valid_ratio, valid_hh_ratio, graph_val_ratio, graph_hh_val_ratio = eval_sample_qualities(ref_atoms, [recon_atoms])

            heavy_ged.append(graph_val_ratio)
            all_ged.append(graph_hh_val_ratio)

            all_valid_ratios.append(valid_hh_ratio)
            heavy_valid_ratios.append(valid_ratio)
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

    true_xyzs = torch.cat(true_xyzs).numpy()
    recon_xyzs = torch.cat(recon_xyzs).numpy()
    cg_xyzs = torch.cat(cg_xyzs).numpy()

    all_valid_ratio = np.array(all_valid_ratios).mean()
    heavy_valid_ratio = np.array(heavy_valid_ratios).mean()

    all_ged = np.array(all_ged).mean()
    heavy_ged = np.array(heavy_ged).mean()
    
    return true_xyzs, recon_xyzs, cg_xyzs, all_valid_ratio, heavy_valid_ratio, all_ged, heavy_ged

def dump_numpy2xyz(xyzs, atomic_nums, path):
    trajs = [Atoms(positions=xyz, numbers=atomic_nums.ravel()) for xyz in xyzs]
    io.write(path, trajs)
