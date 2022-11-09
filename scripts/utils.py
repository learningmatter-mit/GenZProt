import os 
import sys
from tqdm import tqdm 
import torch
import numpy as np
from ase import Atoms, io 
import networkx as nx
from datetime import date
import torch.autograd.profiler as profiler
from sklearn.utils import shuffle
from pynvml import *

from sampling import *
from utils_ic import * 

nvmlInit()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

EPS = 1e-6

def shuffle_traj(traj):
    full_idx = list(range(len(traj)))
    full_idx = shuffle(full_idx)
    return traj[full_idx]

def annotate_job(task, job_name, N_cg):
    today = date.today().strftime('%m-%d')
    return "{}_{}_{}_N{}".format(job_name, today, task, N_cg)

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


def loop(loader, optimizer, device, model, beta, epoch, 
        gamma, eta=0.0, kappa=0.0, train=True, looptext='', tqdm_flag=True, ic_flag=False, info_dict=None):
    
    total_loss = []
    recon_loss = []
    orient_loss = []
    norm_loss = []
    graph_loss = []
    kl_loss = []
    
    if train:
        model.train()
        mode = '{} train'.format(looptext)
    else:
        model.train() # yes, still set to train when reconstructing
        mode = '{} valid'.format(looptext)

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='({} epoch #{})'.format(mode, epoch))
    
    delta = 0.0
    gamma = 0.0
    beta = 10.0
    if epoch > 4:
        delta, gamma = 10, 0.5

    for i, batch in enumerate(loader):
        batch = batch_to(batch, device)

        if ic_flag:
            S_mu, S_sigma, H_prior_mu, H_prior_sigma, ic, ic_recon = model(batch)
        else:
            S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)
   
        if S_mu is not None:
            loss_kl = KL(S_mu, S_sigma, H_prior_mu, H_prior_sigma) 
            kl_loss.append(loss_kl.item())
        else:
            loss_kl = 0.0
            kl_loss.append(0.0)

        if ic_flag:            
            mask_batch = torch.cat([batch['mask']])
            natom_batch = mask_batch.sum()

            loss_bond = ((torch.absolute(ic_recon[:,:,0]) - ic[:,:,0]).reshape(-1)) * mask_batch
            # loss_angle = ((torch.cos(ic_recon[:,:,1]/2) - torch.cos(ic[:,:,1]/2))).reshape(-1) * mask_batch
            loss_angle = (2*(1 - torch.cos(ic[:,:,1] - ic_recon[:,:,1]))).reshape(-1) * mask_batch 
            # loss_torsion_ = ((torch.cos(ic_recon[:,:,2]/2) - torch.cos(ic[:,:,2]/2))).reshape(-1) * mask_batch 
            loss_torsion_ = (2*(1 - torch.cos(ic[:,:,2] - ic_recon[:,:,2]))).reshape(-1) * mask_batch 

            loss_bond = loss_bond.pow(2).sum()/natom_batch
            loss_angle = loss_angle.sum()/natom_batch
            loss_torsion = loss_torsion_.sum()/natom_batch
            print("bond, angle, torsion: ", loss_bond, loss_angle, loss_torsion)

            loss_torsions = loss_torsion_.reshape(-1, 13)
            mask_per = mask_batch.reshape(-1,13)
            print("N", mask_per[:,0].sum(), loss_torsions[:,0].sum()/mask_per[:,0].sum())
            print("C", mask_per[:,1].sum(), loss_torsions[:,1].sum()/mask_per[:,1].sum())
            print("O", mask_per[:,2].sum(), loss_torsions[:,2].sum()/mask_per[:,2].sum())
            print("CB", mask_per[:,3].sum(), loss_torsions[:,3].sum()/mask_per[:,3].sum())
            print("CG", mask_per[:,4].sum(), loss_torsions[:,4].sum()/mask_per[:,4].sum())
            print("CD", mask_per[:,5].sum(), loss_torsions[:,5].sum()/mask_per[:,5].sum())
            print("CE", mask_per[:,6].sum(), loss_torsions[:,6].sum()/mask_per[:,6].sum())

            loss_recon = (loss_bond + loss_angle + loss_torsion) 
            print("ic_loss: ", loss_recon)
            # loss_recon += loss_xyz * 0.1

        else:
            loss_recon = (xyz_recon - xyz).pow(2).mean()

        if batch['prot_idx'][0] == batch['prot_idx'][-1]:
            nres = batch['num_CGs'][0]+2
            xyz = batch['nxyz'][:, 1:]

            OG_CG_nxyz = batch['OG_CG_nxyz'].reshape(-1, nres, 4)
            ic_recon = ic_recon.reshape(-1, nres-2, 13, 3)

            info = info_dict[int(batch['prot_idx'][0])]
            xyz_recon = ic_to_xyz(OG_CG_nxyz, ic_recon, info).reshape(-1,3)
            loss_xyz = (xyz_recon - xyz).pow(2).mean()
            print("xyz_loss: ", loss_xyz)

            if delta != 0.0:
                nbr_list = batch['nbr_list']
                nbr_dist = ((xyz_recon[nbr_list[:, 0]] - xyz_recon[nbr_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
                loss_nbr = torch.maximum(0.95 - nbr_dist,torch.Tensor([0.0]).to(xyz.device)).mean()
                print("nbr_loss: ", loss_nbr)
                loss_recon += loss_nbr * delta

            # add graph loss 
            if gamma != 0.0:
                edge_list = batch['bond_edge_list']#.to("cpu") 
                gen_dist = ((xyz_recon[edge_list[:, 0 ]] - xyz_recon[edge_list[:, 1 ]]).pow(2).sum(-1) + EPS).sqrt()
                data_dist = ((xyz[edge_list[:, 0 ]] - xyz[edge_list[:, 1 ]]).pow(2).sum(-1) + EPS).sqrt()#.to(xyz_recon.device)
                loss_graph = (gen_dist - data_dist).pow(2).mean()
                print("graph_loss: ", loss_graph)
                loss_recon += loss_graph * gamma 
            else:
                loss_graph = torch.Tensor([0.0]).to(device)
        else:
            loss_graph = torch.Tensor([0.0]).to(device)
        
        loss =  loss_recon + loss_kl * beta

        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total/1024**2}')
        print(f'free     : {info.free/1024**2}')
        print(f'used     : {info.used/1024**2}')
        # memory = 0

        # if loss.item() >= gamma * 200.0 or torch.isnan(loss):
        if loss.item() >= (gamma+3)*2000.0 or torch.isnan(loss):
            print("weird batch ", loss.item())
            del loss, loss_graph, loss_kl, loss_recon, S_mu, S_sigma, H_prior_mu, H_prior_sigma
            continue 

        # optimize 
        if train:
            optimizer.zero_grad()
            loss.backward()

            # perfrom gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

        else:
            loss.backward()

        # logging 
        recon_loss.append(loss_recon.item())
        graph_loss.append(loss_graph.item())
        total_loss.append(loss.item())
        
        mean_kl = np.array(kl_loss).mean()
        mean_recon = np.array(recon_loss).mean()
        mean_graph = np.array(graph_loss).mean()
        mean_total_loss = np.array(total_loss).mean()

        postfix = ['total={:.3f}'.format(mean_total_loss),
                    'KL={:.4f}'.format(mean_kl) , 
                   'recon={:.4f}'.format(mean_recon),
                   'graph={:.4f}'.format(mean_graph) , 
                   'memory ={:.4f} Mb'.format(memory) 
                   ]
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

        del loss, loss_graph, loss_kl, loss_recon, S_mu, S_sigma, H_prior_mu, H_prior_sigma
        # torch.cuda.empty_cache()
        
    for result in postfix:
        print(result)
    
    return mean_total_loss, mean_kl, mean_recon, mean_graph, xyz, xyz_recon 

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

            info = info_dict[0]
            xyz_recon = ic_to_xyz(OG_CG_nxyz, ic_recon, info).reshape(-1,3)

        true_xyzs.append(xyz.detach().cpu()) 
        recon_xyzs.append(xyz_recon.detach().cpu())
        cg_xyzs.append(batch['CG_nxyz'][:, 1:].detach().cpu())

        recon = xyz_recon.detach().cpu()
        data = xyz.detach().cpu()

        recon_x_split =  torch.split(recon, batch['num_atoms'].tolist())
        data_x_split =  torch.split(data, batch['num_atoms'].tolist())
        atomic_nums_split = torch.split(atomic_nums, batch['num_atoms'].tolist())

        del S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon, batch

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
