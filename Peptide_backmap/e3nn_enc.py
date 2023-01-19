"""
e3nn encoder 
borrowed from DiffDock (https://github.com/gcorso/DiffDock)
"""
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter, scatter_mean, scatter_add

from e3nn import o3
from e3nn.nn import BatchNorm

from conv import make_directed

class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=False, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out



class e3nnEncoder(torch.nn.Module):
    def __init__(self, device, n_atom_basis, n_cgs=None, in_edge_features=4, cross_max_distance=30,
                 sh_lmax=2, ns=12, nv=4, num_conv_layers=3, atom_max_radius=12, cg_max_radius=30,
                 distance_embed_dim=8, cross_distance_embed_dim=8, use_second_order_repr=False, batch_norm=False,
                 dropout=0.0, lm_embedding_type=None):
        super(e3nnEncoder, self).__init__()
        
        self.in_edge_features = in_edge_features
        self.atom_max_radius = atom_max_radius
        self.cg_max_radius = cg_max_radius
        self.distance_embed_dim = distance_embed_dim
        self.cross_max_distance = cross_max_distance

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.device = device

        self.num_conv_layers = num_conv_layers

        self.atom_node_embedding = nn.Embedding(30, ns, padding_idx=0)
        self.atom_edge_embedding = nn.Sequential(
                                   nn.Linear(2 + in_edge_features + distance_embed_dim, ns),
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.cg_node_embedding = nn.Embedding(30, ns, padding_idx=0)
        self.cg_edge_embedding = nn.Sequential(
                                   nn.Linear(2 + in_edge_features + distance_embed_dim, ns),
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(
                                   nn.Linear(cross_distance_embed_dim, ns),
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.atom_distance_expansion = GaussianSmearing(0.0, atom_max_radius, distance_embed_dim)
        self.cg_distance_expansion = GaussianSmearing(0.0, cg_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        atom_conv_layers, cg_conv_layers, cg_to_atom_conv_layers, atom_to_cg_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            atom_layer = TensorProductConvLayer(**parameters)
            atom_conv_layers.append(atom_layer)
            cg_layer = TensorProductConvLayer(**parameters)
            cg_conv_layers.append(cg_layer)
            cg_to_atom_layer = TensorProductConvLayer(**parameters)
            cg_to_atom_conv_layers.append(cg_to_atom_layer)
            atom_to_cg_layer = TensorProductConvLayer(**parameters)
            atom_to_cg_conv_layers.append(atom_to_cg_layer)

        self.atom_conv_layers = nn.ModuleList(atom_conv_layers)
        self.cg_conv_layers = nn.ModuleList(cg_conv_layers)
        self.cg_to_atom_conv_layers = nn.ModuleList(cg_to_atom_conv_layers)
        self.atom_to_cg_conv_layers = nn.ModuleList(atom_to_cg_conv_layers)

        self.dense = nn.Sequential(nn.Linear(84, n_atom_basis),
                                   nn.Tanh(), 
                                   nn.Linear(n_atom_basis, n_atom_basis))

    def forward(self, z, xyz, cg_z, cg_xyz, mapping, nbr_list, cg_nbr_list):

        # build atom graph
        atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(z, xyz, nbr_list)        
        atom_src, atom_dst = atom_edge_index
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

        # build cg graph
        cg_node_attr, cg_edge_index, cg_edge_attr, cg_edge_sh = self.build_cg_conv_graph(cg_z, cg_xyz, cg_nbr_list)
        cg_src, cg_dst = cg_edge_index
        cg_node_attr = self.cg_node_embedding(cg_node_attr)
        cg_edge_attr = self.cg_edge_embedding(cg_edge_attr)
        
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(xyz, cg_xyz, mapping)
        cross_atom, cross_cg = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for l in range(len(self.atom_conv_layers)):
            # intra graph message passing
            atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[atom_src, :self.ns], atom_node_attr[atom_dst, :self.ns]], -1) #(506470,48)
            atom_intra_update = self.atom_conv_layers[l](atom_node_attr, atom_edge_index, atom_edge_attr_, atom_edge_sh) #(6688,28)
            
            # inter graph message passing
            cg_to_atom_edge_attr_ = torch.cat([cross_edge_attr, atom_node_attr[cross_atom, :self.ns], cg_node_attr[cross_cg, :self.ns]], -1) #(6688,48)
            atom_inter_update = self.cg_to_atom_conv_layers[l](cg_node_attr, cross_edge_index, cg_to_atom_edge_attr_, cross_edge_sh,
                                                              out_nodes=atom_node_attr.shape[0]) #(6688,28)
            
            if l != len(self.atom_conv_layers) - 1:
                cg_edge_attr_ = torch.cat([cg_edge_attr, cg_node_attr[cg_src, :self.ns], cg_node_attr[cg_dst, :self.ns]], -1)
                cg_intra_update = self.cg_conv_layers[l](cg_node_attr, cg_edge_index, cg_edge_attr_, cg_edge_sh)

                atom_to_cg_edge_attr_ = torch.cat([cross_edge_attr, atom_node_attr[cross_atom, :self.ns], cg_node_attr[cross_cg, :self.ns]], -1)
                cg_inter_update = self.atom_to_cg_conv_layers[l](atom_node_attr, (cross_cg, cross_atom), atom_to_cg_edge_attr_,
                                                                  cross_edge_sh, out_nodes=cg_node_attr.shape[0])

            # padding original features
            atom_node_attr = F.pad(atom_node_attr, (0, atom_intra_update.shape[-1] - atom_node_attr.shape[-1]))

            # update features with residual updates
            atom_node_attr = atom_node_attr + atom_intra_update + atom_inter_update

            if l != len(self.atom_conv_layers) - 1:
                cg_node_attr = F.pad(cg_node_attr, (0, cg_intra_update.shape[-1] - cg_node_attr.shape[-1]))
                cg_node_attr = cg_node_attr + cg_intra_update + cg_inter_update

        node_attr = torch.cat([atom_node_attr, cg_node_attr[mapping]], -1)
        node_attr = scatter_mean(node_attr, mapping, dim=0) 
        node_attr = self.dense(node_attr)
        return node_attr, None

    def build_atom_conv_graph(self, z, xyz, nbr_list):
        nbr_list, _ = make_directed(nbr_list) 
        
        node_attr = z.long() 
        edge_attr = torch.cat([
            z[nbr_list[:,0]].unsqueeze(-1), z[nbr_list[:,1]].unsqueeze(-1),
            torch.zeros(nbr_list.shape[0], self.in_edge_features, device=z.device)
        ], -1) 

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]
        edge_length_emb = self.atom_distance_expansion(r_ij.norm(dim=-1)) 
        edge_attr = torch.cat([edge_attr, edge_length_emb], -1) 
        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_ij, normalize=True, normalization='component')
        
        nbr_list = nbr_list[:,0], nbr_list[:,1]
        return node_attr, nbr_list, edge_attr, edge_sh

    def build_cg_conv_graph(self, cg_z, cg_xyz, cg_nbr_list):
        cg_nbr_list, _ = make_directed(cg_nbr_list)
        node_attr = cg_z.long()
        edge_attr = torch.cat([
            cg_z[cg_nbr_list[:,0]].unsqueeze(-1), cg_z[cg_nbr_list[:,1]].unsqueeze(-1),
            torch.zeros(cg_nbr_list.shape[0], self.in_edge_features, device=cg_z.device)
        ], -1)

        r_IJ = cg_xyz[cg_nbr_list[:, 1]] - cg_xyz[cg_nbr_list[:, 0]]
        edge_length_emb = self.cg_distance_expansion(r_IJ.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_IJ, normalize=True, normalization='component')
        cg_nbr_list = cg_nbr_list[:,0], cg_nbr_list[:,1]
        return node_attr, cg_nbr_list, edge_attr, edge_sh

    def build_cross_conv_graph(self, xyz, cg_xyz, mapping):
        cross_nbr_list = torch.arange(len(mapping)).to(cg_xyz.device), mapping 
        r_iI = (xyz - cg_xyz[mapping])
        edge_attr = self.cross_distance_expansion(r_iI.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_iI, normalize=True, normalization='component')
        return cross_nbr_list, edge_attr, edge_sh


class e3nnPrior(torch.nn.Module):
    def __init__(self, device, n_atom_basis, n_cgs=None, in_edge_features=4,
                 sh_lmax=2, ns=12, nv=4, num_conv_layers=3, cg_max_radius=30,
                 distance_embed_dim=8, use_second_order_repr=False, batch_norm=False,
                 dropout=0.0, lm_embedding_type=None):
        super(e3nnPrior, self).__init__()
        
        self.in_edge_features = in_edge_features
        self.cg_max_radius = cg_max_radius
        self.distance_embed_dim = distance_embed_dim

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.device = device

        self.num_conv_layers = num_conv_layers

        self.cg_node_embedding = nn.Embedding(30, ns, padding_idx=0)
        self.cg_edge_embedding = nn.Sequential(
                                   nn.Linear(2 + in_edge_features + distance_embed_dim, ns),
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.cg_distance_expansion = GaussianSmearing(0.0, cg_max_radius, distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        cg_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            cg_layer = TensorProductConvLayer(**parameters)
            cg_conv_layers.append(cg_layer)
            
        self.cg_conv_layers = nn.ModuleList(cg_conv_layers)
        
        self.mu = nn.Sequential(nn.Linear(48, n_atom_basis), 
                                   nn.Tanh(), 
                                   nn.Linear(n_atom_basis, n_atom_basis))
        self.sigma = nn.Sequential(nn.Linear(48, n_atom_basis), 
                                   nn.Tanh(), 
                                   nn.Linear(n_atom_basis, n_atom_basis))

    def forward(self, cg_z, cg_xyz, cg_nbr_list):
        # build cg graph
        cg_node_attr, cg_edge_index, cg_edge_attr, cg_edge_sh = self.build_cg_conv_graph(cg_z, cg_xyz, cg_nbr_list)
        cg_src, cg_dst = cg_edge_index
        cg_node_attr = self.cg_node_embedding(cg_node_attr)
        cg_edge_attr = self.cg_edge_embedding(cg_edge_attr)

        for l in range(len(self.cg_conv_layers)):
            # intra graph message passing
            cg_edge_attr_ = torch.cat([cg_edge_attr, cg_node_attr[cg_src, :self.ns], cg_node_attr[cg_dst, :self.ns]], -1)
            cg_intra_update = self.cg_conv_layers[l](cg_node_attr, cg_edge_index, cg_edge_attr_, cg_edge_sh)
            
            cg_node_attr = F.pad(cg_node_attr, (0, cg_intra_update.shape[-1] - cg_node_attr.shape[-1]))
            cg_node_attr = cg_node_attr + cg_intra_update 

        H_mu = self.mu(cg_node_attr)
        H_logvar = self.sigma(cg_node_attr)

        H_sigma = 1e-9 + torch.exp(H_logvar / 2)
        return H_mu, H_sigma

    def build_cg_conv_graph(self, cg_z, cg_xyz, cg_nbr_list):
        cg_nbr_list, _ = make_directed(cg_nbr_list)

        node_attr = cg_z.long()
        edge_attr = torch.cat([
            cg_z[cg_nbr_list[:,0]].unsqueeze(-1), cg_z[cg_nbr_list[:,1]].unsqueeze(-1),
            torch.zeros(cg_nbr_list.shape[0], self.in_edge_features, device=cg_z.device)
        ], -1)

        r_IJ = cg_xyz[cg_nbr_list[:, 1]] - cg_xyz[cg_nbr_list[:, 0]]
        edge_length_emb = self.cg_distance_expansion(r_IJ.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_IJ, normalize=True, normalization='component')
        cg_nbr_list = cg_nbr_list[:,0], cg_nbr_list[:,1]
        return node_attr, cg_nbr_list, edge_attr, edge_sh


class TensorProductConvBlock(torch.nn.Module):
    def __init__(self, in_edge_features=4,
                 sh_lmax=2, ns=16, nv=4, num_conv_layers=2, cg_max_radius=6,
                 distance_embed_dim=8, use_second_order_repr=False, batch_norm=False,
                 dropout=0.0, lm_embedding_type=None):
        super(TensorProductConvBlock, self).__init__()
        
        self.in_edge_features = in_edge_features
        self.cg_max_radius = cg_max_radius
        self.distance_embed_dim = distance_embed_dim

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv

        self.num_conv_layers = num_conv_layers

        self.cg_node_embedding = nn.Sequential(nn.Linear(32, ns), nn.Tanh(), nn.Linear(ns, ns))
        self.cg_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.cg_distance_expansion = GaussianSmearing(0.0, cg_max_radius, distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        cg_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            cg_layer = TensorProductConvLayer(**parameters)
            cg_conv_layers.append(cg_layer)
            
        self.cg_conv_layers = nn.ModuleList(cg_conv_layers)

    def forward(self, cg_z, cg_xyz, cg_nbr_list):
        # build cg graph
        cg_node_attr, cg_edge_index, cg_edge_attr, cg_edge_sh = self.build_cg_conv_graph(cg_z, cg_xyz, cg_nbr_list)
        cg_src, cg_dst = cg_edge_index
        cg_node_attr = self.cg_node_embedding(cg_node_attr)
        cg_edge_attr = self.cg_edge_embedding(cg_edge_attr)

        for l in range(len(self.cg_conv_layers)):
            # intra graph message passing
            cg_edge_attr_ = torch.cat([cg_edge_attr, cg_node_attr[cg_src, :self.ns], cg_node_attr[cg_dst, :self.ns]], -1)
            cg_intra_update = self.cg_conv_layers[l](cg_node_attr, cg_edge_index, cg_edge_attr_, cg_edge_sh)
            
            cg_node_attr = F.pad(cg_node_attr, (0, cg_intra_update.shape[-1] - cg_node_attr.shape[-1]))
            cg_node_attr = cg_node_attr + cg_intra_update 

        return cg_node_attr

    def build_cg_conv_graph(self, cg_z, cg_xyz, cg_nbr_list):
        cg_nbr_list, _ = make_directed(cg_nbr_list)

        node_attr = cg_z

        r_IJ = cg_xyz[cg_nbr_list[:, 1]] - cg_xyz[cg_nbr_list[:, 0]]
        edge_length_emb = self.cg_distance_expansion(r_IJ.norm(dim=-1))
        edge_attr = edge_length_emb

        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_IJ, normalize=True, normalization='component')
        cg_nbr_list = cg_nbr_list[:,0], cg_nbr_list[:,1]
        return node_attr, cg_nbr_list, edge_attr, edge_sh


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))