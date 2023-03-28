import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add

from conv import *
from e3nn_enc import *


class ZmatInternalDecoder(nn.Module):
    """
    Invariance Message Passing + Dense decoder for internal coordinates
    """
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, cross_flag=True):   
        nn.Module.__init__(self)
        res_embed_dim = 4
        self.res_embed = nn.Embedding(25, res_embed_dim)
        self.message_blocks = nn.ModuleList(
                [InvariantMessage(in_feat_dim=n_atom_basis+res_embed_dim,
                                    out_feat_dim=n_atom_basis+res_embed_dim, 
                                    activation=activation,
                                    n_rbf=n_rbf,
                                    cutoff=cutoff,
                                    dropout=0.0)
                 for _ in range(num_conv)]
            )
        
        self.dense_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim), 
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim))
                 for _ in range(num_conv)]
            )

        self.backbone_dist = nn.Embedding(25, 3)
        self.sidechain_dist = nn.Embedding(25, 10)

        self.backbone_angle = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis+res_embed_dim, 3), 
                                    to_module(activation), 
                                    nn.Linear(3, 3))
        self.sidechain_angle = nn.Embedding(25, 10)

        self.backbone_torsion = nn.Sequential(to_module(activation), 
                                nn.Linear(n_atom_basis+res_embed_dim+3, 3), 
                                to_module(activation), 
                                nn.Linear(3, 3))

        self.sidechain_torsion_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim), 
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim))
                 for _ in range(num_conv)]
            )
        
        self.final_torsion = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis+res_embed_dim, 10), 
                                    to_module(activation), 
                                    nn.Linear(10, 10))

    def forward(self, cg_z, cg_xyz, CG_nbr_list, mapping, S, mask=None):   
        CG_nbr_list, _ = make_directed(CG_nbr_list)

        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        dist, unit = preprocess_r(r_ij)

        bb_dist = self.backbone_dist(cg_z).unsqueeze(-1)
        sc_dist = self.sidechain_dist(cg_z).unsqueeze(-1)
        sc_angle = self.sidechain_angle(cg_z)

        S = torch.cat([S, self.res_embed(cg_z)],axis=-1)
        graph_size = S.shape[0]
        
        for i, message_block in enumerate(self.message_blocks):
            inv_out = message_block(s_j=S,
                                    dist=dist,
                                    nbrs=CG_nbr_list)

            v_i = scatter_add(src=inv_out,
                    index=CG_nbr_list[:, 0],
                    dim=0,
                    dim_size=graph_size)

            S = S + self.dense_blocks[i](v_i)
        
        
        bb_angle = self.backbone_angle(S)
        bb_torsion = self.backbone_torsion(torch.cat([S, bb_angle], axis=-1))
        
        for i, torsion_block in enumerate(self.sidechain_torsion_blocks):
            S = S + torsion_block(S)
        sc_torsion = self.final_torsion(S)

        ic_bb = torch.cat([bb_dist, bb_angle.unsqueeze(-1), bb_torsion.unsqueeze(-1)], axis=-1)
        ic_sc = torch.cat([sc_dist, sc_angle.unsqueeze(-1), sc_torsion.unsqueeze(-1)], axis=-1)
        ic_recon = torch.cat([ic_bb, ic_sc], axis=-2) 
        return None, ic_recon


class GenZProt(nn.Module):
    def __init__(self, encoder, equivaraintconv, 
                     atom_munet, atom_sigmanet,
                     n_cgs, feature_dim,
                    prior_net=None, 
                    det=False, equivariant=True, offset=True):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet

        self.n_cgs = n_cgs
        self.prior_net = prior_net
        self.det = det

        self.offset = offset
        self.equivariant = equivariant
        if equivariant == False:
            self.euclidean = nn.Linear(self.encoder.n_atom_basis, self.encoder.n_atom_basis * 3)
        
    def get_inputs(self, batch):
        # training (all info)
        if 'nxyz' in batch.keys():
            xyz = batch['nxyz'][:, 1:]
            z = batch['nxyz'][:, 0] # atom type
            nbr_list = batch['nbr_list']
            ic = batch['ic']
        # inference (CG info only)
        else:
            xyz, z, nbr_list, ic = None, None, None, None

        cg_xyz = batch['CG_nxyz'][:, 1:]
        cg_z = batch['CG_nxyz'][:, 0].long()
        mapping = batch['CG_mapping']
        CG_nbr_list = batch['CG_nbr_list']
        num_CGs = batch['num_CGs']

        return z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, ic
        
    def reparametrize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        S_I = eps.mul(sigma).add_(mu)

        return S_I

    def CG2ChannelIdx(self, CG_mapping):

        CG2atomChannel = torch.zeros_like(CG_mapping).to("cpu")

        for cg_type in torch.unique(CG_mapping): 
            cg_filter = CG_mapping == cg_type
            num_contri_atoms = cg_filter.sum().item()
            CG2atomChannel[cg_filter] = torch.LongTensor(list(range(num_contri_atoms)))#.to(CG_mapping.device)

        return CG2atomChannel.detach()
            
    def decoder(self, cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask=None):
        _, ic_recon = self.equivaraintconv(cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask)
        
        return ic_recon
        
    def forward(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, ic = self.get_inputs(batch)

        S_I, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list)

        # get prior based on CG conv 
        if self.prior_net:
            H_prior_mu, H_prior_sigma = self.prior_net(cg_z, cg_xyz, CG_nbr_list)
        else:
            H_prior_mu, H_prior_sigma = None, None 
        
        z = S_I

        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)

        print("prior", H_prior_mu.mean(), H_prior_sigma.mean())
        print("encoder", mu.mean(), sigma.mean())
        
        if not self.det: 
            z_sample = self.reparametrize(mu, sigma)
        else:
            z_sample = z

        S_I = z_sample # s_i not used in decoding 
        ic_recon = self.decoder(cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask=None)
        
        return mu, sigma, H_prior_mu, H_prior_sigma, ic, ic_recon
    