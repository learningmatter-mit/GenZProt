
import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add

from conv import *
from e3nn_enc import *


class EquivariantPsuedoDecoder(nn.Module):
    """
    CGVAE (Wang et al., ICML2022) Decoder
    """
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, breaksym=False):   
        
        nn.Module.__init__(self)
        
        self.message_blocks = nn.ModuleList(
                [EquiMessagePsuedo(feat_dim=n_atom_basis,
                              activation=activation,
                              n_rbf=n_rbf,
                              cutoff=cutoff,
                              dropout=0.0)
                 for _ in range(num_conv)]
            )

        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation=activation,
                         dropout=0.0)
             for _ in range(num_conv)]
        )

        self.breaksym = breaksym
        self.n_atom_basis = n_atom_basis

    
    def forward(self, cg_xyz, CG_nbr_list, mapping, S):
    
        CG_nbr_list, _ = make_directed(CG_nbr_list)
        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]

        V = torch.zeros(S.shape[0], S.shape[1], 3 ).to(S.device)
        if self.breaksym:
            Sbar = torch.ones(S.shape[0], S.shape[1]).to(S.device)
        else:
            Sbar = torch.zeros(S.shape[0], S.shape[1]).to(S.device)
        Vbar = torch.zeros(S.shape[0], S.shape[1], 3 ).to(S.device)

        for i, message_block in enumerate(self.message_blocks):
            
            # message block
            dS, dSbar, dV, dVbar = message_block(s_j=S,
                                                   sbar_j = Sbar,
                                                   v_j=V,
                                                   vbar_j=Vbar,
                                                   r_ij=r_ij,
                                                   nbrs=CG_nbr_list,
                                                   edge_wgt=None
                                                   )
            S = S + dS
            Sbar = Sbar + dSbar
            V = V + dV
            Vbar = Vbar + dVbar 

            # update block
            dS_update, dV_update = self.update_blocks[i](s_i=S,
                                                v_i=V)

            S = S + dS_update
            V = V + dV_update

        return S, V 


class InternalDecoder52(nn.Module):
    """
    Invariance Message Passing + Dense decoder for internal coordinates
    """
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, cross_flag=True):   
        nn.Module.__init__(self)

        self.message_blocks = nn.ModuleList(
                [InvariantMessage(in_feat_dim=n_atom_basis,
                                    out_feat_dim=n_atom_basis, 
                                    activation=activation,
                                    n_rbf=n_rbf,
                                    cutoff=cutoff,
                                    dropout=0.0)
                 for _ in range(num_conv)]
            )
        
        self.dense_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis, n_atom_basis), 
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis, n_atom_basis))
                 for _ in range(num_conv)]
            )

        self.distance = nn.Embedding(25, 10)
        self.backbone_dist = nn.Embedding(25, 3)

        self.backbone_angle = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis, 3), 
                                    to_module(activation), 
                                    nn.Linear(3, 3))
        self.backbone_torsion = nn.Sequential(to_module(activation), 
                                nn.Linear(n_atom_basis+3, 3), 
                                to_module(activation), 
                                nn.Linear(3, 3))

        self.sidechain_angle = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis, 10), 
                                    to_module(activation), 
                                    nn.Linear(10, 10))
        
        self.sidechain_torsion_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis, n_atom_basis), 
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis, n_atom_basis))
                 for _ in range(num_conv)]
            )
        
        self.final_torsion = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis, 10), 
                                    to_module(activation), 
                                    nn.Linear(10, 10))

    def forward(self, cg_z, cg_xyz, CG_nbr_list, mapping, S, mask=None):   
        CG_nbr_list, _ = make_directed(CG_nbr_list)

        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        dist, unit = preprocess_r(r_ij)
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
        
        bb_dist = self.backbone_dist(cg_z).unsqueeze(-1)
        bb_angle = self.backbone_angle(S)
        bb_torsion = self.backbone_torsion(torch.cat([S, bb_angle], axis=-1))
        ic_bb = torch.cat([bb_dist, bb_angle.unsqueeze(-1), bb_torsion.unsqueeze(-1)], axis=-1)

        sc_dist = self.distance(cg_z).unsqueeze(-1)
        sc_angle = self.sidechain_angle(S)

        for i, torsion_block in enumerate(self.sidechain_torsion_blocks):
            S = S + torsion_block(S)

        sc_torsion = self.final_torsion(S)
        ic_sc = torch.cat([sc_dist, sc_angle.unsqueeze(-1), sc_torsion.unsqueeze(-1)], axis=-1)

        ic_recon = torch.cat([ic_bb, ic_sc], axis=-2) 
        return None, ic_recon


class InternalDecoder56(nn.Module):
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


def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.constant_(module.bias, 0.)


class InternalDecoder55(nn.Module):
    """
    Invariance Message Passing + Dense decoder for internal coordinates
    """
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, cross_flag=True):   
        nn.Module.__init__(self)
        res_embed_dim = 6
        feature_dim = n_atom_basis + res_embed_dim - 3
        self.res_embed = nn.Embedding(25, res_embed_dim)

        self.message_blocks = nn.ModuleList(
                [InvariantMessage(in_feat_dim=feature_dim,
                                    out_feat_dim=feature_dim, 
                                    activation=activation,
                                    n_rbf=n_rbf,
                                    cutoff=cutoff,
                                    dropout=0.0)
                 for _ in range(num_conv)]
            )
        
        self.dense_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(feature_dim, feature_dim), 
                                   to_module(activation), 
                                   nn.Linear(feature_dim, feature_dim))
                 for _ in range(num_conv)]
            )

        self.backbone_dist = nn.Embedding(25, 3)
        self.sidechain_dist = nn.Embedding(25, 10)

        self.backbone_angle = nn.Sequential(to_module(activation), 
                                    nn.Linear(4, 3), 
                                    to_module(activation), 
                                    nn.Linear(3, 3))

        self.sidechain_angle = nn.Embedding(25, 10)

        self.backbone_torsion = nn.Sequential(to_module(activation), 
                                nn.Linear(feature_dim, 3), 
                                to_module(activation), 
                                nn.Linear(3, 3))

        self.sidechain_torsion_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(feature_dim, feature_dim), 
                                   to_module(activation), 
                                   nn.Linear(feature_dim, feature_dim))
                 for _ in range(num_conv)]
            )
        
        self.final_torsion = nn.Sequential(to_module(activation), 
                                    nn.Linear(feature_dim, 10),
                                    to_module(activation), 
                                    nn.Linear(10, 10))

    def forward(self, cg_z, cg_xyz, CG_nbr_list, mapping, S, mask=None):   
        CG_nbr_list, _ = make_directed(CG_nbr_list)

        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        dist, unit = preprocess_r(r_ij)
        graph_size = S.shape[0]

        bb_dist = torch.abs(self.backbone_dist(cg_z).unsqueeze(-1))
        sc_dist = torch.abs(self.sidechain_dist(cg_z).unsqueeze(-1))

        bb_dist = torch.abs(self.backbone_dist(cg_z).unsqueeze(-1))
        sc_dist = torch.abs(self.sidechain_dist(cg_z).unsqueeze(-1))
        bb_angle = self.backbone_angle(torch.concat([S[:,:3], cg_z.unsqueeze(-1)], axis=-1))
        sc_angle = self.sidechain_angle(cg_z)

        S = torch.cat([S[:,3:], self.res_embed(cg_z)],axis=-1)
        for i, message_block in enumerate(self.message_blocks):
            inv_out = message_block(s_j=S,
                                    dist=dist,
                                    nbrs=CG_nbr_list)

            v_i = scatter_add(src=inv_out,
                    index=CG_nbr_list[:, 0],
                    dim=0,
                    dim_size=graph_size)

            S = S + self.dense_blocks[i](v_i)
    
        for i, torsion_block in enumerate(self.sidechain_torsion_blocks):
            S = S + torsion_block(S)

        # bb_torsion = self.backbone_torsion(torch.cat([S, bb_angle], axis=-1))
        bb_torsion = self.backbone_torsion(S)
        sc_torsion = self.final_torsion(S)

        ic_bb = torch.cat([bb_dist, bb_angle.unsqueeze(-1), bb_torsion.unsqueeze(-1)], axis=-1)
        ic_sc = torch.cat([sc_dist, sc_angle.unsqueeze(-1), sc_torsion.unsqueeze(-1)], axis=-1)
        ic_recon = torch.cat([ic_bb, ic_sc], axis=-2) 
        return None, ic_recon


class EquiEncoder(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_rbf,
             activation,
             cutoff,
             dir_mp=False,
             cg_mp=False):
        super().__init__()

        self.atom_embed = nn.Embedding(50, int(n_atom_basis/2), padding_idx=0)
        self.res_embed = nn.Embedding(30, int(n_atom_basis/2), padding_idx=0)
        
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [EquiMessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          dropout=0.0)
             for _ in range(n_conv)]
        )

        self.cgmessage_layers = nn.ModuleList(
        [ContractiveMessageBlock(feat_dim=n_atom_basis,
                                         activation=activation,
                                         n_rbf=n_rbf,
                                         cutoff=12.5,
                                         dropout=0.0)
         for _ in range(n_conv)])

        self.n_conv = n_conv
        self.dir_mp = dir_mp
        self.cg_mp = cg_mp
        self.n_atom_basis = n_atom_basis
    
    def forward(self, z, xyz, cg_z, cg_xyz, mapping, nbr_list, cg_nbr_list):
        
        # atomic embedding
        if not self.dir_mp:
            nbr_list, _ = make_directed(nbr_list)
        cg_nbr_list, _ = make_directed(cg_nbr_list)
        
        h_atom = self.atom_embed(z.long())
        h_res = self.res_embed(cg_z[mapping].long())
        h = torch.cat([h_atom, h_res], axis=-1)

        v = torch.zeros(h.shape[0], h.shape[1], 3).to(h.device)

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]
        
        # edge features
        r_iI = (xyz - cg_xyz[mapping])
        
        for i in range(self.n_conv):
            ds_message, dv_message = self.message_blocks[i](s_j=h,
                                                   v_j=v,
                                                   r_ij=r_ij,
                                                   nbrs=nbr_list)
            h = h + ds_message
            v = v + dv_message

            # contruct atom messages 
            if i == 0:
                H = scatter_mean(h, mapping, dim=0)
                V = scatter_mean(v, mapping, dim=0) 

            # CG message passing 
            dH, dV = self.cgmessage_layers[i](h, v, r_iI, mapping)

            H = H + dH
            V = V + dV
        
        return H, h


class CGprior(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_rbf,
             activation,
             cutoff,
             dir_mp=False):
        super().__init__()

        self.res_embed = nn.Embedding(30, n_atom_basis, padding_idx=0)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [EquiMessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          dropout=0.0)
             for _ in range(n_conv)]
        )

        self.mu = nn.Sequential(nn.Linear(n_atom_basis, n_atom_basis), nn.Tanh(), nn.Linear(n_atom_basis, n_atom_basis))
        self.sigma = nn.Sequential(nn.Linear(n_atom_basis, n_atom_basis), nn.Tanh(), nn.Linear(n_atom_basis, n_atom_basis))
        
        self.n_conv = n_conv
        self.dir_mp = dir_mp
    
    def forward(self, cg_z, cg_xyz, cg_nbr_list):
        
        # atomic embedding
        # if not self.dir_mp:
        cg_nbr_list, _ = make_directed(cg_nbr_list)

        h = self.res_embed(cg_z.long())
        v = torch.zeros(h.shape[0], h.shape[1], 3).to(h.device)

        r_ij = cg_xyz[cg_nbr_list[:, 1]] - cg_xyz[cg_nbr_list[:, 0]]

        for i in range(self.n_conv):
            ds_message, dv_message = self.message_blocks[i](s_j=h,
                                                   v_j=v,
                                                   r_ij=r_ij,
                                                   nbrs=cg_nbr_list)
            h = h + ds_message
            v = v + dv_message

        H_mu = self.mu(h)
        H_sigma = self.sigma(h)

        H_std = 1e-9 + torch.exp(H_sigma / 2)

        return H_mu, H_std


class CGequiVAE(nn.Module):
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

        xyz = batch['nxyz'][:, 1:]

        cg_xyz = batch['CG_nxyz'][:, 1:]

        cg_z = batch['CG_nxyz'][:, 0]
        z = batch['nxyz'][:, 0]

        mapping = batch['CG_mapping']

        nbr_list = batch['nbr_list']
        CG_nbr_list = batch['CG_nbr_list']
        
        num_CGs = batch['num_CGs']
        
        return z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs
        
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

    def decoder(self, cg_xyz, CG_nbr_list, S_I, s_i, mapping, num_CGs):
        cg_s, cg_v = self.equivaraintconv(cg_xyz, CG_nbr_list, mapping,S_I)
        CG2atomChannel = self.CG2ChannelIdx(mapping)

        # implement an non-equivariant decoder 
        if self.equivariant == False: 
            dv = self.euclidean(cg_s).reshape(cg_s.shape[0], cg_s.shape[1], 3)
            xyz_rel = dv[mapping, CG2atomChannel, :]
        else:
            xyz_rel = cg_v[mapping, CG2atomChannel, :]
            
        #this constraint is only true for geometrical mean
        # need to include weights 

        if self.offset:
          decode_offsets = scatter_mean(xyz_rel, mapping, dim=0)
          xyz_rel = xyz_rel - decode_offsets[mapping]

        # sequential generation
        xyz_recon = xyz_rel + cg_xyz[mapping]

        return xyz_recon
        
    def forward(self, batch):

        atomic_nums, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs= self.get_inputs(batch)

        S_I, s_i = self.encoder(atomic_nums, xyz, cg_xyz, mapping, nbr_list, CG_nbr_list)
        
        # get prior based on CG conv 
        if self.prior_net:
            H_prior_mu, H_prior_sigma = self.prior_net(cg_z, cg_xyz, CG_nbr_list)
        else:
            H_prior_mu, H_prior_sigma = None, None 

        z = S_I

        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)

        if not self.det: 
            z_sample = self.reparametrize(mu, sigma)
        else:
            z_sample = z

        S_I = z_sample # s_i not used in decoding 
        xyz_recon = self.decoder(cg_xyz, CG_nbr_list, S_I, s_i, mapping, num_CGs)
        
        return mu, sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon


class peptideCGequiVAE(nn.Module):
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

        xyz = batch['nxyz'][:, 1:]

        cg_xyz = batch['CG_nxyz'][:, 1:]
        cg_z = batch['CG_nxyz'][:, 0].long()
        
        z = batch['nxyz'][:, 0] # atom type

        mapping = batch['CG_mapping']

        nbr_list = batch['nbr_list']
        CG_nbr_list = batch['CG_nbr_list']
        
        num_CGs = batch['num_CGs']

        ic = batch['ic']
        
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


class peptideCGequiVAE2(nn.Module):
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

        xyz = batch['nxyz'][:, 1:]

        cg_xyz = batch['CG_nxyz'][:, 1:]
        cg_z = batch['CG_nxyz'][:, 0].long()
        
        z = batch['nxyz'][:, 0] # atom type

        mapping = batch['CG_mapping']

        nbr_list = batch['nbr_list']
        CG_nbr_list = batch['CG_nbr_list']
        
        num_CGs = batch['num_CGs']

        ic = batch['ic']
        
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

        S_I, _ = self.encoder(ic, z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list)

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
    

class hyperCGequiVAE(CGequiVAE):
    def __init__(self, encoder, equivaraintconv, 
                     atom_munet, atom_sigmanet,
                     n_cgs, feature_dim, n_res=None,
                    prior_net=None, 
                    det=False, equivariant=True, offset=True):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet

        self.n_cgs = n_cgs
        self.n_res = n_res
        self.prior_net = prior_net
        self.det = det

        self.offset = offset
        self.equivariant = equivariant
        if equivariant == False:
            self.euclidean = nn.Linear(self.encoder.n_atom_basis, self.encoder.n_atom_basis * 3)
        
    def get_inputs(self, batch):

        xyz = batch['OG_CG_nxyz'][:, 1:]
        cg_xyz = batch['hyper_CG_nxyz'][:, 1:]

        cg_z = batch['hyper_CG_nxyz'][:, 0]
        z = batch['OG_CG_nxyz'][:, 0]

        mapping = batch['hyper_CG_mapping']

        nbr_list = batch['hyper_nbr_list']
        CG_nbr_list = batch['hyper_CG_nbr_list']
        
        num_CGs = batch['num_hyper_CGs']
        
        return z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs
    
    def forward(self, batch):
        atomic_nums, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs= self.get_inputs(batch)

        S_I, s_i = self.encoder(atomic_nums, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list)

        # get prior based on CG conv 
        if self.prior_net:
            H_prior_mu, H_prior_sigma = self.prior_net(cg_z, cg_xyz, CG_nbr_list)
        else:
            H_prior_mu, H_prior_sigma = None, None 

        z = S_I

        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)
        
        if not self.det: 
            z_sample = self.reparametrize(mu, sigma)
        else:
            z_sample = z

        S_I = z_sample # s_i not used in decoding 
        xyz_recon = self.decoder(cg_xyz, CG_nbr_list, S_I, s_i, mapping, num_CGs)
        return mu, sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon