"""
functions partially adapted and modified from CGVAE (Wang et al., ICML2022) 
https://github.com/wwang2/CoarseGrainingVAE/datasets.py
"""

import sys
sys.path.append("../scripts/")
import glob
import itertools
import random
from tqdm import tqdm
import time
import math

import numpy as np 
import pickle
import networkx as nx
from sklearn.utils import shuffle

from moleculekit.molecule import Molecule
import mdtraj as md
from ase import Atoms

import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean, scatter_add

from data import * 
from utils_ic import * 
from utils import shuffle_traj


THREE_LETTER_TO_ONE = {
    "ARG": "R", 
    "HIS": "H", 
    "HID": "H",
    "LYS": "K", 
    "ASP": "D", 
    "GLU": "E", 
    "SER": "S", 
    "THR": "T", 
    "ASN": "N", 
    "GLN": "Q", 
    "CYS": "C", 
    "GLY": "G", 
    "PRO": "P", 
    "ALA": "A", 
    "VAL": "V", 
    "ILE": "I", 
    "LEU": "L", 
    "MET": "M", 
    "PHE": "F", 
    "TYR": "Y", 
    "TRP": "W",
    "TPO": "O",
    "SEP": "B"
}

RES2IDX = {'N': 0,
             'H': 1,
             'A': 2,
             'G': 3,
             'R': 4,
             'M': 5,
             'S': 6,
             'I': 7,
             'E': 8,
             'L': 9,
             'Y': 10,
             'D': 11,
             'V': 12,
             'W': 13,
             'Q': 14,
             'K': 15,
             'P': 16,
             'F': 17,
             'C': 18,
             'T': 19,
             'O': 20,
             'B': 21}

atomic_num_dict = {'C':6, 'H':1, 'O':8, 'N':7, 'S':16, 'Se': 34}

PROTEINFILES = {}
PED_PDBs = glob.glob(f'../data/*.pdb')
for PDBfile in PED_PDBs:
    ID = PDBfile.split('/')[-1].split('.')[0][3:]
    dct = {ID: {'pdb_path': PDBfile,
            'traj_paths': PDBfile,
            'file_type': 'pdb'
                    }
                    }
    PROTEINFILES.update(dct)

def random_rotate_xyz_cg(xyz, cg_xyz ): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    cgatoms = Atoms(positions=cg_xyz, numbers=list( range(cg_xyz.shape[0]) ))
    
    # generate rotation paramters 
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    
    # rotate 
    atoms.rotate(angle, nvec)
    cgatoms.rotate(angle, nvec)
    
    return atoms.positions, cgatoms.positions


def random_rotation(xyz): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    atoms.rotate(angle, nvec)
    return atoms.positions


def load_protein_traj(label, ntraj=200, PROTEINFILES=None): 
    traj_files = glob.glob(PROTEINFILES[label]['traj_paths'])[:ntraj]
    pdb_file = PROTEINFILES[label]['pdb_path']
    file_type = PROTEINFILES[label]['file_type']
    
    if file_type == 'xtc':
        trajs = [md.load_xtc(file,
                    top=pdb_file) for file in traj_files]
    elif file_type == 'dcd':
        trajs = [md.load_dcd(file,
                    top=pdb_file) for file in traj_files]  
    elif file_type == 'pdb':
        trajs = [md.load_pdb(file) for file in traj_files]
    else:
        raise ValueError("file type {} not recognized".format(file_type))
                
    traj = md.join(trajs)
                   
    return traj


def get_alpha_mapping(top):
    mappings = []
    table, _ = top.to_dataframe()
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']
    reslist = list(set(list(table.newSeq)))
    reslist.sort()

    j = 0
    for i in range(len(table)):
        if table.iloc[i].newSeq == reslist[j]:
            mappings.append(j)
        else:
            j += 1
            mappings.append(j)

    mapping = np.array(mappings)
    return mapping


def get_cg_and_xyz(traj, params, cg_method='backbone', n_cgs=None, mapshuffle=0.0, mapping=None):
    """
    from Wang et al. (ICML 2022)
    changed cg_method in ['minimal', alpha']
    """
    atomic_nums, protein_index = get_atomNum(traj)
    n_atoms = len(atomic_nums)
    skip = 200

    frames = traj.xyz[:, protein_index, :] * 10.0 

    if cg_method in ['minimal', 'alpha']:
        mappings = []
        print("Note, using CG method {}, user-specified N_cg will be overwritten".format(cg_method))

        table, _ = traj.top.to_dataframe()
        table['newSeq'] = table['resSeq'] + 5000*table['chainID']
        reslist = list(set(list(table.newSeq)))
        reslist.sort()

        j = 0
        for i in range(len(table)):
            if table.iloc[i].newSeq == reslist[j]:
                mappings.append(j)
            else:
                j += 1
                mappings.append(j)

        n_cgs = len(reslist)

        cg_coord = None
        mapping = np.array(mappings)
        print("generated mapping: ", traj)
        frames = shuffle(frames)

    elif cg_method =='newman':

        if n_cgs is None:
            raise ValueError("need to provided number of CG sites")

        protein_top = traj.top.subset(protein_index)
        g = protein_top.to_bondgraph()
        paritions = get_partition(g, n_cgs)
        mapping = parition2mapping(paritions, n_atoms)
        mapping = np.array(mapping)
        cg_coord = None

        # randomly shuffle map 
        perm_percent = mapshuffle

        if mapshuffle > 0.0:
            ran_idx = random.sample(range(mapping.shape[0]), int(perm_percent * mapping.shape[0])  )
            idx2map = mapping[ran_idx]
            mapping[ran_idx] = shuffle(idx2map)

        frames = shuffle(frames)

    elif cg_method == 'backbonepartition': 
        mapping = backbone_partition(traj, n_cgs)
        cg_coord = None


    elif cg_method == 'seqpartition':
        partition = random.sample(range(n_atoms), n_cgs - 1 )
        partition = np.sort(partition)
        mapping = np.zeros(n_atoms)
        mapping[partition] = 1
        mapping = np.cumsum(mapping)

        cg_coord = None
        frames = shuffle(frames)

    elif cg_method =='random':

        mapping = get_random_mapping(n_cgs, n_atoms)
        cg_coord = None
        frames = shuffle(frames)

    else:
        raise ValueError("{} coarse-graining option not available".format(cg_method))

    # print coarse graining summary 
    print("CG method: {}".format(cg_method))
    print("Number of CG sites: {}".format(mapping.max() + 1))

    mapping = torch.LongTensor( mapping)
    
    return mapping, frames, cg_coord


def get_atomNum(traj):
    
    atomic_nums = [atom.element.number for atom in traj.top.atoms]
    
    protein_index = traj.top.select("protein")
    protein_top = traj.top.subset(protein_index)

    atomic_nums = [atom.element.number for atom in protein_top.atoms]
    
    return np.array(atomic_nums), protein_index

def compute_nbr_list(frame, cutoff):
    
    dist = (frame[None, ...] - frame[:, None, :]).pow(2).sum(-1).sqrt()
    nbr_list = torch.nonzero(0 < dist < cutoff).numpy()
    
    return nbr_list

def parition2mapping(partitions, n_nodes):
    # generate mapping 
    mapping = np.zeros(n_nodes)
    
    for k, group in enumerate(partitions):
        for node in group:
            mapping[node] = k
            
    return mapping.astype(int)

def get_partition(G, n_partitions):
    
    # adj = [tuple(pair) for pair in nbr_list]
    # G = nx.Graph()
    # G.add_edges_from(adj)

    G = nx.convert_node_labels_to_integers(G)
    comp = nx.community.girvan_newman(G)

    for communities in itertools.islice(comp, n_partitions-1):
            partitions = tuple(sorted(c) for c in communities)
        
    return partitions 

def compute_mapping(atomic_nums, traj, cutoff, n_atoms, n_cgs, skip):

    # get bond graphs 
    g = traj.top.to_bondgraph()
    paritions = get_partition(g, n_cgs)
    mapping = parition2mapping(paritions, n_atoms)

    return mapping


def get_random_mapping(n_cg, n_atoms):

    mapping = torch.LongTensor(n_atoms).random_(0, n_cg)
    i = 1
    while len(mapping.unique()) != n_cg and i <= 10000000:
        i += 1
        mapping = torch.LongTensor(n_atoms).random_(0, n_cg)

    return mapping


def get_traj(pdb, files, n_frames, shuffle=False):
    feat = pyemma.coordinates.featurizer(pdb)
    traj = pyemma.coordinates.load(files, features=feat)
    traj = np.concatenate(traj)

    peptide_element = [atom.element.symbol for atom in pdb.top.atoms]

    if shuffle: 
        traj = shuffle(traj)
        
    traj_reshape = traj.reshape(-1, len(peptide_element),  3)[:n_frames] * 10.0 # Change from nanometer to Angstrom 
    atomic_nums = np.array([atomic_num_dict[el] for el in peptide_element] )
    
    return atomic_nums, traj_reshape


def get_high_order_edge(edges, order, natoms):

    # get adj 
    adj = torch.zeros(natoms, natoms)
    adj[edges[:,0], edges[:,1]] = 1
    adj[edges[:,1], edges[:,0]] = 1

    # get higher edges 
    edges = torch.triu(get_higher_order_adj_matrix(adj, order=order)).nonzero()

    return edges 


# Building dataset
bb_list = ['CA', 'C', 'N', 'O', 'H']
allow_list = ['NO', 'ON', 'SN', 'NS', 'SO', 'OS', 'SS', 'NN', 'OO']
ring_list = ['PHE', 'TYR', 'TRP', 'HIS']
ring_name_list = ['CG', 'CZ', 'CE1', 'CE2']
ion_list = ['ASP', 'GLU', 'ARG', 'LYS']
ion_name_list = ['OD1', 'OD2', 'NH1', 'NH2', 'NZ']


def build_ic_peptide_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=1, cg_traj=None, prot_idx=None):
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []
    bond_edge_list = []
    
    table, _ = top.to_dataframe()
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']

    endpoints = []
    for idx, chainID in enumerate(np.unique(np.array(table.chainID))):
        tb_chain = table.loc[table.chainID==chainID]
        first = tb_chain.newSeq.min()
        last = tb_chain.newSeq.max()
        endpoints.append(first)
        endpoints.append(last)
 
    print(f'traj has {table.chainID.max()+1} chains')
    nfirst = len(table.loc[table.newSeq==table.newSeq.min()])
    nlast = len(table.loc[table.newSeq==table.newSeq.max()])

    _top = top.subset(np.arange(top.n_atoms)[nfirst:-nlast])
    bondgraph = _top.to_bondgraph()
    indices = table.loc[table.name=='CA'].index

    edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
    edges = get_high_order_edge(edges, order, atomic_nums.shape[0])

    for xyz in tqdm(traj, desc='generate all atom', file=sys.stdout):
        xyz = random_rotation(xyz)
        nxyz = torch.cat((torch.Tensor(atomic_nums[..., None]), torch.Tensor(xyz) ), dim=-1)
        nxyz_data.append(nxyz)
        num_atoms_list.append(torch.LongTensor( [len(nxyz)-(nfirst+nlast)]))
        bond_edge_list.append(edges)

    CG_res = list(top.residues)
    CG_res = CG_res[:len(mapping.unique())]
    CG_res = torch.LongTensor([RES2IDX[THREE_LETTER_TO_ONE[res.name[:3]]] for res in CG_res]).reshape(-1,1)
    # Aggregate CG coorinates 
    for i, nxyz in enumerate(tqdm(nxyz_data, desc='generate CG', file=sys.stdout)):
        xyz = torch.Tensor(nxyz[:, 1:]) 
        if cg_traj is not None:
            CG_xyz = torch.Tensor( cg_traj[i] )
        else:
            CG_xyz = xyz[indices]
        CG_nxyz = torch.cat((CG_res, CG_xyz), dim=-1)
        CG_nxyz_data.append(CG_nxyz)
        num_CGs_list.append(torch.LongTensor( [len(CG_nxyz)-2]) )
        CG_mapping_list.append(mapping[nfirst:-nlast]-1)

    # delete first and last residue
    nxyz_data = [nxyz[nfirst:-nlast,:] for nxyz in nxyz_data] 
    trim_CG_nxyz_data = [nxyz[1:-1,:] for nxyz in CG_nxyz_data] 

    res_list = np.unique(np.array(table.newSeq))
    n_res = len(res_list)
    mask = torch.zeros(n_res-2,13)
    for i in tqdm(range(1,n_res-1), desc='generate mask', file=sys.stdout):
        if res_list[i] not in endpoints:
            num_atoms = len(table.loc[table.newSeq==res_list[i]])-1
            mask[i-1][:num_atoms] = torch.ones(num_atoms)    
    mask = mask.reshape(-1)

    interm_endpoints = set(endpoints)-set([table.newSeq.min(), table.newSeq.max()])
    mask_xyz_list = []
    for res in interm_endpoints:
        mask_xyz_list += list(table.loc[table.newSeq==res].index)
    mask_xyz = torch.LongTensor(np.array(mask_xyz_list) - nfirst)
    
    mask_list = [mask for _ in range(len(nxyz_data))]
    mask_xyz_list = [mask_xyz for _ in range(len(nxyz_data))]
    
    prot_idx_list = [torch.Tensor([prot_idx]) for _ in range(len(nxyz_data))]
    
    st = time.time()
    print(f"generate ic start")
    bb_ic = torch.Tensor(get_backbone_ic(md.Trajectory(traj, top)))
    sc_ic = torch.Tensor(get_sidechain_ic(md.Trajectory(traj, top)))    
    ic_list = torch.cat((bb_ic, sc_ic), axis=2)
    ic_list[:,:,:,1:] = ic_list[:,:,:,1:]%(2*math.pi) 
    ic_list = [ic_list[i] for i in range(len(ic_list))]
    
    print(f"generate ic end {time.time()-st} sec")
    
    props = {'nxyz': nxyz_data,
             'CG_nxyz': trim_CG_nxyz_data,
             'OG_CG_nxyz': CG_nxyz_data,
             'num_atoms': num_atoms_list, 
             'num_CGs':num_CGs_list,
             'CG_mapping': CG_mapping_list, 
             'bond_edge_list':  bond_edge_list,
             'ic': ic_list,
             'mask': mask_list,
             'mask_xyz_list': mask_xyz_list,
             'prot_idx': prot_idx_list
            }
    
    dataset = props.copy()
    dataset = CGDataset(props.copy())
    dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)

    batch_interaction_list = []
    batch_pi_pi_list = []
    batch_pi_ion_list = []
    batch_bb_NO_list = []

    name_list = np.array(list(table['name']))[nfirst:-nlast]
    element_list = np.array(list(table['element']))[nfirst:-nlast]
    res_list = np.array(list(table['resName']))[nfirst:-nlast]
    resSeq_list = np.array(list(table['newSeq']))[nfirst:-nlast]
    for i in tqdm(range(len(nxyz_data)), desc='building interaction list', file=sys.stdout):
        
        # HB, ion-ion interactions
        n = nxyz_data[i].size(0)
        dist = (nxyz_data[i][:,1:].expand(n, n, 3) - nxyz_data[i][:,1:].expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

        src, dst = torch.where((dist <=3.3) & (dist > 0.93))
        src_name, dst_name = name_list[src], name_list[dst]
        src_element, dst_element = element_list[src], element_list[dst]
        src_res, dst_res = res_list[src], res_list[dst]
        elements = [src_element[i]+dst_element[i] for i in range(len(src_element))]
        src_seq, dst_seq = resSeq_list[src], resSeq_list[dst]

        cond1 = (src_seq != dst_seq) & (src_seq != (dst_seq + 1)) & (dst_seq != (src_seq + 1))
        cond2 = ~np.isin(src_name, bb_list) | ~np.isin(dst_name, bb_list)
        cond3 = np.isin(elements, allow_list)
        all_cond = (cond1 & cond2 & cond3)

        interaction_list = torch.stack([src[all_cond], dst[all_cond]], axis=-1).long()
        interaction_list = interaction_list[interaction_list[:, 1] > interaction_list[:, 0]]
        batch_interaction_list.append(interaction_list)

        # pi-pi interactions
        src, dst = torch.where((dist <=8.0) & (dist > 1.5))
        src_res, dst_res = res_list[src], res_list[dst]
        src_name, dst_name = name_list[src], name_list[dst]
        src_seq, dst_seq = resSeq_list[src], resSeq_list[dst]

        cond1 = src_seq == dst_seq
        cond2 = np.isin(src_res, ['PHE', 'TYR', 'TRP']) & np.isin(src_name, ['CD1']) & np.isin(dst_name, ['CD2'])
        cond3 = np.isin(src_res, ['HIS']) & np.isin(src_name, ['CD1']) & np.isin(dst_name, ['ND1'])
        
        all_cond = (cond1 & (cond2 | cond3))
        ring_end1, ring_end2 = src[all_cond], dst[all_cond]
        ring_centers = (nxyz_data[i][:,1:][ring_end1] + nxyz_data[i][:,1:][ring_end2])/2
        n = len(ring_centers)
        ring_dist = (ring_centers.expand(n, n, 3) - ring_centers.expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

        src, dst = torch.where((ring_dist <= 5.5) & (ring_dist >= 2.0))
        pi_pi_list = torch.stack([ring_end1[src], ring_end2[src], ring_end1[dst], ring_end2[dst]], axis=-1).long()  
        pi_pi_list = pi_pi_list[pi_pi_list[:, 1] > pi_pi_list[:, 0]]
        pi_pi_list = pi_pi_list[pi_pi_list[:, 3] > pi_pi_list[:, 2]]
        pi_pi_list = pi_pi_list[pi_pi_list[:, 0] > pi_pi_list[:, 2]]
        batch_pi_pi_list.append(pi_pi_list)   

        # N-O distances
        src, dst = torch.where((dist <=4.0) & (dist > 1.5))
        src_name, dst_name = name_list[src], name_list[dst]
        src_seq, dst_seq = resSeq_list[src], resSeq_list[dst]
        
        cond1 = src_seq == (dst_seq + 1)
        cond2 = (src_name == 'N') & (dst_name == 'O')
        all_cond = (cond1 & cond2)

        bb_NO_list = torch.stack([src[all_cond], dst[all_cond]], axis=-1).long()
        batch_bb_NO_list.append(bb_NO_list)
    
    dataset.props['interaction_list'] = batch_interaction_list
    dataset.props['pi_pi_list'] = batch_pi_pi_list
    dataset.props['bb_NO_list'] = batch_bb_NO_list

    print("finished creating dataset")
    return dataset


def create_info_dict(dataset_label_list, PROTEINFILES):
    n_cg_list, traj_list, info_dict = [], [], {} 
    cnt = 0
    for idx, label in enumerate(tqdm(dataset_label_list)): 
        traj = shuffle_traj(load_protein_traj(label, PROTEINFILES=PROTEINFILES))
        (permute, atom_idx, atom_orders), n_cg = traj_to_info(traj)
        info_dict[cnt] = (permute, atom_idx, atom_orders)
        n_cg_list.append(n_cg)
        traj_list.append(traj)
        cnt += 1
    return n_cg_list, traj_list, info_dict

def traj_to_info(traj):
    table, _ = traj.top.to_dataframe()
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']
    reslist = list(set(list(table.newSeq)))
    reslist.sort()

    n_cg = len(reslist)
    atomic_nums, protein_index = get_atomNum(traj)

    atomn = [list(table.loc[table.newSeq==res].name) for res in reslist][1:-1]
    resn = list(table.loc[table.name=='CA'].resName)[1:-1]

    atom_idx = []
    permute = []
    permute_idx, atom_idx_idx = 0, 0
    
    for i in range(len(resn)):  
        p = [np.where(np.array(core_atoms[resn[i]])==atom)[0][0]+permute_idx for atom in atomn[i]]
        permute.append(p)  
        atom_idx.append(np.arange(atom_idx_idx, atom_idx_idx+len(atomn[i])))
        permute_idx += len(atomn[i])
        atom_idx_idx += 14

    atom_orders1 = [[] for _ in range(10)]
    atom_orders2 = [[] for _ in range(10)]
    atom_orders3 = [[] for _ in range(10)]
    for res_idx, res in enumerate(resn):
        atom_idx_list = atom_order_list[res]
        for i in range(10):
            if i <= len(atom_idx_list)-1:
                atom_orders1[i].append(atom_idx_list[i][0])
                atom_orders2[i].append(atom_idx_list[i][1])
                atom_orders3[i].append(atom_idx_list[i][2])
            else:
                atom_orders1[i].append(0)
                atom_orders2[i].append(1)
                atom_orders3[i].append(2)
    atom_orders1 = torch.LongTensor(np.array(atom_orders1))
    atom_orders2 = torch.LongTensor(np.array(atom_orders2))
    atom_orders3 = torch.LongTensor(np.array(atom_orders3))
    atom_orders = torch.stack([atom_orders1, atom_orders2, atom_orders3], axis=-1) # 10, n_res, 3
    
    permute = torch.LongTensor(np.concatenate(permute)).reshape(-1)
    atom_idx = torch.LongTensor(np.concatenate(atom_idx)).reshape(-1)
    info = (permute, atom_idx, atom_orders)
    return info, n_cg


def build_cg_dataset(mapping, cg_traj, aa_top, atom_cutoff, cg_cutoff, atomic_nums, order=1, prot_idx=None):
    CG_nxyz_data = []
    num_CGs_list = []
    CG_mapping_list = []
    
    table, _ = aa_top.to_dataframe()
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']

    endpoints = []
    for idx, chainID in enumerate(np.unique(np.array(table.chainID))):
        tb_chain = table.loc[table.chainID==chainID]
        first = tb_chain.newSeq.min()
        last = tb_chain.newSeq.max()
        endpoints.append(first)
        endpoints.append(last)
 
    print(f'traj has {table.chainID.max()+1} chains')
    nfirst = len(table.loc[table.newSeq==table.newSeq.min()])
    nlast = len(table.loc[table.newSeq==table.newSeq.max()])

    _top = aa_top.subset(np.arange(aa_top.n_atoms)[nfirst:-nlast])

    bondgraph = _top.to_bondgraph()
    indices = table.loc[table.name=='CA'].index

    edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
    edges = get_high_order_edge(edges, order, atomic_nums.shape[0])

    CG_res = list(aa_top.residues)
    CG_res = CG_res[:len(mapping.unique())]
    CG_res = torch.LongTensor([RES2IDX[THREE_LETTER_TO_ONE[res.name[:3]]] for res in CG_res]).reshape(-1,1)
    # Aggregate CG coorinates 
    for i in range(len(cg_traj)):
        CG_xyz = torch.Tensor(cg_traj[i].xyz[0])
        CG_nxyz = torch.cat((CG_res, CG_xyz), dim=-1)
        CG_nxyz_data.append(CG_nxyz)
        num_CGs_list.append(torch.LongTensor([len(CG_nxyz)-2]))
        CG_mapping_list.append(mapping[nfirst:-nlast]-1)

    # delete first and last residue
    trim_CG_nxyz_data = [nxyz[1:-1,:] for nxyz in CG_nxyz_data] 

    res_list = np.unique(np.array(table.newSeq))
    n_res = len(res_list)
    mask = torch.zeros(n_res-2,13)
    for i in tqdm(range(1,n_res-1), desc='generate mask', file=sys.stdout):
        if res_list[i] not in endpoints:
            num_atoms = len(table.loc[table.newSeq==res_list[i]])-1
            mask[i-1][:num_atoms] = torch.ones(num_atoms)    
    mask = mask.reshape(-1)

    interm_endpoints = set(endpoints)-set([table.newSeq.min(), table.newSeq.max()])
    mask_xyz_list = []
    for res in interm_endpoints:
        mask_xyz_list += list(table.loc[table.newSeq==res].index)
    mask_xyz = torch.LongTensor(np.array(mask_xyz_list) - nfirst)
    
    mask_list = [mask for _ in range(len(cg_traj))]
    mask_xyz_list = [mask_xyz for _ in range(len(cg_traj))]
    prot_idx_list = [torch.Tensor([prot_idx]) for _ in range(len(cg_traj))]

    num_atoms_list = [torch.LongTensor([aa_top.n_atoms-(nfirst+nlast)]) for _ in range(len(num_CGs_list))]
    props = {'CG_nxyz': trim_CG_nxyz_data,
             'OG_CG_nxyz': CG_nxyz_data,
             
             'num_atoms': num_atoms_list,
             'num_CGs':num_CGs_list,
             'CG_mapping': CG_mapping_list, 

             'mask': mask_list,
             'mask_xyz_list': mask_xyz_list,
             'prot_idx': prot_idx_list
            }
    
    dataset = props.copy()
    dataset = CGDataset_inf(props.copy())
    dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)

    print("finished creating dataset")
    return dataset