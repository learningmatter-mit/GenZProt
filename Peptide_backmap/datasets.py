import sys
sys.path.append("../scripts/")
import glob
import itertools
import random
from tqdm import tqdm
import time

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
from cgae import *
from utils_ic import * 
from utils import shuffle_traj
from sampling import *

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

PROTEINFILES = {'covid': {'traj_paths': "../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA-00*.dcd", 
                              'pdb_path': '../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA.pdb', 
                              'file_type': 'dcd'},

                'chignolin': {'traj_paths': "../data/filtered/e1*/*.xtc", 
                              'pdb_path': '../data/filtered/filtered.pdb', 
                              'file_type': 'xtc'}
                }

# PROTEINFILES = {'val_chignolin': {'traj_paths': '/home/gridsan/sjyang/backmap_exp/data/use_files/val_chignolin.pdb', 
#                                   'pdb_path': '/home/gridsan/sjyang/backmap_exp/data/use_files/val_chignolin.pdb', 
#                                   'file_type': 'pdb'},
#                 'train_chignolin': {'traj_paths': '/home/soojungy/backmap_exp/data/use_files/train_chignolin.pdb', 
#                                   'pdb_path': '/home/soojungy/backmap_exp/data/use_files/train_chignolin.pdb', 
#                                   'file_type': 'pdb'}
#                 }

PROTEINFILES = {'val_chignolin': {'traj_paths': '/home/soojungy/backmap_exp/data/use_files/val_chignolin.pdb', 
                                  'pdb_path': '/home/soojungy/backmap_exp/data/use_files/val_chignolin.pdb', 
                                  'file_type': 'pdb'},
                'train_chignolin': {'traj_paths': '/home/soojungy/backmap_exp/data/use_files/train_chignolin.pdb', 
                                  'pdb_path': '/home/soojungy/backmap_exp/data/use_files/train_chignolin.pdb', 
                                  'file_type': 'pdb'}
                }

prefixs = ['PED']
# PED_PDBs = [glob.glob(f'../data/processed/{prefix}*.pdb') for prefix in prefixs]
PED_PDBs = [glob.glob(f'/home/gridsan/sjyang/backmap_exp/data/use_files/{prefix}*.pdb') for prefix in prefixs] + \
           [glob.glob(f'/home/soojungy/backmap_exp/data/use_files/{prefix}*.pdb') for prefix in prefixs]

for idx, prefix_files in enumerate(PED_PDBs):
    for PDBfile in prefix_files:
        ID = PDBfile.split('/')[-1].split('.')[0][len(prefixs[0]):]
        dct = {ID: {'pdb_path': PDBfile,
                'traj_paths': PDBfile,
                'file_type': 'pdb'
                        }
                        }
        PROTEINFILES.update(dct)

def get_backbone(top):
    backbone_index = []
    for atom in top.atoms:
        if atom.is_backbone:
            backbone_index.append(atom.index)
    return np.array(backbone_index)

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

def backbone_partition(traj, n_cgs, skip=100):
    atomic_nums, protein_index = get_atomNum(traj)
    #indices = traj.top.select_atom_indices('minimal')
    indices = get_backbone(traj.top)

    if indices.shape[0] < n_cgs:
        raise ValueError("N_cg = {} is larger than N_backbone = {}".format(n_cgs, indices.shape[0]) )

    if len(indices) == n_cgs:
        partition = list(range(1, n_cgs))
    else:
        partition = random.sample(range(indices.shape[0]), n_cgs - 1 )
        partition = np.array(partition)
        partition = np.sort(partition)
        segment_sizes = (partition[1:] - partition[:-1]).tolist() + [indices.shape[0] - partition[-1]] + [partition[0]]

    mapping = np.zeros(indices.shape[0])
    mapping[partition] = 1
    mapping = np.cumsum(mapping)

    backbone_cgxyz = scatter_mean(torch.Tensor(traj.xyz[:, indices]), 
                          index=torch.LongTensor(mapping), dim=1).numpy()

    mappings = []
    for i in protein_index:
        dist = traj.xyz[::skip, [i], ] - backbone_cgxyz[::skip]
        map_index = np.argmin( np.sqrt( np.sum(dist ** 2, -1)).mean(0) )
        mappings.append(map_index)

    cg_coord = None
    mapping = np.array(mappings)

    return mapping 


def get_diffpool_data(N_cg, trajs, n_data, edgeorder=1, shift=False, pdb=None, rotate=False):
    props = {}

    num_cgs = []
    num_atoms = []

    z_data = []
    xyz_data = []
    bond_data = []
    angle_data = []
    dihedral_data = []
    hyperedge_data = []

    # todo: not quite generalizable to different proteins
    if pdb is not None:
        mol = Molecule(pdb, guess=['bonds', 'angles', 'dihedrals'] )  
        dihedrals = torch.LongTensor(mol.dihedrals.astype(int))
        angles = torch.LongTensor(mol.angles.astype(int))
    else:
        dihedrals = None
        angles = None

    for traj in trajs:
        atomic_nums, protein_index = get_atomNum(traj)
        n_atoms = len(atomic_nums)
        frames = traj.xyz[:, protein_index, :] * 10.0 # from nm to Angstrom

        bondgraph = traj.top.to_bondgraph()
        bond_edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
        hyper_edges = get_high_order_edge(bond_edges, edgeorder, n_atoms)

        for xyz in frames[:n_data]: 
            if shift:
                xyz = xyz - np.random.randn(1, 3)
            if rotate:
                xyz = random_rotation(xyz)
            z_data.append(torch.Tensor(atomic_nums))
            coord = torch.Tensor(xyz)

            xyz_data.append(coord)
            bond_data.append(bond_edges)
            hyperedge_data.append(hyper_edges)

            angle_data.append(angles)
            dihedral_data.append(dihedrals)

            num_cgs.append(torch.LongTensor([N_cg]))
            num_atoms.append(torch.LongTensor([n_atoms]))

    props = {'z': z_data[:n_data],
         'xyz': xyz_data[:n_data],
         'num_atoms': num_atoms[:n_data], 
         'num_CGs':num_cgs[:n_data],
         'bond': bond_data[:n_data],
         'hyperedge': hyperedge_data[:n_data],
        }

    return props

def load_protein_traj(label, ntraj=200): 
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


def learn_map(traj, reg_weight, n_cgs, n_atoms ,
              n_data=1000, n_epochs=1500, 
              lr=4e-3, batch_size=32, device=0):

    props = get_diffpool_data(n_cgs, [traj], n_data=n_data, edgeorder=1)
    dataset = DiffPoolDataset(props)
    dataset.generate_neighbor_list(8.0)
    train_index, test_index = train_test_split(list(range(len(traj)))[:n_data], test_size=0.1)
    trainset = get_subset_by_indices(train_index,dataset)
    testset = get_subset_by_indices(test_index,dataset)

    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True, pin_memory=True)
    
    ae = cgae(n_atoms, n_cgs).to(device)
    optimizer = torch.optim.Adam(list(ae.parameters()), lr=lr)
    
    
    tau = 1.0

    for epoch in range(n_epochs):
        all_loss = []
        all_reg = []
        all_recon = [] 

        trainloader = trainloader

        for i, batch in enumerate(trainloader):

            batch = batch_to(batch, device)
            xyz = batch['xyz']

            shift = xyz.mean(1)
            xyz = xyz - shift.unsqueeze(1)

            xyz, xyz_recon, M, cg_xyz = ae(xyz, tau)
            xyz_recon = torch.einsum('bnj,ni->bij', cg_xyz, ae.decode)
            X_lift = torch.einsum('bij,ni->bnj', cg_xyz, M)

            loss_reg = (xyz - X_lift).pow(2).sum(-1).mean()
            loss_recon = (xyz - xyz_recon).pow(2).mean() 
            loss = loss_recon + reg_weight * loss_reg

            all_reg.append(loss_reg.item())
            all_recon.append(loss_recon.item())
            all_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

        if tau >= 0.025:
            tau -= 0.001

        if epoch % 50 == 0:
            print(epoch, tau, np.array(all_recon).mean(), np.array(all_reg).mean())

    return ae.assign_map.argmax(-1).detach().cpu()


def get_calpha(table):
    bead_mappings = []

    reslist = list(set(list(table.resSeq)))
    reslist.sort()

    j = 0
    for i in range(len(table)):
        if table.iloc[i].resSeq == reslist[j]:
            bead_mappings.append(j)
        else:
            j += 1
            bead_mappings.append(j)

    # n_cgs = len(reslist)
    bead_mapping = np.array(bead_mappings)
    return bead_mapping


def get_hier_cg_and_xyz(traj, params, cg_method='backbone', n_cgs=None, mapshuffle=0.0, bead_mapping=None, hyperbead_mapping=None):

    atomic_nums, protein_index = get_atomNum(traj)
    n_atoms = len(atomic_nums)

    frames = traj.xyz[:, protein_index, :] * 10.0

    table, _ = traj.top.to_dataframe()
    indices = table.loc[table.name=='CA'].index
    traj = traj.atom_slice(indices)

    if hyperbead_mapping != None and bead_mapping != None:
        return hyperbead_mapping, bead_mapping, frames, None

    # hyperbead: backbone partition only
    # bead: alpha only
    cg_method = 'backbone'
    hyperbead_mapping = backbone_partition(traj, n_cgs, skip=1)
    cg_coord = None
    bead_mapping = get_calpha(table)

    # print coarse graining summary 
    print("CG method: {}".format(cg_method))
    print("Number of CG sites: {}".format(hyperbead_mapping.max() + 1))
    print("Number of residues: {}".format(bead_mapping.max() + 1))

    #assert len(list(set(mapping.tolist()))) == n_cgs

    hyperbead_mapping = torch.LongTensor( hyperbead_mapping)
    bead_mapping = torch.LongTensor( bead_mapping)
    frames = shuffle(frames)
    
    return hyperbead_mapping, bead_mapping, frames, cg_coord


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

        # indices = traj.top.select_atom_indices(cg_method)
        # for i in protein_index:
        #     dist = traj.xyz[::skip, [i], ] - traj.xyz[::skip, indices, :]
        #     map_index = np.argmin( np.sqrt( np.sum(dist ** 2, -1)).mean(0) )
        #     mappings.append(map_index)

        table, _ = traj.top.to_dataframe()
        table['newSeq'] = table['resSeq'] + 5000*table['chainID']
        # reslist = list(set(list(table.resSeq)))
        reslist = list(set(list(table.newSeq)))
        reslist.sort()

        j = 0
        for i in range(len(table)):
            # if table.iloc[i].resSeq == reslist[j]:
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

    elif cg_method == 'cgae':
        
        if mapping == None:
            print("learning CG mapping")
            mapping = learn_map(traj, reg_weight=params['cgae_reg_weight'], n_cgs=n_cgs, n_atoms=n_atoms, batch_size=32)
            print(mapping)
        else:
            mapping = mapping 

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

    #assert len(list(set(mapping.tolist()))) == n_cgs

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

# def get_mapping(label, cutoff, n_atoms, n_cgs, skip=200):

#     peptide = get_peptide_top(label)

#     files = mdshare.fetch(DATALABELS[label]['xtc'], working_directory='data')

#     atomic_nums, traj = get_traj(peptide, files, n_frames=20000)
#     peptide_element = [atom.element.symbol for atom in peptide.top.atoms]

#     if len(traj) < skip:
#         skip = len(traj)

#     mappings = compute_mapping(atomic_nums, traj,  cutoff,  n_atoms, n_cgs, skip)

#     return mappings.long()

def get_random_mapping(n_cg, n_atoms):

    mapping = torch.LongTensor(n_atoms).random_(0, n_cg)
    i = 1
    while len(mapping.unique()) != n_cg and i <= 10000000:
        i += 1
        mapping = torch.LongTensor(n_atoms).random_(0, n_cg)

    return mapping

# def get_peptide_top(label):

#     pdb = mdshare.fetch(DATALABELS[label]['pdb'], working_directory='data')
#     peptide = md.load("data/{}".format(DATALABELS[label]['pdb']))

#     return peptide

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

# need a function to get mapping, and CG coordinates simultanesouly. We can have alpha carbon as the CG site


def get_high_order_edge(edges, order, natoms):

    # get adj 
    adj = torch.zeros(natoms, natoms)
    adj[edges[:,0], edges[:,1]] = 1
    adj[edges[:,1], edges[:,0]] = 1

    # get higher edges 
    edges = torch.triu(get_higher_order_adj_matrix(adj, order=order)).nonzero()

    return edges 

def build_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=1, cg_traj=None):
    
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []
    bond_edge_list = []
    bondgraph = top.to_bondgraph()

    edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
    edges = get_high_order_edge(edges, order, atomic_nums.shape[0])

    for xyz in traj:
        xyz = random_rotation(xyz)
        nxyz = torch.cat((torch.Tensor(atomic_nums[..., None]), torch.Tensor(xyz) ), dim=-1)
        nxyz_data.append(nxyz)
        num_atoms_list.append(torch.LongTensor( [len(nxyz)]))
        bond_edge_list.append(edges)

    # Aggregate CG coorinates 
    for i, nxyz in enumerate(nxyz_data):
        xyz = torch.Tensor(nxyz[:, 1:]) 
        if cg_traj is not None:
            CG_xyz = torch.Tensor( cg_traj[i] )
        else:
            CG_xyz = scatter_mean(xyz, mapping, dim=0)
        CG_nxyz = torch.cat((torch.LongTensor(list(range(len(CG_xyz))))[..., None], CG_xyz), dim=-1)
        CG_nxyz_data.append(CG_nxyz)
        num_CGs_list.append(torch.LongTensor( [len(CG_nxyz)]) )
        CG_mapping_list.append(mapping)

    props = {'nxyz': nxyz_data,
             'CG_nxyz': CG_nxyz_data,
             'num_atoms': num_atoms_list, 
             'num_CGs':num_CGs_list,
             'CG_mapping': CG_mapping_list, 
             'bond_edge_list':  bond_edge_list
            }

    dataset = props.copy()
    
    return dataset


def build_hier_dataset(bead_mapping, hyperbead_mapping, traj, atom_cutoff, cg_cutoff, hyper_cg_cutoff, atomic_nums, top, order=1, cg_traj=None, prot_idx=None):
    
    dataset = build_ic_peptide_dataset(bead_mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=order, cg_traj=cg_traj, prot_idx=prot_idx)
    
    traj = md.Trajectory(traj, top)
    table, _ = traj.top.to_dataframe()
    indices = table.loc[table.name=='CA'].index
    traj = traj.atom_slice(indices)
    ca_top = traj.top
    for i in range(len(indices)-1):
        ca_top.add_bond(ca_top.residue(i), ca_top.residue(i+1), order=1)

    hb_bond_edge_list = []
    hb_bondgraph = ca_top.to_bondgraph()
    hb_edges = torch.LongTensor( [[e[0].index, e[1].index] for e in hb_bondgraph.edges] )# list of edge list 

    num_res_list, res_edge_list = [], []
    for res_nxyz in dataset.props['OG_CG_nxyz']:
        num_res_list.append(torch.LongTensor( [len(res_nxyz)]) )
        res_edge_list.append(hb_edges)

    # Aggregate CG coorinates 
    hb_nxyz_data, num_hbs_list, hb_mapping_list = [], [], []
    first_occurences_list = []
    for i, res_nxyz in enumerate(dataset.props['OG_CG_nxyz']):
        res_xyz = torch.Tensor(res_nxyz[:, 1:]) 

        hb_xyz = scatter_mean(res_xyz, hyperbead_mapping, dim=0)
        hb_nxyz = torch.cat((torch.LongTensor(list(range(len(hb_xyz))))[..., None], hb_xyz), dim=-1)
        hb_nxyz_data.append(hb_nxyz)
        num_hbs_list.append(torch.LongTensor( [len(hb_nxyz)]) )
        hb_mapping_list.append(hyperbead_mapping)

    dataset.props['num_hyper_atoms'] = num_res_list
    dataset.props['hyper_CG_nxyz'] = hb_nxyz_data
    dataset.props['num_hyper_CGs'] = num_hbs_list
    dataset.props['hyper_CG_mapping'] = hb_mapping_list
    dataset.props['hyper_bond_edge_list'] = res_edge_list

    dataset.generate_neighbor_list(atom_cutoff=cg_cutoff, cg_cutoff=hyper_cg_cutoff, prefix='hyper_')

    return dataset

bb_list = ['CA', 'C', 'N', 'O', 'H']
allow_list = ['NO', 'ON', 'SN', 'NS', 'SO', 'OS', 'SS', 'NN', 'OO']
ring_list = ['PHE', 'TYR', 'TRP', 'HIS']
ring_name_list = ['CG', 'CZ', 'CE1', 'CE2']
ion_list = ['ASP', 'GLU', 'ARG', 'LYS']
ion_name_list = ['OD1', 'OD2', 'NH1', 'NH2', 'NZ']
import math

def real_number_batch_to_one_hot_vector_bins(real_numbers, bins):
    """Converts a batch of real numbers to a batch of one hot vectors for the bins the real numbers fall in."""
    _, indexes = (real_numbers.view(-1, 1) - bins.view(1, -1)).abs().min(dim=1)
    return indexes_to_one_hot(indexes, n_dims=bins.shape[0])
    # return indexes

def indexes_to_one_hot(indexes, n_dims=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size()[0], n_dims)
    one_hots = one_hots.scatter_(1, indexes, 1)
    # one_hots = one_hots.view(*indexes.shape, -1)
    return one_hots

from scipy.ndimage import gaussian_filter


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


def create_info_dict(dataset_label_list):
    n_cg_list, traj_list, info_dict = [], [], {} 
    cnt = 0
    for idx, label in enumerate(tqdm(dataset_label_list)): 
        traj = shuffle_traj(load_protein_traj(label))
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

        info_dict[cnt] = (permute, atom_idx, atom_orders)
        n_cg_list.append(n_cg)
        traj_list.append(traj)
        cnt += 1

    return n_cg_list, traj_list, info_dict


def create_ring_info_dict(dataset_label_list):
    n_cg_list, traj_list, info_dict = [], [], {} 
    cnt = 0
    for idx, label in enumerate(tqdm(dataset_label_list)): 
        traj = shuffle_traj(load_protein_traj(label))
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
        
        ring_cnt_idx = 0
        ring_ic_list = []
        ring_ic_idx = []
        for i in range(len(resn)):  
            p = [np.where(np.array(core_atoms[resn[i]])==atom)[0][0]+permute_idx for atom in atomn[i]]
            permute.append(p)  
            atom_idx.append(np.arange(atom_idx_idx, atom_idx_idx+len(atomn[i])))

            if resn[i] in ['TYR', 'TRP', 'HIS', 'PHE']:
                for idx in ring_idx[resn[i]]:
                    ring_ic_list.append(ring_ic[resn[i]][idx])
                    ring_ic_idx.append(ring_cnt_idx+idx)
            
            permute_idx += len(atomn[i])
            atom_idx_idx += 14
            ring_cnt_idx += 13        
        ring_ic_idx = torch.LongTensor(ring_ic_idx)
        
        if len(ring_ic_list) > 0:
            ring_ic_list = torch.stack(ring_ic_list)
        else:
            ring_ic_list = None

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

        info_dict[cnt] = (permute, atom_idx, atom_orders, ring_ic_list, ring_ic_idx)
        n_cg_list.append(n_cg)
        traj_list.append(traj)
        cnt += 1

    return n_cg_list, traj_list, info_dict