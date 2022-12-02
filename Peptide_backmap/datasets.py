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

import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean, scatter_add

from data import * 
from cgae import *
from utils_ic import * 
from utils import shuffle_traj

THREE_LETTER_TO_ONE = {
    "ARG": "R", 
    "HIS": "H", 
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
    "ACE": "ACE", #terminal
    "NME": "NME", #terminal
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
             'ACE': 20,
             'NME': 21}

atomic_num_dict = {'C':6, 'H':1, 'O':8, 'N':7, 'S':16, 'Se': 34}

PROTEINFILES = {'covid': {'traj_paths': "../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA-00*.dcd", 
                              'pdb_path': '../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA.pdb', 
                              'file_type': 'dcd'},

                'chignolin': {'traj_paths': "../data/filtered/e1*/*.xtc", 
                              'pdb_path': '../data/filtered/filtered.pdb', 
                              'file_type': 'xtc'}, 
                'chignolin_1': {'traj_paths': "../data/chig_partial_1.xtc", 
                              'pdb_path': '../data/filtered/filtered.pdb', 
                              'file_type': 'xtc'}, 
                'chignolin_2': {'traj_paths': "../data/chig_partial_2.xtc", 
                              'pdb_path': '../data/filtered/filtered.pdb', 
                              'file_type': 'xtc'},
                'chignolin_3': {'traj_paths': "../data/chig_partial_3.xtc", 
                              'pdb_path': '../data/filtered/filtered.pdb', 
                              'file_type': 'xtc'},
                'cc_chignolin': {'traj_paths': "../data/cc_chignolin.pdb", 
                              'pdb_path': '../data/cc_chignolin.pdb', 
                              'file_type': 'pdb'},
                'red_chignolin': {'traj_paths': "../data/red_chignolin.pdb", 
                              'pdb_path': '../data/red_chignolin.pdb', 
                              'file_type': 'pdb'},
                              
                'dipeptide': 
                            {'pdb_path': '../data/alanine-dipeptide-nowater.pdb', 
                            'traj_paths': '../data/alanine-dipeptide-*-250ns-nowater.xtc',
                            'file_type': 'xtc'
                             },
                'pentapeptide': 
                            {'pdb_path': '../data/pentapeptide-impl-solv.pdb',
                             'traj_paths': '../data/pentapeptide-*-500ns-impl-solv.xtc',
                             'file_type': 'xtc'
                            },
                'fs':
                            {'pdb_path': '../data/fs_peptide/onlyprot.pdb',
                             'traj_paths': '../data/fs_peptide/all_reduced.xtc',
                             'file_type': 'xtc'
                            },
                'coiled_coil':
                            {'pdb_path': '../data/pep_trajs/P5-coiled-coil/coiled_coil.pdb',
                             'traj_paths': '../data/pep_trajs/P5-coiled-coil/free*xtc',
                             'file_type': 'xtc'
                            },
                'nucleocapsid':
                            {'pdb_path': '../data/nucleocapsid/onlyprot.pdb',
                             'traj_paths': '../data/nucleocapsid/all_nucleocapsid.xtc',
                             'file_type': 'xtc'
                            },
                }

prefixs = ['PED']
PED_PDBs = [glob.glob(f'../data/processed/{prefix}*.pdb') for prefix in prefixs]
for idx, prefix_files in enumerate(PED_PDBs):
    for PDBfile in prefix_files:
        ID = PDBfile.split('/')[-1].split('.')[0][len(prefixs[idx]):]
        dct = {ID: {'pdb_path': PDBfile,
                'traj_paths': PDBfile,
                'file_type': 'pdb'
                        }
                        }
        PROTEINFILES.update(dct)
# print(PROTEINFILES)

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

    #z_data, xyz_data, num_atoms, num_cgs, bond_data, hyperedge_data, angle_data, dihedral_data = shuffle( z_data, xyz_data, num_atoms, num_cgs, bond_data, hyperedge_data, angle_data, dihedral_data)


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


def get_cg_and_xyz(traj, params, cg_method='backbone', n_cgs=None, mapshuffle=0.0, mapping=None):

    atomic_nums, protein_index = get_atomNum(traj)
    n_atoms = len(atomic_nums)
    skip = 200
    # get alpha carbon only 

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
        reslist = list(set(list(table.resSeq)))
        reslist.sort()

        j = 0
        for i in range(len(table)):
            if table.iloc[i].resSeq == reslist[j]:
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

    elif cg_method =='martini':
        mapping = None
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
    nbr_list = torch.nonzero(dist < cutoff).numpy()
    
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


def build_multiprotein_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=1, cg_traj=None):
    
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

    CG_res = list(top.residues)#[:len(mapping.unique())]
    CG_res = CG_res[:len(mapping.unique())]
    # table, bonds = top.to_dataframe()
    CG_res = torch.LongTensor([RES2IDX[THREE_LETTER_TO_ONE[res.name[:3]]] for res in CG_res]).reshape(-1,1)

    # Aggregate CG coorinates 
    for i, nxyz in enumerate(nxyz_data):
        xyz = torch.Tensor(nxyz[:, 1:]) 
        if cg_traj is not None:
            CG_xyz = torch.Tensor( cg_traj[i] )
        else:
            CG_xyz = scatter_mean(xyz, mapping, dim=0)
        
        CG_nxyz = torch.cat((CG_res, CG_xyz), dim=-1)
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
    # dataset = CGDataset(props.copy())
    # dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)
    
    return dataset


def build_ic_multiprotein_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=1, cg_traj=None, prot_idx=None):
    
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []
    bond_edge_list = []
    
    table, _ = top.to_dataframe()
    nfirst = len(table.loc[table.resSeq==table.resSeq.min()])
    nlast = len(table.loc[table.resSeq==table.resSeq.max()])

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

    res_list = list(set(list(table.resSeq)))
    res_list.sort()
    n_res = len(res_list)
    mask = torch.zeros(n_res-2,13)

    for i in tqdm(range(1,n_res-1), desc='generate mask', file=sys.stdout):
        num_atoms = len(table.loc[table.resSeq==res_list[i]])-1
        mask[i-1][:num_atoms] = torch.ones(num_atoms)
    mask = mask.reshape(-1)
    mask_list = [mask for _ in range(len(nxyz_data))]
    prot_idx_list = [torch.Tensor([prot_idx]) for _ in range(len(nxyz_data))]
    
    st = time.time()
    print(f"generate ic start")
    bb_ic = torch.Tensor(get_backbone_ic(md.Trajectory(traj, top)))
    sc_ic = torch.Tensor(get_sidechain_ic(md.Trajectory(traj, top)))    
    ic_list = torch.cat((bb_ic, sc_ic), axis=2)
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
             'prot_idx': prot_idx_list
            }
    
    dataset = props.copy()
    dataset = CGDataset(props.copy())
    dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)
    
    filter_list = ['CA', 'C', 'N', 'O', 'H']
    allow_list = ['NO', 'ON', 'SN', 'NS', 'SS']

    EPS = 1e-6
    batch_interaction_list = []
    
    name_list = np.array(list(table['name']))
    element_list = np.array(list(table['element']))
    for i in tqdm(range(len(nxyz_data)), desc='building interaction list', file=sys.stdout):
        interaction_list = []
        n = nxyz_data[i].size(0)
        dist = (nxyz_data[i][:,1:].expand(n, n, 3) - nxyz_data[i][:,1:].expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

        src, dst = torch.where((dist <=3.3) & (dist > 0.93))
        src_name, dst_name = name_list[src], name_list[dst]
        src_element, dst_element = element_list[src], element_list[dst]
        elements = [src_element[i]+dst_element[i] for i in range(len(src_element))]

        cond1 = CG_mapping_list[i][src] != CG_mapping_list[i][dst]
        cond2 = ~np.isin(src_name, filter_list)
        cond3 = ~np.isin(dst_name, filter_list)
        cond4 = np.isin(elements, allow_list)
        all_cond = (cond1 & (cond2 | cond3) & cond4)

        interaction_list = torch.stack([src[all_cond], dst[all_cond]], axis=-1).long()

        batch_interaction_list.append(interaction_list)
    dataset.props['interaction_list'] = batch_interaction_list
    print("finished creating dataset")
    return dataset


def create_info_dict(dataset_label_list):
    n_cg_list, traj_list, info_dict = [], [], {} 
    cnt = 0
    for idx, label in enumerate(tqdm(dataset_label_list)): 
        try:
            traj = shuffle_traj(load_protein_traj(label))
            table, _ = traj.top.to_dataframe()
            reslist = list(set(list(table.resSeq)))
            reslist.sort()
            
            n_cg = len(reslist)
            atomic_nums, protein_index = get_atomNum(traj)

            atomn = [list(table.loc[table.resSeq==res].name) for res in reslist][1:-1]
            resn = list(table.loc[table.name=='CA'].resName)[1:-1]
        
            atom_idx = []
            permute = []
            permute_idx, atom_idx_idx = 0, 0
            for i in range(len(resn)):  
                atom = atomn[i][0]
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
        except:
            print(f'failed to load {label}')
    return n_cg_list, traj_list, info_dict