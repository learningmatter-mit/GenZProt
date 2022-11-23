import torch
import mdtraj as md
import numpy as np
import copy

core_sc = \
{'ALA': ['CB'],
 'ARG': ['CB', 'CG', 'CD', 'NE', 'HE', 'CZ', 'NH1', 'NH2'],
 'ASP': ['CB', 'CG', 'OD1', 'OD2'],
 'ASN': ['CB', 'CG', 'OD1', 'ND2'],
 'CYS': ['CB', 'SG'],
 'GLU': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
 'GLN': ['CB', 'CG', 'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
 'GLY': [],
 'HIS': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
 'ILE': ['CB', 'CG2', 'CG1', 'CD1'],
 'LEU': ['CB', 'CG', 'CD1', 'CD2'],
 'LYS': ['CB', 'CG', 'CD', 'CE', 'NZ'],
 'MET': ['CB', 'CG', 'SD', 'CE'],
 'PHE': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
 'PRO': ['CB', 'CG', 'CD'],
 'SER': ['CB', 'OG', 'HG'],
 'THR': ['CB', 'OG1', 'HG1', 'CG2'],
 'TRP': ['CB', 'CG', 'CD2', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3'],
 'TYR': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2', 'OH', 'HH', ],
 'VAL': ['CB', 'CG1', 'CG2']
}

copy_core = copy.deepcopy(core_sc)
for key in copy_core.keys():
    copy_core[key] = np.array(['O', 'N', 'C', 'CA'] + copy_core[key])

# TRP ['CB', 'CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2']

IDX2THR = {0: 'ASN',
 1: 'HIS',
 2: 'ALA',
 3: 'GLY',
 4: 'ARG',
 5: 'MET',
 6: 'SER',
 7: 'ILE',
 8: 'GLU',
 9: 'LEU',
 10: 'TYR',
 11: 'ASP',
 12: 'VAL',
 13: 'TRP',
 14: 'GLN',
 15: 'LYS',
 16: 'PRO',
 17: 'PHE',
 18: 'CYS',
 19: 'THR'}

def get_ic(names, traj):
    A4, A3, A2, A1 = names 
    A1 = traj.top.select(f'name {A1}')
    A2 = traj.top.select(f'name {A2}')
    A3 = traj.top.select(f'name {A3}')
    A4 = traj.top.select(f'name {A4}')
    distance = md.compute_distances(traj, np.stack((A1, A2), axis=-1))
    angle = md.compute_angles(traj, np.stack((A1, A2, A3), axis=-1))
    torsion = md.compute_dihedrals(traj, np.stack((A1, A2, A3, A4), axis=-1))
    return distance, angle, torsion

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return (vector.T /np.linalg.norm(vector, axis=-1)).T

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2).T
    return np.diagonal(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def dihedral(p):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = (b1.T / np.linalg.norm(b1, axis=-1)).T

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - (b1.T * np.diagonal(np.dot(b0, b1.T))).T
    w = b2 - (b1.T * np.diagonal(np.dot(b2, b1.T))).T

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.diagonal(np.dot(v, w.T))
    y = np.diagonal(np.dot(np.cross(b1, v), w.T))
    return np.arctan2(y, x)

def get_sidechain_ic(traj):
    table, _ = traj.top.to_dataframe()   
    res_list = list(set(list(table.resSeq)))
    res_list.sort()
    n_res = len(res_list)
    tot_ic = []
    
    for i in range(1,n_res-1):
        rest = table.loc[table.resSeq==res_list[i]]
        if len(rest) > 0: 
            resn = rest.resName.values[0]
            atom_list = ['N', 'C', 'CA'] + core_sc[resn] 
            atom_idx = [rest.loc[rest.name==atom].index[0] for atom in atom_list]
            restraj = traj.atom_slice(atom_idx)
            tb, _ = restraj.top.to_dataframe()
            resxyz = restraj.xyz
            
            ic_i = np.zeros((10, resxyz.shape[0], 3))
            for j in range(len(core_sc[resn])):
                A1, A2, A3, A4 = resxyz[:, j+3], resxyz[:, j+2], resxyz[:, j+1], resxyz[:, j]      
                dist = np.sqrt(((A1 - A2)**2).sum(-1))
                ang = angle_between(A1 - A2, A3 - A2)
                tor = dihedral([A1, A2, A3, A4])
                tor = ((tor + np.pi) % (2 * np.pi)) - np.pi
                ic_i[j] = np.stack([dist, ang, tor],axis=-1).reshape(resxyz.shape[0], 3)
            tot_ic.append(ic_i)

    tot_ic = np.stack(tot_ic, axis=0).transpose(2, 0, 1, 3)
    return tot_ic


def get_backbone_ic(traj):
    CA = traj.top.select('name CA')
    C = traj.top.select('name C')
    N = traj.top.select('name N')
    O = traj.top.select('name O')
    
    C_torsion = md.compute_dihedrals(traj, np.stack((C[1:-1], CA[1:-1], CA[2:], CA[:-2]), axis=-1))[..., None]
    N_torsion = md.compute_dihedrals(traj, np.stack((N[1:-1], CA[1:-1], CA[:-2], CA[2:]), axis=-1))[..., None]
    O_torsion = md.compute_dihedrals(traj, np.stack((O[1:-1], C[1:-1], CA[1:-1], N[1:-1]), axis=-1))[..., None]

    C_angle = md.compute_angles(traj, np.stack((C[1:-1], CA[1:-1], CA[2:]), axis=-1))[..., None]
    N_angle = md.compute_angles(traj, np.stack((N[1:-1], CA[1:-1], CA[:-2]), axis=-1))[..., None]
    O_angle = md.compute_angles(traj, np.stack((O[1:-1], C[1:-1], CA[1:-1]), axis=-1))[..., None]

    C_dist = md.compute_distances(traj, np.stack((C[1:-1], CA[1:-1]), axis=-1))[..., None]
    N_dist = md.compute_distances(traj, np.stack((N[1:-1], CA[1:-1]), axis=-1))[..., None]
    O_dist = md.compute_distances(traj, np.stack((O[1:-1], C[1:-1]), axis=-1))[..., None]

    C_ic = np.stack((C_dist, C_angle, C_torsion), axis=-1) # 75, 126, 1, 3
    N_ic = np.stack((N_dist, N_angle, N_torsion), axis=-1) # 75, 126, 1, 3
    O_ic = np.stack((O_dist, O_angle, O_torsion), axis=-1) # 75, 126, 1, 3

    all_ic = np.stack((N_ic, C_ic, O_ic), axis=2).squeeze() # 75, 126, 3, 3
    return all_ic  


def rotation_matrix(axis, angle):
    """
    Euler-Rodrigues formula for rotation matrix
    """
    # Normalize the axis
    axis = axis / torch.sqrt((axis * axis).sum(-1)).unsqueeze(-1)
    a = torch.cos(angle / 2).squeeze(-1)
    res = -axis * torch.sin(angle / 2)
    b, c, d = res[:,:,0], res[:,:,1], res[:,:,2]

    rx = torch.stack((a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)),axis=-1)
    ry = torch.stack((2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)),axis=-1)
    rz = torch.stack((2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c),axis=-1)
    return torch.stack((rx, ry, rz),axis=-2)


def add_atom_to_xyz(newatom_ic, ref_atoms):
    atom1, atom2, atom3 = ref_atoms[0], ref_atoms[1], ref_atoms[2]
    distance, angle, dihedral = newatom_ic[:,:,0].unsqueeze(-1), newatom_ic[:,:,1].unsqueeze(-1), newatom_ic[:,:,2].unsqueeze(-1)
    
    # Vector pointing from atom1 to atom2
    a = atom2 - atom1
    # Vector pointing from atom3 to atom2
    b = atom2 - atom3

    # Vector of length distance pointing from atom1 to atom2
    d = torch.absolute(distance) * a / torch.sqrt((a * a).sum(-1)).unsqueeze(-1)

    # Vector normal to plane defined by atom1, atom2, atom3
    normal = torch.cross(a, b)
    # Rotate d by the angle around the normal to the plane defined by atom1, atom2, atom3
    d = d.unsqueeze(-1)
    d = torch.matmul(rotation_matrix(normal, angle), d)
    # Rotate d around a by the dihedral
    d = torch.matmul(rotation_matrix(a, dihedral), d).squeeze(-1)
    # Add d to the position of atom1 to get the new coord 
    p = atom1 + d

    return p


def ic_to_xyz(CG_nxyz, ic_recon, info):
    CG_xyz = CG_nxyz[:, :, 1:]

    atomn, resn = info
    
    N = add_atom_to_xyz(ic_recon[:,:,0], [CG_xyz[:,1:-1], CG_xyz[:,:-2], CG_xyz[:,2:]])
    C = add_atom_to_xyz(ic_recon[:,:,1], [CG_xyz[:,1:-1], CG_xyz[:,2:], CG_xyz[:,:-2]])
    O = add_atom_to_xyz(ic_recon[:,:,2], [C, CG_xyz[:,1:-1], N])
    sc_xyz = torch.stack((N, C, CG_xyz[:,1:-1]), axis=2)
    
    for i in range(10):
        new_atom = add_atom_to_xyz(ic_recon[:,:,3+i], [sc_xyz[:,:,-1], sc_xyz[:,:,-2], sc_xyz[:,:,-3]]).unsqueeze(-2) #.reshape(1, 3)
        sc_xyz = torch.cat((sc_xyz, new_atom),axis=2)
        
    sc_xyz = torch.cat((O.unsqueeze(2), sc_xyz),axis=2)
    
    atom_idx = []
    permute = []
    permute_idx, atom_idx_idx = 0, 0
    for i in range(len(resn)):  
        p = torch.LongTensor([np.where(copy_core[resn[i]]==atom)[0][0]+permute_idx for atom in atomn[i]])
        permute.append(p)  
        atom_idx.append(torch.LongTensor(np.arange(atom_idx_idx, atom_idx_idx+len(atomn[i]))))
        permute_idx += len(atomn[i])
        atom_idx_idx += 14
        
    atom_idx = torch.cat(atom_idx).reshape(-1)
    permute = torch.cat(permute).reshape(-1)
    sc_xyz = sc_xyz.reshape(sc_xyz.shape[0],-1,3)[:,atom_idx,:]
    sc_xyz = sc_xyz[:,permute,:]
    return sc_xyz

