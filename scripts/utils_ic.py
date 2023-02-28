import copy
import torch
import numpy as np
import mdtraj as md


core_atoms = \
{
'ALA': ['O','N','C','CA','CB'],
'ARG': ['O','N','C','CA','CB','CG','CD','NE','CZ','NH1','NH2'],
'ASP': ['O','N','C','CA','CB','CG','OD1','OD2'],
'ASN': ['O','N','C','CA','CB','CG','OD1','ND2'],
'CYS': ['O','N','C','CA','CB','SG'],
'GLU': ['O','N','C','CA','CB','CG','CD','OE1','OE2'],
'GLN': ['O','N','C','CA','CB','CG','CD','OE1','NE2'],
'GLY': ['O','N','C','CA'],
'HIS': ['O','N','C','CA','CB','CG','CD2','ND1','NE2','CE1'],
'ILE': ['O','N','C','CA','CB','CG2','CG1','CD1'],
'LEU': ['O','N','C','CA','CB','CG','CD1','CD2'],
'LYS': ['O','N','C','CA','CB','CG','CD','CE','NZ'],
'MET': ['O','N','C','CA','CB','CG','SD','CE'],
'PHE': ['O','N','C','CA','CB','CG','CD1','CE1','CZ','CD2','CE2'],
'PRO': ['O','N','C','CA','CB','CG','CD'],
'SER': ['O','N','C','CA','CB','OG'],
'THR': ['O','N','C','CA','CB','OG1','CG2'],
'TRP': ['O','N','C','CA','CB','CG','CD1','CD2','NE1','CE2','CZ2','CH2','CE3','CZ3'],
'TYR': ['O','N','C','CA','CB','CG','CD1','CD2','CE2','CZ','CE1','OH'],
'VAL': ['O','N','C','CA','CB','CG1','CG2'],
'TPO': ['O','N','C','CA','CB','OG1','CG2', 'P', 'OE1', 'OE2', 'OE3'],
'SEP': ['O','N','C','CA','CB','OG', 'P', 'OE1', 'OE2', 'OE3'],
}

# original version
atom_order_list = \
{'ALA': [[1, 2, 3]],
 'ARG': [[1, 2, 3],
  [2, 3, 4],
  [3, 4, 5],
  [4, 5, 6],
  [5, 6, 7],
  [6, 7, 8],
  [7, 8, 9]],
 'ASP': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
 'ASN': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
 'CYS': [[1, 2, 3], [2, 3, 4]],
 'GLU': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
 'GLN': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
 'GLY': [],
 'HIS': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 4, 5], [7, 5, 6], [5, 6, 8]],
 'ILE': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 4, 6]],
 'LEU': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
 'LYS': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
 'MET': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
 'PHE': [[1, 2, 3],
  [2, 3, 4],
  [3, 4, 5],
  [4, 5, 6],
  [5, 6, 7],
  [3, 4, 5],
  [4, 5, 9]],
 'PRO': [[1, 2, 3], [1, 3, 4], [4, 3, 1]],
 'SER': [[1, 2, 3], [2, 3, 4]],
 'THR': [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
 'TRP': [[1, 2, 3],
  [2, 3, 4],
  [3, 4, 5],
  [3, 4, 5],
  [7, 5, 6],
  [6, 5, 7],
  [5, 7, 9],
  [7, 9, 10],
  [10, 9, 7],
  [9, 7, 12]],
 'TYR': [[1, 2, 3],
  [2, 3, 4],
  [3, 4, 5],
  [3, 4, 5],
  [6, 5, 7],
  [5, 7, 8],
  [7, 8, 9],
  [7, 8, 9]],
 'VAL': [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
 'TPO': [[1, 2, 3], [2, 3, 4], [2, 3, 4], [6, 4, 5], [4, 5, 7], [4, 5, 7], [4, 5, 7]],
 'SEP': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [4, 5, 6], [4, 5, 6]]}


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
 19: 'THR',
 20: 'TPO',
 21: 'SEP'}

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
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']
    res_list = np.unique(np.array(table.newSeq))
    res_list.sort()
    n_res = len(res_list)
    tot_ic = []
    for i in range(1,n_res-1):
        rest = table.loc[table.newSeq==res_list[i]]
        if len(rest) > 0:             
            resn = rest.resName.values[0]
            atom_list = core_atoms[resn]
            atom_idx = [rest.loc[rest.name==atom].index[0] for atom in atom_list]
            restraj = traj.atom_slice(atom_idx)
            tb, _ = restraj.top.to_dataframe()
            resxyz = restraj.xyz
            
            ic_i = np.zeros((10, resxyz.shape[0], 3))
            for j in range(len(core_atoms[resn])-4):
                order = atom_order_list[resn][j]      
                A1, A2, A3, A4 = resxyz[:, j+4], resxyz[:, order[2]], resxyz[:, order[1]], resxyz[:, order[0]]       
                # get dist, ang, tor
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

    all_ic = np.stack((N_ic, C_ic, O_ic), axis=2).squeeze(axis=-2) # 75, 126, 3, 3
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
    permute, atom_idx, atom_orders = info
    atom_orders = atom_orders.to(ic_recon.device)
    CG_xyz = CG_nxyz[:, :, 1:]
    
    N = add_atom_to_xyz(ic_recon[:,:,0], [CG_xyz[:,1:-1], CG_xyz[:,:-2], CG_xyz[:,2:]])
    C = add_atom_to_xyz(ic_recon[:,:,1], [CG_xyz[:,1:-1], CG_xyz[:,2:], CG_xyz[:,:-2]])
    O = add_atom_to_xyz(ic_recon[:,:,2], [C, CG_xyz[:,1:-1], N])
    xyz_recon = torch.stack((O, N, C, CG_xyz[:,1:-1]), axis=2) # O, N, C, CA
    
    bs = CG_xyz.shape[0]
    for i in range(10):
        current_ic = ic_recon[:,:,3+i]

        current_order1 = atom_orders[i,:,2].reshape(1, -1, 1, 1).repeat(bs, 1, 1, 3)
        current_order2 = atom_orders[i,:,1].reshape(1, -1, 1, 1).repeat(bs, 1, 1, 3)
        current_order3 = atom_orders[i,:,0].reshape(1, -1, 1, 1).repeat(bs, 1, 1, 3)

        atom1 = torch.gather(xyz_recon, dim=2, index=current_order1).squeeze()
        atom2 = torch.gather(xyz_recon, dim=2, index=current_order2).squeeze()
        atom3 = torch.gather(xyz_recon, dim=2, index=current_order3).squeeze()
        
        new_atom = add_atom_to_xyz(current_ic, [atom1, atom2, atom3]).unsqueeze(2)
        xyz_recon = torch.cat([xyz_recon, new_atom], axis=2)
        
    xyz_recon = xyz_recon.reshape(xyz_recon.shape[0],-1,3)[:,atom_idx,:][:,permute,:]
    return xyz_recon
