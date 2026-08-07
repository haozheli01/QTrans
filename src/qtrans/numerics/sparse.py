"""QTrans numerics.sparse helpers."""

from qtrans.data.basis import basis_def_dicts
import math
import numpy as np


def build_sparse_matrix_soc(species, cell_shift, nao_max, Hon, Hoff, iHon, iHoff, edge_index, Ham_type:str='openmx', return_raw_mat:bool=False):
    """
    Args:
        cell_shift (_type_): _description_
        natoms (_type_): _description_
        nao_max (_type_): _description_
        Hon (_type_): _description_
        Hoff (_type_): _description_
        edge_index (_type_): _description_
    """
    # Convert each row in cell_shift to a tuple.
    cell_shift_tuple_edge = [tuple(row) for row in cell_shift] # list(map(lambda x: tuple(x), cell_shift.tolist())) # len: (nedges, )
    if (0,0,0) not in cell_shift_tuple_edge:
        cell_shift_tuple = [(0,0,0)]+cell_shift_tuple_edge
    else:
        cell_shift_tuple = cell_shift_tuple_edge
    # Convert cell_shift_tuple to a set to remove duplicates
    cell_shift_set = set(cell_shift_tuple) # len: (ncells, Tuple[3])
    cell_shift_list = sorted(list(cell_shift_set)) # len: (ncells, Tuple[3])
    cell_shift_array = np.array(cell_shift_list) # shape: (ncells, 3)
    ncells = len(cell_shift_list)

    # Map each cell shift to its index in cell_shift_list (built first for O(1) lookups)
    cell_index_map = {icell: i for i, icell in enumerate(cell_shift_list)}
    # Get indices of elements in cell_shift_tuple_edge from cell_shift_list
    cell_index = [cell_index_map[icell] for icell in cell_shift_tuple_edge] # len: (nedges, )
    # Find indices of elements in -cell_shift_array from cell_shift_list
    inv_cell_index = [cell_index_map[tuple(vec)] for vec in -cell_shift_array] # len: (ncells,)

    # soc build
    natoms = len(species)
    na = np.arange(natoms)
    H_cell = np.zeros((ncells, 2, natoms, nao_max, 2, natoms, nao_max), dtype=complex)
    H_cell[cell_index, :, edge_index[0], :, :, edge_index[1], :] = (Hoff + 1.0j*iHoff).reshape(-1, 2, nao_max, 2, nao_max)
    H_cell[cell_index_map[(0,0,0)], :, na, :, :, na, :] = (Hon + 1.0j*iHon).reshape(-1, 2, nao_max, 2, nao_max)

    if return_raw_mat:
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index)
    else:
        H_cell = H_cell.reshape(ncells, 2*natoms*nao_max, 2*natoms*nao_max)
        basis_label = f"{Ham_type}_{nao_max}"
        # parse the Atomic Orbital Basis Sets
        basis_definition = np.zeros((99, nao_max))
        # key is the atomic number, value is the index of the occupied orbits.
        basis_def = basis_def_dicts[basis_label]
        for k in basis_def.keys():
            basis_definition[k][basis_def[k]] = 1

        orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max]
        orb_mask = np.concatenate([orb_mask, orb_mask], axis=0) # shape: [2*natoms*nao_max]
        orb_idx = np.nonzero(orb_mask)[0]
        ix = np.ix_(orb_idx, orb_idx)

        H_cell = H_cell[:, ix[0], ix[1]]

        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index)

def build_sparse_matrix_collinear_split(species, cell_shift, nao_max, Hon, Hoff, edge_index, Ham_type:str='openmx', return_raw_mat:bool=False):
    """
    Args:
        cell_shift (_type_): _description_
        natoms (_type_): _description_
        nao_max (_type_): _description_
        Hon (_type_): _description_
        Hoff (_type_): _description_
        edge_index (_type_): _description_
    """
    # Initialize cell index
    cell_shift_tuple_edge = list(map(tuple, cell_shift.tolist())) # len: (nedges, )
    cell_shift_set = set(cell_shift_tuple_edge)
    cell_shift_set.add((0,0,0))
    cell_shift_list = sorted(cell_shift_set) # len: (ncells,)
    cell_shift_array = np.array(cell_shift_list) # shape: (ncells, 3)
    ncells = len(cell_shift_list)

    cell_index_map = {icell: i for i, icell in enumerate(cell_shift_list)}
    cell_index = np.array([cell_index_map[icell] for icell in cell_shift_tuple_edge]) # len: (nedges, )
    inv_cell_index = [cell_index_map[tuple(v)] for v in (-cell_shift_array).tolist()] # len: (ncells,)

    natoms = len(species)
    na = np.arange(natoms)
    H_cell = np.zeros((2, ncells, natoms, nao_max, natoms, nao_max))
    H_cell[0, cell_index, edge_index[0], :, edge_index[1], :] = Hoff[:,0,:,:]
    H_cell[1, cell_index, edge_index[0], :, edge_index[1], :] = Hoff[:,1,:,:]
    H_cell[0, cell_index_map[(0,0,0)], na, :, na, :] = Hon[:,0,:,:]
    H_cell[1, cell_index_map[(0,0,0)], na, :, na, :] = Hon[:,1,:,:]
    
    if return_raw_mat:
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index) 
    else:
        H_cell = H_cell.reshape(2, ncells, natoms*nao_max, natoms*nao_max)
        basis_label = f"{Ham_type}_{nao_max}"
        # parse the Atomic Orbital Basis Sets
        basis_definition = np.zeros((99, nao_max), dtype=bool)
        basis_def = basis_def_dicts[basis_label]
        for k, v in basis_def.items():
            basis_definition[k][v] = True

        orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max]
        orb_mask_2d = orb_mask[:, None] & orb_mask[None, :]  # shape: [natoms*nao_max, natoms*nao_max]

        H_cell = H_cell[:, :, orb_mask_2d]
        norbs = int(math.sqrt(H_cell.size/2/ncells))
        H_cell = H_cell.reshape(2, -1, norbs, norbs)

        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index)

def build_sparse_matrix(species, cell_shift, nao_max, Hon, Hoff, edge_index, Ham_type:str='openmx', return_raw_mat:bool=False):
    """
    Args:
        cell_shift (_type_): _description_
        natoms (_type_): _description_
        nao_max (_type_): _description_
        Hon (_type_): _description_
        Hoff (_type_): _description_
        edge_index (_type_): _description_
    """
    # Convert each row in cell_shift to a tuple.
    cell_shift_tuple_edge = list(map(tuple, cell_shift.tolist())) # len: (nedges, )
    cell_shift_set = set(cell_shift_tuple_edge)
    cell_shift_set.add((0,0,0))
    cell_shift_list = sorted(cell_shift_set) # len: (ncells, Tuple[3])
    cell_shift_array = np.array(cell_shift_list) # shape: (ncells, 3)
    ncells = len(cell_shift_list)

    cell_index_map = {icell: i for i, icell in enumerate(cell_shift_list)}
    cell_index = np.array([cell_index_map[icell] for icell in cell_shift_tuple_edge]) # len: (nedges, )
    inv_cell_index = [cell_index_map[tuple(v)] for v in (-cell_shift_array).tolist()] # len: (ncells,)

    natoms = len(species)
    na = np.arange(natoms)
    edge_index_0 = np.asarray(edge_index[0])
    edge_index_1 = np.asarray(edge_index[1])
    Hoff_2d = Hoff.reshape(-1, nao_max, nao_max)
    Hon_2d = Hon.reshape(-1, nao_max, nao_max)
    # Allocate directly in target layout to avoid swapaxes copy
    H_cell = np.zeros((ncells, natoms, nao_max, natoms, nao_max))
    H_cell[cell_index, edge_index_0, :, edge_index_1, :] = Hoff_2d
    H_cell[cell_index_map[(0,0,0)], na, :, na, :] = Hon_2d
    
    if return_raw_mat:
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index) 
    else:
        H_cell = H_cell.reshape(-1, natoms*nao_max, natoms*nao_max)
        # parse the Atomic Orbital Basis Sets
        basis_label = f"{Ham_type}_{nao_max}"
        # parse the Atomic Orbital Basis Sets
        basis_definition = np.zeros((99, nao_max), dtype=bool)
        basis_def = basis_def_dicts[basis_label]
        for k, v in basis_def.items():
            basis_definition[k][v] = True

        orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max]
        orb_mask_2d = orb_mask[:, None] & orb_mask[None, :]  # shape: [natoms*nao_max, natoms*nao_max]

        H_cell = H_cell[:, orb_mask_2d]
        norbs = int(math.sqrt(H_cell.size/ncells))
        H_cell = H_cell.reshape(-1, norbs, norbs)
        return (H_cell, cell_shift_array, cell_index, cell_index_map, inv_cell_index)

def build_reciprocal_from_sparseMat(H_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, norbs, norbs)
    """
    ncells, norbs, _ = H_cell.shape
    phase = np.exp(2j * np.pi * (k_vec @ nbr_shift_of_cell.T))  # (nk, ncells)
    HK = (phase @ H_cell.reshape(ncells, -1)).reshape(-1, norbs, norbs)  # (nk, norbs, norbs)
    return HK

def build_reciprocal_from_sparseMat_collinear_split(H_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (2, ncells, norbs, norbs)
    """
    _, ncells, norbs, _ = H_cell.shape
    phase = np.exp(2j * np.pi * (k_vec @ nbr_shift_of_cell.T))  # (nks, ncells)
    # (1, nks, ncells) @ (2, ncells, norbs*norbs) -> (2, nks, norbs*norbs)
    HK = (phase[np.newaxis] @ H_cell.reshape(2, ncells, -1)).reshape(2, -1, norbs, norbs)
    return HK

def build_reciprocal_from_sparseMat3(M_cell, k_vec, nbr_shift_of_cell):
    """Build reciprocal-space matrix with velocity component.

    Args:
        M_cell: (ncells, norbs, norbs, 3)
        k_vec: (nk, 3)
        nbr_shift_of_cell: (ncells, 3)

    Returns:
        MK: (nk, norbs, norbs, 3)
    """
    phase = np.exp(2j * np.pi * (k_vec @ nbr_shift_of_cell.T))  # (nk, ncells)
    MK = 1.0j * np.tensordot(phase, M_cell, axes=([1], [0]))  # (nk, norbs, norbs, 3)
    return MK

def build_reciprocal_from_sparseMat_soc(H_cell, k_vec, nbr_shift_of_cell):
    """_summary_

    Args:
        H_cell (_type_): (ncells, 2*norbs, 2*norbs)
    """

    # phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)
    # HK = np.einsum('ijk, ni->njk', H_cell, phase) # (nk, norbs, norbs,)
    
    phase = np.exp(2j*np.pi*np.sum(nbr_shift_of_cell[None,:,:]*k_vec[:,None,:], axis=-1)) # shape (nk, ncells)    
    norbs = int(H_cell.shape[1]/2)
    Hsoc = [H_cell[:,:norbs,:norbs], H_cell[:,:norbs,norbs:], H_cell[:,norbs:,:norbs], H_cell[:,norbs:,norbs:]]
    
    HK_list = []
    for H in Hsoc:
        HK = np.einsum('ijk, ni->njk', H, phase) # (nk, norbs, norbs,)
        HK_list.append(HK)

    HK = np.block([[HK_list[0],HK_list[1]],[HK_list[2],HK_list[3]]])
    
    return HK
