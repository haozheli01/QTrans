"""Mechanically extracted QTrans methods."""

from copy import deepcopy
from pymatgen.core.periodic_table import Element
from qtrans.core.constants import QTcts
from qtrans.io.abacus import read_abacus_input
from qtrans.numerics.arrays import average_close_values
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
import numpy as np
import os
import re
import torch
import warnings


class _AbacusMixin:
    def _gen_graph_data_abacus(self, folder:str, basis_def:dict):
        # Define paths for the required files
        output_folder_path = os.path.join(folder, self.abacus_out_folder)
        scf_log_path = os.path.join(output_folder_path, self.abacus_scf_log)
        stru_file_path = os.path.join(folder, 'STRU')
        pattern_eng_abacus = re.compile(r'final etot is\s*(-?\d+\.?\d*)')
        pattern_md_abacus = re.compile(r'ELEC=\s*(\d+)')
        
        # Read energy and SCF iteration data
        try:
            with open(scf_log_path, 'r') as f:
                log_content = f.read().strip()
                #energy = float(pattern_eng_abacus.findall(log_content)[0])
                energy = -999.99 # this parameter is useless here!
                max_scf_iterations = int(pattern_md_abacus.findall(log_content)[-1])
        except Exception as e:
            print(f"Error reading SCF log file: {e}. Please check the convergence! Skipping...")

        # Read crystal structure parameters
        try:
            species, lattice, positions, magmoms = read_abacus_input(stru_file_path)
            atomic_numbers = [Element(iatom).Z for iatom in species]
            lattice = lattice * QTcts.ANGtoBOHR # convert to atomic unit
            positions = positions * QTcts.ANGtoBOHR # convert to atomic unit
            if self.spinor_mode == 'collinear':
                spin_length = np.abs(magmoms[:, 2])
                is_magnetic = spin_length > self.magnetic_tolerance
                spin_length[~is_magnetic] = 0.0
                spin_vec = np.zeros_like(magmoms, dtype=float)
                spin_vec[is_magnetic, 2] = np.sign(magmoms[is_magnetic, 2])
                spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
            elif self.spinor_mode == 'nc':
                spin_length = np.linalg.norm(magmoms, axis=1)
                spin_vec = np.zeros_like(magmoms, dtype=float)
                is_magnetic = spin_length > self.magnetic_tolerance
                spin_length[~is_magnetic] = 0.0
                spin_vec[is_magnetic, :] = magmoms[is_magnetic, :] / spin_length[is_magnetic][:, None]
                spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
        except Exception as e:
            print(f"Error reading STRU file: {e}. Skipping...")
        # Read hopping and overlap parameters
        try:
            # Load sparse Hamiltonian and overlap matrices
            HR_spin0 = os.path.join(output_folder_path, 'data-HR-sparse_SPIN0.csr')
            SR = os.path.join(output_folder_path, 'data-SR-sparse_SPIN0.csr')
            # Generate graphs for Hamiltonian and overlap
            if self.spinor_mode == 'nc':
                graph_h_spin0 = self.get_abacus_hamiltonian_graph(HR_spin0, atomic_numbers, lattice, positions, basis_def,
                                                            graph={}, isH=True, skip=False, isSOC = True)
                graph_s = self.get_abacus_hamiltonian_graph(SR, atomic_numbers, lattice, positions, basis_def,
                                                            graph=graph_h_spin0, skip=True, isSOC = True)
            elif self.spinor_mode == 'collinear':
                graph_h_spin0 = self.get_abacus_hamiltonian_graph(HR_spin0, atomic_numbers, lattice, positions, basis_def,
                                                            graph={}, isH=True, skip=False)
                HR_spin1 = os.path.join(output_folder_path, 'data-HR-sparse_SPIN1.csr')
                graph_h_spin1 = self.get_abacus_hamiltonian_graph(HR_spin1, atomic_numbers, lattice, positions, basis_def,
                                                            graph={}, isH=True, skip=False)
                graph_s = self.get_abacus_hamiltonian_graph(SR, atomic_numbers, lattice, positions, basis_def,
                                                            graph=graph_h_spin0, skip=True)
            else:
                graph_h_spin0 = self.get_abacus_hamiltonian_graph(HR_spin0, atomic_numbers, lattice, positions, basis_def,
                                                            graph={}, isH=True, skip=False)
                graph_s = self.get_abacus_hamiltonian_graph(SR, atomic_numbers, lattice, positions, basis_def,
                                                            graph=graph_h_spin0, skip=True)
            # Extract graph properties
            pos = graph_h_spin0['pos']
            edge_index = graph_h_spin0['edge_index']
            inv_edge_idx = graph_h_spin0['inv_edge_idx']
            nbr_shift = graph_h_spin0['nbr_shift']
            cell_shift = graph_h_spin0['cell_shift']
            # Validate edge indices
            if len(inv_edge_idx) != len(edge_index[0]):
                raise ValueError(f"Mismatch in lengths: len(inv_edge_idx) ({len(inv_edge_idx)}) != len(edge_index[0]) ({len(edge_index[0])})")
            # Extract relevant data from graph_h and graph_s
            Hon_spin0 = graph_h_spin0['Hon']
            Hoff_spin0 = graph_h_spin0['Hoff']
            if self.spinor_mode == 'collinear':
                Hon_spin1 = graph_h_spin1['Hon']
                Hoff_spin1 = graph_h_spin1['Hoff']
            Son = graph_s['Hon'][0]
            Soff = graph_s['Hoff'][0]
            # Initialize matrices
            num_sub_matrix = pos.shape[0] + edge_index.shape[1]
            if self.spinor_mode == 'collinear':
                H = np.zeros((num_sub_matrix, 2, self.nao_max**2), dtype=np.float32)
            elif self.spinor_mode == 'nc':
                H = np.zeros((num_sub_matrix, (2*self.nao_max)**2), dtype=np.float32)
                iH= np.zeros((num_sub_matrix, (2*self.nao_max)**2), dtype=np.float32)
            else:
                H = np.zeros((num_sub_matrix, self.nao_max**2), dtype=np.float32)
            S = np.zeros((num_sub_matrix, self.nao_max**2), dtype=np.float32)
            # Fill in on-site terms for Hamiltonian and overlap
            for i, src in enumerate(atomic_numbers):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                mask[basis_def[src][:, None], basis_def[src][None, :]] = 1
                mask = mask.astype(bool)  # Ensure mask is binary
                if self.spinor_mode == 'collinear':
                    H[i,0][mask.flatten()] = Hon_spin0[0][i]
                    H[i,1][mask.flatten()] = Hon_spin1[0][i]
                elif self.spinor_mode == 'nc':
                    tH = np.zeros((2*self.nao_max, 2*self.nao_max), dtype=np.complex64)
                    tH[:self.nao_max,:self.nao_max][mask] = Hon_spin0[0][i] # uu
                    tH[:self.nao_max,self.nao_max:][mask] = Hon_spin0[1][i] # ud
                    tH[self.nao_max:,:self.nao_max][mask] = Hon_spin0[2][i] # du
                    tH[self.nao_max:,self.nao_max:][mask] = Hon_spin0[3][i] # dd
                    H[i] = tH.real.flatten()
                    iH[i]= tH.imag.flatten()
                else:
                    H[i][mask.flatten()] = Hon_spin0[0][i]
                S[i][mask.flatten()] = Son[i].real
            # Fill in off-site terms for Hamiltonian and overlap
            for num, (src, tar) in enumerate(zip(edge_index[0], edge_index[1])):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                mask[basis_def[atomic_numbers[src]][:, None], basis_def[atomic_numbers[tar]][None, :]] = 1
                mask = mask.astype(bool)
                if self.spinor_mode == 'collinear':
                    H[num + len(atomic_numbers),0][mask.flatten()] = Hoff_spin0[0][num]
                    H[num + len(atomic_numbers),1][mask.flatten()] = Hoff_spin1[0][num]
                elif self.spinor_mode == 'nc':
                    tH = np.zeros((2*self.nao_max, 2*self.nao_max), dtype=np.complex64)
                    tH[:self.nao_max,:self.nao_max][mask] = Hoff_spin0[0][num] # uu
                    tH[:self.nao_max,self.nao_max:][mask] = Hoff_spin0[1][num] # ud
                    tH[self.nao_max:,:self.nao_max][mask] = Hoff_spin0[2][num] # du
                    tH[self.nao_max:,self.nao_max:][mask] = Hoff_spin0[3][num] # dd
                    H[num + len(atomic_numbers)] = tH.real.flatten()
                    iH[num + len(atomic_numbers)]= tH.imag.flatten()
                else:
                    H[num + len(atomic_numbers)][mask.flatten()] = Hoff_spin0[0][num]
                # Fill in overlap matrix for off-site terms
                S[num + len(atomic_numbers)][mask.flatten()] = Soff[num].real
        except Exception as e:
            print(f"Error reading Hamiltonian or overlap matrices: {e}. Skipping...")
        # save in Data
        graph_tmp = Data(z=torch.LongTensor(atomic_numbers),
                        cell = torch.Tensor(lattice[None,:,:]),
                        total_energy=torch.Tensor([energy]),
                        pos=torch.FloatTensor(pos),
                        node_counts=torch.LongTensor([len(atomic_numbers)]),
                        edge_index=torch.LongTensor(edge_index),
                        inv_edge_idx=torch.LongTensor(inv_edge_idx),
                        nbr_shift=torch.FloatTensor(nbr_shift),
                        cell_shift=torch.LongTensor(cell_shift),
                        Son = torch.FloatTensor(S[:pos.shape[0],:]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]))
        if self.spinor_mode == 'collinear':
            graph_tmp['Hon'] = torch.FloatTensor(H[:pos.shape[0],:,:])
            graph_tmp['Hoff'] = torch.FloatTensor(H[pos.shape[0]:,:,:])
            graph_tmp['spin_length'] = torch.FloatTensor(spin_length)
            graph_tmp['spin_vec'] = torch.FloatTensor(spin_vec)
        elif self.spinor_mode == 'nc':
            graph_tmp['Hon'] = torch.FloatTensor(H[:pos.shape[0],:])
            graph_tmp['Hoff'] = torch.FloatTensor(H[pos.shape[0]:,:])
            graph_tmp['iHon'] = torch.FloatTensor(iH[:pos.shape[0],:])
            graph_tmp['iHoff'] = torch.FloatTensor(iH[pos.shape[0]:,:])
            graph_tmp['overlap'] = torch.FloatTensor(S)
            graph_tmp['spin_length'] = torch.FloatTensor(spin_length)
            graph_tmp['spin_vec'] = torch.FloatTensor(spin_vec)
        else:
            graph_tmp['Hon'] = torch.FloatTensor(H[:pos.shape[0]])
            graph_tmp['Hoff'] = torch.FloatTensor(H[pos.shape[0]:])
        return graph_tmp

    def get_abacus_hamiltonian_graph(self, file_path, atomic_numbers, lattice, positions, basis_def,
                                    graph: dict = None, skip = False, isH = False, isSOC = False, tojson = False) -> dict:
        def find_matching_column_index(matrix, target_column_values):
            """
            Find the index of a column in a 2D numpy array that exactly matches a given target column of values.
            
            This function compares each column of the matrix with the target column values and returns the index of
            the first column that matches. If no match is found, it returns None.

            Parameters:
                matrix (np.ndarray): A 2D numpy array (5xN), where N is the number of rows and 5 is the number of columns.
                target_column_values (list or np.ndarray): A 1D array or list containing the target column values to match.
                
            Returns:
                int or None: The index of the matching column if found, otherwise None.
            """
            # Ensure the target column is a numpy array for consistency and correct shape
            target_column_values = np.asarray(target_column_values)
            
            # Validate that the target column has the same number of rows as the matrix
            if target_column_values.shape[0] != matrix.shape[0]:
                raise ValueError("The number of elements in the target column must match the number of rows in the matrix.")
            
            # Compare each column of the matrix with the target column using broadcasting
            column_matches = np.all(matrix == target_column_values[:, None], axis=0)
            
            # If a match is found, return the index of the first matching column; otherwise, return None
            return np.argmax(column_matches) if column_matches.any() else None
        
        def find_inverse_edge_index(edge_index, cell_shifts):
            """
            Find the indices of inverse edges in a given set of edges with associated cell shifts.

            This function takes a list of edges (defined by their start and end nodes) and a list of 
            corresponding periodic cell shifts. It returns an array of indices where each entry 
            contains the index of the inverse edge for the corresponding edge. If no inverse edge is found,
            the index will be -1.

            Parameters:
            -----------
            edge_index : np.ndarray
                A 2xN numpy array where the first row contains the start nodes of edges, and the second 
                row contains the end nodes of edges.

            cell_shifts : np.ndarray
                A Nx3 numpy array where each row represents a periodic shift vector associated with the 
                corresponding edge in edge_index.

            Returns:
            --------
            inverse_indices : np.ndarray
                A 1D numpy array where each element is the index of the inverse edge for the corresponding 
                edge. If an inverse edge does not exist, the entry is -1.
            """
            
            # Combine edge information with cell shifts and ensure the shifts are tuples (for immutability)
            edges_with_shifts = list(zip(edge_index[0], edge_index[1], map(tuple, cell_shifts)))
            
            # Create dictionaries to map edges and inverse edges to their indices
            edge_to_index = {edge: idx for idx, edge in enumerate(edges_with_shifts)}
            inverse_edge_to_index = {
                (end, start, tuple(-np.array(shift))): idx
                for idx, (start, end, shift) in enumerate(edges_with_shifts)
            }
            
            # Initialize an array to hold the indices of the inverse edges, defaulting to -1
            inverse_indices = np.full(len(edges_with_shifts), -1, dtype=int)
            
            # For each edge, check if its inverse exists and store the inverse's index
            for edge, idx in edge_to_index.items():
                if edge in inverse_edge_to_index:
                    inverse_indices[idx] = inverse_edge_to_index[edge]
            
            # Raise an exception if any inverse index was not found (i.e., is still -1)
            # if np.any(inverse_indices == -1):
            #     raise RuntimeError("Some edges do not have corresponding inverse edges.")
            if np.any(inverse_indices == -1):
                num_missing = np.sum(inverse_indices == -1)
                warnings.warn(
                    f"Some edges do not have corresponding inverse edges. "
                    f"Found {num_missing} missing edges, matrix might not be positive definite.",
                    RuntimeWarning
                )
            return inverse_indices

        def convert_to_int(value):
            """
            Convert a value or a collection of values to integers.
            
            Parameters:
            value: int, list, or np.ndarray
                The value to convert. Can be a single integer, a list of values, or a NumPy array.
                
            Returns:
            int, list, or np.ndarray
                The converted integer value(s). If the input is a list or array, it returns the converted values in the same structure.
            """
            if isinstance(value, list):
                return [convert_to_int(item) for item in value]
            elif isinstance(value, np.ndarray):
                return value.astype(int).tolist()
            elif isinstance(value, (np.int32, np.int64)):
                return int(value)
            return value

        def convert_to_float(value):
            """
            Convert a value or a collection of values to floats.
            
            Parameters:
            value: float, list, or np.ndarray
                The value to convert. Can be a single float, a list of values, or a NumPy array.
                
            Returns:
            float, list, or np.ndarray
                The converted float value(s). If the input is a list or array, it returns the converted values in the same structure.
            """
            if isinstance(value, list):
                return [convert_to_float(item) for item in value]
            elif isinstance(value, np.ndarray):
                return value.astype(float).tolist()
            elif isinstance(value, (np.float32, np.float64)):
                return float(value)
            return value

        def convert_complex(value):
            """
            Convert a complex number or a collection of complex numbers to their real and imaginary parts.
            
            Parameters:
            value: complex, list, or np.ndarray
                The complex value to convert. Can be a single complex number, a list of complex values, or a NumPy array.
                
            Returns:
            tuple
                A tuple containing lists of real and imaginary parts of the complex value(s).
            """
            def extract_real(value):
                if isinstance(value, list):
                    return [extract_real(item) for item in value]
                elif isinstance(value, np.ndarray):
                    return value.real.tolist()
                elif isinstance(value, (np.complex64, np.complex128)):
                    return float(value.real)
                return value

            def extract_imaginary(value):
                if isinstance(value, list):
                    return [extract_imaginary(item) for item in value]
                elif isinstance(value, np.ndarray):
                    return value.imag.tolist()
                elif isinstance(value, (np.complex64, np.complex128)):
                    return float(value.imag)
                return value

            real_part = extract_real(deepcopy(value))
            imaginary_part = extract_imaginary(deepcopy(value))
            return real_part, imaginary_part

        num_atoms_unit_cell = len(atomic_numbers)
        if graph is None:
            graph = {}

        assert (not graph and not skip) or (graph and skip)

        dtype = np.float32 if not isSOC else np.complex64
        repeat = 1 if not isSOC else 2
        nspin = 1 if not isSOC else 4
        
        edge_idx_src, edge_idx_dst, cell_shift, nbr_shift = [], [], [], []
        Hon = [[]] if not isSOC else [[], [], [], []]
        Hoff = [[]] if not isSOC else [[], [], [], []]

        noff = 0
        if skip:
            graph_ = deepcopy(graph)
            noff = len(graph_['inv_edge_idx'])
            edge_idx_src = graph_['edge_index'][0]
            edge_idx_dst = graph_['edge_index'][1]
            cell_shift = graph_['cell_shift']
            Hoff = graph_['Hoff']
            for ispin in range(nspin):
                for ioff in range(noff):
                    Hoff[ispin][ioff] = np.zeros_like(Hoff[ispin][ioff], dtype=dtype)

        with open(file_path, 'r') as fp:
            line = fp.readline()
            if 'STEP' in line:
                num_tot_orbits = int(fp.readline().split()[-1])
            else:
                num_tot_orbits = int(line.split()[-1])
            ncell_shift = int(fp.readline().split()[-1])

            # basis number of each atom
            orbitals_per_atom = [len(basis_def[src]) for src in atomic_numbers]
            orbitals_per_atom = np.array(orbitals_per_atom, dtype=int) * repeat
            if orbitals_per_atom.sum() != num_tot_orbits:
                print("STRU parse error! Mismatch in total number of orbitals.")
                raise RuntimeError("Total number of orbitals mismatch")
            # the index of atomic orbital in the whole matrix
            orbital_indices = np.zeros_like(orbitals_per_atom, dtype=int)
            orbital_indices[1:] = np.cumsum(orbitals_per_atom[:-1])

            edge_info_array = None
            if skip:
                # shape: (5, N) -> src, dst, cx, cy, cz
                edge_info_array = np.concatenate([np.array(graph_['edge_index']), np.array(cell_shift).T], axis=0)
            
            while True:
                line = fp.readline()
                if not line: break
                # skip the empty line
                tmp = line.split()
                if not tmp: continue
                # cell shift
                cx, cy, cz = int(tmp[0]), int(tmp[1]), int(tmp[2])
                nh = int(tmp[3]) # number of nonzero elements
                # All the matrix elements are zero, skip!
                if nh == 0: continue

                # if nh == 0:
                #     hamilt = csr_matrix(
                #         (num_tot_orbits, num_tot_orbits),
                #         dtype=dtype
                #     )
                else:
                    val_str = fp.readline()
                    col = list(map(int, fp.readline().split()))
                    row = list(map(int, fp.readline().split()))

                    if not isSOC:
                        val = list(map(float, val_str.split()))
                    else:
                        val_raw = re.findall(r'[\-\+\d\.eE]+', val_str)
                        val_raw = np.asarray(val_raw, dtype=np.float32)
                        val = val_raw[0::2] + 1j * val_raw[1::2]

                    hamilt = csr_matrix(
                        (val, col, row),
                        shape=(num_tot_orbits, num_tot_orbits),
                        dtype=dtype
                    )

                if isH:
                    hamilt *= 1/2 # 1 Ry = 1/2 Hartree
                # Process Hamiltonian and populate graph data
                for ia in range(num_atoms_unit_cell):
                    for ja in range(num_atoms_unit_cell):
                        ham = hamilt[orbital_indices[ia]:orbital_indices[ia] + orbitals_per_atom[ia], 
                                     orbital_indices[ja]:orbital_indices[ja] + orbitals_per_atom[ja]]
                        # Onsite Hamiltonian (same cell and same atom)
                        if ia == ja and cx == 0 and cy == 0 and cz == 0:
                            if not isSOC:
                                Hon[0].append(ham.toarray().flatten())
                            else:
                                Hon[0].append(ham[0::2, 0::2].toarray().flatten()) # uu
                                Hon[1].append(ham[0::2, 1::2].toarray().flatten()) # ud
                                Hon[2].append(ham[1::2, 0::2].toarray().flatten()) # du
                                Hon[3].append(ham[1::2, 1::2].toarray().flatten()) # dd
                        # Offsite Hamiltonian
                        elif ham.getnnz() > 0:
                            if not skip:
                                if not isSOC:
                                    Hoff[0].append(ham.toarray().flatten())
                                else:
                                    Hoff[0].append(ham[0::2, 0::2].toarray().flatten())
                                    Hoff[1].append(ham[0::2, 1::2].toarray().flatten())
                                    Hoff[2].append(ham[1::2, 0::2].toarray().flatten())
                                    Hoff[3].append(ham[1::2, 1::2].toarray().flatten())
                                
                                edge_idx_src.append(ia)
                                edge_idx_dst.append(ja)
                                cell_shift.append(np.array([cx, cy, cz], dtype=int))
                                nbr_shift.append(np.array([cx, cy, cz]) @ lattice)
                            else:
                                ioff = find_matching_column_index(edge_info_array, [ia, ja, cx, cy, cz])
                                
                                if ioff is not None:
                                    if not isSOC:
                                        Hoff[0][ioff] = ham.toarray().flatten()
                                    else:
                                        Hoff[0][ioff] = ham[0::2, 0::2].toarray().flatten()  # uu
                                        Hoff[1][ioff] = ham[0::2, 1::2].toarray().flatten()  # ud
                                        Hoff[2][ioff] = ham[1::2, 0::2].toarray().flatten()  # du
                                        Hoff[3][ioff] = ham[1::2, 1::2].toarray().flatten()  # dd
        if not skip:
            # Construct the edges and graph
            edge_index = [edge_idx_src, edge_idx_dst]
            inv_edge_idx = find_inverse_edge_index(np.array(edge_index), np.array(cell_shift))

            graph_ = {}
            if tojson:
                graph_['edge_index'] = edge_index 
                graph_['inv_edge_idx'] = convert_to_int(inv_edge_idx)
                graph_['cell_shift'] = convert_to_int(cell_shift)
                graph_['nbr_shift'] = convert_to_float(nbr_shift)
                graph_['pos'] = convert_to_float(positions)
            else:
                graph_['edge_index'] = np.array(edge_index)
                graph_['inv_edge_idx'] = inv_edge_idx
                graph_['cell_shift'] = np.array(cell_shift)
                graph_['nbr_shift'] = np.array(nbr_shift)
                graph_['pos'] = positions

        if not tojson:
            graph_['Hon'] = Hon
            graph_['Hoff'] = Hoff
        else:
            if not isSOC:
                graph_['Hon'] = convert_to_float(Hon)
                graph_['Hoff'] = convert_to_float(Hoff)
            else:
                graph_['Hon'], graph_['iHon'] = convert_complex(Hon)
                graph_['Hoff'], graph_['iHoff'] = convert_complex(Hoff)

        return graph_
