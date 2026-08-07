"""Mechanically extracted QTrans methods."""

from easydict import EasyDict
from qtrans.core.constants import QTcts
from qtrans.core.errors import InputParameterError
from qtrans.data.valence import num_val_abacus
from qtrans.data.valence import num_val_openmx
from qtrans.data.valence import num_val_siesta
from qtrans.numerics.arrays import average_close_values
from qtrans.numerics.sparse import build_sparse_matrix
from qtrans.numerics.sparse import build_sparse_matrix_collinear_split
from qtrans.numerics.sparse import build_sparse_matrix_soc
import math
import numpy as np


class _GraphMixin:
    def _load_magnetic_data(self, graph_data):
        """
        Load magnetic atom data from graph_data into self.graph_data.
        """
        self.graph_data.mag_atom_index = np.where(graph_data.spin_length.numpy() > self.magnetic_tolerance)[0]
        self.graph_data.spin_vec = graph_data.spin_vec.numpy()[self.graph_data.mag_atom_index]
        self.graph_data.spin_length = graph_data.spin_length.numpy()[self.graph_data.mag_atom_index] * QTcts.BOHRMAGtoHARTREE
        self.natoms_mag = len(self.graph_data.spin_length)

    def _load_hamiltonian(self, graph_data, *shape):
        """Load real-part Hamiltonian matrices (Hon/Hoff) with the given reshape dimensions."""
        self.graph_data.Hon = graph_data.Hon.numpy().reshape(*shape)
        self.graph_data.Hoff = graph_data.Hoff.numpy().reshape(*shape)

    def _load_soc_hamiltonian(self, graph_data):
        """Load SOC (spin-orbit coupling) Hamiltonian matrices from graph_data."""
        nao2 = 2 * self.nao_max
        self._load_hamiltonian(graph_data, -1, nao2, nao2)
        self.graph_data.iHon = graph_data.iHon.numpy().reshape(-1, nao2, nao2)
        self.graph_data.iHoff = graph_data.iHoff.numpy().reshape(-1, nao2, nao2)

    def _load_graph_data_uc(self) -> None:
        graph_data = np.load(self.graph_data_path_uc, allow_pickle=True)
        graph_data = graph_data['graph'].item()
        graph_data = list(graph_data.values())[0]

        self.graph_data = EasyDict()
        self.graph_data.pos = graph_data.pos.numpy()
        self.graph_data.Son = graph_data.Son.numpy().reshape(-1, self.nao_max, self.nao_max)
        self.graph_data.Soff = graph_data.Soff.numpy().reshape(-1, self.nao_max, self.nao_max)
        
        self.spinor_mode = 'none'
        if self.soc_switch:
            self._load_soc_hamiltonian(graph_data)
            if self.magnetic_type == 'none':
                self._log("Running calculation for a non-magnetic system with SOC.")
            elif self.magnetic_type == 'nc':
                self._load_magnetic_data(graph_data)
                self._log("Running calculation for a non-collinear magnetic system with SOC.")
            else:
                raise InputParameterError(5024, 'advanced', 'magnetic_type', f"The magnetic type {self.magnetic_type} is not supported in SOC systems.")
            self.spinor_mode = 'nc'
        else:
            if self.magnetic_type == 'none':
                self._load_hamiltonian(graph_data, -1, self.nao_max, self.nao_max)
                self._log("Running calculation for a non-magnetic system without SOC.")
            elif self.magnetic_type == 'collinear':
                self._load_hamiltonian(graph_data, -1, 2, self.nao_max, self.nao_max)
                self._load_magnetic_data(graph_data)
                self.graph_data.moment_direct = np.ones(self.natoms_mag)
                # The magnetic moment of the spin is opposite.
                mask_up_spin = self.graph_data.spin_vec[:, 2] > 0.0
                self.graph_data.moment_direct[mask_up_spin] = -1.0
                spin_up_atom_index = np.where(mask_up_spin)[0]
                spin_down_atom_index = np.where(~mask_up_spin)[0]
                natoms_mag_spin_up = len(spin_up_atom_index)
                natoms_mag_spin_down = len(spin_down_atom_index)
                spin_vec_with_length = self.graph_data.spin_vec[:, 2] * self.graph_data.spin_length
                pure_moment = np.abs(np.sum(spin_vec_with_length))
                self.spin_system_type = None
                self._log("Running calculation for a collinear magnetic system without SOC.")
                if natoms_mag_spin_up == 0 and natoms_mag_spin_down == 0:
                    self._log("Warning: This collinear system has no magnetic order.\n")
                elif natoms_mag_spin_up == 0:
                    self.spin_first_index = spin_down_atom_index[0]
                    self.spin_system_type = 'FM'
                    self._log("Dealing with a standard ferromagnetic system, and the spin direction is z-axis down.\n")
                elif natoms_mag_spin_down == 0:
                    self.spin_first_index = spin_up_atom_index[0]
                    self.spin_system_type = 'FM'
                    self._log("Dealing with a standard ferromagnetic system, and the spin direction is z-axis up.\n")
                elif pure_moment < QTcts.TENPM3:
                    if natoms_mag_spin_up == natoms_mag_spin_down:
                        self.spin_first_index = [spin_down_atom_index[0], spin_up_atom_index[0]]
                        self.spin_system_type = 'AFM'
                        self._log("Dealing with a standard antiferromagnetic system.\n")
                    else:
                        self.spin_first_index = [spin_down_atom_index[0], spin_up_atom_index[0]]
                        self.spin_system_type = 'CFM'
                        self._log("Dealing with a compensated ferromagnetic system.")
                else:
                    self.spin_first_index = [spin_down_atom_index[0], spin_up_atom_index[0]]
                    self.spin_system_type = 'FiM'
                    self._log("Dealing with a ferrimagnetism system.")
                self.spinor_mode = 'collinear'
            elif self.magnetic_type == 'nc':
                self._load_soc_hamiltonian(graph_data)
                self._load_magnetic_data(graph_data)
                self.spinor_mode = 'nc'
                self._log("Running calculation for a non-collinear magnetic system without SOC.")
            else:
                raise InputParameterError(5024, 'advanced', 'magnetic_type', f"The magnetic type {self.magnetic_type} is not supported.")
        if self.magnetic_type in ['collinear', 'nc']:
            self.graph_data.spin_length = average_close_values(self.graph_data.spin_length, self.moment_similar_tolerence)
            self.graph_data.spin_length = self.graph_data.spin_length * self.spin_length_scale
        # if exist Pon and Poff, save them in self.graph_data for electron-phonon-coupling long range correction
        # Pon shape: (natoms, nao_max^2, 3)
        # Poff shape: (num_edges, nao_max^2, 3)
        if ('Pon' in graph_data.keys) and ('Poff' in graph_data.keys):
            self.graph_data.Pon = graph_data.Pon.numpy().reshape(-1, self.nao_max, self.nao_max, 3)
            self.graph_data.Poff = graph_data.Poff.numpy().reshape(-1, self.nao_max, self.nao_max, 3)

        # Fallback: some graph_data files store the position-matrix elements under
        # r_mat_on / r_mat_off (transposed layout). Convert to the Pon/Poff layout.
        if ('r_mat_on' in graph_data.keys) and ('r_mat_off' in graph_data.keys):
            # r_mat_on  shape: (3, natoms, nao_max^2)
            # r_mat_off shape: (3, num_edges, nao_max^2)
            self.graph_data.Pon  = graph_data.r_mat_on.numpy().transpose(1, 2, 0)   # (natoms, nao_max^2, 3)
            self.graph_data.Poff = graph_data.r_mat_off.numpy().transpose(1, 2, 0)  # (num_edges, nao_max^2, 3)
            self.graph_data.Pon  = self.graph_data.Pon.reshape(-1, self.nao_max, self.nao_max, 3)
            self.graph_data.Poff = self.graph_data.Poff.reshape(-1, self.nao_max, self.nao_max, 3)
        # a_{i} = latt[i]
        self.graph_data.latt = graph_data.cell.numpy().reshape(3,3)
        self.graph_data.inv_latt =  np.linalg.inv(self.graph_data.latt)
        # b_{i} = 2PI * rec_latt[i]
        self.graph_data.rec_latt = self.graph_data.inv_latt.T
        self.graph_data.cell_shift = graph_data.cell_shift.numpy()
        self.graph_data.nbr_shift = graph_data.nbr_shift.numpy()
        self.graph_data.edge_index = graph_data.edge_index.numpy()
        self.graph_data.species = graph_data.z.numpy()

        if self.Ham_type == 'openmx':
            self.graph_data.num_electrons = np.sum(num_val_openmx[self.graph_data.species])
        elif self.Ham_type in ['siesta', 'honpas']:
            self.graph_data.num_electrons = np.sum(num_val_siesta[self.graph_data.species])
        elif self.Ham_type == 'abacus':
            self.graph_data.num_electrons = np.sum(num_val_abacus[self.graph_data.species])

        self.has_unpair_electron = bool(self.graph_data.num_electrons & 1)
        if self.has_unpair_electron and self.cal_mode in ['breakdown']:
            self._log("Warnning: The system has an odd number of electrons. Please make sure it is an insulator.")

        # CBM and VBM
        if self.spinor_mode == 'nc':
            self.VBM_band_index = self.graph_data.num_electrons - 1
            self.CBM_band_index = self.graph_data.num_electrons
        else:
            self.VBM_band_index = math.ceil(self.graph_data.num_electrons / 2) - 1
            if self.has_unpair_electron:
                self.CBM_band_index = self.VBM_band_index
            else:
                self.CBM_band_index = self.VBM_band_index + 1

        # cell volume
        if self.system_2D:
            a1 = self.graph_data.latt[0]
            a2 = self.graph_data.latt[1]
            self.volume_uc = np.linalg.norm(np.cross(a1, a2))
        else:
            self.volume_uc = np.linalg.det(self.graph_data.latt)
        self.inv_cell = 1.0 / self.volume_uc
        # Real space Hamiltonian matrix
        if self.spinor_mode == 'nc':
            # H_cell shape: [ncells, 2*norbs, 2*norbs]
            H_cell, cell_shift_array, _, cell_index_map, _ = build_sparse_matrix_soc(
                self.graph_data.species, self.graph_data.cell_shift, self.nao_max,
                self.graph_data.Hon, self.graph_data.Hoff, self.graph_data.iHon, self.graph_data.iHoff,
                self.graph_data.edge_index, return_raw_mat=False, Ham_type=self.Ham_type)
        elif self.spinor_mode == 'none':
            # H_cell shape: [ncells, norbs, norbs]
            H_cell, cell_shift_array, _, cell_index_map, _ = build_sparse_matrix(
                self.graph_data.species, self.graph_data.cell_shift, self.nao_max,
                self.graph_data.Hon, self.graph_data.Hoff,
                self.graph_data.edge_index, return_raw_mat=False, Ham_type=self.Ham_type)
        elif self.spinor_mode == 'collinear':
            # H_cell shape: [2, ncells, norbs, norbs]
            H_cell, cell_shift_array, _, cell_index_map, _ = build_sparse_matrix_collinear_split(
                self.graph_data.species, self.graph_data.cell_shift, self.nao_max,
                self.graph_data.Hon, self.graph_data.Hoff,
                self.graph_data.edge_index, return_raw_mat=False, Ham_type=self.Ham_type)
        # S_cell shape: [ncells, norbs, norbs]
        S_cell, _, _, _, _ = build_sparse_matrix(
            self.graph_data.species, self.graph_data.cell_shift, self.nao_max,
            self.graph_data.Son, self.graph_data.Soff,
            self.graph_data.edge_index, return_raw_mat=False, Ham_type=self.Ham_type)
        nbr_shift_of_cell = np.einsum('ni, ij -> nj', cell_shift_array, self.graph_data.latt) # shape: (ncells, 3)
        self.norbs = H_cell.shape[-1]
        self.graph_data.H_cell = H_cell
        self.graph_data.S_cell = S_cell
        self.graph_data.cell_shift_array = cell_shift_array
        self.graph_data.nbr_shift_of_cell = nbr_shift_of_cell
        self.cell_index_map = cell_index_map
        self.graph_data.Hv_cell = self._Hv_cell_prepare()
        self.graph_data.Sv_cell = self._Sv_cell_prepare()
        self.natoms = len(self.graph_data.species)
