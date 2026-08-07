"""Mechanically extracted QTrans methods."""

from phonopy.structure.atoms import atom_data
from qtrans.calculator.registry import CALCULATION_MODES
from qtrans.core.constants import QTcts
from qtrans.core.defaults import calculator_default_parameters
from qtrans.core.errors import InputParameterError
from qtrans.core.validators import validate_bool
from qtrans.core.validators import validate_existed_dir
from qtrans.core.validators import validate_int
from qtrans.core.validators import validate_int_list
from qtrans.core.validators import validate_non_negative_num
from qtrans.core.validators import validate_num
from qtrans.core.validators import validate_num_list
from qtrans.core.validators import validate_numpy_array
from qtrans.core.validators import validate_positive_int
from qtrans.core.validators import validate_positive_int_list
from qtrans.core.validators import validate_positive_num
from qtrans.core.validators import validate_readable_path
from qtrans.core.validators import validate_str_list
from qtrans.data.basis import basis_def_dicts
from qtrans.io.filesystem import check_path_exist
import numpy as np
import os
import phonopy


class _ConfigMixin:
    def _initial_basic(self):
        self.graph_data_path_uc = str(self.graph_data_path_uc)
        self.Ham_type = str(self.Ham_type).lower()
        self.outdir = str(self.outdir)

        if self.cal_mode not in CALCULATION_MODES:
            raise InputParameterError(
                5002, 'basic', 'cal_mode',
                f"The calculation mode {self.cal_mode} is not supported."
            )

        self._set('graph_data_path_uc', validate_readable_path, 'basic', 5003)

        VALID_HAM_TYPES = {'openmx', 'siesta', 'honpas', 'abacus'}
        if self.Ham_type not in VALID_HAM_TYPES:
            raise InputParameterError(
                5004, 'basic', 'Ham_type',
                f"The Hamiltonian type {self.Ham_type} is not supported."
            )

        self.basis_label = f"{self.Ham_type}_{self.nao_max}"
        if self.basis_label not in basis_def_dicts:
            raise InputParameterError(
                5005, 'basic', 'nao_max',
                f"The atomic basis {self.basis_label} is not supported."
            )

        self._log(f"Using {self.Ham_type.upper()} with a {self.nao_max}-orbital definition dictionary.\n")

        self.outdir = os.path.abspath(self.outdir)
        try:
            check_path_exist(self.outdir, check_type='dir', create_if_missing=True)
        except Exception:
            raise InputParameterError(
                5006, 'basic', 'outdir',
                f"The specified output directory '{self.outdir}' is not accessible or cannot be created."
            )

        try:
            self._load_graph_data_uc()
        except Exception as e:
            raise InputParameterError(
                5003, 'basic', 'graph_data_path_uc',
                f"Failed to read {self.graph_data_path_uc}.\n{e}"
            )

    def _initial_advanced(self):
        self.magnetic_type = str(self.magnetic_type).lower()
        if self.magnetic_type not in ('none', 'collinear', 'nc'):
            raise InputParameterError(
                5023, 'advanced', 'magnetic_type',
                f"The {self.magnetic_type} is not supported."
            )

        self._set('read_sparse_grad_mat',   validate_bool,         'advanced', 5021)
        self._set('soc_switch',             validate_bool,         'advanced', 5022)
        self._set('magnetic_tolerance',     validate_num,          'advanced', 5024)
        self._set('moment_similar_tolerence', validate_num,        'advanced', 5025)
        self._set('doping_charge',          validate_num,          'advanced', 5026)
        self._set('system_2D',              validate_bool,         'advanced', 5027)
        self._set('tetra_integral',         validate_bool,         'advanced', 5028)

        if self.system_2D:
            self._log('Dealing with a 2D system.\n')

    def _initial_phonon(self):
        self._set('supercell_matrix',      validate_numpy_array,      'phonon', 5041, (3, 3), float)
        self._set('unitcell_filename',     validate_readable_path,    'phonon', 5042)
        self._set('force_infos_filename',  validate_readable_path,    'phonon', 5043)
        self._set('apply_correction',      validate_bool,             'phonon', 5044)
        self._set('quadrupole_correction', validate_bool,             'phonon', 5045)
        self._set('thickness',             validate_non_negative_num, 'phonon', 5049)
        self._set('chiral_projection',     validate_bool,             'phonon', 5050)

        extension = os.path.splitext(self.force_infos_filename)[1].lower()
        common_params = dict(
            supercell_matrix=self.supercell_matrix,
            primitive_matrix=np.eye(3),
            unitcell_filename=self.unitcell_filename,
        )
        try:
            if extension == '':
                with open(self.force_infos_filename, 'r') as fin:
                    n_fields = len(fin.readline().strip().split())
                if n_fields == 1:
                    self.phonon = phonopy.load(**common_params, force_sets_filename=self.force_infos_filename)
                elif n_fields == 2:
                    self.phonon = phonopy.load(**common_params, force_constants_filename=self.force_infos_filename)
                else:
                    raise NotImplementedError
            elif extension == '.yaml':
                self.phonon = phonopy.load(self.force_infos_filename)
            elif extension in ('.h5', '.hdf5'):
                self.phonon = phonopy.load(**common_params, force_constants_filename=self.force_infos_filename)
            else:
                raise NotImplementedError
        except Exception:
            raise InputParameterError(
                5043, 'phonon', 'force_infos_filename',
                f"The format of phonon force infos file '{self.force_infos_filename}' is not supported."
            )


        if self.chiral_projection and not self.system_2D:
            self._log("Warning: You are using chiral phonon projection in 3D system, please make sure you really want this.")

        if self.apply_correction:
            self._set('BECs', validate_numpy_array, 'phonon', 5047, (self.natoms, 3, 3), float)
            self._set('DL',   validate_numpy_array, 'phonon', 5048, (3, 3), float)
            self.thickness = self.thickness * QTcts.ANGtoBOHR
            self.phonon.nac_params = {
                'born': self.BECs,
                'factor': QTcts.BOHR * QTcts.HARTREE,
                'dielectric': self.DL,
            }

        if self.quadrupole_correction:
            self._set('DL',     validate_numpy_array,   'phonon', 5048, (3, 3), float)
            self._set('Q_file', validate_readable_path, 'phonon', 5046)
            # input format is (natoms, 3, 6) in Voigt notation, convert to (natoms, 3, 3, 3)
            self.Q_tensor = np.zeros((self.natoms, 3, 3, 3), dtype=float)

            # Voigt index mapping: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
            voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]

            if self.rank == 0:
                with open(self.Q_file, 'r') as fin:
                    lines = fin.readlines()

                header_line = None
                for i, line in enumerate(lines):
                    if line.split()[0] == "atom":
                        header_line = i
                        break
                if header_line is None:
                    raise RuntimeError(f"Cannot find 'atom' keyword in {self.Q_file}.")

                for j in range(3 * self.natoms):
                    atom_idx, dir_idx = divmod(j, 3)
                    split_line = lines[header_line + 1 + j].split()
                    if len(split_line) < 8:
                        raise RuntimeError(f"Line {header_line+j+2} in {self.Q_file} does not contain enough data for quadrupole tensor.")
                    Q_values = [float(v) for v in split_line[2:8]]
                    for val, (a, b) in zip(Q_values, voigt_map):
                        self.Q_tensor[atom_idx, dir_idx, a, b] = val
                        self.Q_tensor[atom_idx, dir_idx, b, a] = val

                print("#" * 80)
                print(f"Quadrupole tensor loaded from file {self.Q_file} :")
                print("     atom    dir        Qxx         Qyy         Qzz         Qyz         Qxz         Qxy")
                for iatom in range(self.natoms):
                    for d in range(3):
                        Q = self.Q_tensor[iatom, d]
                        print(f"     {iatom+1:3d}     {d+1}     {Q[0,0]:10.6f}  {Q[1,1]:10.6f}  {Q[2,2]:10.6f}  {Q[1,2]:10.6f}  {Q[0,2]:10.6f}  {Q[0,1]:10.6f}")
                print("#" * 80)
            
            self.Q_tensor = self.comm.bcast(self.Q_tensor, root=0)       

        self.atomic_mass = np.array([atom_data[ia][3] for ia in self.graph_data.species]) * QTcts.AMU / QTcts.MASS_E  # in atomic unit, i.e. mass of electron.

    def _extract_dispersion_select_index(self, index_str: str, mode: str):
        """Parse two-part dispersion_select_index for epc modes."""
        index_str = index_str.strip()
        disp_select_index = [None, None]
        try:
            parts = index_str.split(',')
            if len(parts) != 2:
                raise RuntimeError(f"No index for the selected final state could be identified.")
            for i, each in enumerate(parts):
                if '-' in each:
                    lo, hi = each.split('-')
                    disp_select_index[i] = np.arange(int(lo) - 1, int(hi))
                else:
                    disp_select_index[i] = np.array([int(each) - 1])
                if np.any(disp_select_index[i] >= self.norbs):
                    raise RuntimeError(
                        f"The dispersion_select_index is over the total number of bands, which is {self.norbs}")
        except Exception as e:
                raise InputParameterError(
                    5091, 'dispersion', 'dispersion_select_index',
                    f"The dispersion_select_index must set as 'n, m' or 'n1-n2, m1-m2', while using '{mode}' calculation mode.\n{e}"
                )
        return disp_select_index

    def _parse_select_index(self, nmodes_max, label):
        """Parse dispersion_select_index string into a sorted unique array of indices."""
        if self.dispersion_select_index:
            try:
                raw = self.dispersion_select_index.strip()
                indices = []
                for each in raw.split(','):
                    lo, hi = each.split('-')
                    indices.extend(range(int(lo) - 1, int(hi)))
                result = np.unique(np.array(indices, dtype="int64"))
                if (result > nmodes_max - 1).any():
                    raise RuntimeError(f"The dispersion_select_index is over the total number of {label}, which is {nmodes_max}")
            except Exception as e:
                raise InputParameterError(
                    5091, 'dispersion', 'dispersion_select_index',
                    f"The dispersion_select_index must be specified as 'n1-n2, n3-n4, ...'.\n{e}"
                )
        else:
            result = np.arange(0, nmodes_max, dtype="int64")
        return result

    def _initial_dispersion(self):
        """
        Initialize dispersion block.
        """
        self._set('disp_renormalize', validate_bool,         'dispersion', 5081)
        self._set('disp_lambda',      validate_bool,         'dispersion', 5082)
        self._set('disp_nesting',     validate_bool,         'dispersion', 5083)
        self._set('nks_path',         validate_positive_int, 'dispersion', 5086)

        disp_source = self.cal_mode
        if disp_source == 'self-energy':
            disp_source = 'band' if self.SE_type == 'electron' else self.SE_type

        # EPC grid mode: skip path generation, parse two-part dispersion_select_index
        if disp_source == 'epc' and self.epc_grid_mode:
            if not self.dispersion_select_index:
                raise InputParameterError(
                    5091, 'dispersion', 'dispersion_select_index',
                    "dispersion_select_index must be specified for epc grid mode.")
            self.dispersion_select_index = self._extract_dispersion_select_index(
                self.dispersion_select_index, 'epc')
            return
        # Band grid mode: skip path generation
        if disp_source == 'band' and self.band_grid_mode:
            self.dispersion_select_index = self._parse_select_index(self.norbs, 'bands')
            return

        if self.disp_renormalize:
            if disp_source in ['band', 'phonon']:
                self._log(f"Calculating {disp_source} dispersion relation with renormalization.")
            else:
                raise InputParameterError(5081, 'dispersion', 'disp_renormalize', f"Renormalization is not implemented for calculation mode {disp_source}.")
        if self.disp_lambda:
            if disp_source == 'phonon':
                self._log(f"Calculating {disp_source} dispersion relation with lambda.")
            else:
                raise InputParameterError(5082, 'dispersion', 'disp_lambda', f"Lambda calculation is not implemented for calculation mode {disp_source}.")

        if self.plot_2D_axes:
            self._set('high_symmetry_points', validate_positive_int_list, 'dispersion', 5084, 2)
            self._set('plot_2D_axes',         validate_str_list,          'dispersion', 5087, 2)
            self._set('plot_2D_zcut',         validate_num,               'dispersion', 5088)
            self._set('plot_2D_ncuts',        validate_positive_int,      'dispersion', 5089)
            self._set('plot_2D_smear',        validate_positive_num,      'dispersion', 5090)

            self._log(f"Plot 2D figures as {self.plot_2D_axes[0]}-{self.plot_2D_axes[1]}")

            plot_2D_x = np.linspace(-0.5, 0.5, self.high_symmetry_points[0])
            plot_2D_y = np.linspace(-0.5, 0.5, self.high_symmetry_points[1])
            plot_2D_x, plot_2D_y = np.meshgrid(plot_2D_x, plot_2D_y)
            plot_2D_x, plot_2D_y = plot_2D_x.ravel(), plot_2D_y.ravel()

            self.high_symmetry_k_vecs = np.full((len(plot_2D_x), 3), self.plot_2D_zcut)
            for iaxis in range(2):
                axis_key = self.plot_2D_axes[iaxis].lower()
                if axis_key not in QTcts.AXIS_MAP:
                    raise InputParameterError(5087, 'dispersion', 'plot_2D_axes', f"The plot_2D_axes of '{self.plot_2D_axes!r}' is not supported.")
                self.plot_2D_axes[iaxis] = QTcts.AXIS_MAP[axis_key]
            self.high_symmetry_k_vecs[:, self.plot_2D_axes[0]] = plot_2D_x
            self.high_symmetry_k_vecs[:, self.plot_2D_axes[1]] = plot_2D_y
            self.plot_2D_smear = self.plot_2D_smear * QTcts.MEVtoHARTREE
        else:
            self.high_symmetry_auto = False
            if not isinstance(self.high_symmetry_points, list) or len(self.high_symmetry_points) == 0:
                self.high_symmetry_auto = True
            else:
                nhsks = len(self.high_symmetry_points)
                self._set('high_symmetry_points', validate_numpy_array, 'dispersion', 5084, (nhsks, 3), float)
                self._set('high_symmetry_labels', validate_str_list,    'dispersion', 5085, nhsks)
            self.high_symmetry_k_vecs, self.high_symmetry_k_dist, self.high_symmetry_k_nodes, self.high_symmetry_labels = \
                self._get_hsk_path(nks_path=self.nks_path, hsk_points=self.high_symmetry_points, hsk_labels=self.high_symmetry_labels)

        if disp_source == 'band':
            self.dispersion_select_index = self._parse_select_index(self.norbs, 'bands')
        elif disp_source == 'phonon':
            self.dispersion_select_index = self._parse_select_index(3 * self.natoms, 'branches')
        elif disp_source == 'epc':
            self._set('epc_path_fix_k', validate_numpy_array, 'dispersion', 5092, (3, ), float)
            if self.dispersion_select_index:
                self.dispersion_select_index = self._extract_dispersion_select_index(self.dispersion_select_index, disp_source)
            else:
                raise InputParameterError(
                    5091, 'dispersion', 'dispersion_select_index',
                    f"The dispersion_select_index must be specified, while using '{disp_source}' calcultion mode."
                )

        VALID_SAVER = {
            'none', 'ascii'
        }
        self.save_eigen_vectors = str(self.save_eigen_vectors).lower()
        if self.save_eigen_vectors not in VALID_SAVER:
            raise InputParameterError(
                5093, 'dispersion', 'save_eigen_vectors',
                f"The format of {self.save_eigen_vectors} is not supported."
            )
        
        if self.rank == 0:
            np.savetxt(os.path.join(self.outdir, 'high_symmetry_path.dat'),
                       self.high_symmetry_k_vecs, fmt="%.10f", delimiter="    ",
                       header=f"nks = {len(self.high_symmetry_k_vecs)}")

    def _initial_dos(self):
        """
        Initialize dos block.
        """
        self.dos_qp = str(self.dos_qp).lower()
        if self.dos_qp not in ('electron', 'phonon'):
            raise InputParameterError(5101, 'dos', 'dos_qp', f"The {self.dos_qp} is not supported in DOS calculations.")

        self._set('dos_npoints', validate_positive_int, 'dos', 5102)
        self._set('dos_erange', validate_num_list, 'dos', 5103, 2)
        self._set('dos_mesh_size', validate_positive_int_list, 'dos', 5104, 3)
        self._set('dos_smear', validate_positive_num, 'dos', 5105)
        self._set('dos_gauss_type', validate_int, 'dos', 5106)

        self.dos_erange = np.array(self.dos_erange, dtype=float) * QTcts.EVtoHARTREE
        self.dos_smear *= QTcts.MEVtoHARTREE
        self.inv_dos_smear = 1.0 / self.dos_smear

    def _initial_epc(self):
        """
        Initialize epc block.
        """
        self._set('grad_mat_path',    validate_readable_path, 'epc', 5141)
        self._set('mat_info_sc_path', validate_readable_path, 'epc', 5142)
        self._set('cell_range_cut',   validate_int_list,      'epc', 5143, 3)

        try:
            self.mat_info_sc = np.load(self.mat_info_sc_path, allow_pickle=True).item()
            self.cell_shift_array_reduced = self.mat_info_sc['cell_shift_array'] # shape: (ncells, 3)
            self.cell_index_map_reduced = self.mat_info_sc['cell_index_map'] # len: (ncells,) # index each cell_shift
            self.p2s_indices_reduced = self.mat_info_sc['p2s_indices'] # shape: (natoms_uc,)
            self.s2u_list_reduced = self.mat_info_sc['s2u_list'] # shape: (natoms_sc,)
            self.cell_shift_of_each_atom_in_sc = self.mat_info_sc['cell_shift_of_each_atom'] # shape: (natoms_sc,3)
        except Exception as e:
            raise InputParameterError(5142, 'epc', 'mat_info_sc_path', f"Fail to read {self.mat_info_sc_path!r}.\n{e}")
        self.cell_cut_list = []
        for icell, cell_shift in enumerate(self.cell_shift_array_reduced):
            if (np.abs(cell_shift[0])<self.cell_range_cut[0]) and (np.abs(cell_shift[1])<self.cell_range_cut[1]) and (np.abs(cell_shift[2])<self.cell_range_cut[2]):
                self.cell_cut_list.append(icell)
        self.cell_cut_array = np.array(self.cell_cut_list)

        try:
            if self.read_sparse_grad_mat:
                self._log(f"Loading sparse gradient matrix from {self.grad_mat_path}...")
                self.grad_mat_raw = np.load(self.grad_mat_path, allow_pickle=True)
                cell_cut_set = set(self.cell_cut_list)
                if self.spinor_mode == 'collinear':
                    self.grad_mat = [[], []]
                    for ispin in range(2):
                        self.grad_mat[ispin] = {key: value for key, value in self.grad_mat_raw[ispin].items() if ((key[0] in cell_cut_set) and (key[1] in cell_cut_set))}
                else:
                    self.grad_mat = {key: value for key, value in self.grad_mat_raw.item().items() if ((key[0] in cell_cut_set) and (key[1] in cell_cut_set))}
            else:
                self._log(f"Loading dense gradient matrix from {self.grad_mat_path}...")
                if self.spinor_mode == 'collinear':
                    self.grad_mat = np.load(self.grad_mat_path)[:,self.cell_cut_array[:,None], self.cell_cut_array[None,:]]
                else:
                    self.grad_mat = np.load(self.grad_mat_path)[self.cell_cut_array[:,None], self.cell_cut_array[None,:]]
        except Exception as e:
            raise InputParameterError(5141, 'epc', 'grad_mat_path', f"Fail to read {self.grad_mat_path!r}.\n{e}")
                        
        self.nbr_shift_of_cell_sc = np.einsum('ni, ij -> nj', self.cell_shift_array_reduced, self.graph_data.latt) # shape: (ncells, 3)

        if self.apply_correction or self.quadrupole_correction:
            self._set('q_cut', validate_num, 'epc', 5145)
            if self.LRC_taylor_order in (0, 1):
                self._log(f"The long-range correction is calculated using a Taylor expansion up to order {self.LRC_taylor_order}.")
            else:
                raise InputParameterError(5144, 'epc', 'LRC_taylor_order', f"The Taylor expansion order {self.LRC_taylor_order!r} is not supported.")
            self.q_cut = self.q_cut * np.linalg.norm(self.graph_data.rec_latt[0])
            self.ewald_param = 4.0 * QTcts.EWALD_SCALE * np.power(QTcts.TWOPI / np.linalg.norm(self.graph_data.latt[0]), 2)
            if self.system_2D:
                self.area_2D = np.linalg.norm(np.cross(self.graph_data.latt[0,:2], self.graph_data.latt[1,:2]))
                self._set('range_separation_length_L', validate_num, 'epc', 5146)
                self._set('vaccum_length',             validate_num, 'epc', 5147)
                self.range_separation_length_L = self.range_separation_length_L * QTcts.ANGtoBOHR
                self.vaccum_length             = self.vaccum_length             * QTcts.ANGtoBOHR
                self.macro_polarizability_inplane    = (self.DL[0:2, 0:2] - np.eye(2)) * self.vaccum_length / QTcts.FOURPI
                self.macro_polarizability_outofplane = (self.DL[2, 2] - 1.0)           * self.vaccum_length / QTcts.FOURPI
                self.offsets_333   = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
                self.G_vectors_333 = self.offsets_333 @ np.array(self.graph_data.rec_latt)
                # P_cell is required by _LR_correction_2D for the z-component projection
                self.graph_data.P_cell = self._P_cell_prepare()
            else:
                if self.LRC_taylor_order == 1:
                    if self.spinor_mode == 'none':
                        self.graph_data.P_cell = self._P_cell_prepare()
                    else:
                        raise InputParameterError(5144, 'epc', 'LRC_taylor_order', "The LRC_taylor_order = 1 is supported only for non-magnetic systems without SOC.")

    def _initial_transport(self):
        """
        Initialize transport block.
        """
        self._set('k_size',             validate_positive_int_list, 'transport', 5182, 3)
        self._set('q_size',             validate_positive_int_list, 'transport', 5183, 3)
        self._set('fermi_maxiter',      validate_positive_int,      'transport', 5185)
        self._set('fermi_shift',        validate_num,               'transport', 5186)
        self._set('temperature',        validate_positive_num,      'transport', 5187)
        self._set('frequency_cutoff',   validate_num,               'transport', 5188)
        self._set('smeark',             validate_positive_num,      'transport', 5189)
        self._set('smearq',             validate_positive_num,      'transport', 5190)
        self._set('gauss_type',         validate_int,               'transport', 5191)
        self._set('e_thr',              validate_positive_num,      'transport', 5192)
        self._set('is_hole',            validate_bool,              'transport', 5193)
        self._set('mode_resolved',      validate_bool,              'transport', 5194)


        try:
            if type(self.bands_indices) == list:
                self.bands_indices = np.array([each - int(1) for each in self.bands_indices], dtype="int64")
            elif type(self.bands_indices) == str:
                tmp = [np.arange(int(each.split('-')[0])-1, int(each.split('-')[1]), dtype="int64") \
                                                for each in self.bands_indices.split(',')]
                self.bands_indices = []
                for each in tmp:
                    self.bands_indices.extend(each.tolist())
                    self.bands_indices = np.unique(np.array(self.bands_indices, dtype="int64"))
            else:
                raise TypeError("The bands_indices should be a list of integers or a string.")
        except Exception as e:
            raise InputParameterError(5184, 'transport', 'bands_indices',
                                        f"The bands_indices must be a list of integers or a string in the format 'n1-n2,n3-n4,...', but got {self.bands_indices!r}.\n{e}")

        self.temperature = self.temperature * QTcts.KELVINtoHARTREE
        self.inv_smeartemp = 1.0 / self.temperature
        self.smeark = self.smeark * QTcts.MEVtoHARTREE
        self.inv_smeark = 1.0 / self.smeark
        self.smearq = self.smearq * QTcts.MEVtoHARTREE
        self.inv_smearq = 1.0 / self.smearq
        self.e_thr = self.e_thr * QTcts.MEVtoHARTREE
        self.frequency_cutoff = self.frequency_cutoff * QTcts.MEVtoHARTREE
        self.fermi_shift = self.fermi_shift * QTcts.EVtoHARTREE

    def _initial_conductivity(self):
        """
        Initialize conductivity block.
        """
        self._set('over_band_edge', validate_num, 'conductivity', 5201)
        self._set('ncarrier', validate_num, 'conductivity', 5207)

        self.MC_sampling = str(self.MC_sampling).lower()
        if self.MC_sampling in ('cauchy', 'uniform'):
            self._set('sampling_seed', validate_num,          'conductivity', 5205)
            self._set('nsamples',      validate_positive_int, 'conductivity', 5206)
            self._log(f"Using importance sampling method with {self.MC_sampling.upper()} sampling distribution...")
            if self.MC_sampling == 'cauchy':
                self._set('cauchy_scale', validate_positive_num, 'conductivity', 5204)
                self._log(f"The Cauchy scale is {self.cauchy_scale:10.6f}.\n")
        elif self.MC_sampling == 'none':
            self._log("Using uniform mesh sampling method...")
        else:
            raise InputParameterError(5202, 'conductivity', 'MC_sampling', f"The MC sampling method {self.MC_sampling!r} is not supported.")

        self.polar_split = str(self.polar_split).lower()
        if self.polar_split == 'polar':
            self._log("Considering only the polar part of the scattering rate...\n")
        elif self.polar_split == 'rmp':
            self._log("Considering only the remainder part of the scattering rate...\n")
        elif self.polar_split == 'none':
            pass
        else:
            raise InputParameterError(5203, 'conductivity', 'polar_split',
                                      f"The 'polar_split' must be one of 'polar', 'rmp', or 'none', but got {self.polar_split!r}.")

        self.mob_level = str(self.mob_level).lower()
        if self.mob_level == 'erta':
            self._log(f"Using {self.mob_level.upper()} method to calculate conductivity...\n")
        elif self.mob_level == 'ibte':
            self._set('ibte_error', validate_positive_num, 'conductivity', 5214)
            self._set('ibte_cycle', validate_positive_int, 'conductivity', 5215)
            self._log(f"Using {self.mob_level.upper()} method to calculate conductivity...\n")
            if self.MC_sampling != 'none':
                raise InputParameterError(5208, 'conductivity', 'mob_level',
                                          f"IBTE calculation with importance sampling method is not supported.")
        else:
            raise InputParameterError(5208, 'conductivity', 'mob_level', f"The {self.mob_level!r} approximation for conductivity is not supported.")

        if self.polar_rate_path:
            self._set('polar_rate_path', validate_readable_path, 'conductivity', 5209)
        if self.rmp_rate_path:
            self._set('rmp_rate_path', validate_readable_path, 'conductivity', 5210)
        if self.epc_rate_path:
            self._set('epc_rate_path', validate_readable_path, 'conductivity', 5211)            
        if self.scattering_channel_dir:
            self._set('scattering_channel_dir', validate_existed_dir, 'conductivity', 5213)

        self.carrier_density = 0.0
        self.over_band_edge = self.over_band_edge * QTcts.EVtoHARTREE
        if self.system_2D:
            self.ncarrier = self.ncarrier / (QTcts.CMtoBOHR ** 2)
        else:
            self.ncarrier = self.ncarrier / (QTcts.CMtoBOHR ** 3)
        # For collinear case, the carrier density is split to spin-up and spin-down electrons equally.
        if self.spinor_mode == 'collinear':
            self.ncarrier = self.ncarrier / 2.0

    def _initial_resistivity(self):
        """
        Initialize resistivity block.
        """
        self._set('ziman_nc',    validate_num,          'resistivity', 5221)
        self._set('omega_range', validate_numpy_array,  'resistivity', 5222, (2, ), float)
        self._set('omega_step',  validate_positive_num, 'resistivity', 5223)

        self.omega_range *= QTcts.MEVtoHARTREE
        self.omega_step *= QTcts.MEVtoHARTREE
        self.ziman_nc /= self.volume_uc

    def _initial_superconductivity(self):
        """
        Initialize superconductivity block.
        """
        self._set('mus',         validate_numpy_array,  'superconductivity', 5241, (3,), float)
        self._set('omega_range', validate_numpy_array,  'superconductivity', 5242, (2,), float)
        self._set('omega_step',  validate_positive_num, 'superconductivity', 5243)

        self.mus = np.arange(self.mus[0], self.mus[1], self.mus[2])
        self.omega_range *= QTcts.MEVtoHARTREE
        self.omega_step *= QTcts.MEVtoHARTREE

    def _initial_pomega(self):
        """
        Initialize pomega block.
        """
        self._set('omega_range', validate_numpy_array,  'pomega', 5441, (2,), float)
        self._set('omega_step',  validate_positive_num, 'pomega', 5442)

        self.mus = np.arange(self.mus[0], self.mus[1], self.mus[2])
        self.omega_range *= QTcts.MEVtoHARTREE
        self.omega_step *= QTcts.MEVtoHARTREE

    def _initial_breakdown(self):
        """
        Initialize breakdown block.
        """
        self._set('fbd_erange',         validate_numpy_array,  'breakdown', 5261, (3,), float)
        self._set('fbd_effective_mass', validate_positive_num, 'breakdown', 5263)
        self._set('fbd_exp_bandgap',    validate_positive_num, 'breakdown', 5264)

        if self.fbd_rate_file:
            self._set('fbd_rate_file', validate_readable_path, 'breakdown', 5262)

        self.fbd_erange *= QTcts.EVtoHARTREE
        self.fbd_exp_bandgap *= QTcts.EVtoHARTREE

    def _initial_boltzmann(self):
        """
        Initialize boltzmann block.
        """
        self._set('over_band_edge',  validate_num,          'boltzmann', 5281)
        self._set('ncarrier',        validate_num,          'boltzmann', 5282)
        self._set('temperature_lat', validate_positive_num, 'boltzmann', 5283)
        self._set('time_step',       validate_positive_num, 'boltzmann', 5284)
        self._set('energy_step',     validate_positive_num, 'boltzmann', 5285)
        self._set('num_step',        validate_positive_int, 'boltzmann', 5286)
        self._set('norm_dist',       validate_bool,         'boltzmann', 5287)
        self._set('output_step',     validate_positive_int, 'boltzmann', 5289)
        self._set('efield',          validate_numpy_array,  'boltzmann', 5290, (3,), float)

        self.boltz_solver = str(self.boltz_solver).lower()
        if self.boltz_solver in ('euler', 'rk4'):
            self._log(f"Using {self.boltz_solver.upper()} solver to solve real-time Boltzmann transport equation...\n")
        else:
            raise InputParameterError(5288, 'boltzmann', 'boltz_solver', f"The rt-BTE solver {self.boltz_solver!r} is not supported.")

        self.carrier_density = 0.0
        self.over_band_edge *= QTcts.EVtoHARTREE
        if self.system_2D:
            self.ncarrier /= (QTcts.CMtoBOHR ** 2)
        else:
            self.ncarrier /= (QTcts.CMtoBOHR ** 3)
        self.temperature_lat *= QTcts.KELVINtoHARTREE
        self.time_step *= (1.0E-15 * QTcts.HARTREE / QTcts.HBAR_EV)
        self.energy_step *= QTcts.MEVtoHARTREE
        if np.linalg.norm(self.efield) >= QTcts.TENPM5:
            self.adv_cal = True
        else:
            self.adv_cal = False
        efield_str = "[" + ", ".join("{:.4f}".format(x) for x in self.efield) + "]"
        self._log('The strength of external electric field (V/cm):', efield_str)
        self.efield *= (100 / (QTcts.HARTREEtoEV / QTcts.BOHRtoM))

    def _initial_self_energy(self):
        """
        Initialize self-energy block.
        """
        self._set('SE_erange',      validate_numpy_array, 'self-energy', 5302, (3,), float)
        self._set('over_band_edge', validate_num,         'self-energy', 5303)
        self._set('ncarrier',       validate_num,         'self-energy', 5304)

        self.SE_type = str(self.SE_type).lower()
        if self.SE_type in ('electron', 'phonon'):
            self._log(f"Calculate the self-energy for {self.SE_type}.\n")
        else:
            raise InputParameterError(5301, 'self-energy', 'SE_type', f"Self-energy cannot be calculated for {self.SE_type!r}.")

        if self.epc_se_path:
            self._set('epc_se_path', validate_readable_path, 'self-energy', 5305)
        if self.disp_path:
            self._set('disp_path', validate_readable_path, 'self-energy', 5307)

        if self.cal_mode == 'band':
            self.SE_type = 'electron'
        elif self.cal_mode == 'phonon':
            self.SE_type = 'phonon'
        self.SE_erange *= QTcts.EVtoHARTREE
        if self.system_2D:
            self.ncarrier = self.ncarrier / (QTcts.CMtoBOHR ** 2)
        else:
            self.ncarrier = self.ncarrier / (QTcts.CMtoBOHR ** 3)
        # For collinear case, the carrier density is split to spin-up and spin-down electrons equally.
        if self.spinor_mode == 'collinear':
            self.ncarrier = self.ncarrier / 2.0
        self.over_band_edge *= QTcts.EVtoHARTREE

    def _initial_fermi_surface(self):
        """
        Initialize fermi_surface block.
        """
        self._set('mesh_file', validate_readable_path, 'fermi_surface', 5321)

    def _print_banner(self, title: str, width: int = 100):
        """Print a centered '#' banner for a calculation mode."""
        title_str = f' {title} '
        left = (width - len(title_str)) // 2
        right = width - len(title_str) - left
        self._log('#' * left + title_str + '#' * right)

    def _init_scattering_blocks(self, config: dict):
        """
        Parse and initialize scattering-related blocks.

        Args:
            config (dict): The config dict.
            supported (tuple): Supported scattering sources for this calculation mode.
        """
        # Parse and initialize phonon and electron-phonon coupling blocks.
        self._parse_input_optional(config=config, block_name='phonon')
        self._initial_phonon()
        self._parse_input_optional(config=config, block_name='epc')
        self._initial_epc()
        config.pop('phonon', 0)
        config.pop('epc', 0)

    def _parse_input(self, config: dict):
        """
        Parse input config in terms of cal_mode.

        Args:
            config (dict): The config dict.
        """

        if 'advanced' in config:
            self._parse_input_optional(config=config, block_name='advanced')
            self._initial_advanced()
            config.pop('advanced')

        self._parse_input_basic(config=config)
        self._initial_basic()
        config.pop('basic')

        if self.cal_mode == 'dos':
            self._print_banner('DOS Calculation')
            self._parse_input_optional(config=config, block_name='dos')
            self._initial_dos()
            config.pop('dos', 0)
            if self.dos_qp == 'electron':
                self._log("Calculate density of states for electron.")
            elif self.dos_qp == 'phonon':
                self._log("Calculate density of states for phonon.")
                self._parse_input_optional(config=config, block_name='phonon')
                self._initial_phonon()
                config.pop('phonon', 0)
            
        elif self.cal_mode == 'boltzmann':
            self._print_banner('Boltzmann Calculation')
            self._parse_input_optional(config=config, block_name='transport')
            self._init_scattering_blocks(config)
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='boltzmann')
            self._initial_boltzmann()
            config.pop('transport', 0)
            config.pop('boltzmann', 0)

        elif self.cal_mode == 'breakdown':
            self._print_banner('Breakdown Calculation')
            self._init_scattering_blocks(config)
            self._parse_input_optional(config=config, block_name='transport')
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='breakdown')
            self._initial_breakdown()
            config.pop('transport', 0)
            config.pop('breakdown', 0)

        elif self.cal_mode == 'conductivity':
            self._print_banner('Conductivity Calculation')
            self._parse_input_optional(config=config, block_name='transport')
            self._init_scattering_blocks(config)
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='conductivity')
            self._initial_conductivity()
            config.pop('transport', 0)
            config.pop('conductivity', 0)

        elif self.cal_mode == 'superconductivity':
            self._print_banner('Superconductivity Calculation')
            self._parse_input_optional(config=config, block_name='transport')
            self._init_scattering_blocks(config)
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='superconductivity')
            self._initial_superconductivity()
            config.pop('transport', 0)
            config.pop('superconductivity', 0)

        elif self.cal_mode == 'pomega':
            self._print_banner('Pomega Calculation')
            self._parse_input_optional(config=config, block_name='transport')
            self._init_scattering_blocks(config)
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='pomega')
            self._initial_pomega()
            config.pop('transport', 0)
            config.pop('pomega', 0)

        elif self.cal_mode == 'resistivity':
            self._print_banner('Resistivity Calculation')
            self._parse_input_optional(config=config, block_name='transport')
            self._init_scattering_blocks(config)
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='resistivity')
            self._initial_resistivity()
            config.pop('transport', 0)
            config.pop('resistivity', 0)

        elif self.cal_mode == 'band':
            self._print_banner('Band Calculation')
            self._parse_input_optional(config=config, block_name='dispersion')
            self._initial_dispersion()
            if self.disp_renormalize:
                self._parse_input_optional(config=config, block_name='transport')
                self._initial_transport()
                self._init_scattering_blocks(config)
                self._parse_input_optional(config=config, block_name='self-energy')
                self._initial_self_energy()
                config.pop('transport', 0)
                config.pop('self-energy', 0)
            elif self.disp_nesting:
                self._parse_input_optional(config=config, block_name='transport')
                self._initial_transport()
                config.pop('transport', 0)
            config.pop('dispersion', 0)

        elif self.cal_mode == 'phonon':
            self._print_banner('Phonon Calculation')
            self._parse_input_optional(config=config, block_name='phonon')
            self._initial_phonon()
            self._parse_input_optional(config=config, block_name='dispersion')
            self._initial_dispersion()
            if self.disp_renormalize:
                self._parse_input_optional(config=config, block_name='epc')
                self._initial_epc()
                self._parse_input_optional(config=config, block_name='transport')
                self._initial_transport()
                self._parse_input_optional(config=config, block_name='self-energy')
                self._initial_self_energy()
                config.pop('epc', 0)
                config.pop('transport', 0)
                config.pop('self-energy', 0)
            if self.disp_lambda:
                self._parse_input_optional(config=config, block_name='epc')
                self._initial_epc()
                self._parse_input_optional(config=config, block_name='transport')
                self._initial_transport()
                config.pop('epc', 0)
                config.pop('transport', 0)
            config.pop('phonon', 0)
            config.pop('dispersion', 0)

        elif self.cal_mode == 'epc':
            self._print_banner('EPC Calculation')
            self._parse_input_optional(config=config, block_name='transport')
            self._init_scattering_blocks(config)
            self._initial_transport()
            self._parse_input_optional(config=config, block_name='dispersion')
            self._initial_dispersion()
            config.pop('transport', 0)
            config.pop('dispersion', 0)
        
        elif self.cal_mode == 'fermi_surface':
            self._print_banner('Fermi Surface Calculation')
            self._parse_input_optional(config=config, block_name='fermi_surface')
            self._initial_fermi_surface()
            config.pop('fermi_surface', 0)

        if self.rank == 0:
            for key in config:
                print(f"Ignore block {key}.")

    def _parse_input_basic(self, config: dict):
        """
        Parse the basic block of input configuration.

        Extracts required parameters from the 'basic' block of the configuration
        dictionary and assigns them as instance attributes. Fields 'cal_mode' and
        'Ham_type' are normalized to lowercase.

        Args:
            config (dict): The full configuration dictionary, which must contain
                a 'basic' key with the required subfields.

        Raises:
            InputParameterError: If the 'basic' block is missing (code 5001),
                or if any required field is missing inside it (code 5002).

        Attributes set:
            cal_mode (str): Calculation mode (lowercased).
            graph_data_path_uc (str): Path to the graph data for the unit cell.
            nao_max (int): Maximum number of numerical atomic orbitals.
            Ham_type (str): Hamiltonian type (lowercased).
            outdir (str): Output directory path.
        """
        if 'basic' not in config:
            raise InputParameterError(5001, 'basic')

        config_basic = config['basic']

        required_fields = {
            'cal_mode':           lambda v: v.lower(),
            'graph_data_path_uc': None,
            'nao_max':            None,
            'Ham_type':           lambda v: v.lower(),
            'outdir':             None,
        }

        for key, transform in required_fields.items():
            try:
                value = config_basic[key]
            except KeyError:
                raise InputParameterError(5001, 'basic', key, f"Cannot find {key} in basic blcok.")
            setattr(self, key, transform(value) if transform else value)

    def _parse_input_optional(self, config: dict, block_name: str):
        """
        Parse an optional block of input configuration.

        Reads key-value pairs from the specified optional block and assigns
        recognized parameters as instance attributes. Unrecognized parameters
        are silently ignored (with a warning printed on rank 0).

        Args:
            config (dict): The full configuration dictionary.
            block_name (str): The name of the optional block to parse
                (e.g., 'solver', 'output').

        Notes:
            Only parameters listed in ``calculator_default_parameters[block_name]``
            are accepted. Any parameter not in that registry is skipped and a
            warning is printed on the root process (rank 0).
        """
        for property_name, value in config.get(block_name, {}).items():
            if property_name in calculator_default_parameters[block_name]:
                setattr(self, property_name, value)
            elif self.rank == 0:
                print(f"Ignore {property_name} in {block_name} block.")
