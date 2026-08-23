"""Mechanically extracted QTrans methods."""

from collections import OrderedDict
from pathlib import Path
from phonopy.interface.vasp import write_vasp
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from qtrans.core.constants import QTcts
from qtrans.core.defaults import preparer_default_parameters
from qtrans.core.errors import InputParameterError
from qtrans.core.logging import print_banner
from qtrans.data.basis import basis_def_dicts
from qtrans.io.abacus import read_abacus_input
from qtrans.io.abacus import write_abacus_input
from qtrans.io.filesystem import check_path_exist
from qtrans.io.openmx import read_openmx_input
from qtrans.io.openmx import write_openmx_input
from qtrans.io.structure import read_structure_from_graph_data
from qtrans.io.structure import read_xsf
from qtrans.numerics.arrays import average_close_values
from qtrans.preparer.registry import PREPARATION_MODES
import copy
import glob
import numpy as np
import os
import phonopy
import pickle
try:
    import natsort
except ImportError:
    natsort = None


class _PreparerConfigMixin:
    def _parse_input(self, config:dict):
        """
        Parse input config in terms of cal_mode.

        Args:
            config (EasyDict): The config dict.
        """
        if 'basic' in config.keys():
            self._parse_input_basic(config=config)
            config.pop('basic')
        else:
            raise InputParameterError(1001, 'basic')
        
        # Processing different block accroding to the prep_mode
        if self.prep_mode == 'vasp-ph':
            if self.rank == 0:
                print_banner('VASP Phonon Preparation')
            self._parse_input_optional(config, "software_path")
            self._initial_software_path()
            self._parse_input_optional(config, "material")
            self._initial_material()
            self._parse_input_optional(config, "scf_params")
            self._initial_scf_params()
            self._parse_input_optional(config, "phonon_prep")
            self._initial_phonon_prep()
            config.pop('software_path', 0)
            config.pop('material', 0 )
            config.pop('scf_params', 0)
            config.pop('phonon_prep', 0)
        elif self.prep_mode == 'grad-epc':
            if self.rank == 0:
                print_banner('EPC Gradient Matrix Preparation')
            self._parse_input_optional(config, "software_path")
            self._initial_software_path()
            self._parse_input_optional(config, "material")
            self._initial_material()
            self._parse_input_optional(config, "scf_params")
            self._initial_scf_params()
            self._parse_input_optional(config, "grad_epc")
            self._initial_grad_epc()
            config.pop('software_path', 0)
            config.pop('material', 0 )
            config.pop('scf_params', 0)
            config.pop('grad_epc', 0)
        elif self.prep_mode == 'graph-pack':
            if self.rank == 0:
                print_banner('Graph Data Packing')
            self._parse_input_optional(config, "software_path")
            self._initial_software_path()
            self._parse_input_optional(config, "material")
            self._initial_material()
            self._parse_input_optional(config, "graph_pack")
            self._initial_graph_pack()
            config.pop('software_path', 0)
            config.pop('material', 0 )
            config.pop('graph_pack', 0)

        if self.rank == 0: 
            for key in config.keys():
                print(f"Ignore block {key}.")

    def _parse_input_basic(self, config:dict):
        """
        Parse the basic block of input config.

        Args:
            config (EasyDict): The config dict.
        """
        config_basic = config['basic']
        self.prep_mode = config_basic['prep_mode'].lower()
        self.outdir = config_basic['outdir']
        check_path_exist(self.outdir, check_type="dir", create_if_missing=True)
        self.outdir = os.path.abspath(self.outdir)
        if self.prep_mode not in PREPARATION_MODES:
            raise InputParameterError(1002, 'basic', 'prep_mode')

    def _parse_input_optional(self, config:dict, block_name:str):
        """
        Parse an optional block of input config.

        Args:
            config (EasyDict): The config dict.
            block_name (str): The name of the optional block.
        """
        if block_name in config.keys():
            for property_name, value in config[block_name].items():
                if property_name in preparer_default_parameters[block_name].keys():
                    setattr(self, property_name, value)
                else:
                    if self.rank == 0:
                        print(f"Ignore {property_name} in {block_name} block.")

    def _initial_software_path(self):
        # check software
        self.software_subable = dict()
        # VASP
        self.vasp_path = os.path.abspath(self.vasp_path)
        _, is_subable = check_path_exist(self.vasp_path, check_access='rx')
        self.software_subable['vasp'] = is_subable
        # vaspkit
        self.vaspkit_path = os.path.abspath(self.vaspkit_path)
        _, is_subable = check_path_exist(self.vaspkit_path, check_access='rx')
        self.software_subable['vaspkit'] = is_subable
        # OpenMX
        self.openmx_path = os.path.abspath(self.openmx_path)
        _, is_subable = check_path_exist(self.openmx_path, check_access='rx')
        self.software_subable['openmx'] = is_subable
        # openmx_postprocess
        self.openmx_pp_path = os.path.abspath(self.openmx_pp_path)
        _, is_subable = check_path_exist(self.openmx_pp_path, check_access='rx')
        self.software_subable['openmx_pp'] = is_subable
        # read_openmx
        self.openmx_read_path = os.path.abspath(self.openmx_read_path)
        _, is_subable = check_path_exist(self.openmx_read_path, check_access='rx')
        self.software_subable['openmx_read'] = is_subable
        # ABACUS
        self.abacus_path = os.path.abspath(self.abacus_path)
        _, is_subable = check_path_exist(self.abacus_path, check_access='rx')
        self.software_subable['abacus'] = is_subable
        # abacus_postprocess
        self.abacus_pp_path = os.path.abspath(self.abacus_pp_path)
        _, is_subable = check_path_exist(self.abacus_pp_path, check_access='rx')
        self.software_subable['abacus_pp'] = is_subable
        # JX
        self.jx_path = os.path.abspath(self.jx_path)
        _, is_subable = check_path_exist(self.jx_path, check_access='rx')
        self.software_subable['jx'] = is_subable
        # check pseudo-potential
        # VASP Pseudo-potential
        self.pot_reachable = dict()
        self.vasp_pot_dir = os.path.abspath(self.vasp_pot_dir)
        is_exist = check_path_exist(self.vasp_pot_dir, check_type='dir')
        self.pot_reachable['vasp'] = is_exist
        # Openmx  Pseudo-potential
        self.openmx_dftdata_dir = os.path.abspath(self.openmx_dftdata_dir)
        is_exist = check_path_exist(self.openmx_dftdata_dir, check_type='dir')
        self.pot_reachable['openmx'] = is_exist
        # Abacus  Pseudo-potential
        self.abacus_dftdata_dir = os.path.abspath(self.abacus_dftdata_dir)
        is_exist = check_path_exist(self.abacus_dftdata_dir, check_type='dir')
        self.pot_reachable['abacus_pot'] = is_exist
        # Abacus  Pseudo-potential
        self.abacus_orb_dir = os.path.abspath(self.abacus_orb_dir)
        is_exist = check_path_exist(self.abacus_orb_dir, check_type='dir')
        self.pot_reachable['abacus_orb'] = is_exist
        # print situation
        if self.rank == 0:
            print("Software Subable:")
            for key, value in self.software_subable.items():
                print(f"{key:<10} : {value}")
            print()
            print("Pseudo-potential reachable:")
            for key, value in self.pot_reachable.items():
                print(f"{key:<10} : {value}")
            print()

    def _initial_material(self):
        # software and orbits definition
        self.DFT_software:str = self.DFT_software.lower()
        self.magnetic_tolerance:float = self.magnetic_tolerance
        self.moment_similar_tolerence:float = self.moment_similar_tolerence
        self.xc_functional:str = self.xc_functional.upper()
        if self.DFT_software not in ['openmx', 'abacus']:
            raise InputParameterError(1008, "material", "DFT_software")
        self.basis_label = f"{self.DFT_software}_{self.nao_max}"
        if self.basis_label not in basis_def_dicts.keys():
            raise InputParameterError(1009, "material", "nao_max")
        if self.rank == 0:
            print(f"Using {self.DFT_software.upper()} with a {self.nao_max}-orbital definition dictionary.\n")
        if self.DFT_software == 'openmx':
            if self.xc_functional in ['LDA', 'GGA-PBE', 'LSDA-CA']:
                if self.rank == 0:
                    print(f"Using {self.xc_functional} Functional.")
            else:
                raise InputParameterError(1009, "material", "xc_functional")
        # spinor type
        self.magnetic_type:str = self.magnetic_type.lower()
        self.spinor_mode = 'none'
        if self.soc_switch:
            self.spinor_mode = 'nc'
            if self.magnetic_type == 'collinear':
                raise InputParameterError(1007, "material", "magnetic_type")
            if self.rank == 0:
                print("Dealing with a SOC system.")
                print(f"Dealing with a {self.magnetic_type.upper()} magnetic system.\n")
        else:
            if self.magnetic_type == 'collinear':
                self.spinor_mode = 'collinear'
            elif self.magnetic_type == 'nc':
                self.spinor_mode = 'nc'
            if self.magnetic_type in ['none', 'collinear', 'nc']:
                if self.rank == 0:
                    print(f"Dealing with a {self.magnetic_type.upper()} magnetic system.\n")
            else:
                raise InputParameterError(1007, "material", "magnetic_type")
        # structure
        if self.struct_path is not None:
            if not check_path_exist(self.struct_path):
                raise InputParameterError(1003, "material", "struct_path")
            try:
                _, ext = os.path.splitext(self.struct_path)
                ext = ext.lower()
                self.atomic_u_enhances_uc = None
                self.orbital_angles = None
                self.atomic_mag_constrained = None
                if self.DFT_software == 'abacus' and os.path.basename(self.struct_path) == "STRU":
                    atomic_symbols, latt, atomic_positions, magmoms_uc = \
                        read_abacus_input(self.struct_path)
                    self.unitcell = Structure(latt, atomic_symbols, atomic_positions, coords_are_cartesian=True)
                elif ext == '.xsf':
                    latt, atomic_symbols, atomic_positions, magmoms_uc = read_xsf(self.struct_path)
                    self.unitcell = Structure(latt, atomic_symbols, atomic_positions, coords_are_cartesian=True)
                elif ext == '.dat':
                    atomic_symbols, latt, atomic_positions, magmoms_uc, \
                        self.orbital_angles, self.atomic_mag_constrained, self.atomic_u_enhances_uc = \
                        read_openmx_input(self.struct_path)
                    self.unitcell = Structure(latt, atomic_symbols, atomic_positions, coords_are_cartesian=True)
                elif ext == '.npz':
                    atomic_symbols, latt, atomic_positions, magmoms_uc = read_structure_from_graph_data(self.struct_path)
                    self.unitcell = Structure(latt, atomic_symbols, atomic_positions, coords_are_cartesian=True)
                else:
                    self.unitcell = Structure.from_file(self.struct_path)
                    magmoms_uc = np.zeros((len(self.unitcell.species), 3))

                if self.spinor_mode == 'collinear':
                    spin_length = np.abs(magmoms_uc[:, 2])
                    is_magnetic = spin_length > self.magnetic_tolerance
                    spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
                    self.magmoms_uc = np.zeros_like(magmoms_uc, dtype=float)
                    self.magmoms_uc[is_magnetic, 2] = spin_length[is_magnetic] * np.sign(magmoms_uc[is_magnetic, 2])
                elif self.spinor_mode == 'nc':
                    spin_length = np.linalg.norm(magmoms_uc, axis=1)
                    is_magnetic = spin_length > self.magnetic_tolerance
                    self.magmoms_uc = np.zeros_like(magmoms_uc, dtype=float)
                    self.magmoms_uc[is_magnetic, :] = magmoms_uc[is_magnetic, :] / spin_length[is_magnetic][:, None]
                    spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
                    self.magmoms_uc[is_magnetic, :] = self.magmoms_uc[is_magnetic, :] * spin_length[is_magnetic][:, None]
                    if self.rank == 0:
                        print(f"Structure Loaded from {self.struct_path}")
                elif self.spinor_mode == 'none':
                    self.magmoms_uc = magmoms_uc
            except Exception as e:
                print(e)
                raise InputParameterError(1003, "material", "struct_path")
            sga = SpacegroupAnalyzer(self.unitcell, symprec=self.symm_tol)
            sg_symbol = sga.get_space_group_symbol()
            sg_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
            if self.rank == 0:
                print("Structure Information")
                print(f"{'Crystal system':20}: {crystal_system}")
                print(f"{'Space group':20}: {sg_symbol} [{sg_number}]")
            if self.std_symm_cell:
                self.unitcell = sga.get_primitive_standard_structure()
            self.unitcell_latt = self.unitcell.lattice.matrix * QTcts.ANGtoBOHR
            self.atoms_pos_cart = self.unitcell.cart_coords * QTcts.ANGtoBOHR
            self.atoms_pos_frac = self.unitcell.frac_coords
            self.atoms_label = self.unitcell.species
            self.atoms_label = [each.name for each in self.atoms_label]
            self.natoms_uc = len(self.atoms_label)
            if self.rank == 0:
                print("Lattice vectors (ang):")
                for vec in self.unitcell_latt*QTcts.BOHRtoANG:
                    print("  {:10.6f}  {:10.6f}  {:10.6f}".format(*vec))
                print()
                print(f"  {'Index':<6} {'Element':<8}  {'Frac_x':>10}  {'Frac_y':>10}  {'Frac_z':>10}")
                print('-' * 60)
                for iatom in range(self.natoms_uc):
                    print("    {:<6}  {:<8}  {:10.5f} {:10.5f} {:10.5f}".format(iatom+1, self.atoms_label[iatom], *self.atoms_pos_frac[iatom]))
                print('-' * 60)
                print()
                print(f"  {'Index':<6} {'Element':<8}  {'Cart_x (ang)':>12}  {'Cart_y (ang)':>12}  {'Cart_z (ang)':>12}")
                print('-' * 60)
                for iatom in range(self.natoms_uc):
                    print("    {:<6}  {:<8}  {:12.5f} {:12.5f} {:12.5f}".format(iatom+1, self.atoms_label[iatom], *(self.atoms_pos_cart[iatom]*QTcts.BOHRtoANG)))
                print('-' * 60)
                print()
                self.unitcell.to(fmt="poscar", filename=os.path.join(self.outdir, "QT.UNITCELL"))

    def _initial_scf_params(self):
        self.scf_rec_density = self.scf_rec_density * QTcts.ANGtoBOHR
        if type(self.scf_uc_kpt) == list:
            if len(self.scf_uc_kpt) != 3:
                raise InputParameterError(1005, "scf_params", "scf_uc_kpt")
            self.scf_uc_kpt = np.array(self.scf_uc_kpt)
        elif type(self.scf_uc_kpt) == str and (self.scf_uc_kpt.lower().strip()[0] == 'a'):
            latt_abc = np.array(self.unitcell.lattice.abc) * QTcts.ANGtoBOHR
            self.scf_uc_kpt = np.array([max(1, int(np.ceil(self.scf_rec_density / each))) for each in latt_abc])
            if self.rank == 0:
                print("Automatically generate k sampling grid for SCF calculation.")
        else:
            raise InputParameterError(1005, "scf_params", "scf_uc_kpt")
        if self.rank == 0:
            # already angstrom here
            rec_density = np.min(self.scf_uc_kpt * np.array(self.unitcell.lattice.abc))
            print(f"Reciprocal sampling density (nks*a) is {rec_density:6.2f}.")
            print()

    def _initial_phonon_prep(self):
        self.ph_sc_auto_length = self.ph_sc_auto_length * QTcts.ANGtoBOHR
        self.ph_sc_disp = self.ph_sc_disp * QTcts.ANGtoBOHR
        unitcell_phonopy = PhonopyAtoms(symbols=self.atoms_label,
                                        cell=self.unitcell_latt*QTcts.BOHRtoANG,
                                        scaled_positions=self.atoms_pos_frac)
        if type(self.ph_sc_matrix) == list:
            if len(self.ph_sc_matrix) != 3:
                raise InputParameterError(1004, "phonon_prep", "ph_sc_matrix")
            self.ph_sc_matrix = np.array(self.ph_sc_matrix)
        elif type(self.ph_sc_matrix) == str and (self.ph_sc_matrix.lower().strip()[0] == 'a'):
            latt_abc = np.array(self.unitcell.lattice.abc) * QTcts.ANGtoBOHR
            self.ph_sc_matrix = np.array([max(1, int(np.ceil(self.ph_sc_auto_length / each))) for each in latt_abc])
            if self.rank == 0:
                print("Automatically generate supercell matrix for phonon calculation.")
        else:
            raise InputParameterError(1004, "phonon_prep", "ph_sc_matrix")
        if self.rank == 0:
            print("Phonon Supercell:  {:<3} * {:<3} * {:<3}".format(*self.ph_sc_matrix))
            print()
        self.phonon = phonopy.Phonopy(unitcell_phonopy, self.ph_sc_matrix)
        self.phonon.generate_displacements(distance=self.ph_sc_disp*QTcts.BOHRtoANG)
        supercells = self.phonon.get_supercells_with_displacements()
        self.phonon_nstructs = len(supercells)
        index_dig = len(str(self.phonon_nstructs))
        sc_filenames = [os.path.join(self.outdir, f"POSCAR-QTph-{istruct+1:0{index_dig}d}") for istruct in range(self.phonon_nstructs)]
        # split supercells into each core
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(self.phonon_nstructs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        supercells_all = np.split(supercells, indices_or_sections=split_sections, axis=0)
        sc_filenames_all = np.split(sc_filenames, indices_or_sections=split_sections, axis=0)
        if supercells_all[self.rank].size>0:
            supercells = supercells_all[self.rank]
            sc_filenames =  sc_filenames_all[self.rank]
        else:
            supercells = []
            sc_filenames = []
        try:
            for istruct, name in enumerate(sc_filenames):
                write_vasp(name, supercells[istruct], direct=True)
        except Exception as e:
            raise IOError(f"Error(2001): Failed to write {name}: {e}")

    def _initial_grad_epc(self):
        self.dSon_delta_r = self.dSon_delta_r * QTcts.ANGtoBOHR
        self.epc_sc_auto_length = self.epc_sc_auto_length * QTcts.ANGtoBOHR
        self.epc_sc_disp = self.epc_sc_disp * QTcts.ANGtoBOHR
        atomic_numbers = [Element(each).Z for each in self.atoms_label]
        unique_atomic_numbers, atomic_inverse = np.unique(atomic_numbers, return_inverse=True)
        nelements = len(unique_atomic_numbers)
        if self.graph_data_sc_path:
            # calculate gradient matrix
            # find orb_mask
            basis_def = basis_def_dicts[self.basis_label]
            basis_list = [basis_def[key] for key in unique_atomic_numbers]
            basis_definition = np.zeros((nelements, self.nao_max), dtype=int)
            for iele, ele_orbits in enumerate(basis_list):
                basis_definition[iele, ele_orbits] = 1
            self.orb_mask = basis_definition[atomic_inverse].reshape(-1) # shape: (natoms_uc*nao_max, )
            self.norbs = int(np.sum(self.orb_mask))
            # find atomic_orb_len_list
            self.atomic_orb_len_list = []
            for iatom in range(self.natoms_uc):
                self.atomic_orb_len_list.append(len(basis_list[atomic_inverse[iatom]]))
            # load mat_info_sc
            if self.mat_info_sc_path:
                try:
                    self.mat_info_sc = np.load(self.mat_info_sc_path, allow_pickle=True).item()
                except:
                    raise InputParameterError(1010, "grad_epc", "mat_info_sc_path")
            else:
                raise InputParameterError(1010, "grad_epc", "mat_info_sc_path")
            # load graph_data_sc
            try:
                if self.graph_data_sc_path.endswith(".npz"):
                    graph_data = np.load(self.graph_data_sc_path, allow_pickle=True)['graph'].item()
                elif self.graph_data_sc_path.endswith(".pkl"):
                    with open(self.graph_data_sc_path, 'rb') as f:
                        graph_data = pickle.load(f)
                else:
                    raise ValueError("Unsupported graph data format")
                # in case of wrong index!
                self.graph_dataset = [graph_data[k] for k in sorted(graph_data)]
            except Exception as e:
                print(f"QTrans Failed to load graph data: {e}")
                raise InputParameterError(1011, "grad_epc", "graph_data_sc_path")
            if self.fd_method == 'forward' and len(self.graph_dataset) != self.natoms_uc*3+1:
                raise InputParameterError(1011, "grad_epc", "graph_data_sc_path")
            elif self.fd_method == 'central' and len(self.graph_dataset) != self.natoms_uc*3*2+1:
                raise InputParameterError(1011, "grad_epc", "graph_data_sc_path")
            # load pred_H_sc if use_prediction
            if self.use_prediction:
                try:
                    if self.spinor_mode == 'nc':
                        self.pred_H_sc = np.load(self.pred_H_sc_path).reshape(-1, self.nao_max * 2, self.nao_max * 2)
                    else:
                        self.pred_H_sc = np.load(self.pred_H_sc_path).reshape(-1, self.nao_max, self.nao_max)
                except:
                    raise InputParameterError(1012, "grad_epc", "pred_H_sc_path")
            
        else:
            # Generate perturbed supercells
            if type(self.epc_sc_matrix) == list:
                if len(self.epc_sc_matrix) != 3:
                    raise InputParameterError(1006, "grad_epc", "epc_sc_matrix")
                if not np.all((np.array(self.epc_sc_matrix) > 0) & (np.array(self.epc_sc_matrix) % 2 == 1)):
                    raise InputParameterError(1013, "grad_epc", "epc_sc_matrix")
                self.epc_sc_matrix = np.array(self.epc_sc_matrix)
            elif type(self.epc_sc_matrix) == str and (self.epc_sc_matrix.lower().strip()[0] == 'a'):
                latt_abc = np.array(self.unitcell.lattice.abc) * QTcts.ANGtoBOHR
                self.epc_sc_matrix = np.array([ 
                    n if n % 2 == 1 else n + 1
                    for n in (  
                        max(1, int(np.ceil(self.epc_sc_auto_length / each))) for each in latt_abc)])
                if self.rank == 0:
                    print("Automatically generate supercell matrix for electron-phonon-coupling gradient matrix calculation.")
            else:
                raise InputParameterError(1006, "grad_epc", "epc_sc_matrix")
            if self.fd_method == 'forward':
                num_supercells = 1 + 3 * self.natoms_uc
            elif self.fd_method == 'central':
                num_supercells = 1 + 3 * self.natoms_uc * 2
            else:
                raise InputParameterError(1018, "grad_epc", "fd_method")
            if self.rank == 0:
                print("Electron-Phonon Coupling Settings:")
                print("  Supercell Size          : {:<3} * {:<3} * {:<3}".format(*self.epc_sc_matrix))
                print("  Finite Displacement     : {:.6f} angstrom".format(self.epc_sc_disp*QTcts.BOHRtoANG))
                print("  Method of FD            :", self.fd_method)
                print()

            # make dir <perturb>
            perturb_folder_path = Path(os.path.join(self.outdir, 'perturb'))
            try:
                perturb_folder_path.mkdir(parents=False, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Error(3004): Failed to create {perturb_folder_path}: {e}")

            # expand cell and save supercell
            sc_atoms_dict = dict()
            sc_positions_dict = OrderedDict()
            if self.spinor_mode == 'collinear':
                sc_magmoms_dict = OrderedDict()
                sc_uenhances_dict = OrderedDict()
            elif self.spinor_mode == 'nc':
                sc_magmoms_dict = OrderedDict()
                sc_uenhances_dict = OrderedDict()
                sc_orbangles_dict = OrderedDict()
                sc_magconstrains_dict = OrderedDict()
            s2u_map = OrderedDict()
            cell_shift_map = OrderedDict()
            cell_index_map = OrderedDict()
            nbr_shift_of_cell = []
            cell_index = 0
            # unique_labels = np.unique(self.atoms_label) # Must keep caution of this !!!
            unique_temp, idx_temp = np.unique(self.atoms_label, return_index=True)
            unique_labels = unique_temp[np.argsort(idx_temp)]
            for each in unique_labels:
                sc_positions_dict[each] = []
                s2u_map[each] = []
                cell_shift_map[each] = []
                if self.spinor_mode == 'collinear':
                    sc_magmoms_dict[each] = []
                    sc_uenhances_dict[each] = []
                elif self.spinor_mode == 'nc':
                    sc_magmoms_dict[each] = []
                    sc_uenhances_dict[each] = []
                    sc_orbangles_dict[each] = []
                    sc_magconstrains_dict[each] = []
            Nrange = self.epc_sc_matrix // 2
            for i in range(-Nrange[0],Nrange[0]+1):
                for j in range(-Nrange[1],Nrange[1]+1):
                    for k in range(-Nrange[2],Nrange[2]+1):
                        cell_index_map[(i,j,k)] = cell_index # index each cell_shift
                        cell_index += 1
                        cell_shift_tmp = (i+Nrange[0])*self.unitcell_latt[0] \
                                        + (j+Nrange[1])*self.unitcell_latt[1] + (k+Nrange[2])*self.unitcell_latt[2]
                        nbr_shift_of_cell.append(cell_shift_tmp)
                        for iatom in range(self.natoms_uc):
                            this_label = self.atoms_label[iatom]
                            sc_positions_dict[this_label].append(self.atoms_pos_cart[iatom]+cell_shift_tmp)
                            s2u_map[this_label].append(iatom)
                            cell_shift_map[this_label].append((i,j,k))
                            if self.spinor_mode == 'collinear':
                                sc_magmoms_dict[this_label].append(self.magmoms_uc[iatom])
                                if self.DFT_software == 'openmx':
                                    sc_uenhances_dict[this_label].append(self.atomic_u_enhances_uc[iatom])
                            elif self.spinor_mode == 'nc':
                                sc_magmoms_dict[this_label].append(self.magmoms_uc[iatom])
                                if self.DFT_software == 'openmx':
                                    sc_orbangles_dict[this_label].append(self.orbital_angles[iatom])
                                    sc_uenhances_dict[this_label].append(self.atomic_u_enhances_uc[iatom])
                                    sc_magconstrains_dict[this_label].append(self.atomic_mag_constrained[iatom])
            sc_atoms_dict['chemical_symbols'] = [z for z in sc_positions_dict.keys() for _ in range(len(sc_positions_dict[z]))]
            sc_atoms_dict['positions'] = np.concatenate([sc_positions_dict[z] for z in sc_positions_dict.keys()], axis=0)
            sc_atoms_dict['cell'] = self.epc_sc_matrix[:,None] * self.unitcell_latt
            if self.spinor_mode == 'collinear':
                sc_atoms_dict['magmoms'] = np.concatenate([sc_magmoms_dict[z] for z in sc_magmoms_dict.keys()], axis=0)
                if self.DFT_software == 'openmx':
                    sc_atoms_dict['atomic_u_enhances'] = np.concatenate([sc_uenhances_dict[z] for z in sc_uenhances_dict.keys()], axis=0)
            elif self.spinor_mode == 'nc':
                sc_atoms_dict['magmoms'] = np.concatenate([sc_magmoms_dict[z] for z in sc_magmoms_dict.keys()], axis=0)
                if self.DFT_software == 'openmx':
                    sc_atoms_dict['orbital_angles'] = np.concatenate([sc_orbangles_dict[z] for z in sc_orbangles_dict.keys()], axis=0)
                    sc_atoms_dict['atomic_u_enhances'] = np.concatenate([sc_uenhances_dict[z] for z in sc_uenhances_dict.keys()], axis=0)
                    sc_atoms_dict['atomic_mag_constrained'] = np.concatenate([sc_magconstrains_dict[z] for z in sc_magconstrains_dict.keys()], axis=0)
            
            nbr_shift_of_cell = np.array(nbr_shift_of_cell) # shape: (ncells, 3)
            # get openmx input file names that need to generate
            index_dig = len(str(num_supercells - 1))
            QTrans_input_names = [os.path.join(perturb_folder_path, f"QTepc-{istruct:0{index_dig}d}.dat") for istruct in range(num_supercells)]
            # index 0 structure is the unperturbed supercell
            kpt_size = (np.ceil(np.array(self.scf_uc_kpt) / np.array(self.epc_sc_matrix))).astype(int)
            if self.rank == 0:
                if self.DFT_software == 'openmx':
                    write_openmx_input(QTrans_input_names[0],
                                    sc_atoms_dict, kpt_size, self.openmx_dftdata_dir,
                                    spinor_mode=self.spinor_mode, soc_switch=self.soc_switch, cwd='./',
                                    magnetic_tolerance=self.magnetic_tolerance, xc_func=self.xc_functional)
                elif self.DFT_software == 'abacus':
                    write_abacus_input(QTrans_input_names[0],
                                    sc_atoms_dict, kpt_size, spinor_mode=self.spinor_mode, soc_switch=self.soc_switch,
                                    pseudo_dir = self.abacus_dftdata_dir,orbital_dir=self.abacus_orb_dir)
            # Initialize the mapping between supercell and unitcell
            cell_shift_list = [] # len: (natoms_sc, 3)
            s2u_list = [] # len: (natoms_sc,)
            for z in cell_shift_map.keys():
                cell_shift_list += cell_shift_map[z]
                s2u_list += s2u_map[z]
            p2s_indices  = [index for (index, item) in enumerate(cell_shift_list) if item == (0,0,0)]
            # Save matrix information
            cell_shift_array = np.array(list(cell_index_map.keys()))
            cell_shift_of_each_atom = np.array(cell_shift_list)
            mat_info =  {'species':atomic_numbers,
                        'cell_shift_array':cell_shift_array, # shape: (ncells, 3)
                        'cell_index_map':cell_index_map, # len: (ncells,)
                        'p2s_indices': np.array(p2s_indices), # shape: (natoms_uc,)
                        's2u_list': np.array(s2u_list), # shape: (natoms_sc,)
                        'cell_shift_of_each_atom': cell_shift_of_each_atom # shape: (natoms_sc,)
                        }
            if self.rank == 0:
                np.save(os.path.join(perturb_folder_path, 'mat_info_sc.npy'), mat_info)
                # perturb the atoms in the (0,0,0) cell of the supercell
                # TODO: mpi write openmx input files
                vectors = np.identity(3) * self.epc_sc_disp
                for i, idx in enumerate(p2s_indices):
                    for j in range(3):
                        temp = copy.deepcopy(sc_atoms_dict)
                        if self.fd_method == 'forward':
                            # forward
                            temp['positions'][idx] += vectors[j]
                            istruct = 3*i + j + 1
                            if self.DFT_software == 'openmx':
                                write_openmx_input(QTrans_input_names[istruct],
                                                temp, kpt_size, self.openmx_dftdata_dir,
                                                spinor_mode=self.spinor_mode, soc_switch=self.soc_switch, cwd='./',
                                                magnetic_tolerance=self.magnetic_tolerance, xc_func=self.xc_functional)
                            elif self.DFT_software == 'abacus':
                                write_abacus_input(QTrans_input_names[istruct],
                                                temp, kpt_size,
                                                spinor_mode=self.spinor_mode, soc_switch=self.soc_switch,
                                                pseudo_dir = self.abacus_dftdata_dir,orbital_dir=self.abacus_orb_dir)
                        elif self.fd_method == 'central':
                            # forward
                            temp['positions'][idx] += vectors[j]
                            istruct = 3*i + j + 1
                            if self.DFT_software == 'openmx':
                                write_openmx_input(QTrans_input_names[istruct],
                                                temp, kpt_size, self.openmx_dftdata_dir,
                                                spinor_mode=self.spinor_mode, soc_switch=self.soc_switch, cwd='./',
                                                magnetic_tolerance=self.magnetic_tolerance, xc_func=self.xc_functional)
                            elif self.DFT_software == 'abacus':
                                write_abacus_input(QTrans_input_names[istruct],
                                                temp, kpt_size,
                                                spinor_mode=self.spinor_mode, soc_switch=self.soc_switch,
                                                pseudo_dir = self.abacus_dftdata_dir,orbital_dir=self.abacus_orb_dir)
                            # backward
                            temp['positions'][idx] -= vectors[j] * 2 # -vectors[j] * 1 back to pristine supercell
                            istruct = 3*i + j + 1 + 3 * self.natoms_uc
                            if self.DFT_software == 'openmx':
                                write_openmx_input(QTrans_input_names[istruct],
                                                temp, kpt_size, self.openmx_dftdata_dir,
                                                spinor_mode=self.spinor_mode, soc_switch=self.soc_switch, cwd='./',
                                                magnetic_tolerance=self.magnetic_tolerance, xc_func=self.xc_functional)
                            elif self.DFT_software == 'abacus':
                                write_abacus_input(QTrans_input_names[istruct],
                                                temp, kpt_size,
                                                spinor_mode=self.spinor_mode, soc_switch=self.soc_switch,
                                                pseudo_dir = self.abacus_dftdata_dir,orbital_dir=self.abacus_orb_dir)

    def _initial_graph_pack(self):
        self.pack_atomic_force = self.pack_atomic_force
        if self.DFT_software == 'abacus':
            if bool(self.abacus_scf_log):
                self.is_pred_graph = False
                self.with_nonscf = False
            else:
                raise InputParameterError(1014, "graph_pack", "abacus_scf_log")
        else:
            if bool(self.vbout_filename):
                self.is_pred_graph = False
                if bool(self.HSout_filename):
                    if bool(self.nonscf_filename):
                        self.with_nonscf = True
                    else:
                        self.with_nonscf = False
                else:
                    raise InputParameterError(1014, "graph_pack", "HSout_filename")
            else:
                self.is_pred_graph = True
                self.with_nonscf = True
                if not bool(self.nonscf_filename):
                    raise InputParameterError(1015, "graph_pack", "nonscf_filename")
                if not bool(self.stdin_filename):
                    raise InputParameterError(1016, "graph_pack", "stdin_filename")
            if self.is_pred_graph and self.pack_atomic_force:
                raise InputParameterError(1017, "graph_pack", "pack_atomic_force")

        self.data_folder = self.data_folder.strip()
        self.data_folder = os.path.join(self.data_folder, '')
        if self.rank == 0:
            data_folders = glob.glob(self.data_folder)
            data_folders = natsort.natsorted(data_folders)
            nfolders = len(data_folders)
            data_folders = self._check_files_for_graph(data_folders)
            nfolders_useful = len(data_folders)
            if nfolders == 0:
                raise InputParameterError(1013, "graph_pack", "data_folder")
            print(f"Packing {'non-scf' if self.is_pred_graph else 'scf'} graph data {'with' if self.with_nonscf else 'without'} non-scf Hamiltonian.")
            if self.pack_atomic_force:
                print("Packing also atomic forces into the graph data.")
            print(f"Found {nfolders} folders while packing graph data.")
            print(f"Inside {nfolders_useful} folders are useful.\n")
            indexed_folders = list(enumerate(data_folders))  # [(0, folder0), (1, folder1), ...]
            chunks = [[] for _ in range(self.rank_size)]
            for i, item in enumerate(indexed_folders):
                chunks[i % self.rank_size].append(item)
        else:
            chunks = None
        if self.comm is None:
            indexed_folders_local = chunks[0]
        else:
            indexed_folders_local = self.comm.scatter(chunks, root=0)
        self.data_folders_index_local = [ifolder for (ifolder, _) in indexed_folders_local]
        self.data_folders_local = [folder for (_, folder) in indexed_folders_local]
        # convert to abs path
        self.data_folders_local = [os.path.abspath(each) for each in self.data_folders_local]
