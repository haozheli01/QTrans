"""QTrans io.abacus helpers."""

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from qtrans.core.constants import QTcts
from qtrans.data.species import abacus_pao_dict
from qtrans.data.species import abacus_pot_pbe_dict
from typing import Union
import numpy as np
import re


def read_abacus_input(filepath: str):
    # return unit is angstrom, \mu_B and Hartree/Bohr
    def parse_orbitals(orbital_data: str) -> int:
        # Parse the orbital string and calculate the total number of orbitals for the species.
        s, p, d, f = 0, 0, 0, 0
        if 's' in orbital_data:
            s = int(re.findall(r'\d', orbital_data)[0]) * 1
        if 'p' in orbital_data:
            p = int(re.findall(r'\d', orbital_data)[1]) * 3
        if 'd' in orbital_data:
            d = int(re.findall(r'\d', orbital_data)[2]) * 5
        if 'f' in orbital_data:
            f = int(re.findall(r'\d', orbital_data)[3]) * 7
        return s + p + d + f
    
    def _process_atomic_positions(fp, line, num_orbitals):
        """
        Process atomic positions block and parse relevant data.
        Returns: pos_type (str), num_atoms_per_species (list), positions (list)
        """
        pos_type = line.strip().lower()
        num_atoms_per_species = []
        positions = []
        magmoms = []
        
        for is_ in range(len(num_orbitals)):
            # element
            while True:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() != '':
                    break
            # mag
            while True:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() != '':
                    break
            # num
            while True:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() != '':
                    na = int(line)
                    num_atoms_per_species.append(na)
                    break
            ia = 0
            while ia < na:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() == '':
                    continue
                tmp = line.split()
                positions.append([float(tmp[0]), float(tmp[1]), float(tmp[2])])
                mag_keywords = ['mag', 'magmom']
                mag_value = 0.0
                for key in mag_keywords:
                    if key in tmp:
                        idx = tmp.index(key)
                        if idx + 1 < len(tmp):
                            try:
                                mag_value = float(tmp[idx + 1])
                                break
                            except ValueError:
                                pass
                magmoms.append(mag_value)
                ia += 1
                
        return pos_type, num_atoms_per_species, positions, magmoms

    def convert_to_cartesian(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """
        Convert atomic positions from direct to Cartesian coordinates.
        """
        cartesian_positions = np.dot(positions, cell)
        return cartesian_positions

    species = []
    num_orbitals = []  # Number of orbitals per species.
    num_atoms_per_species = []  # Number of atoms per species.
    cell = []  # Lattice vectors.
    positions = []  # Atomic positions.
    latconst = 1.0  # Default lattice constant
    active_block = None
    pos_type = ''

    with open(filepath, 'r') as fp:
        for line in fp:
            line = line.split('//')[0].split('#')[0].strip()  # Remove comments and whitespace
            if not line:
                continue

            if 'ATOMIC_SPECIES' in line:
                active_block = 'ATOMIC_SPECIES'
            elif 'NUMERICAL_ORBITAL' in line:
                active_block = 'NUMERICAL_ORBITAL'
            elif 'LATTICE_CONSTANT' in line:
                active_block = 'LATTICE_CONSTANT'
            elif 'LATTICE_VECTORS' in line:
                active_block = 'LATTICE_VECTORS'
            elif 'ATOMIC_POSITIONS' in line:
                active_block = 'ATOMIC_POSITIONS'
                
            elif active_block == 'ATOMIC_SPECIES':
                species.append(line.split()[0])

            elif active_block == 'NUMERICAL_ORBITAL':
                # Parse the number of orbitals for each species
                # 'element_2s2p1d.orb'
                orbital_file = line.split()[0]
                orbital_data = orbital_file.split('.orb')[0].split('_')[-1]
                num = parse_orbitals(orbital_data)
                num_orbitals.append(num)

            elif active_block == 'LATTICE_CONSTANT':
                latconst = float(line.split()[0])

            elif active_block == 'LATTICE_VECTORS':
                lattice_vector = list(map(float, line.split()))
                cell.append(lattice_vector)

            elif active_block == 'ATOMIC_POSITIONS':
                pos_type, num_atoms_per_species, positions, magmoms = _process_atomic_positions(fp, line, num_orbitals)
                break 
    # After reading all lines, finalize the data
    cell = np.array(cell) * latconst * QTcts.BOHRtoANG # ang
    positions = np.array(positions)
    
    # Convert positions to Cartesian if they are in 'direct' format
    if pos_type == 'direct':
        positions = convert_to_cartesian(positions, cell) # ang
    else: 
        positions = positions * latconst * QTcts.BOHRtoANG # ang
    # Generate atomic numbers (Z values)
    species = [spec for spec, na in zip(species, num_atoms_per_species) for _ in range(na)]
    # Extend magmoms to three dimension...
    magmoms = np.array(magmoms)[:, np.newaxis]
    magmoms = magmoms * np.array([0, 0, 1])
    
    return species, cell, positions, magmoms


def write_abacus_input(filepath:str, structure:Union[Structure, dict], kpts:list,
                       spinor_mode:str='none', soc_switch:bool=False,
                       magmoms:np.ndarray=None, write_mode:str='STRU', stru_file:str='STRU',  kpoint_file:str='KPT',
                       pseudo_dir:str='./', orbital_dir:str='./', is_hse:bool=False, dSon_cal:bool=False):
    file_str = ""
    if type(structure) == dict:
        chemical_symbols = structure['chemical_symbols']
        # species = set(chemical_symbols) # there will be some random indices
        species = list(dict.fromkeys(chemical_symbols))
        positions = structure['positions'] * QTcts.BOHRtoANG
        cell = structure['cell'] * QTcts.BOHRtoANG
        # TODO: Better approach to make sure positions inside the cell
        unitcell = Structure(cell, chemical_symbols, positions, coords_are_cartesian=True)
        cell = unitcell.lattice.matrix
        positions = unitcell.cart_coords
        chemical_symbols = unitcell.species
        chemical_symbols = [each.name for each in chemical_symbols]
        if magmoms is None:
            magmoms = structure.get('magmoms')
    elif type(structure) == Structure:        
        chemical_symbols = structure.species
        chemical_symbols = [each.name for each in chemical_symbols]
        # species = set(chemical_symbols) # there will be some random indices
        species = list(dict.fromkeys(chemical_symbols))
        positions = structure.cart_coords
        cell = structure.lattice.matrix
        species = list(species)
        # species.sort()
    elif write_mode == 'KPT':
        structure = None
    else:
        raise RuntimeError(f"Failed to write ABACUS input file: unsupported write mode '{write_mode}' for the given structure. \
                           Please check the structure format and ensure the write mode is correct.")
    if write_mode == 'STRU':
        file_str = "ATOMIC_SPECIES\n"
        for iele, each in enumerate(species):
            file_str += "{:2s} {:8.4f}  {:s}\n".format(each, Element(each).atomic_mass, abacus_pot_pbe_dict[each])
        file_str += "\nNUMERICAL_ORBITAL\n"
        for iele, each in enumerate(species):
            file_str += f"{abacus_pao_dict[each]}\n"
        file_str += f"\nLATTICE_CONSTANT\n{QTcts.ANGtoBOHR}\n"
        file_str += "\nLATTICE_VECTORS\n"
        file_str += " {:19.15f} {:19.15f} {:19.15f}\n {:19.15f} {:19.15f} {:19.15f}\n {:19.15f} {:19.15f} {:19.15f}\n".format(*cell[0], *cell[1], *cell[2])
        atomic_group_dict = {}
        for each in species:
            atomic_group_dict[each] = []
        for iatom, label in enumerate(chemical_symbols):
            atomic_group_dict[label].append(iatom)
        file_str += "\nATOMIC_POSITIONS\nCartesian\n"
        for each in species:
            natoms_ele = len(atomic_group_dict[each])
            file_str += f"{each}\n0.0\n{natoms_ele}\n"
            for index in atomic_group_dict[each]:
                if spinor_mode == 'collinear':
                    file_str += " {:15.10f} {:15.10f} {:15.10f} ".format(*positions[index])
                    file_str += " magmom {:10.7f} \n".format(magmoms[index, 2])
                elif spinor_mode == 'nc':
                    file_str += " {:15.10f} {:15.10f} {:15.10f} ".format(*positions[index])
                    file_str += " magmom {:10.7f} {:10.7f} {:10.7f} \n".format(*magmoms[index])
                else:
                    file_str += " {:15.10f} {:15.10f} {:15.10f} \n".format(*positions[index])
    elif write_mode == 'KPT':
        if dSon_cal:
            kpts = [1,1,1]
        file_str += "K_POINTS\n"
        file_str += "0\n"
        file_str += "Gamma\n"
        file_str += "{:d} {:d} {:d}  0 0 0\n".format(*kpts)
    elif write_mode == 'INPUT':
        file_str += "INPUT_PARAMETERS\n"
        if spinor_mode == 'none':
            nspin = 1
        elif spinor_mode == 'collinear':
            nspin = 2
        elif spinor_mode == 'nc':
            nspin = 4
        else:
            raise RuntimeError(f"Failed to write ABACUS input file: unexpected spinor mode {spinor_mode}.")
        if is_hse:
            dft_functional = 'hse'
        else:
            dft_functional = 'pbe'
        setting_dict = {}
        setting_dict['suffix'] = 'QTrans'
        setting_dict['pseudo_dir'] = pseudo_dir
        setting_dict['orbital_dir'] = orbital_dir
        setting_dict['ecutwfc'] = 100.0
        setting_dict['scf_thr'] =  '1e-9'
        if dSon_cal:
            setting_dict['scf_nmax'] = 1
        else:
            setting_dict['scf_nmax'] = 300
        setting_dict['ntype'] = len(species)
        setting_dict['stru_file'] = stru_file
        setting_dict['kpoint_file'] = kpoint_file
        setting_dict['basis_type'] = 'lcao'
        setting_dict['calculation'] = 'scf'
        setting_dict['nspin'] = nspin
        setting_dict['ks_solver'] = 'genelpa'
        setting_dict['min_dist_coef'] = 0.000001
        if spinor_mode == 'nc':
            setting_dict['noncolin'] = True
            setting_dict['symmetry'] = -1
        if soc_switch:
            setting_dict['lspinorb'] = True
        setting_dict['smearing_method'] = 'gauss'
        setting_dict['smearing_sigma'] = 0.015
        setting_dict['mixing_type'] = 'broyden'
        setting_dict['out_mat_hs2'] = True
        setting_dict['dft_functional'] = dft_functional
        max_key_length = max(len(key) for key in setting_dict)
        for key, value in setting_dict.items():
            file_str += f"{key:<{max_key_length}}    {value}\n"
    else:
        raise RuntimeError(f"Failed to write ABACUS input file: unexpected writing mode {write_mode}.")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(file_str)
