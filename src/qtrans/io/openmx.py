"""QTrans io.openmx helpers."""

from pymatgen.core.structure import Structure
from qtrans.core.constants import QTcts
from qtrans.data.species import openmx_pao_dict
from qtrans.data.species import openmx_pot_dict
from qtrans.data.species import openmx_spin_sets
from typing import Pattern
from typing import Union
import numpy as np
import re


def get_openmx_re_pattern(mode:str, spinor_mode:str='none', natoms:int=None) -> Pattern:
    # check spinor_mode
    if spinor_mode not in ['none', 'collinear', 'nc']:
        raise RuntimeError(f"The spinor_mode {spinor_mode} is not supported in OpenMX regular expression pattern!")
    if mode in ['atomic_force', 'magmom'] and (natoms is None):
        raise RuntimeError(f"The OpenMX regular expression pattern of {mode} needs the number of atoms")
    # generate pattern
    if mode == 'latt':
        pattern = re.compile(r'<Atoms\.UnitVectors[^\n]*\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms\.UnitVectors>')
    elif mode == 'coordinate':
        if spinor_mode == 'none':
            pattern = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+')
        elif spinor_mode == 'collinear':
            pattern = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\w+')
        elif spinor_mode == 'nc':
            pattern = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\d\s+\w+')
    elif mode == 'enpy':
        pattern = re.compile(r'Enpy.\s+(\-\d+\.?\d*)')
    elif mode == 'niter_scf':
        pattern = re.compile(r'SCF=\s+(\d+)\s+NormRD')
    elif mode == 'atomic_force':
        match_force = r"""<coordinates.forces\s+"""
        match_force += rf"\s+{natoms}\s+"
        for i in range(natoms):
            match_force += r"\s+\d+\s+\w+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+"
        match_force += r"""coordinates.forces>"""
        pattern = re.compile(match_force)
    elif mode == 'magmom':
        if spinor_mode == 'none':
            pattern = None
        elif spinor_mode == 'collinear':
            match_spin = r"""Up spin      Down spin     Sum           Diff+"""
            for i in range(natoms):
                match_spin += r"\s+\d+\s+\w+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+(\-?\d+\.?\d+)\s"
            pattern = re.compile(match_spin)
        elif spinor_mode == 'nc':
            match_spin = r"""Up       Down      Sum      Diff        theta      phi+"""
            for i in range(natoms):
                match_spin += r"\s+\d+\s+\w+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s"
            pattern = re.compile(match_spin)
    else:
        raise RecursionError(f"Cannot find OpenMX regular expression pattern for {mode}")
    return pattern

def read_openmx_input(filepath:str):
    # return unit is angstrom and \mu_B
    # Initialize default values
    spinor_mode = 'none'
    latt_unit = 'au'
    pos_unit = 'au'
    # Flags to ensure we only capture the first occurrence
    spin_polarization_found = False
    pos_unit_found = False
    latt_unit_found = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue # Skip empty lines and comments
            parts = line.split()
            # Check for scf.SpinPolarization (only if not found yet)
            if not spin_polarization_found and len(parts) >= 2 and parts[0] == 'scf.SpinPolarization':
                if parts[1].lower() == 'on':
                    spinor_mode = 'collinear'
                elif parts[1].lower() == 'nc':
                    spinor_mode = 'nc'
                spin_polarization_found = True # Mark as found
            # Check for Atoms.SpeciesAndCoordinates.Unit (position unit, only if not found yet)
            elif not pos_unit_found and len(parts) >= 2 and parts[0] == 'Atoms.SpeciesAndCoordinates.Unit':
                if parts[1].lower().startswith('ang'):
                    pos_unit = 'ang'
                pos_unit_found = True
            # Check for Atoms.UnitVectors.Unit (lattice unit, only if not found yet)
            elif not latt_unit_found and len(parts) >= 2 and parts[0] == 'Atoms.UnitVectors.Unit':
                if parts[1].lower().startswith('ang'):
                    latt_unit = 'ang'
                latt_unit_found = True
            # Optional: If all three parameters are found, we can break early
            # This makes the parsing even more efficient for long files
            if spin_polarization_found and pos_unit_found and latt_unit_found:
                break

    pattern_latt = re.compile(r'<Atoms\.UnitVectors[^\n]*\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms\.UnitVectors>')
    if spinor_mode == 'none':
        pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+')
    elif spinor_mode == 'collinear':
        pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\w+)')
    elif spinor_mode == 'nc':
        pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\d)\s+(\w+)')
    with open(filepath,'r') as f:
        content = f.read()
    speciesAndCoordinates = pattern_coor.findall((content).strip())
    latt = pattern_latt.findall((content).strip())[0]
    latt = np.array([float(var) for var in latt]).reshape(-1, 3)
    if latt_unit == 'au':
        latt = latt * QTcts.BOHRtoANG
    species = []
    coordinates = []
    orbital_angles = None
    atomic_mag_constrained = None
    atomic_u_enhances = None
    if spinor_mode == 'none':
        for item in speciesAndCoordinates:
            species.append(item[0])
            coordinates.extend(item[1:4])
        magmoms = np.zeros((len(species), 3))
    elif spinor_mode == 'collinear':
        nelecs = []
        atomic_u_enhances = []
        for item in speciesAndCoordinates:
            species.append(item[0])
            coordinates.extend(item[1:4])
            nelecs.extend(item[4:6])
            atomic_u_enhances.append(item[6])
        nelecs = np.array([float(nelec) for nelec in nelecs]).reshape(-1, 2)
        magmoms = np.zeros((len(nelecs), 3))
        # if collinear system, spin is projected into z axis.
        magmoms[:, 2] = nelecs[:, 0] - nelecs[:, 1]
    elif spinor_mode == 'nc':
        nelecs = []
        theta = []
        phi = []
        orbital_angles = []
        atomic_u_enhances = []
        atomic_mag_constrained = []
        for item in speciesAndCoordinates:
            species.append(item[0])
            coordinates.extend(item[1:4])
            nelecs.extend(item[4:6])
            theta.append(item[6])
            phi.append(item[7])
            orbital_angles.extend(item[8:10])
            atomic_mag_constrained.append(item[10])
            atomic_u_enhances.append(item[11])
        nelecs = np.array([float(nelec) for nelec in nelecs]).reshape(-1, 2)
        orbital_angles = np.array([float(angle) for angle in orbital_angles]).reshape(-1,2)
        atomic_mag_constrained = np.array([int(each) for each in atomic_mag_constrained], dtype=bool)
        theta = np.radians([float(each) for each in theta])
        phi = np.radians([float(each) for each in phi])
        magmoms = np.zeros((len(nelecs), 3))
        vx = np.sin(theta) * np.cos(phi)
        vy = np.sin(theta) * np.sin(phi)
        vz = np.cos(theta)
        spin_length = nelecs[:, 0] - nelecs[:, 1]
        magmoms[:, 0] = spin_length * vx
        magmoms[:, 1] = spin_length * vy
        magmoms[:, 2] = spin_length * vz
    coordinates = np.array([float(pos) for pos in coordinates]).reshape(-1, 3)
    if pos_unit == 'au':
        coordinates = coordinates * QTcts.BOHRtoANG
    return species, latt, coordinates, magmoms, orbital_angles, atomic_mag_constrained, atomic_u_enhances

def read_openmx_out(filepath: str,
                    return_iter: bool = False,
                    return_magmom: bool = False,
                    return_total_energy: bool = False,
                    return_atomic_force: bool = False):
    # return unit is angstrom, \mu_B and Hartree/Bohr
    # Initialize default values
    spinor_mode = 'none'
    latt_unit = 'au'
    pos_unit = 'au'
    max_niter = 40 # The default maximum number of SCF iterations in openmx
    # Flags to ensure we only capture the first occurrence
    spin_polarization_found = False
    pos_unit_found = False
    latt_unit_found = False
    max_niter_found = not return_iter

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue # Skip empty lines and comments
            parts = line.split()
            # Check for scf.SpinPolarization (only if not found yet)
            if not spin_polarization_found and len(parts) >= 2 and parts[0] == 'scf.SpinPolarization':
                if parts[1].lower() == 'on':
                    spinor_mode = 'collinear'
                elif parts[1].lower() == 'nc':
                    spinor_mode = 'nc'
                spin_polarization_found = True
            # Check for Atoms.SpeciesAndCoordinates.Unit (position unit, only if not found yet)
            elif not pos_unit_found and len(parts) >= 2 and parts[0] == 'Atoms.SpeciesAndCoordinates.Unit':
                if parts[1].lower().startswith('ang'):
                    pos_unit = 'ang'
                pos_unit_found = True
            # Check for Atoms.UnitVectors.Unit (lattice unit, only if not found yet)
            elif not latt_unit_found and len(parts) >= 2 and parts[0] == 'Atoms.UnitVectors.Unit':
                if parts[1].lower().startswith('ang'):
                    latt_unit = 'ang'
                latt_unit_found = True
            # Check for scf.SpinPolarization (only if not found yet)
            elif not max_niter_found and len(parts) >= 2 and parts[0] == 'scf.maxIter':
                max_niter = int(parts[1])
                max_niter_found = True
            # Optional: If all three parameters are found, we can break early
            # This makes the parsing even more efficient for long files
            if spin_polarization_found and pos_unit_found and latt_unit_found and max_niter_found:
                break

    # begin regular expression matching
    with open(filepath,'r') as f:
        content = f.read()
    # since OpenMX will repeat the input file in its outputs, we split <content> into input part and output part to speed up the matching
    cut_pos = content.find("SCF history at MD= 1")
    if cut_pos != -1:
        content_before = content[:cut_pos]
        content_after = content[cut_pos:]
    else:
        raise RuntimeError("Failed to parse OpenMX output: unexpected format.")
    # total energy
    if return_total_energy:
        pattern_enpy = get_openmx_re_pattern('enpy')
        match_enpy = pattern_enpy.search(content_after)
        if match_enpy:
            enpy = float(match_enpy.group(1))
        else:
            raise RuntimeError("Failed to parse OpenMX output: unexpected format.")
    # number of scf iteration
    if return_iter:
        pattern_niter_scf = get_openmx_re_pattern('niter_scf')
        last_match = None
        for match in pattern_niter_scf.finditer(content_after):
            last_match = match
        if last_match:
            niter_scf = int(last_match.group(1))
    # lattice matrix
    pattern_latt = get_openmx_re_pattern('latt')
    latt = pattern_latt.findall((content_before).strip())[0]
    latt = np.array([float(var) for var in latt]).reshape(-1, 3)
    if latt_unit == 'au':
        latt = latt * QTcts.BOHRtoANG
    # atomic position
    pattern_coor = get_openmx_re_pattern('coordinate', spinor_mode)
    speciesAndCoordinates = pattern_coor.findall((content_before).strip())
    species = []
    coordinates = []
    for item in speciesAndCoordinates:
        species.append(item[0])
        coordinates.extend(item[1:4])
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    if pos_unit == 'au':
        coordinates = coordinates * QTcts.BOHRtoANG
    # magmoms
    if return_magmom:
        magmoms = None
        pattern_spin = get_openmx_re_pattern('magmom', spinor_mode, natoms=len(coordinates))
        if spinor_mode == 'none':
            magmoms = np.zeros((len(coordinates), 3))
        elif spinor_mode == 'collinear':
            spin_info = pattern_spin.findall(content_after.strip())
            spin_config_z = np.asarray(spin_info[0], dtype=float).reshape(-1)
            magmoms = np.zeros((len(spin_config_z), 3))
            magmoms[:, 2] = spin_config_z
        elif spinor_mode == 'nc':
            spin_info = pattern_spin.findall(content_after.strip())
            spin_config = np.asarray(spin_info[0], dtype=float).reshape(-1, 3)
            spin_length = spin_config[:,0]
            theta, phi = np.radians(spin_config[:, 1]), np.radians(spin_config[:, 2])
            vx = np.sin(theta) * np.cos(phi)
            vy = np.sin(theta) * np.sin(phi)
            vz = np.cos(theta)
            magmoms = np.hstack((vx[:,None], vy[:,None], vz[:,None])) * spin_length[:, None]
    if return_atomic_force:
        pattern_force = get_openmx_re_pattern('atomic_force', natoms=len(coordinates))
        force_info = pattern_force.findall(content_after.strip())
        atomic_forces = np.asarray(force_info[0], dtype=float).reshape(-1, 3) # shape (natoms, 3)

    # returns
    results_dict = {
        'species': species,
        'latt': latt,
        'coordinates': coordinates,
        'spinor_mode': spinor_mode,
    }
    if return_magmom:
        results_dict['magmoms'] = magmoms
    if return_iter:
        results_dict['max_niter'] = max_niter
        results_dict['niter_scf'] = niter_scf
    if return_total_energy:
        results_dict['total_energy'] = enpy
    if return_atomic_force:
        results_dict['atomic_forces'] = atomic_forces
    return results_dict


def write_openmx_input(filepath:str, structure:Union[Structure, dict], kpts:list, dftdata_dir:str,
                       spinor_mode:str='none', soc_switch:bool=False, cwd:str=None,
                       magmoms:np.ndarray=None, orbital_angles:np.ndarray=None,
                       atomic_mag_constrained:np.ndarray=None, atomic_u_enhances:np.ndarray=None, 
                       magnetic_tolerance:float=QTcts.TENPM5, xc_func:str='GGA-PBE'):
    # if type(structure) == 'Structure': units are angstrom
    # if type(structure) == 'dict': units are Bohr
    # magmoms in \mu_B 
    # generate setting dict
    kpt_size = "{:d} {:d} {:d}".format(*kpts)
    setting_dict = {
        'System.CurrrentDirectory': cwd,
        'System.Name': 'QTrans',
        'DATA.PATH': dftdata_dir,
        'HS.fileout': 'on',
        'scf.XcType': xc_func,
        'scf.partialCoreCorrection': 'on',
        'scf.ElectronicTemperature': '300.0',
        'scf.energycutoff': '200.0',
        'scf.maxIter': '300',
        'scf.EigenvalueSolver': 'Band',
        'scf.Kgrid': kpt_size,
        'scf.Mixing.Type': 'rmm-diis',
        'scf.Init.Mixing.Weight': '0.001',
        'scf.Min.Mixing.Weight': '0.0001', 
        'scf.Max.Mixing.Weight': '0.3',
        'scf.Mixing.History': '50',
        'scf.Mixing.StartPulay': '30',
        'scf.Mixing.EveryPulay': '1',
        'scf.criterion' : '1.0e-10',
    }
    if spinor_mode == 'none':
        setting_dict['scf.SpinPolarization'] = 'off'
    elif spinor_mode == 'collinear':
        setting_dict['scf.SpinPolarization'] = 'on'
    elif spinor_mode == 'nc':
        setting_dict['scf.SpinPolarization'] = 'NC'
    if soc_switch:
        setting_dict['scf.SpinOrbit.Coupling'] = 'on'

    # generate atomic info
    openmx_atom_info = ""
    if type(structure) == dict:
        chemical_symbols = structure['chemical_symbols']
        # species = set(chemical_symbols) # there will be some random indices 
        species = list(dict.fromkeys(chemical_symbols))
        positions = structure['positions'] * QTcts.BOHRtoANG
        cell = structure['cell'] * QTcts.BOHRtoANG
        if magmoms is None:
            magmoms = structure.get('magmoms')
        if orbital_angles is None:
            orbital_angles = structure.get('orbital_angles')
        if atomic_mag_constrained is None:
            atomic_mag_constrained = structure.get('atomic_mag_constrained')
        if atomic_u_enhances is None:
            atomic_u_enhances = structure.get('atomic_u_enhances')
    elif type(structure) == Structure:        
        chemical_symbols = structure.species
        chemical_symbols = [each.name for each in chemical_symbols]
        # species = set(chemical_symbols) # there will be some random indices
        species = list(dict.fromkeys(chemical_symbols))
        positions = structure.cart_coords
        cell = structure.lattice.matrix
    natoms = len(chemical_symbols)
    openmx_atom_info += "#\n# Definition of Atomic Species\n#\n"
    openmx_atom_info += f'Species.Number       {len(species)}\n'
    openmx_atom_info += '<Definition.of.Atomic.Species\n'
    for s in species:
        openmx_atom_info += f"{s}   {openmx_pao_dict[s]}       {openmx_pot_dict[xc_func][s]}\n"
    openmx_atom_info += "Definition.of.Atomic.Species>\n\n"
    openmx_atom_info += "#\n# Atoms\n#\n"
    openmx_atom_info += "Atoms.Number%12d" % len(chemical_symbols)
    openmx_atom_info += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
    openmx_atom_info += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
    if spinor_mode == 'none':
        for iatom, symbol in enumerate(chemical_symbols):
            openmx_atom_info += "\n{:<3d}  {:<2s}  {:10.7f}  {:10.7f}  {:10.7f}   {:.2f}   {:.2f}".format(
                                    iatom + 1, symbol, *positions[iatom], *openmx_spin_sets[chemical_symbols[iatom]])
    elif spinor_mode == 'collinear':
        # if system collinear, consider only z projection of spin vector
        spin_length = magmoms[:, 2]
        is_magnetic = np.abs(spin_length) > magnetic_tolerance
        spin_length = spin_length / 2.0
        if atomic_u_enhances is None:
            atomic_u_enhances = ['off'] * natoms
            atomic_u_enhances[is_magnetic] = 'on'
        for iatom, symbol in enumerate(chemical_symbols):
            tmp_nelecs = openmx_spin_sets[chemical_symbols[iatom]].copy()
            if is_magnetic[iatom]:
                tmp_nelecs[0] = tmp_nelecs[0] + spin_length[iatom]
                tmp_nelecs[1] = tmp_nelecs[1] - spin_length[iatom]
            openmx_atom_info += "\n{:<3d}  {:<2s}  {:10.7f}  {:10.7f}  {:10.7f}   {:.7f}   {:.7f}  {:<5s}".format(
                                    iatom + 1, symbol, *positions[iatom], *tmp_nelecs, atomic_u_enhances[iatom])
    elif spinor_mode == 'nc':
        spin_length = np.linalg.norm(magmoms, axis=1)
        # spin_vec = magmoms / spin_length[:,None]
        spin_vec = np.zeros_like(magmoms)
        nonzero_mask = spin_length > 0
        spin_vec[nonzero_mask] = magmoms[nonzero_mask] / spin_length[nonzero_mask, None]
        # theta = np.arccos(spin_vec[:,2])
        theta = np.zeros(len(spin_vec))
        theta[nonzero_mask] = np.arccos(spin_vec[nonzero_mask, 2])
        theta = np.degrees(theta)
        # phi = np.arctan2(spin_vec[:,1], spin_vec[:,0])
        phi = np.zeros(len(spin_vec))
        phi[nonzero_mask] = np.arctan2(spin_vec[nonzero_mask, 1], spin_vec[nonzero_mask, 0])
        phi[phi < 0] += QTcts.TWOPI
        phi = np.degrees(phi)
        is_magnetic = spin_length > magnetic_tolerance
        spin_length = spin_length / 2.0
        if orbital_angles is None:
            orbital_angles = np.zeros((natoms, 2))
        if atomic_mag_constrained is None:
            atomic_mag_constrained = np.zeros(natoms, dtype=int)
            atomic_mag_constrained[is_magnetic] = 1
        if atomic_u_enhances is None:
            atomic_u_enhances = ['off'] * natoms
            atomic_u_enhances[is_magnetic] = 'on'
        for iatom, symbol in enumerate(chemical_symbols):
            tmp_nelecs = openmx_spin_sets[chemical_symbols[iatom]].copy()
            if is_magnetic[iatom]:
                tmp_nelecs[0] = tmp_nelecs[0] + spin_length[iatom]
                tmp_nelecs[1] = tmp_nelecs[1] - spin_length[iatom]
            openmx_atom_info += "\n{:<3d}  {:<2s}  {:10.7f}  {:10.7f}  {:10.7f}   {:.7f}   {:.7f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:<2d}  {:<5s}".format(
                                    iatom + 1, symbol, *positions[iatom],
                                    *tmp_nelecs,
                                    theta[iatom], phi[iatom],
                                    *orbital_angles[iatom],
                                    atomic_mag_constrained[iatom],
                                    atomic_u_enhances[iatom])
    openmx_atom_info += "\nAtoms.SpeciesAndCoordinates>"
    openmx_atom_info += "\nAtoms.UnitVectors.Unit             Ang #  Ang|AU"
    openmx_atom_info += "\n<Atoms.UnitVectors                     # unit=Ang."
    openmx_atom_info += "\n      {:10.7f}  {:10.7f}  {:10.7f}\n      {:10.7f}  {:10.7f}  {:10.7f}\n      {:10.7f}  {:10.7f}  {:10.7f}".format(*cell[0], *cell[1], *cell[2])
    openmx_atom_info += "\nAtoms.UnitVectors>"
    openmx_atom_info += '\n'
    max_key_length = max(len(key) for key in setting_dict)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('#\n')
        f.write("# Openmx input file generate by QTrans\n")
        f.write('#\n')
        f.write('\n')
        for key, value in setting_dict.items():
            f.write(f"{key:<{max_key_length}}    {value}\n")
        f.write(openmx_atom_info)
