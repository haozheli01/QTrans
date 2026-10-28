"""QTrans io.abacus helpers."""

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from qtrans.core.constants import QTcts
from qtrans.data.species import abacus_pao_dict
from qtrans.data.species import abacus_pot_pbe_dict
from typing import Union
import numpy as np
import re


def read_abacus_image_shifts(log_path, stru_path, species, lattice, positions):
    """Recover ABACUS images relative to STRU (row-vector cell, all in Bohr).

    Only a single fixed-structure SCF is supported. Rounded log lattice vectors
    validate the reference cell; they are never used to recover image integers.
    """
    context = f"{log_path} (STRU: {stru_path})"
    with open(stru_path) as handle:
        stru_lines = [line.split('#')[0].split('//')[0].strip() for line in handle]
    stru_lines = [line for line in stru_lines if line]
    lat0 = float(stru_lines[stru_lines.index('LATTICE_CONSTANT') + 1])
    pos_type = stru_lines[stru_lines.index('ATOMIC_POSITIONS') + 1].lower()
    if pos_type not in ('direct', 'cartesian'):
        raise ValueError(f"{context}: unsupported STRU coordinate unit {pos_type!r}")
    lattice = np.asarray(lattice, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if (not np.isfinite(lat0) or lat0 <= 0 or lattice.shape != (3, 3)
            or positions.shape != (len(species), 3)
            or not np.all(np.isfinite(lattice)) or not np.all(np.isfinite(positions))):
        raise ValueError(f"{context}: invalid STRU lattice/positions (Bohr)")
    with open(log_path) as handle:
        content = handle.read()
    lines = content.splitlines()
    modes = re.findall(r'^\s*calculation\b\s*(?:=|:)?\s*(\S+)', content, re.M | re.I)
    if any(mode.lower() != 'scf' for mode in modes):
        raise ValueError(f"{context}: only fixed-structure SCF is supported, got {modes}")
    ions = {int(value) for value in re.findall(r'\bION\s*=\s*(\d+)', content)}
    if ions and ions != {1}:
        raise ValueError(f"{context}: multiple/unsupported ionic frames {sorted(ions)}")
    counts = re.findall(r'TOTAL ATOM NUMBER\s*=\s*(\d+)', content)
    if len(counts) != 1 or int(counts[0]) != len(species):
        raise ValueError(f"{context}: expected one TOTAL ATOM NUMBER = {len(species)}, got {counts}")
    headers = [i for i, line in enumerate(lines)
               if re.match(r'^\s*(?:CARTESIAN|DIRECT) COORDINATES\b', line)]
    if len(headers) != 1:
        raise ValueError(f"{context}: expected exactly one coordinate table, found {len(headers)}")
    header = lines[headers[0]].strip()
    cartesian = re.fullmatch(
        r'CARTESIAN COORDINATES\s*\(\s*UNIT\s*=\s*(\S+)\s+Bohr\s*\)\.?', header)
    if cartesian:
        scale = float(cartesian[1])
        if not np.isfinite(scale) or abs(scale - lat0) > 1e-10:
            raise ValueError(f"{context}: Cartesian UNIT {scale} Bohr != STRU {lat0} Bohr (tolerance 1e-10 Bohr)")
        prefix = 'tauc'
    elif re.fullmatch(r'DIRECT COORDINATES\s*\.?', header):
        scale, prefix = 1.0, 'taud'
    else:
        raise ValueError(f"{context}: unknown coordinate header/unit {header!r}")
    # Check the printed cell using its deliberately coarser output precision.
    lattice_headers = [i for i, line in enumerate(lines) if line.strip().startswith('Lattice vectors:')]
    if len(lattice_headers) != 1:
        raise ValueError(f"{context}: expected one log lattice, found {len(lattice_headers)}")
    index = lattice_headers[0]
    if lines[index].strip() != 'Lattice vectors: (Cartesian coordinate: in unit of a_0)':
        raise ValueError(f"{context}: unknown log lattice unit {lines[index].strip()!r}")
    try:
        printed_cell = np.array([[float(v) for v in row.split()]
                                 for row in lines[index + 1:index + 4]], dtype=np.float64) * lat0
    except ValueError as exc:
        raise ValueError(f"{context}: malformed log lattice (Bohr)") from exc
    if (printed_cell.shape != (3, 3) or not np.all(np.isfinite(printed_cell))
            or np.any(np.abs(printed_cell - lattice) > 1e-8 + 1e-5 * np.abs(lattice))):
        raise ValueError(f"{context}: log lattice differs from STRU (Bohr; component tolerance 1e-8 + 1e-5*abs(reference))")
    row_indices = []
    actual, type_counts = [], {}
    index = headers[0] + 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines) and re.match(r'^\s*atom\s+x\s+y\s+z\b', lines[index]):
        index += 1
    for atom, element in enumerate(species, 1):
        if index >= len(lines):
            raise ValueError(f"{context}: missing coordinate row for atom {atom} ({element})")
        fields = lines[index].split()
        type_counts[element] = type_counts.get(element, 0) + 1
        expected = f'{prefix}_{element}{type_counts[element]}'
        if len(fields) < 4 or fields[0] != expected:
            raise ValueError(f"{context}: atom {atom}: expected {expected}, got {lines[index]!r}")
        try:
            xyz = np.array(fields[1:4], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"{context}: invalid coordinates at atom {atom} ({expected})") from exc
        if not np.all(np.isfinite(xyz)):
            raise ValueError(f"{context}: non-finite coordinates at atom {atom} ({expected})")
        actual.append(xyz)
        row_indices.append(index)
        index += 1
    all_rows = [i for i, line in enumerate(lines) if re.match(r'^\s*tau[cd]_', line)]
    if row_indices != all_rows:
        raise ValueError(f"{context}: extra/orphan coordinate rows or multiple frames")
    actual = np.asarray(actual, dtype=np.float64)
    actual = actual * scale if prefix == 'tauc' else actual @ lattice
    delta = actual - positions
    try:
        fractional = np.linalg.solve(lattice.T, delta.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context}: singular STRU lattice (Bohr)") from exc
    if not np.all(np.isfinite(fractional)) or np.any(np.abs(fractional) >= 2**52):
        raise ValueError(f"{context}: invalid/unrepresentable periodic image")
    images = np.rint(fractional).astype(np.int64)
    residual = np.linalg.norm(delta - images @ lattice, axis=1)
    if np.any(residual > 1e-5):
        atom = int(np.argmax(residual))
        raise ValueError(f"{context}: atom {atom + 1} ({species[atom]}): periodic-image residual {residual[atom]:.12g} Bohr exceeds 1e-5 Bohr")
    return images


def correct_abacus_graph_images(graph, images, lattice):
    """Relabel edges after matching the raw H/S keys; preserve matrix values/order."""
    edges = np.asarray(graph['edge_index'], dtype=np.int64).reshape(2, -1)
    shifts = np.asarray(graph['cell_shift'], dtype=np.int64).reshape(-1, 3)
    images = np.asarray(images, dtype=np.int64)
    correction = images[edges[1]] - images[edges[0]]
    if not np.any(correction):
        return graph
    corrected = shifts + correction
    neighbours = corrected @ np.asarray(lattice, dtype=np.float64)
    as_lists = isinstance(graph['cell_shift'], list)
    graph['cell_shift'] = corrected.tolist() if as_lists else corrected
    graph['nbr_shift'] = neighbours.tolist() if as_lists else neighbours
    return graph


def align_abacus_spin_graph(reference, other, context):
    """Align a second collinear channel using the original (i,j,R) keys."""
    if reference['_abacus_csr_step'] != other['_abacus_csr_step']:
        raise ValueError(f"{context}: collinear CSR STEP mismatch")
    def keys(graph):
        return [(int(i), int(j), *map(int, r)) for (i, j), r in
                zip(np.asarray(graph['edge_index']).T, graph['cell_shift'])]
    target, source = keys(reference), keys(other)
    lookup = {key: i for i, key in enumerate(source)}
    if len(lookup) != len(source) or len(set(target)) != len(target) or set(target) != set(source):
        missing = next(iter(set(target) - set(source)), None)
        extra = next(iter(set(source) - set(target)), None)
        raise ValueError(f"{context}: collinear edge key sets differ (missing {missing}, extra {extra}; zero-based atoms)")
    return [[channel[lookup[key]] for key in target] for channel in other['Hoff']]


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
        if len(species) != len(num_orbitals):
            raise ValueError(f"{filepath}: ATOMIC_SPECIES/NUMERICAL_ORBITAL type count mismatch")

        def next_nonempty():
            for raw_line in fp:
                value = raw_line.split('//')[0].split('#')[0]
                if value.strip():
                    return value
            raise ValueError(f"{filepath}: incomplete ATOMIC_POSITIONS block")
        
        for is_ in range(len(num_orbitals)):
            # element
            line = next_nonempty()
            if line.strip() != species[is_]:
                raise ValueError(f"{filepath}: ATOMIC_POSITIONS species {is_ + 1}: expected {species[is_]}, got {line.strip()!r}")
            # mag
            line = next_nonempty()
            # num
            line = next_nonempty()
            na = int(line)
            if na <= 0:
                raise ValueError(f"{filepath}: invalid atom count for species {species[is_]}: {na}")
            num_atoms_per_species.append(na)
            ia = 0
            while ia < na:
                line = next_nonempty()
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
