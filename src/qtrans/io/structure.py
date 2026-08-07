"""QTrans io.structure helpers."""

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from qtrans.core.constants import QTcts
import numpy as np


def read_xsf(filepath:str):
    # return unit is angstrom and \mu_B
    lattice_vectors = []
    atomic_positions = []
    atomic_symbols = []
    magmoms = []

    in_primvec = False
    in_primcoord = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('CRYSTAL'):
                continue
            elif line.startswith('BEGIN_BLOCK_DATAGRID_3D_UNKNOWN'):
                in_primvec = False
                in_primcoord = False
            elif line.startswith('END_BLOCK_DATAGRID_3D_UNKNOWN'):
                pass
            elif line.startswith('PRIMVEC'):
                in_primvec = True
                in_primcoord = False
                lattice_vectors = []
            elif in_primvec:
                try:
                    vec = list(map(float, line.split()))
                    if len(vec) == 3:
                        lattice_vectors.append(vec)
                    if len(lattice_vectors) == 3:
                        in_primvec = False
                except ValueError:
                    in_primvec = False  
            elif line.startswith('PRIMCOORD') or line.startswith('ATOMS'):
                in_primcoord = True
                in_primvec = False
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    pass
                atomic_positions = []
                atomic_symbols = []
                magmoms = []
            elif in_primcoord:
                # specie x y z mx my mz
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        symbol = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        atomic_symbols.append(symbol)
                        atomic_positions.append([x, y, z])
                        if len(parts) >= 7:
                            mx, my, mz = float(parts[4]), float(parts[5]), float(parts[6])
                            magmoms.append([mx, my, mz])
                        elif len(parts) == 5:
                            mz = float(parts[4])
                            magmoms.append([0.0, 0.0, mz])
                        else:
                            magmoms.append([0.0, 0.0, 0.0])
                    except ValueError:
                        in_primcoord = False
                else:
                    in_primcoord = False

    lattice_vectors = np.array(lattice_vectors)
    atomic_symbols = np.array(atomic_symbols)
    atomic_positions = np.array(atomic_positions)
    magmoms = np.array(magmoms)

    return lattice_vectors, atomic_symbols, atomic_positions, magmoms


def read_structure_from_graph_data(filepath:str):
    # return unit is angstrom and \mu_B
    graph_data = np.load(filepath, allow_pickle=True)['graph'].item()[0]
    species = graph_data['z'].numpy()
    species = np.array([Element.from_Z(each).name for each in species], dtype=str)
    latt = graph_data['cell'].numpy() * QTcts.BOHRtoANG
    coordinates = graph_data['pos'].numpy() * QTcts.BOHRtoANG
    if 'spin_length' in graph_data.keys:
        spin_length = graph_data['spin_length'].numpy()
        spin_vec = graph_data['spin_vec'].numpy()   # (natoms, 3)
        magmoms = spin_length[:,None] * spin_vec
    else:
        magmoms =np.zeros((len(species), 3))
    return species, latt, coordinates, magmoms


def write_mcif(filepath:str, structure:Structure, magmoms:np.ndarray, sga:SpacegroupAnalyzer, mag_symm_dict:dict):
    # magmoms: (natoms, 3) or (natoms,) in \mu_B
    abc = structure.lattice.abc
    angles = structure.lattice.angles
    reduced_formula, formula_units_Z = structure.composition.get_integer_formula_and_factor()
    atoms_label = structure.species
    atoms_label = [each.name for each in atoms_label]
    frac_coors = structure.frac_coords
    mcif_info = f"""# generated using QTrans
    data_{reduced_formula}
    
    # Global data for the structure
    _chemical_formula_sum        '{structure.formula}'
    _chemical_formula_structural '{reduced_formula}'
    _cell_formula_units_Z  '{formula_units_Z}'

    # Crystallographic data (standard CIF part)
    _cell_length_a               {abc[0]}
    _cell_length_b               {abc[1]}
    _cell_length_c               {abc[2]}
    _cell_angle_alpha            {angles[0]}
    _cell_angle_beta             {angles[1]}
    _cell_angle_gamma            {angles[2]}
    _cell_volume                 {structure.volume}

    _symmetry_space_group_name_H-M     '{sga.get_space_group_symbol()}'
    _symmetry_Int_Tables_number         {sga.get_space_group_number()}

    # Magnetic data (MCIF specific part)
    # Magnetic space group (Shubnikov Group)
    _magnetic_space_group_name_Shubnikov 'P1'

    # Loop for atomic sites and their magnetic moments
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    _atom_site_moment.direction_x # Magnetic moment component along x
    _atom_site_moment.direction_y # Magnetic moment component along y
    _atom_site_moment.direction_z # Magnetic moment component along z
    """
    for i, magmom in enumerate(magmoms):
        if isinstance(magmom, (float, int)):
            mag_vec = (0.0, 0.0, magmom)
        else:
            mag_vec = magmom
        mcif_info += f"{atoms_label[i]}{i} {atoms_label[i]}  {frac_coors[i, 0]} {frac_coors[i, 1]} {frac_coors[i, 2]}  1  {mag_vec[0]:.6f} {mag_vec[1]:.6f} {mag_vec[2]:.6f}\n"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(mcif_info)
