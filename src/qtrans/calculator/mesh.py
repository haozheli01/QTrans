"""Mechanically extracted QTrans methods."""

from collections import Counter
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from qtrans.core.constants import QTcts
import math
import numpy as np
import spglib


class _MeshMixin:
    def _get_monkhorst_pack(self, mesh, shift=[0,0,0], return_frac:bool=False):
        """
        Construct a uniform sampling grid of given size in k-space.
        Note that here 2*pi constant is ignored here.
        
        Args:
            mesh (list or np.array): The size of the grid.
            shift (list or np.array): Offset of the grid relative to the Brillouin zone center.
            return_frac (bool): If true, the function will return the fractional coordinates of k points.
        
        Returns:
            k_grids: The coordinates of the k sampling points.
        """     
        struct = Structure(lattice=self.graph_data.latt*QTcts.BOHRtoANG,
                           species=[Element.from_Z(k).symbol for k in self.graph_data.species],
                           coords=self.graph_data.pos*QTcts.BOHRtoANG, coords_are_cartesian=True)
        positions = struct.frac_coords
        cell = (self.graph_data.latt*QTcts.BOHRtoANG, positions, self.graph_data.species)
        _, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift)
        # Irreducible k-points
        k_grids = grid / np.array(mesh, dtype=float) # (nk, 3)
        if return_frac:
            return k_grids
        else:
            k_grids = np.tensordot(k_grids, self.graph_data.rec_latt, axes=1) # (nk, 3)
            return k_grids

    def _frac2car(self, k_grids):
        """2*pi constant is missed.
        Args:
            k_grids (_type_): _description_

        Returns:
            _type_: _description_
        """
        k_grids = k_grids.reshape(-1, 3)
        k_vec = np.tensordot(k_grids, self.graph_data.rec_latt, axes=1)
        return k_vec

    def _car2frac(self, k_grids):
        k_grids = k_grids.reshape(-1, 3)
        k_vec = np.tensordot(k_grids, self.graph_data.latt.T, axes=1)
        return k_vec

    def _get_ir_reciprocal_mesh(self, mesh, shift=[0,0,0], auxiliary_info=False, return_frac:bool=False):
        """Calculate the k-point grid and weights in the irreducible Brillouin zone

        Args:
            mesh: list or np.array
            shift: list or np.array

        Returns:
            k_vec : kpoint coordinates in the irreducible zone, unit: Bohr^-1 
            weight: the weight of each irreducible kpoint
        """
        struct = Structure(lattice=self.graph_data.latt*QTcts.BOHRtoANG,
            species=[Element.from_Z(k).symbol for k in self.graph_data.species],
            coords=self.graph_data.pos*QTcts.BOHRtoANG, coords_are_cartesian=True)
        # positions = struct.frac_coords
        positions = np.round(struct.frac_coords,7)

        if self.soc_switch:
            if self.magnetic_type == 'none': # Nonmagnetic system with SOC, the symmetry is the same as the non-SOC case.
                cell = (self.graph_data.latt*QTcts.BOHRtoANG, positions, self.graph_data.species)
                mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift)
            else: # Magnetic system with SOC, the symmetry is generally lower than the non-SOC case, and we directly use Monkhorst-Pack grid without symmetry reduction.
                k_vec = self._get_monkhorst_pack(mesh, shift=shift, return_frac=return_frac)
                weight = np.ones(len(k_vec), dtype=float) / len(k_vec)
                if auxiliary_info:
                    return k_vec, weight, k_vec, None
                else:
                    return k_vec, weight
        else:
            if self.spinor_mode == 'none': # Non-spin-polarized case, the symmetry is the same as the non-SOC case.
                cell = (self.graph_data.latt*QTcts.BOHRtoANG, positions, self.graph_data.species)
                mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift)
            elif self.spinor_mode == 'collinear': # Collinear case, the symmetry is generally lower than the non-spin-polarized case.
                # a little stupid to use the magmom tag to break the symmetry, but it works.
                magtag_numbers = []
                magmom_tag_map = {}
                new_tag = 1000
                magmoms = self.graph_data.spin_length * self.graph_data.moment_direct
                prec_order = -math.floor(np.log10(np.abs(self.moment_similar_tolerence))) + 1
                numbers_data = self.graph_data.species.copy()
                for i, (number, magmom) in enumerate(zip(numbers_data[self.graph_data.mag_atom_index], magmoms)):
                    key = (number, round(magmom, prec_order))
                    if key not in magmom_tag_map:
                        magmom_tag_map[key] = new_tag
                        new_tag += 1
                    magtag_numbers.append(magmom_tag_map[key])
                numbers_data[self.graph_data.mag_atom_index] = magtag_numbers
                cell = (self.graph_data.latt*QTcts.BOHRtoANG, positions, numbers_data)
                mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift, is_time_reversal=False)
        ird_grids = grid[np.unique(mapping)] / np.array(mesh, dtype=float) # (nks, 3)
        self._log(f"\nThe symmetry for setting up kpoint is: {spglib.get_spacegroup(cell, symprec=1e-5)}")
        
        if not return_frac:
            ird_grids = np.tensordot(ird_grids, self.graph_data.rec_latt, axes=1) # (nks, 3)
        res = Counter(mapping)
        weight = []
        for i in np.unique(mapping):
            weight.append(res[i])
        weight = np.array(weight)
        weight = weight/np.sum(weight)
        if auxiliary_info:
            ir_ids = np.unique(mapping)
            ir_idx_dict = {}
            for i, id in enumerate(ir_ids):
                ir_idx_dict[id] = i
            grid2ir_idx = []
            for ir_gp_id in mapping:
                grid2ir_idx.append(ir_idx_dict[ir_gp_id])
            grid2ir_idx = np.array(grid2ir_idx)
            grid = grid / np.array(mesh, dtype=float)
            if not return_frac:
                grid = np.tensordot(grid, self.graph_data.rec_latt, axes=1) # (nks, 3)
            return ird_grids, weight, grid, grid2ir_idx
        else:
            return ird_grids, weight
