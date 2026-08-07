"""Mechanically extracted QTrans methods."""

from pathlib import Path
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet
from qtrans.io.external import run_vaspkit
from typing import List
import numpy as np
import os
import shutil


class _VaspMixin:
    def _get_vasp_ncore(self, natoms:int):
        if natoms < 50:
            ncore = 1
        elif natoms < 100:
            ncore = min(2, self.min_ntasks_per_node)
        elif natoms < 200:
            ncore = min(4, self.min_ntasks_per_node)
        elif natoms < 400:
            ncore = min(8, self.min_ntasks_per_node)
        else:
            ncore = min(16, self.min_ntasks_per_node)
        return ncore

    def _gen_vasp_incar(self, structure:Structure) -> Incar:
        mp_sets = MPRelaxSet(structure)
        incar = mp_sets.incar
        incar['EDIFF'] = incar['EDIFF'] / 10.0
        incar['IBRION'] = -1
        incar['ISMEAR'] = 0
        incar['NSW'] = 0
        incar['NCORE'] = self._get_vasp_ncore(len(structure.sites))
        if self.magnetic_type == 'collinear':
            if ("ISPIN" not in incar) or ("MAGMOM" not in incar):
                raise RuntimeError(f"Error(3002): Failed to generate INCAR file using pymatgen.")
        elif self.magnetic_type == 'nc':
            raise NotImplementedError("Noncollinear system is not supported now.")
        elif self.magnetic_type == 'none':
            incar.pop("ISPIN", None)
            incar.pop("MAGMOM", None)
        return incar

    def _gen_vasp_kpoints(self, sc_size:list=None) -> Kpoints:
        if sc_size is None:
            kpt_size = self.scf_uc_kpt.tolist()
        else:
            kpt_size = np.ceil(self.scf_uc_kpt / np.array(sc_size)).astype(int).tolist()
        kpoints = Kpoints(style="Gamma", kpts=[kpt_size])
        return kpoints

    def _gen_vasp_potcar(self, cwd:str):
        _, stderr, rcode = run_vaspkit(self.vaspkit_path, ["103"], cwd=cwd)
        if rcode != 0:
            raise RuntimeError(f"Error(3001): Failed to generate POTCAR using VASPKIT: {stderr}")

    def _gen_vasp_inputs(self, structure:Structure, cwd:str, gmode:List[str]=[], sc_size:list=None):
        """
        Generate INCAR, KPOINTS, POTCAR for a given structure.
        Please notice in this function, it do not.

        Args:
            structure (Structure): The structure.
            cwd (str): Directory in which to generate and save the output files.
        """
        if "kpoints" in gmode:
            # KPOINTS
            try:
                tmp_path = os.path.join(cwd, "KPOINTS")
                kpoints = self._gen_vasp_kpoints(sc_size)
                kpoints.write_file(tmp_path)
            except Exception as e:
                raise IOError(f"Error(2002): Failed to write {tmp_path}: {e}")
        if "potcar" in gmode:
            # POTCAR
            try:
                tmp_path = os.path.join(cwd, "POTCAR")
                self._gen_vasp_potcar(cwd)
            except Exception as e:
                raise IOError(f"Error(2003): Failed to write {tmp_path}: {e}")
        if "incar" in gmode:
            # INCAR
            try:
                tmp_path = os.path.join(cwd, "INCAR")
                incar = self._gen_vasp_incar(structure)
                incar.write_file(tmp_path)
            except Exception as e:
                raise IOError(f"Error(2004): Failed to write {tmp_path}: {e}")

    def prep_force_sets_from_vasp(self):
        """

        """
        # generate vasp inputs for each supercell structure
        index_dig = len(str(self.phonon_nstructs))
        sc_filenames = [os.path.join(self.outdir, f"POSCAR-QTph-{istruct+1:0{index_dig}d}") for istruct in range(self.phonon_nstructs)]
        # split supercells into each core
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(self.phonon_nstructs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        sc_filenames_all = np.split(sc_filenames, indices_or_sections=split_sections, axis=0)
        if sc_filenames_all[self.rank].size>0:
            sc_filenames =  sc_filenames_all[self.rank]
        else:
            sc_filenames = []
        for name in sc_filenames:
            folder_path = Path(os.path.join(self.outdir, name.split('-')[-1]))
            try:
                folder_path.mkdir(parents=False, exist_ok=True)
                shutil.move(name, os.path.join(str(folder_path), "POSCAR"))
            except Exception as e:
                raise RuntimeError(f"Error(3003): Failed to create {folder_path} or move POSCAR into it: {e}")
            structure = Structure.from_file(os.path.join(str(folder_path), "POSCAR"))
            self._gen_vasp_inputs(structure, str(folder_path), ["incar", "kpoints", "potcar"], sc_size=self.ph_sc_matrix)
