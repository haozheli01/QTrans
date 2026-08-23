"""Density-of-states calculation workflows."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.numerics.gaussian import w0gauss
from qtrans.numerics.tetrahedron import tetrahedron_integral
from scipy.spatial import Delaunay
import numpy as np
import os


class _DOSMixin:
    def plot_dos(self):
        if self.tetra_integral:
            k_grid, self.weight_k, grid_all, self.grid2ir_idx = self._get_ir_reciprocal_mesh(
                self.dos_mesh_size, auxiliary_info=True, return_frac=True)
            delaunay = Delaunay(grid_all)
            tetrahedrons = delaunay.simplices  # (ntetras, 4)
        else:
            k_grid, self.weight_k, _, _ = self._get_ir_reciprocal_mesh(
                self.dos_mesh_size, auxiliary_info=True, return_frac=True)
            tetrahedrons = None
        energies = np.linspace(self.dos_erange[0], self.dos_erange[1], self.dos_npoints)
        if self.dos_qp == 'electron':
            if self.spinor_mode == 'none':
                self.weight_k *= 2.0
            dos_list = self._cal_electron_dos(k_grid, energies, tetrahedrons=tetrahedrons)
            dos_list = dos_list / QTcts.HARTREEtoEV
            # shift the energy reference to fermi level
            energies = energies * QTcts.HARTREEtoEV
            if self.rank == 0:
                filepath = os.path.join(self.outdir, 'electron_dos.dat')
                if self.spinor_mode == 'collinear':
                    output_data = np.column_stack((energies, dos_list.T))
                    np.savetxt(filepath, output_data, fmt="%12.8e", header="Energy(eV)    DOS-up(1/eV)    DOS-down(1/eV)")
                else:
                    output_data = np.column_stack((energies, dos_list))
                    np.savetxt(filepath, output_data, fmt="%12.8e", header="Energy(eV)    DOS(1/eV)")
        elif self.dos_qp == 'phonon':
            dos_list = self._cal_phonon_dos(k_grid, energies, tetrahedrons=tetrahedrons)
            dos_list = dos_list / QTcts.HARTREEtoMEV
            energies = energies * QTcts.HARTREEtoMEV
            if self.rank == 0:
                filepath = os.path.join(self.outdir, 'phonon_dos.dat')
                output_data = np.column_stack((energies, dos_list))
                np.savetxt(filepath, output_data, fmt="%12.8e", header="Energy(meV)    DOS(1/meV)")
        else:
            raise RuntimeError(f"Cannot calculate density of states for {self.dos_qp}.", '5732')

    def _get_is_metal(self, enks:np.ndarray, efermi:float):
        """ 
        Decide whether the system is a metal or insulator.
        Args:
            enks (np.ndarray): The energy list to calculate DOS. # shape if collinear (2*norbs, nks), else (norbs, nks)
            efermi (float): The fermi energy.

        Returns:
            is_metal (bool): Return True, if the system is a metal.
        """
        is_metal = False
        nbands, _ = enks.shape
        emins = np.min(enks, axis=1)
        emaxs = np.max(enks, axis=1)
        for iband in range(nbands):
            if emins[iband] < efermi and emaxs[iband] > efermi:
                is_metal = True
                break
        return is_metal

    def _cal_electron_dos(self, k_grid:np.ndarray, energies:np.ndarray, tetrahedrons:np.ndarray=None):
        """ 
        Calculate the density of states for electron corresponding to a special energy array.

        Args:
            k_grid (np.ndarray): The k sampling points. # shape (nks, 3)
            energies (np.ndarray): The energy list to calculate DOS. # shape (nenes, )
            tetrahedrons (np.ndarray): Indices of k-points forming each tetrahedron. # shape (ntetras, 4)

        Returns:
            dos_list (np.ndarray): The DOS corresponding to the energies. # shape (nenes, )
        """
        nks = len(k_grid)
        nenes = len(energies)
        # use k points parallel scheme
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nks):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_all = np.split(k_grid, indices_or_sections=split_sections, axis=0)
        weights_k_all = np.split(self.weight_k, indices_or_sections=split_sections, axis=0)
        if k_grid_all[self.rank].size > 0:            
            # enks_local (norbs, nks_local)
            nks_local = len(k_grid_all[self.rank])
            k_grid_all[self.rank] = self._frac2car(k_grid_all[self.rank])
            enks_local, _ = self._elec_cal_torch(k_grid_all[self.rank])
            weights_k_local = weights_k_all[self.rank]
            if self.spinor_mode == 'collinear':
                # enks_local (2, norbs, nks_local)
                enks_local = enks_local.reshape(2, self.norbs, nks_local)
        else:
            weights_k_local = np.empty((0))
            if self.spinor_mode == 'collinear':
                enks_local = np.empty((2, self.norbs, 0))
            else:
                enks_local = np.empty((self.norbs, 0))
        if self.rank == 0:
            enks_all = [None] * self.rank_size
        else:
            enks_all = None
        self.comm.Barrier()
        if self.tetra_integral:
            ntetras = len(tetrahedrons)
            enks_all = self.comm.allgather(enks_local) 
        else:
            enks_all = self.comm.gather(enks_local, root=0)
        if self.rank == 0:
            enks = np.concatenate(enks_all, axis=-1) # if collinear (2, norbs, nks), else (norbs, nks)
            num_electron_include = self.graph_data.num_electrons - self.doping_charge
            self.efermi = self._get_fermi_level_metal(enks.reshape(-1, nks), self.dos_smear, self.dos_gauss_type, num_electron_include) + self.fermi_shift
            is_metal = self._get_is_metal(enks.reshape(-1, nks), self.efermi)
            if is_metal:
                print("The system is a metal, the Fermi level of {:.4f} eV is chosen as the reference energy.".format(self.efermi*QTcts.HARTREEtoEV))
            else:
                if self.spinor_mode == 'none':
                    emaxs = np.sort(np.max(enks, axis=1).repeat(2))
                elif self.spinor_mode == 'collinear':
                    emaxs = np.sort(np.max(enks.reshape(-1, nks), axis=1))
                elif self.spinor_mode == 'nc':
                    emaxs = np.sort(np.max(enks, axis=1))
                self.efermi = emaxs[self.graph_data.num_electrons-1]
                print("The system is an insulator, the VBM energy of {:.4f} eV is chosen as the reference energy.".format(self.efermi*QTcts.HARTREEtoEV))
            ele_data_tmp = [self.efermi]
        else:
            ele_data_tmp = None
            if self.tetra_integral:
                enks = np.concatenate(enks_all, axis=-1) # if collinear (2, norbs, nks), else (norbs, nks)
        ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
        self.efermi = ele_data_tmp[0]
        enks_local = enks_local - self.efermi
        # calculate the density of states
        if self.tetra_integral:
            enks = enks - self.efermi
            if self.spinor_mode != 'nc':
                enks = enks[..., self.grid2ir_idx]
            # use tetrahedron parallel scheme
            split_sections = np.zeros(self.rank_size, dtype=int)
            for i in range(ntetras):
                split_sections[i%self.rank_size] += 1
            split_sections = np.cumsum(split_sections, axis=0)
            tetrahedron_all = np.split(tetrahedrons, indices_or_sections=split_sections, axis=0)
            if tetrahedron_all[self.rank].size > 0:
                tetrahedron_local = tetrahedron_all[self.rank]
            else:
                tetrahedron_local = np.empty((0, 4))
            if self.spinor_mode == 'collinear':
                dos_list = np.zeros((2, nenes), dtype=float)
                for ispin in range(2):
                    for iene in range(nenes):
                        dos_list[ispin, iene] = tetrahedron_integral(tetrahedron_local, enks[ispin], energies[iene])
                dos_list = dos_list / ntetras
            else:
                dos_list = np.zeros(nenes, dtype=float)
                for iene in range(nenes):
                    dos_list[iene] = tetrahedron_integral(tetrahedron_local, enks, energies[iene])
                if self.spinor_mode == 'nc':
                    dos_list = dos_list / ntetras
                else:
                    dos_list = dos_list / ntetras * 2.0
        else:
            if self.spinor_mode == 'collinear':
                dos_list = np.zeros((2, nenes), dtype=float)
                for ispin in range(2):
                    # delta_func: # shape (nenes, norbs, nks_local)
                    delta_func = w0gauss((enks_local[np.newaxis,ispin,:,:] - energies[:,np.newaxis,np.newaxis]) * self.inv_dos_smear, self.dos_gauss_type) * self.inv_dos_smear
                    dos_list[ispin] = np.einsum("ink, k->i", delta_func, weights_k_local)
            else:
                dos_list = np.zeros(nenes, dtype=float)
                # delta_func: # shape (nenes, norbs, nks_local)
                delta_func = w0gauss((enks_local[np.newaxis,:,:] - energies[:,np.newaxis,np.newaxis]) * self.inv_dos_smear, self.dos_gauss_type) * self.inv_dos_smear
                dos_list = np.einsum("ink, k->i", delta_func, weights_k_local)
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, dos_list, op=MPI.SUM)
        return dos_list

    def _cal_phonon_dos(self, k_grid:np.ndarray, energies:np.ndarray, tetrahedrons:np.ndarray=None):
        """ 
        Calculate the density of states for phonon corresponding to a special energy array.

        Args:
            k_grid (np.ndarray): The k sampling points. # shape (nks, 3)
            energies (np.ndarray): The energy list to calculate DOS. # shape (nenes, )
            tetrahedrons (np.ndarray): Indices of k-points forming each tetrahedron. # shape (ntetras, 4)

        Returns:
            dos_list (np.ndarray): The DOS corresponding to the energies. # shape (nenes, )
        """
        nks = len(k_grid)
        nenes = len(energies)
        nbranches = int(3) * self.natoms
        # use k points parallel scheme
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nks):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_all = np.split(k_grid, indices_or_sections=split_sections, axis=0)
        weights_k_all = np.split(self.weight_k, indices_or_sections=split_sections, axis=0)
        if k_grid_all[self.rank].size > 0:            
            # wqvs_local: # shape (nks_local, nbranches)
            wqvs_local, _ = self._phonon_cal(k_grid_all[self.rank])
            weights_k_local = weights_k_all[self.rank]
        else:
            weights_k_local = np.empty((0))
            wqvs_local = np.empty((0, nbranches))
        dos_list = np.zeros(nenes, dtype=float)
        if self.tetra_integral:
            ntetras = len(tetrahedrons)
            wqvs_local = wqvs_local.transpose()
            wqvs_all = self.comm.allgather(wqvs_local) 
            wqvs = np.concatenate(wqvs_all, axis=-1)
            if self.spinor_mode != 'nc':
                wqvs = wqvs[..., self.grid2ir_idx]
            # use tetrahedron parallel scheme
            split_sections = np.zeros(self.rank_size, dtype=int)
            for i in range(ntetras):
                split_sections[i%self.rank_size] += 1
            split_sections = np.cumsum(split_sections, axis=0)
            tetrahedron_all = np.split(tetrahedrons, indices_or_sections=split_sections, axis=0)
            if tetrahedron_all[self.rank].size > 0:
                tetrahedron_local = tetrahedron_all[self.rank]
            else:
                tetrahedron_local = np.empty((0, 4))
            for iene in range(nenes):
                dos_list[iene] = tetrahedron_integral(tetrahedron_local, wqvs, energies[iene])
            dos_list = dos_list / ntetras
        else:
            # delta_func: # shape (nenes, nks_local, nbranches)
            delta_func = w0gauss((wqvs_local[np.newaxis,:,:] - energies[:,np.newaxis,np.newaxis]) * self.inv_dos_smear, self.dos_gauss_type) * self.inv_dos_smear
            dos_list = np.einsum("ikn, k->i", delta_func, weights_k_local)
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, dos_list, op=MPI.SUM)
        return dos_list
