"""Mechanically extracted QTrans methods."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.numerics.distributions import bose_weight
from qtrans.numerics.distributions import fermi_weight
from qtrans.numerics.distributions import minus_dfermi
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import gather_to_rank_root
from typing import List
import numpy as np
import opt_einsum as oe
import os
import sys


class _TransportSolverMixin:
    def scattering_rate_cal_epc_polar(self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float):
        """
        Calculate the polar long-range contribution to EPC scattering rates.

        This routine evaluates only the long-range polar correction part of the EPC
        scattering rate. It does not use the short-range gradient matrix. Only q points
        marked by ``apply_correction_for_qs`` contribute. If ``self.mode_resolved`` is True,
        per-phonon-branch polar rates are saved to ``rate_mode_resolved.npy`` and the
        corresponding eigenvalues are saved to ``enks_all.npy``.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape (nks_local, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the transport window, shape (nbands,).
            band_edge_energy (float): The energy of the band edge.

        Returns:
            tuple: ``(rate_all, eigen, vnks)``.

            rate_all (np.ndarray): Polar scattering rate, shape (nbands, nks_local).
            eigen (np.ndarray): Eigenvalues on the local k grid, shape (nbands, nks_local).
            vnks (np.ndarray): Band velocities, shape (nks_local, nbands, 3).

        Side Effects:
            If ``self.mode_resolved`` is True, saves ``rate_mode_resolved.npy`` with shape
            (nbands, nks_local, nbranches), and saves ``enks_all.npy`` on rank 0.
        """

        nbranches = self._get_nbranches()
        nbands = len(band_indice)
        nks = len(k_grid)
        mode_resolved = self.mode_resolved

        rate_all = np.full((nbands, nks), np.inf)
        rate_all_resolved = np.full((nbands, nks, nbranches), np.inf) if mode_resolved else None
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy)

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc_polar')

        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq] for iq in range(len(q_grid_local))]
        bose_grid = [bose_weight(freq_grid[iq], self.temperature) for iq in range(len(q_grid_local))]

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik, :]
                if self.is_hole:
                    enk_inside_table = (eig_k >= efocus_threshold)
                else:        
                    enk_inside_table = (eig_k <= efocus_threshold)
                if not enk_inside_table.any():
                    continue
                rate_all[enk_inside_table, ik_all] = 0.0
                if mode_resolved:
                    rate_all_resolved[enk_inside_table, ik_all, :] = 0.0
                ibands_inside = np.where(enk_inside_table)[0]
                for iq, q in enumerate(q_grid_local):
                    if not apply_correction_for_qs[iq]:
                        continue
                    # phonon spectrum
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                    eig_kpq = eig_kpq[band_indice, 0]
                    wave_kpq = wave_kpq[0, band_indice, :]
                    if self.is_hole:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold)
                    else:        
                        emkpq_inside_table = (eig_kpq <= efocus_threshold)
                    if not emkpq_inside_table.any():
                        continue 
                    jbands_inside = np.where(emkpq_inside_table)[0]
                    freq = freq_grid[iq]
                    phvec_wap = phvec_waps_all[iq]
                    match_table = self._get_match_table(eig_k, eig_kpq, freq)
                    if not match_table.any():
                        continue
                    bose_qvs = bose_grid[iq]
                    fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                    # phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    epc_corr = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=False)
                    for iband in ibands_inside:
                        for jband in jbands_inside:
                            for ibranch in range(nbranches):
                                if match_table[iband, jband, ibranch]:
                                    epc = epc_corr[iband, jband, ibranch]
                                    delta_f1 = w0gauss((eig_k[iband] - eig_kpq[jband] + freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                    delta_f2 = w0gauss((eig_k[iband] - eig_kpq[jband] - freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                    rate_tmp = g2_tmp * ((bose_qvs[ibranch] + fermi_kpqs[jband]) * delta_f1 + 
                                                                    (bose_qvs[ibranch] + 1.0 - fermi_kpqs[jband]) * delta_f2) * weights_q_local[iq]
                                    rate_all[iband,ik_all] += rate_tmp
                                    if mode_resolved:
                                        rate_all_resolved[iband,ik_all,ibranch] += rate_tmp
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= QTcts.TWOPI
        if mode_resolved:
            rate_all_resolved *= QTcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
            if mode_resolved:
                self.comm.Allreduce(MPI.IN_PLACE, rate_all_resolved, op=MPI.SUM)
                enks_all = gather_to_rank_root(eigen, self.comm)
            self.comm.Barrier()
        elif mode_resolved:
            enks_all = [eigen]
        if self.rank == 0 and mode_resolved:
            np.save('rate_mode_resolved.npy', rate_all_resolved)
            enks_all = np.concatenate(enks_all, axis=-1) # (nbands, nks)
            np.save('enks_all.npy', enks_all)
        # sanity check!
        rate_all[np.abs(rate_all) < QTcts.TENPM160] = np.inf
        return rate_all, eigen, vnks

    def scattering_rate_cal_epc_polar_collinear(self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float):
        """
        Calculate the polar long-range contribution to EPC scattering rates in collinear mode.

        This is the spin-resolved collinear version of ``scattering_rate_cal_epc_polar``.
        It evaluates only the long-range polar correction and does not use the short-range
        gradient matrix. If ``self.mode_resolved`` is True, per-phonon-branch polar rates
        are saved to ``rate_mode_resolved.npy``.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape (nks_local, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the transport window, shape (nbands,).
            band_edge_energy (float): The energy of the band edge.

        Returns:
            tuple: ``(rate_all, eigen, vnks)``.

            rate_all (np.ndarray): Polar scattering rate, shape (2, nbands, nks_local).
            eigen (np.ndarray): Eigenvalues on the local k grid, shape (2, nbands, nks_local).
            vnks (np.ndarray): Band velocities, shape (2, nks_local, nbands, 3).

        Side Effects:
            If ``self.mode_resolved`` is True, saves ``rate_mode_resolved.npy`` with shape
            (2, nbands, nks_local, nbranches).
        """
        nbranches = self._get_nbranches()
        nbands = len(band_indice)
        nks = len(k_grid)
        mode_resolved = self.mode_resolved

        rate_all = np.full((2, nbands, nks), np.inf)
        rate_all_resolved = np.full((2, nbands, nks, nbranches), np.inf) if mode_resolved else None
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy)

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc_polar_collinear')

        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq] for iq in range(len(q_grid_local))]
        bose_grid = [bose_weight(freq_grid[iq], self.temperature) for iq in range(len(q_grid_local))]

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, :, ik], eigen_vecs_recv[:, ik]
                if self.is_hole:
                    enk_inside_table = (eig_k >= efocus_threshold[:,None])
                else:        
                    enk_inside_table = (eig_k <= efocus_threshold[:,None])
                if not enk_inside_table.any():
                    continue
                rate_all[enk_inside_table, ik_all] = 0.0
                if mode_resolved:
                    for ispin in range(2):
                        rate_all_resolved[ispin, enk_inside_table[ispin], ik_all, :] = 0.0
                ibands_inside_by_spin = [np.where(enk_inside_table[ispin])[0] for ispin in range(2)]
                # phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    if not apply_correction_for_qs[iq]:
                        continue
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                    eig_kpq = np.stack((eig_kpq[band_indice, 0], eig_kpq[band_indice+self.norbs, 0]), axis=0)
                    wave_kpq = np.stack((wave_kpq[0, band_indice], wave_kpq[0, band_indice+self.norbs]), axis=0)
                    if self.is_hole:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold[:, None])
                    else:        
                        emkpq_inside_table = (eig_kpq <= efocus_threshold[:, None])
                    if not emkpq_inside_table.any():
                        continue 
                    jbands_inside_by_spin = [np.where(emkpq_inside_table[ispin])[0] for ispin in range(2)]
                    freq = freq_grid[iq]
                    phvec_wap = phvec_waps_all[iq]
                    bose_qvs = bose_grid[iq]
                    # phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    for ispin in range(2):
                        match_table = self._get_match_table(eig_k[ispin], eig_kpq[ispin], freq)
                        if not match_table.any():
                            continue
                        fermi_kpqs = fermi_weight(eig_kpq[ispin] - self.efermi[ispin], self.temperature)
                        epc_corr = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table[ispin], k, wave_k[ispin], wave_kpq[ispin], return_tmp1=False)
                        for iband in ibands_inside_by_spin[ispin]:
                            for jband in jbands_inside_by_spin[ispin]:
                                for ibranch in range(nbranches):
                                    if match_table[iband, jband, ibranch]:
                                        epc = epc_corr[iband, jband, ibranch]
                                        delta_f1 = w0gauss((eig_k[ispin,iband] - eig_kpq[ispin,jband] + freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                        delta_f2 = w0gauss((eig_k[ispin,iband] - eig_kpq[ispin,jband] - freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                        g2_tmp = np.abs(epc) * np.abs(epc)
                                        rate_tmp = g2_tmp * ((bose_qvs[ibranch] + fermi_kpqs[jband]) * delta_f1 +
                                                                        (bose_qvs[ibranch] + 1.0 - fermi_kpqs[jband]) * delta_f2) * weights_q_local[iq]
                                        rate_all[ispin,iband,ik_all] += rate_tmp
                                        if mode_resolved:
                                            rate_all_resolved[ispin, iband, ik_all, ibranch] += rate_tmp
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= QTcts.TWOPI
        if mode_resolved:
            rate_all_resolved *= QTcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
            if mode_resolved:
                self.comm.Allreduce(MPI.IN_PLACE, rate_all_resolved, op=MPI.SUM)
        if self.rank == 0 and mode_resolved:
            np.save('rate_mode_resolved.npy', rate_all_resolved)
        # sanity check!
        rate_all[np.abs(rate_all) < QTcts.TENPM160] = np.inf
        return rate_all, eigen, vnks

    def _read_rate_file(self, filepath:str, shape:List[int]):
        rate_ir = None
        data = np.atleast_2d(np.loadtxt(filepath, skiprows=2))
        if data.size == 0:
            if self.cal_mode == 'conductivity' and self.magnetic_type == 'collinear':
                return np.full([2] + shape, np.inf)
            else:
                return np.full(shape, np.inf)
        ik_arr = data[:, 0].astype(int)
        ibnd_arr = data[:, 1].astype(int)
        if self.cal_mode == 'conductivity':
            if self.magnetic_type == 'collinear':
                rate_ir = np.full([2] + shape, np.inf)
                rate_ir[0, ibnd_arr, ik_arr] = data[:, 4]
                rate_ir[1, ibnd_arr, ik_arr] = data[:, 6]
            else:
                rate_ir = np.full(shape, np.inf)
                rate_ir[ibnd_arr, ik_arr] = data[:, 4]
        elif self.cal_mode == 'breakdown':
            rate_ir = np.full(shape, np.inf)
            rate_ir[0, ibnd_arr, ik_arr] = data[:, 4]
            rate_ir[1, ibnd_arr, ik_arr] = data[:, 5]

        return rate_ir

    def _write_rate_file(self, filepath:str, weight_k:np.ndarray, enks:np.ndarray, rate:np.ndarray, with_header:bool=True, is_metal:bool=False):
        enks_shape = enks.shape
        nks = enks_shape[-1]
        nbands = enks_shape[-2]
        rate_shape = rate.shape
        out_case = len(enks_shape) + len(rate_shape)

        # Build (ibnd, ik) index grid: ibnd varies fastest (inner loop), ik slowest (outer loop)
        ik_all, ibnd_all = np.meshgrid(np.arange(nks), np.arange(nbands), indexing='ij')
        ik_flat = ik_all.ravel()
        ibnd_flat = ibnd_all.ravel()

        with open(filepath, 'w') as fout:
            if out_case == 4:
                if with_header:
                    if is_metal:
                        fout.write(f"#Fermi level = {self.efermi:.8f} Hartree\n")
                    else:
                        fout.write(f"#Fermi level = {self.efermi:.8f} Hartree    Density of carrier = {self.carrier_density:.8e} \n")
                    fout.write(f"#{'ik_ibz':<8}{'ibnd':<6}{'weight_k':<18}{'enk(Hartree)':<18}{'scattering_rate(Hartree)':<24}\n")
                mask = ~np.isinf(rate[ibnd_flat, ik_flat])
                iks, ibnds = ik_flat[mask], ibnd_flat[mask]
                lines = [f"{ik:<9}{ibnd:<6}{weight_k[ik]:<18.12f}{enks[ibnd, ik]:<18.12f}{rate[ibnd, ik]:<24.16e}"
                         for ik, ibnd in zip(iks, ibnds)]
                fout.write('\n'.join(lines))
                if lines:
                    fout.write('\n')
            elif out_case == 5:
                if with_header:
                    fout.write(f"#{'ik_ibz':<8}{'ibnd':<6}{'weight_k':<18}{'enk(Hartree)':<18}{'scattering_rate(Hartree)':<24}{'energy_loss_rate(Hartree)':<24}\n")
                mask = ~np.isinf(rate[0, ibnd_flat, ik_flat])
                iks, ibnds = ik_flat[mask], ibnd_flat[mask]
                lines = [f"{ik:<9}{ibnd:<6}{weight_k[ik]:<18.12f}{enks[ibnd, ik]:<18.12f}{rate[0, ibnd, ik]:<24.16e}{rate[1, ibnd, ik]:<24.16e}"
                         for ik, ibnd in zip(iks, ibnds)]
                fout.write('\n'.join(lines))
                if lines:
                    fout.write('\n')
            elif out_case == 6:
                if with_header:
                    if is_metal:
                        fout.write(f"#Fermi level = {self.efermi:.8f} Hartree\n")
                    else:
                        fout.write(f"#Fermi level = {self.efermi[0]:.8f} Hartree and {self.efermi[1]:.8f} Hartree   Density of carrier = {self.carrier_density[0]:.8e} and {self.carrier_density[1]:.8e}\n")
                    fout.write(
                        f"#{'ik_ibz':<8}{'ibnd':<6}{'weight_k':<18}"
                        f"{'enk[up](Hartree)':<18}{'scatt_rate[up](Hartree)':<24}"
                        f"{'enk[down](Hartree)':<18}{'scatt_rate[down](Hartree)':<24}\n"
                    )
                mask = ~np.isinf(rate[:, ibnd_flat, ik_flat]).all(axis=0)
                iks, ibnds = ik_flat[mask], ibnd_flat[mask]
                lines = [f"{ik:<9}{ibnd:<6}{weight_k[ik]:<18.12f}"
                         f"{enks[0, ibnd, ik]:<18.12f}"
                         f"{rate[0, ibnd, ik]:<24.16e}"
                         f"{enks[1, ibnd, ik]:<18.12f}"
                         f"{rate[1, ibnd, ik]:<24.16e}"
                         for ik, ibnd in zip(iks, ibnds)]
                fout.write('\n'.join(lines))
                if lines:
                    fout.write('\n')
            else:
                raise NotImplementedError("Cannot output the rate file.", '7101')

    def _write_scattering_channel(self, scat_channel:np.ndarray, output_dir:str="scattering_channels"):
        output_dir = os.path.join(self.outdir, output_dir)
        if self.rank == 0 and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.comm.Barrier()
        filename = f"epc_scat_channel_rank{self.rank}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, scat_channel)

    def _read_scattering_channel(self, folder_path):
        filename = f"epc_scat_channel_rank{self.rank}.npy"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[Rank {self.rank}] Cannot find scat_channel file: {filepath}")
        scat_channel = np.load(filepath, allow_pickle=True)
        return scat_channel

    def _get_epc_func(self):
        """Return the appropriate epc scattering rate method based on spinor_mode and sparse/dense."""
        if self.spinor_mode == 'collinear':
            return self.scattering_rate_cal_epc_collinear_sparse if self.read_sparse_grad_mat \
                else self.scattering_rate_cal_epc_collinear_dense
        else:
            return self.scattering_rate_cal_epc_sparse if self.read_sparse_grad_mat \
                else self.scattering_rate_cal_epc_dense

    def _dispatch_scattering_rate(self, k_grid:np.ndarray, q_grid:np.ndarray, bands_indices:np.ndarray,
                                   energy_ref, is_metal:bool, polar_split:str=None):
        """
        Dispatch the appropriate scattering rate calculation method.

        Selects the function based on spinor_mode, read_sparse_grad_mat,
        mob_level, and polar_split. For IBTE, also writes the scattering channel to disk.

        Args:
            k_grid (np.ndarray): useful IBZ k-points.
            q_grid (np.ndarray): full BZ q-points.
            bands_indices (np.ndarray): band index array.
            energy_ref: reference energy (Fermi level for metal, band edge for insulator).
            is_metal (bool): True for metal, False for insulator.
            polar_split (str or None): polar splitting mode (insulator phonon only).

        Returns:
            rate_useful (np.ndarray): scattering rates on useful k-points.
            eigen_local (np.ndarray): local eigenvalues.
            vnks_local (np.ndarray): local band velocities.
            fout_name (str): output filename for the rate file.
            scat_channel_local: scattering channel data (only for IBTE), or None.
        """
        scat_channel_local = None
        col_suffix = '_collinear' if self.spinor_mode == 'collinear' else ''

        if not is_metal and polar_split == 'polar' and self.mob_level == 'erta':
            fout_name = f'rate_nk_polar{col_suffix}.dat'
            if self.spinor_mode == 'collinear':
                func = self.scattering_rate_cal_epc_polar_collinear
            else:
                func = self.scattering_rate_cal_epc_polar
            rate_useful, eigen_local, vnks_local = func(k_grid, q_grid, bands_indices, energy_ref)
            return rate_useful, eigen_local, vnks_local, fout_name, None

        if not is_metal and polar_split == 'rmp' and self.mob_level == 'erta':
            fout_name = f'rate_nk_rmp{col_suffix}.dat'
            func = self._get_epc_func()
            rate_useful, eigen_local, vnks_local = func(
                k_grid, q_grid, bands_indices, energy_ref, mob_level='erta', is_rmp=True)
            return rate_useful, eigen_local, vnks_local, fout_name, None

        fout_name = f'rate_nk{col_suffix}.dat'
        func = self._get_epc_func()
        kwargs = {'mob_level': self.mob_level}
        if is_metal:
            kwargs['is_metal'] = True
        result = func(k_grid, q_grid, bands_indices, energy_ref, **kwargs)

        if self.mob_level == 'ibte':
            rate_useful, eigen_local, vnks_local, scat_channel_local = result
            self._write_scattering_channel(scat_channel_local)
        else:
            rate_useful, eigen_local, vnks_local = result

        return rate_useful, eigen_local, vnks_local, fout_name, scat_channel_local

    def _read_and_broadcast_rate(self, path:str, label:str, nbands:int, nks_ibz:int):
        """
        Read a single rate file on rank 0 and broadcast to all ranks.

        Args:
            path (str): path to rate file.
            label (str): human-readable label for log messages.
            nbands (int): number of bands.
            nks_ibz (int): number of IBZ k-points.

        Returns:
            rate_ir (np.ndarray): rate array, broadcast to all ranks.
        """
        rate_ir = None
        if self.rank == 0:
            print(f"Reading {label} scattering rate from {path}")
            rate_ir = self._read_rate_file(path, [nbands, nks_ibz])
        rate_ir = self.comm.bcast(rate_ir, root=0)
        return rate_ir

    def _merge_rate_files(self, path_a:str, path_b:str, label_a:str, label_b:str, nbands:int, nks_ibz:int):
        """
        Read two rate files, check inf-mask consistency, merge by addition, and broadcast.

        Args:
            path_a (str): path to first rate file.
            path_b (str): path to second rate file.
            label_a (str): human-readable label for first file.
            label_b (str): human-readable label for second file.
            nbands (int): number of bands.
            nks_ibz (int): number of IBZ k-points.

        Returns:
            rate_ir (np.ndarray): merged rate array, broadcast to all ranks.
        """
        rate_ir = None
        if self.rank == 0:
            print(f"Reading {label_a} scattering rate from {path_a}")
            print(f"Reading {label_b} scattering rate from {path_b}")
            rate_a = self._read_rate_file(path_a, [nbands, nks_ibz])
            rate_b = self._read_rate_file(path_b, [nbands, nks_ibz])
            if not np.array_equal(np.isinf(rate_a), np.isinf(rate_b)):
                print(f"Warning: Please ensure the {label_a} scattering rate and {label_b} "
                      f"scattering rate are calculated using the same parameters, except for sampling.")
            rate_a[np.isinf(rate_a)] = 0.0
            rate_b[np.isinf(rate_b)] = 0.0
            rate_ir = rate_a + rate_b
            del rate_a, rate_b
            rate_ir[np.abs(rate_ir) < QTcts.TENPM160] = np.inf
        rate_ir = self.comm.bcast(rate_ir, root=0)
        return rate_ir

    def _compute_erta(self, eigen_local, vnks_local, rate_local, k_grid_local, nbands,
                      efermi, all_rots_cart, ibz_to_fbz, k_index_local,
                      sigma_unit, mob_unit_base, is_metal,
                      carrier_density=None, inv_cell=None):
        """
        Compute the ERTA (zero-order BTE) solution and print results.

        Args:
            eigen_local: local eigenvalues.
            vnks_local: local band velocities.
            rate_local: local scattering rates.
            k_grid_local: local k-points.
            nbands: number of bands.
            efermi: Fermi level (scalar for metal/non-collinear insulator, array(2) for collinear insulator).
            all_rots_cart, ibz_to_fbz, k_index_local: symmetry data.
            sigma_unit: conductivity unit conversion factor.
            mob_unit_base: mobility unit base factor.
            is_metal: True for conductivity, False for mobility.
            carrier_density: carrier density (insulator only).
            inv_cell: inverse cell volume (insulator only).

        Returns:
            F_0_local: local F_0 distribution function.
            sigma_serta: ERTA conductivity tensor.
            Seebeck_serta: Seebeck coefficient tensor.
            mobility_serta: ERTA mobility tensor (None for metals).
        """
        is_collinear = (self.spinor_mode == 'collinear')
        shape_33 = (2, 3, 3) if is_collinear else (3, 3)

        if k_grid_local.size > 0:
            nks_local = k_grid_local.shape[0]
            if is_collinear:
                F_0_local = np.zeros((2, nks_local, nbands, 3))
                sigma_temp = np.zeros((2, 3, 3))
                sigma_See_temp = np.zeros((2, 3, 3))
                for ispin in range(2):
                    ef = efermi if is_metal else efermi[ispin]
                    mdf = minus_dfermi(eigen_local[ispin] - ef, self.temperature)
                    F_0_local[ispin] = oe.contract('nk, kni, nk->kni', mdf, vnks_local[ispin], 1.0 / rate_local[ispin])
                    sigma_temp[ispin], sigma_See_temp[ispin] = self.get_conductivity_symm(
                        eigen_local[ispin] - ef, vnks_local[ispin], F_0_local[ispin], all_rots_cart, ibz_to_fbz, k_index_local)
            else:
                F_0_local = np.zeros((nks_local, nbands, 3))
                mdf = minus_dfermi(eigen_local - efermi, self.temperature)
                F_0_local = oe.contract('nk, kni, nk->kni', mdf, vnks_local, 1.0 / rate_local)
                sigma_temp, sigma_See_temp = self.get_conductivity_symm(
                    eigen_local - efermi, vnks_local, F_0_local, all_rots_cart, ibz_to_fbz, k_index_local)
        else:
            if is_collinear:
                F_0_local = np.zeros((2, 0, nbands, 3))
            else:
                F_0_local = np.zeros((0, nbands, 3))
            sigma_temp = np.zeros(shape_33)
            sigma_See_temp = np.zeros(shape_33)

        mobility_serta = None
        if is_metal:
            sigma_temp = sigma_temp * sigma_unit
            sigma_See_temp = sigma_See_temp * sigma_unit
        else:
            mobility_temp = sigma_temp * mob_unit_base
            if is_collinear:
                for ispin in range(2):
                    mobility_temp[ispin] = mobility_temp[ispin] / (QTcts.EV * carrier_density[ispin] * inv_cell) * (QTcts.BOHRtoCM ** 3)
            else:
                mobility_temp = mobility_temp / (QTcts.EV * carrier_density * inv_cell) * (QTcts.BOHRtoCM ** 3)

        sigma_serta = np.zeros(shape_33)
        sigma_See_serta = np.zeros(shape_33)
        if self.comm is not None:
            self.comm.Allreduce(sigma_temp, sigma_serta, op=MPI.SUM)
            self.comm.Allreduce(sigma_See_temp, sigma_See_serta, op=MPI.SUM)
        if not is_metal:
            mobility_serta = np.zeros(shape_33)
            if self.comm is not None:
                self.comm.Allreduce(mobility_temp, mobility_serta, op=MPI.SUM)
        Seebeck_serta = np.linalg.pinv(sigma_serta) @ sigma_See_serta / self.temperature / QTcts.HARTREEtoKELVIN * QTcts.HARTREEtoEV

        if not is_collinear and self.spinor_mode == 'none':
            if is_metal:
                sigma_serta = sigma_serta * 2
            else:
                mobility_serta = mobility_serta * 2

        if self.rank == 0:
            print('=' * 100)
            print(' ' * 32 + 'Energy Relaxation Time Approximation')
            print('=' * 100)
            primary = sigma_serta if is_metal else mobility_serta
            primary_label = 'Conductivity(Siemens/m)' if is_metal else 'Mobility(cm^2/V/s)'
            seebeck_label = 'Seebeck coefficient(V/K)' if is_metal else 'Seebeck(Volt/Kelvin)'
            if is_collinear:
                for ispin in range(2):
                    p_label = f"Conductivity[spin-{ispin+1}](Siemens/m)" if is_metal else f"Mobility[spin-{ispin+1}](cm^2/V/s)"
                    s_label = f"Seebeck coefficient[spin-{ispin+1}](V/K)" if is_metal else f"Seebeck[spin-{ispin+1}](Volt/Kelvin)"
                    print(f"{p_label}      x                y                z")
                    print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][0]))
                    print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][1]))
                    print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][2]))
                    print(f"{s_label}      x                y                z")
                    print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_serta[ispin][0]))
                    print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_serta[ispin][1]))
                    print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_serta[ispin][2]))
            else:
                print(f"{primary_label}           x                y                z")
                print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[0]))
                print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[1]))
                print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[2]))
                print(f"{seebeck_label}           x                y                z")
                print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_serta[0]))
                print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_serta[1]))
                print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_serta[2]))
            print('\n')
        sys.stdout.flush()
        return F_0_local, sigma_serta, Seebeck_serta, mobility_serta

    def _run_ibte_iterations(self, F_0_local, eigen_local, vnks_local, rate_useful,
                             k_grid_local, nbands, nks_ibz_usefull, efermi,
                             all_rots_cart, ibz_to_fbz, k_index_local,
                             rot_indices, unique_kpts, sl_start, sl_end,
                             sigma_unit, mob_unit_base,
                             is_metal, sigma_serta, mobility_serta=None,
                             scat_channel_local=None,
                             carrier_density=None, inv_cell=None):
        is_collinear = (self.spinor_mode == 'collinear')
        shape_33 = (2, 3, 3) if is_collinear else (3, 3)

        F_0 = self.comm.allgather(F_0_local)
        F_0 = np.concatenate(F_0, axis=1 if is_collinear else 0)
        F_in = F_0
        conv_in = sigma_serta.copy() if is_metal else mobility_serta.copy()

        cycle = 0
        error = QTcts.TENPP6
        while error > self.ibte_error and cycle < self.ibte_cycle:
            cycle += 1
            sigma_temp = np.zeros(shape_33)
            sigma_See_temp = np.zeros(shape_33)

            if is_collinear:
                if k_grid_local.size > 0:
                    F_out = np.zeros((2, nks_ibz_usefull, nbands, 3))
                    for ispin in range(2):
                        jspin = int(1) - ispin
                        F_out[ispin] = self.get_F_nk(F_in[ispin], rate_useful[ispin], scat_channel_local[ispin],
                                                        rot_indices, all_rots_cart, unique_kpts)
                else:
                    F_out = np.zeros((2, nks_ibz_usefull, nbands, 3))
            else:
                if k_grid_local.size > 0:
                    F_out = self.get_F_nk(F_in, rate_useful, scat_channel_local, rot_indices, all_rots_cart, unique_kpts)
                else:
                    F_out = np.zeros((nks_ibz_usefull, nbands, 3))

            if self.comm is not None:
                self.comm.Allreduce(MPI.IN_PLACE, F_out, op=MPI.SUM)
            F_out += F_0
            F_out_local = F_out[:, sl_start:sl_end] if is_collinear else F_out[sl_start:sl_end]

            if k_grid_local.size > 0:
                if is_collinear:
                    for ispin in range(2):
                        ef = efermi if is_metal else efermi[ispin]
                        sigma_temp[ispin], sigma_See_temp[ispin] = self.get_conductivity_symm(
                            eigen_local[ispin] - ef, vnks_local[ispin], F_out_local[ispin],
                            all_rots_cart, ibz_to_fbz, k_index_local)
                        if is_metal:
                            sigma_temp[ispin] *= sigma_unit
                            sigma_See_temp[ispin] *= sigma_unit
                else:
                    sigma_temp, sigma_See_temp = self.get_conductivity_symm(
                        eigen_local - efermi, vnks_local, F_out_local,
                        all_rots_cart, ibz_to_fbz, k_index_local)
                    if is_metal:
                        sigma_temp *= sigma_unit
                        sigma_See_temp *= sigma_unit

            if not is_metal:
                mobility_temp = sigma_temp * mob_unit_base
                if is_collinear:
                    for ispin in range(2):
                        mobility_temp[ispin] = mobility_temp[ispin] / (QTcts.EV * carrier_density[ispin] * inv_cell) * (QTcts.BOHRtoCM ** 3)
                else:
                    mobility_temp = mobility_temp / (QTcts.EV * carrier_density * inv_cell) * (QTcts.BOHRtoCM ** 3)

            sigma_out = np.zeros(shape_33)
            sigma_See_out = np.zeros(shape_33)
            if self.comm is not None:
                self.comm.Allreduce(sigma_temp, sigma_out, op=MPI.SUM)
                self.comm.Allreduce(sigma_See_temp, sigma_See_out, op=MPI.SUM)
            if not is_metal:
                mobility_out = np.zeros(shape_33)
                if self.comm is not None:
                    self.comm.Allreduce(mobility_temp, mobility_out, op=MPI.SUM)

            Seebeck_out = np.linalg.pinv(sigma_out) @ sigma_See_out / self.temperature / QTcts.HARTREEtoKELVIN * QTcts.HARTREEtoEV

            if not is_collinear and self.spinor_mode == 'none':
                if is_metal:
                    sigma_out = sigma_out * 2
                else:
                    mobility_out = mobility_out * 2

            conv_out = sigma_out if is_metal else mobility_out
            error = np.max(np.abs(conv_out - conv_in))

            if self.rank == 0:
                print('=' * 100)
                print('Iteration number: {:>4d}'.format(cycle))
                print('=' * 100)
                primary = sigma_out if is_metal else mobility_out
                primary_label = 'Conductivity(Siemens/m)' if is_metal else 'Mobility(cm^2/V/s)'
                seebeck_label = 'Seebeck coefficient(V/K)' if is_metal else 'Seebeck(Volt/Kelvin)'
                if is_collinear:
                    for ispin in range(2):
                        p_label = f"Conductivity[spin-{ispin+1}](Siemens/m)" if is_metal else f"Mobility[spin-{ispin+1}](cm^2/V/s)"
                        s_label = f"Seebeck coefficient[spin-{ispin+1}](V/K)" if is_metal else f"Seebeck[spin-{ispin+1}](Volt/Kelvin)"
                        print(f"{p_label}      x                y                z")
                        print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][0]))
                        print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][1]))
                        print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][2]))
                        print(f"{s_label}      x                y                z")
                        print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_out[ispin][0]))
                        print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_out[ispin][1]))
                        print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_out[ispin][2]))
                else:
                    print(f"{primary_label}           x                y                z")
                    print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[0]))
                    print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[1]))
                    print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[2]))
                    print(f"{seebeck_label}           x                y                z")
                    print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_out[0]))
                    print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_out[1]))
                    print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_out[2]))
                print(f"Error of this iteration: {error:.4e}\n")
            sys.stdout.flush()
            F_in = F_out
            conv_in = conv_out

        if self.rank == 0:
            print('=' * 100)
            if error <= self.ibte_error:
                print(' ' * 27 + f"Converged in {cycle} cycles with error = {error:.4e}")
            else:
                print(' ' * 18 + f"Warning: reached max iteration {self.ibte_cycle} with error = {error:.4e}")
            print('=' * 100)

    def _compute_eigen_velocity_local(self, k_grid_local:np.ndarray, nbands:int):
        """
        Compute local eigenvalues and band velocities for k-points assigned to this MPI rank.

        Args:
            k_grid_local (np.ndarray): k-points owned by this rank, shape (nk_local, 3). May have size 0.
            nbands (int): number of bands in self.bands_indices.

        Returns:
            eigen_local (np.ndarray): eigenvalues.
                Collinear: shape (2, nbands, nk_local). Non-collinear: shape (nbands, nk_local).
            vnks_local (np.ndarray): band velocities.
                Collinear: shape (2, nk_local, nbands, 3). Non-collinear: shape (nk_local, nbands, 3).
        """
        if k_grid_local.size > 0:
            eigen_local, eigen_vecs_local = self._elec_cal_torch(k_grid_local)
            if self.spinor_mode == 'collinear':
                eigen_local = np.stack((eigen_local[self.bands_indices], eigen_local[self.bands_indices+self.norbs]), axis=0)
                eigen_vecs_local = np.stack((eigen_vecs_local[:, self.bands_indices], eigen_vecs_local[:, self.bands_indices+self.norbs]), axis=0)
            else:
                eigen_local = eigen_local[self.bands_indices]
                eigen_vecs_local = eigen_vecs_local[:, self.bands_indices]
            if self.spinor_mode == 'collinear':
                vnks_local = np.zeros((2, len(k_grid_local), nbands, 3))
                vnks_local[0] = self._vel_nk_cal_from_HS(eigen_local[0], eigen_vecs_local[0], k_grid_local, collinear_part_index=0)
                vnks_local[1] = self._vel_nk_cal_from_HS(eigen_local[1], eigen_vecs_local[1], k_grid_local, collinear_part_index=1)
            else:
                vnks_local = self._vel_nk_cal_from_HS(eigen_local, eigen_vecs_local, k_grid_local)
        else:
            if self.spinor_mode == 'collinear':
                eigen_local = np.empty((2, nbands, 0))
                vnks_local = np.empty((2, 0, nbands, 3))
            else:
                eigen_local = np.empty((nbands, 0))
                vnks_local = np.empty((0, nbands, 3))
        return eigen_local, vnks_local
