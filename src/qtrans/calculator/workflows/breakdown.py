"""Scattering-rate breakdown calculation workflow."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.extensions import breakdown_rate_cal_EPC_helper_sparse
from qtrans.numerics.distributions import bose_weight
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import print_peak_memeory
import numpy as np
import os


class _BreakdownMixin:
    def scattering_rate_cal_epc_fbd_dense(self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float):
        """
        Calculate the scattering rate and energy loss rate induced by electron-phonon coupling
        in a non-magnetic system, with a dense gradient matrix.

        This routine is used by field-breakdown calculations. It returns both the carrier
        scattering rate and the energy-loss rate for each selected band and k point. If
        ``self.mode_resolved`` is True, it also returns the contribution resolved by phonon
        branch.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape (nks, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the breakdown window,
                shape (nbands,).
            band_edge_energy (float): Band-edge energy used to construct the focus window.

        Returns:
            tuple: ``(rate_all, rate_mode)``.

            rate_all (np.ndarray): Total rates, shape (2, nbands, nks). Axis 0 stores
                scattering rate at index 0 and energy-loss rate at index 1.
            rate_mode (np.ndarray or None): Mode-resolved scattering and energy-loss rates,
                shape (2, nbranches, nbands, nks). None when ``self.mode_resolved`` is False.
        """

        nbranches = int(3) * self.natoms
        nbands = len(band_indice)
        nks = len(k_grid)
        nqs = len(q_grid)

        efocus_threshold, k_grid_all, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal=False)
        rate_all = np.full((2, nbands, nks), np.inf)
        rate_mode = np.zeros((2, nbranches, nbands, nks)) if self.mode_resolved else None

        # Pre-compute q-only quantities (atomic phase * phonon eigenvector, Bose weight)
        # once per local q point; they are reused across the entire k loop and do not
        # change with k. This restores the April-version optimisation.
        q_phases = []
        q_bose_weights = []
        for iq, q_local in enumerate(q_grid_local):
            atomic_phase_q = np.exp(QTcts.JTWOPI * np.sum(self.graph_data.pos * q_local[None, :], axis=-1))  # (natoms,)
            q_phases.append(atomic_phase_q[None, :, None] * phon_vecs[iq])  # (nbranches, natoms, 3)
            q_bose_weights.append(bose_weight(freq_grid[iq], self.temperature))

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc_fbd')

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik, :]
                enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                if not enk_inside_table.any():
                    continue
                rate_all[:, enk_inside_table, ik_all] = 0.0
                phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                    eig_kpq = eig_kpq[band_indice, 0]
                    wave_kpq = wave_kpq[0, band_indice, :]
                    freq = freq_grid[iq]
                    match_table = self._get_match_table(eig_k, eig_kpq, freq)
                    if not match_table.any():
                        continue
                    phvec_wap = q_phases[iq]      # (nbranches, natoms, 3), reused across all k
                    bose_qvs = q_bose_weights[iq] # (nbranches,)
                    apply_correction_for_this_q = apply_correction_for_qs[iq]
                    phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    if apply_correction_for_this_q:
                        epc_corr, tmp1_save = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=True)
                    else:
                        tmp1_save = np.einsum('jm,in -> jimn', np.conj(wave_kpq), wave_k)
                    for iband in (np.where(enk_inside_table)[0]):
                        for jband in range(nbands):
                            tmp1 = tmp1_save[jband, iband]
                            for ibranch in range(nbranches):
                                if match_table[iband, jband, ibranch]:
                                    epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                                tmp1, self.grad_mat, phase_k, phase_kpq,
                                                                dipole_correction=False)
                                    if apply_correction_for_this_q:
                                        epc = epc + epc_corr[iband, jband, ibranch]
                                    delta_f1 = w0gauss((eig_k[iband] - eig_kpq[jband] + freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                    delta_f2 = w0gauss((eig_k[iband] - eig_kpq[jband] - freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                    g2_tmp = (epc * np.conj(epc)).real
                                    scatt_contrib = g2_tmp * (bose_qvs[ibranch] * delta_f1 + \
                                                                    (bose_qvs[ibranch] + 1.0) * delta_f2) * weights_q_local[iq]
                                    eloss_contrib = g2_tmp * ((bose_qvs[ibranch] + 1.0) * delta_f2 - \
                                                                    bose_qvs[ibranch] * delta_f1) * freq[ibranch] * weights_q_local[iq]
                                    rate_all[0, iband, ik_all] += scatt_contrib
                                    rate_all[1, iband, ik_all] += eloss_contrib
                                    if self.mode_resolved:
                                        rate_mode[0, ibranch, iband, ik_all] += scatt_contrib
                                        rate_mode[1, ibranch, iband, ik_all] += eloss_contrib
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= QTcts.TWOPI
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
        if self.mode_resolved:
            rate_mode *= QTcts.TWOPI
            if self.comm is not None:
                self.comm.Allreduce(MPI.IN_PLACE, rate_mode, op=MPI.SUM)
        return rate_all, rate_mode

    def scattering_rate_cal_epc_fbd_sparse(self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float):
        """
        Calculate the scattering rate and energy loss rate induced by electron-phonon coupling
        in a non-magnetic system, with a sparse gradient matrix (C-accelerated).

        This is the sparse-gradient version of ``scattering_rate_cal_epc_fbd_dense``.
        It uses the C helper to accumulate scattering and energy-loss rates, and optionally
        their phonon-branch-resolved contributions.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape (nks, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the breakdown window,
                shape (nbands,).
            band_edge_energy (float): Band-edge energy used to construct the focus window.

        Returns:
            tuple: ``(rate_all, rate_mode)``.

            rate_all (np.ndarray): Total rates, shape (2, nbands, nks). Axis 0 stores
                scattering rate at index 0 and energy-loss rate at index 1.
            rate_mode (np.ndarray or None): Mode-resolved scattering and energy-loss rates,
                shape (2, nbranches, nbands, nks). None when ``self.mode_resolved`` is False.
        """
        nbranches = int(3) * self.natoms
        nbands = len(band_indice)
        nks = len(k_grid)

        efocus_threshold, k_grid_all, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal=False)

        rate_all = np.full((2, nbands, nks), np.inf)
        rate_mode = np.zeros((2, nbranches, nbands, nks)) if self.mode_resolved else None

        # Pre-compute q-only quantities (atomic phase * phonon eigenvector, Bose weight)
        # once per local q point; they are reused across the entire k loop.
        q_phases = []
        q_bose_weights = []
        for iq, q_local in enumerate(q_grid_local):
            atomic_phase_q = np.exp(QTcts.JTWOPI * np.sum(self.graph_data.pos * q_local[None, :], axis=-1))  # (natoms,)
            q_phases.append(atomic_phase_q[None, :, None] * phon_vecs[iq])  # (nbranches, natoms, 3)
            q_bose_weights.append(bose_weight(freq_grid[iq], self.temperature))

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc_fbd_sparse')

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik, :]
                enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                if not enk_inside_table.any():
                    continue
                rate_all[:, enk_inside_table, ik_all] = 0.0
                phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                    eig_kpq = eig_kpq[band_indice, 0]
                    wave_kpq = wave_kpq[0, band_indice, :]
                    freq = freq_grid[iq]
                    match_table = self._get_match_table(eig_k, eig_kpq, freq)
                    if not match_table.any():
                        continue
                    phvec_wap = q_phases[iq]       # (nbranches, natoms, 3), reused across all k
                    bose_qvs = q_bose_weights[iq]  # (nbranches,)
                    apply_correction_for_this_q = apply_correction_for_qs[iq]
                    phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    if apply_correction_for_this_q:
                        epc_corr = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=False)
                    else:
                        epc_corr = np.zeros((nbands, nbands, nbranches), dtype=complex)
                    breakdown_rate_k = np.zeros((2, nbands), dtype=float, order='C')
                    breakdown_rate_mode_k = np.zeros((2, nbranches, nbands), dtype=float, order='C') if self.mode_resolved else None
                    breakdown_rate_cal_EPC_helper_sparse(np.ascontiguousarray(freq), np.ascontiguousarray(self.atomic_mass),
                                                        np.ascontiguousarray(phvec_wap),
                                                        np.ascontiguousarray(phase_k), np.ascontiguousarray(phase_kpq),
                                                        self.grad_mat,
                                                        np.ascontiguousarray(wave_k), np.ascontiguousarray(wave_kpq),
                                                        np.ascontiguousarray(enk_inside_table), np.ascontiguousarray(match_table),
                                                        np.ascontiguousarray(eig_k), np.ascontiguousarray(eig_kpq),
                                                        np.ascontiguousarray(bose_qvs),
                                                        np.ascontiguousarray(epc_corr), breakdown_rate_k,
                                                        breakdown_rate_mode_k,
                                                        self.inv_smearq, weights_q_local[iq])
                    rate_all[:, :, ik_all] += breakdown_rate_k
                    if self.mode_resolved:
                        rate_mode[:, :, :, ik_all] += breakdown_rate_mode_k
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= QTcts.TWOPI
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
        if self.mode_resolved:
            rate_mode *= QTcts.TWOPI
            if self.comm is not None:
                self.comm.Allreduce(MPI.IN_PLACE, rate_mode, op=MPI.SUM)
        return rate_all, rate_mode

    def breakdown_cal(self):
        """
        Calculate the breakdown field strength.

        This driver builds the k/q grids, computes or reads EPC-induced breakdown rates,
        writes k-resolved and energy-resolved rate files, and estimates the breakdown field
        from the energy-dependent scattering and energy-loss rates.

        Workflow:
        1. Generate k/q grids and compute electronic eigenvalues.
        2. Pre-filter k-points based on the energy focus range to reduce computation.
        3. Compute (or read from file) scattering rates and energy loss rates.
        4. If self.mode_resolved is True, save per-phonon-branch rates to 'fbd_rate_mode_nk.npy'.
        5. Compute energy-resolved DOS, scattering rate, and energy loss rate.
        6. Calculate and print the breakdown field strength.

        Uses:
            self.k_size, self.q_size, self.bands_indices, self.CBM_band_index,
            self.fbd_erange, self.fbd_effective_mass, self.fbd_rate_file, self.outdir,
            self.read_sparse_grad_mat, and MPI rank/communicator state.

        Outputs:
            fbd_rate_nk.dat: k-resolved scattering and energy-loss rates.
            fbd_rate_mode_nk.npy: Optional mode-resolved rates with shape
                (2, nbranches, nbands, nks), written only when ``self.mode_resolved`` is True.
            fbd_rate_ene.dat: energy-resolved DOS, scattering rate, and energy-loss rate.

        Returns:
            None.
        """
        k_grid, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
        self.weight_k *= 2.0
        q_grid = self._get_monkhorst_pack(self.q_size, return_frac=True)
        self.weight_q = np.ones(len(q_grid)) / len(q_grid)
        band_edge_index = self.CBM_band_index
        iband_edge = np.where(np.array(self.bands_indices)==band_edge_index)[0][0]

        nks = len(k_grid)
        nbands = len(self.bands_indices)

        # Distribute eigenvalue computation across MPI ranks
        split_sections = np.full(self.rank_size, nks // self.rank_size, dtype=int)
        split_sections[:nks % self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_split = np.split(k_grid, indices_or_sections=split_sections, axis=0)
        k_grid_local = k_grid_split[self.rank]
        if k_grid_local.size > 0:
            enks_local, _ = self._elec_cal_torch(k_grid_local)
            enks_local = enks_local[self.bands_indices, :]
        else:
            enks_local = np.empty((nbands, 0))
        enks_all = self.comm.gather(enks_local, root=0)

        if self.rank == 0:
            enks = np.concatenate(enks_all, axis=-1) # (nbnd, nk)
            ecbm = self._get_ecbm(enks, iband_edge)
            print(f"CBM energy = {ecbm * QTcts.HARTREEtoEV} eV")
        else:
            ecbm = None
            enks = None
        ecbm = self.comm.bcast(ecbm, root=0)

        if self.fbd_rate_file:
            if self.rank == 0:
                rate_ir = self._read_rate_file(self.fbd_rate_file, [nbands, nks])
        else:
            # Pre-filter k-points: keep only those with bands in the energy focus range
            if self.rank == 0:
                max_smear_width = self.smeark * 10
                margin = 5.0 * max_smear_width
                efocus_low = ecbm + self.fbd_erange[0] - margin
                efocus_high = ecbm + self.fbd_erange[1] + margin
                k_mask = np.any((enks >= efocus_low) & (enks <= efocus_high), axis=0)
                k_grid_filtered = k_grid[k_mask]
            else:
                k_mask = None
                k_grid_filtered = None
            k_grid_filtered = self.comm.bcast(k_grid_filtered, root=0)
            nks_useful = len(k_grid_filtered)
            self._print_mesh_mpi_check(np.prod(self.k_size), nks, nks_useful, len(q_grid), reduced_type='k')
            nbranches = 3 * self.natoms
            if self.read_sparse_grad_mat:
                rate_ir_filtered, rate_mode_filtered = self.scattering_rate_cal_epc_fbd_sparse(k_grid_filtered, q_grid, self.bands_indices, ecbm)
            else:
                rate_ir_filtered, rate_mode_filtered = self.scattering_rate_cal_epc_fbd_dense(k_grid_filtered, q_grid, self.bands_indices, ecbm)
            if self.rank == 0:
                rate_ir = np.full((2, nbands, nks), np.inf)
                rate_ir[:, :, k_mask] = rate_ir_filtered
                fout_name = 'fbd_rate_nk.dat'
                filepath = os.path.join(self.outdir, fout_name)
                self._write_rate_file(filepath, self.weight_k, enks, rate_ir, is_metal=False)
                if self.mode_resolved:
                    rate_mode_ir = np.zeros((2, nbranches, nbands, nks))
                    rate_mode_ir[:, :, :, k_mask] = rate_mode_filtered
                    np.save(os.path.join(self.outdir, 'fbd_rate_mode_nk.npy'), rate_mode_ir)

        if self.rank == 0:
            ene_array = np.arange(self.fbd_erange[0], self.fbd_erange[1] + self.fbd_erange[2], self.fbd_erange[2])
            inv_smears_array = 1.0 / np.arange(self.smeark, self.smeark * 10.1, self.smeark)
            dos = np.zeros((len(ene_array), len(inv_smears_array)))
            scatt_rate = np.zeros_like(dos)
            eloss_rate = np.zeros_like(dos)

            rate_ir[np.isinf(rate_ir)] = 0.0
            enks_shifted = enks - ecbm
            inv_smears_3d = inv_smears_array[None, None, :]
            for iene, ene in enumerate(ene_array):
                base = enks_shifted - ene
                delta_nks = w0gauss(base[:, :, None] * inv_smears_3d) * inv_smears_3d
                dos[iene, :] = np.einsum('nks,k->s', delta_nks, self.weight_k)
                scatt_num = np.einsum('nk,nks,k->s', rate_ir[0], delta_nks, self.weight_k)
                eloss_num = np.einsum('nk,nks,k->s', rate_ir[1], delta_nks, self.weight_k)
                valid_dos = dos[iene, :] > 0.0
                scatt_rate[iene, :] = np.divide(scatt_num, dos[iene, :], out=np.full_like(scatt_num, np.nan), where=valid_dos)
                eloss_rate[iene, :] = np.divide(eloss_num, dos[iene, :], out=np.full_like(eloss_num, np.nan), where=valid_dos)

            inv_smears_array = inv_smears_array / QTcts.HARTREEtoMEV
            with open(os.path.join(self.outdir, 'fbd_rate_ene.dat'), 'w') as fout:
                # DOS, scatt_rate, eloss_rate
                header = "energy"
                units  = "  eV  "
                labels = " ---- "
                for inv_smear in inv_smears_array:
                    header += "       DOS          scatt-rate       eloss-rate    "
                    units  += "       1/eV             1/ps            eV/ps      "
                    val = "    {:<6.2f}meV    ".format(1.0 / inv_smear)
                    labels += f"{val:>17}" * 3
                fout.write(header + '\n')
                fout.write(units + '\n')
                fout.write(labels + '\n')
                for iene, ene in enumerate(ene_array):
                    line = f"{ene * QTcts.HARTREEtoEV:<6.4f}"
                    for ismear, _ in enumerate(inv_smears_array):
                        dos_val = dos[iene, ismear] / QTcts.HARTREEtoEV
                        scatt_val = scatt_rate[iene, ismear] * QTcts.HARTREEtoINVPS
                        eloss_val = eloss_rate[iene, ismear] * QTcts.HARTREEtoINVPS * QTcts.HARTREEtoEV
                        line += f"{dos_val:17.8e}{scatt_val:17.8e}{eloss_val:17.8e}"
                    fout.write(line + '\n')

            # Multi-smearing summary table: report Max DOS and Breakdown Field for
            # each broadening in inv_smears_array (restored from the April version),
            # keeping the May NaN/Inf safety guards.
            fbd_prefactor = np.sqrt(3.0 * self.fbd_effective_mass * QTcts.MASS_E) / QTcts.EV
            mask_ene = ene_array <= self.fbd_exp_bandgap

            print(f"Smear of Summation (meV)    Max DOS (1/eV)       Breakdown Field (MV/m)")
            for ismear in range(len(inv_smears_array)):
                power_rate = scatt_rate[:, ismear] * eloss_rate[:, ismear]
                valid_power = np.isfinite(power_rate) & (power_rate >= 0.0)
                fbd_array = np.full_like(power_rate, np.nan)
                fbd_array[valid_power] = np.sqrt(
                    power_rate[valid_power] * QTcts.HARTREEtoINVS * QTcts.HARTREEtoINVS * QTcts.HARTREEtoJ
                ) * fbd_prefactor

                valid_fbd_mask = mask_ene & np.isfinite(fbd_array)
                fbd_max = np.max(fbd_array[valid_fbd_mask]) if valid_fbd_mask.any() else 0.0
                valid_dos_mask = mask_ene & np.isfinite(dos[:, ismear])
                dos_max = np.max(dos[valid_dos_mask, ismear]) if valid_dos_mask.any() else 0.0

                # Diagnostic counters for the first (smallest) smearing only, to avoid noise.
                if ismear == 0:
                    invalid_count = np.count_nonzero(mask_ene & ~np.isfinite(fbd_array))
                    negative_power_count = np.count_nonzero(mask_ene & np.isfinite(power_rate) & (power_rate < 0.0))
                    zero_dos_count = np.count_nonzero(mask_ene & ~(dos[:, 0] > 0.0))
                    if invalid_count:
                        print(
                            "Skipped {} invalid breakdown energy points at smear={:.2f} meV "
                            "({} with negative scatt_rate*eloss_rate, {} with zero DOS).".format(
                                invalid_count, 1.0 / inv_smears_array[0], negative_power_count, zero_dos_count
                            )
                        )

                smear_mev = 1.0 / inv_smears_array[ismear]
                print(f"{smear_mev:>20.2f}{dos_max / QTcts.HARTREEtoEV:>20.4f}{fbd_max * 1.0E-6:>22.4f}")
        print_peak_memeory(self.comm, self.rank)
