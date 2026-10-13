"""Mechanically extracted QTrans methods."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.extensions import scattering_rate_cal_EPC_helper_sparse
from qtrans.extensions import scattering_rate_cal_EPC_helper_sparse_soc
from qtrans.numerics.distributions import bose_weight
from qtrans.numerics.distributions import fermi_weight
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import gather_to_rank_root
import numpy as np


class _ScatteringMixin:
    def scattering_rate_cal_epc_dense(
            self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float, mob_level='erta', is_metal:bool=False, is_rmp:bool=False):
        """
        Calculate EPC scattering rates with a dense gradient matrix.

        This dense-gradient path computes total electron-phonon scattering rates for
        each selected band and k point. If ``mob_level == 'ibte'``, it also records
        transition probabilities for the IBTE solver. If ``self.mode_resolved`` is
        True, it saves the per-phonon-branch rate contributions to
        ``rate_mode_resolved.npy``.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape
                (nks_local, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the transport window,
                shape (nbands,).
            band_edge_energy (float): The energy of the band edge.
            mob_level (str): Transport level. ``'erta'`` returns only rates;
                ``'ibte'`` also returns transition channels.
            is_metal (bool): If True, use the metallic energy-focus window.
            is_rmp (bool): If True, subtract the pure long-range correction
                contribution from |g|^2.

        Returns:
            tuple: For ``mob_level == 'erta'``, returns ``(rate_all, eigen, vnks)``.
                For ``mob_level == 'ibte'``, returns
                ``(rate_all, eigen, vnks, scat_channel)``.

            rate_all (np.ndarray): Total scattering rate, shape (nbands, nks_local).
            eigen (np.ndarray): Eigenvalues on the local k grid, shape
                (nbands, nks_local).
            vnks (np.ndarray): Band velocities, shape (nks_local, nbands, 3).
            scat_channel (np.ndarray): Only for IBTE. Structured array with fields
                ``ik``, ``ikpq_ibz``, ``ikpq_fbz``, ``iband``, ``jband``, and
                ``trans_prob``.

        Side Effects:
            If ``self.mode_resolved`` is True, saves ``rate_mode_resolved.npy`` with
            shape (nbands, nks_local, nbranches).
        """

        nbranches = self._get_nbranches()
        nbands = len(band_indice)
        nks = len(k_grid)
        mode_resolved = self.mode_resolved
        is_ibte = (mob_level == 'ibte')

        if self.rank == 0 and is_ibte:
            print('='*100)
            print(' '*16+'Calculating the scattering rate and transition probability for IBTE!')
            print('='*100 + "\n")

        rate_all = np.full((nbands, nks), np.inf)
        rate_all_resolved = np.full((nbands, nks, nbranches), np.inf) if mode_resolved else None
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal)

        trans_local = []

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc')

        nqs_local = len(q_grid_local)
        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [
            np.ascontiguousarray(atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq])
            for iq in range(nqs_local)
        ]
        bose_grid = [bose_weight(freq_grid[iq], self.temperature) for iq in range(nqs_local)]
        phase_q_all = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ q_grid_local.T))

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik]
                if is_metal:
                    enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                elif self.is_hole:
                    enk_inside_table = (eig_k >= efocus_threshold)
                else:        
                    enk_inside_table = (eig_k <= efocus_threshold)
                if not enk_inside_table.any():
                    continue
                rate_all[enk_inside_table, ik_all] = 0.0
                if mode_resolved:
                    rate_all_resolved[enk_inside_table, ik_all, :] = 0.0
                enk_indices = np.where(enk_inside_table)[0]
                phase_k = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ k)) # shape: (ncells,)
                fermi_ks = fermi_weight(eig_k - self.efermi, self.temperature)
                if is_ibte:
                    # obtain the coordinates of kpq
                    kpq_list = k + q_grid_local # (nqs_local, 3)
                    BZ_index_list = self._get_kpq_BZ_index(kpq_list) # index in FBZ
                    IBZ_index_list = self.ir_indices[BZ_index_list] # index in IBZ

                for iq, q in enumerate(q_grid_local):
                    # calculate the electronic info for k+q
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                    eig_kpq = eig_kpq[band_indice, 0]
                    wave_kpq = wave_kpq[0, band_indice, :]
                    if is_ibte:
                        ikpq_fbz = BZ_index_list[iq] # index in IBZ
                        ikpq_ibz = IBZ_index_list[iq] # index in IBZ
                    if is_metal:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold[0]) & (eig_kpq <= efocus_threshold[1])
                    elif self.is_hole:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold)
                    else:        
                        emkpq_inside_table = (eig_kpq <= efocus_threshold)
                    if not emkpq_inside_table.any():
                        continue 
                    emkpq_indices = np.where(emkpq_inside_table)[0]
                    # phonon spectrum
                    apply_correction_for_this_q = apply_correction_for_qs[iq]
                    freq = freq_grid[iq]
                    match_table = self._get_match_table(eig_k, eig_kpq, freq)
                    if not match_table.any(): # skip the meaningless iterations
                        continue
                    phvec_wap = phvec_waps_all[iq]
                    bose_qvs = bose_grid[iq]
                    fermi_kpqs = fermi_weight(eig_kpq - self.efermi, self.temperature)
                    # cal epc
                    phase_kpq = phase_k * phase_q_all[:, iq] # shape: (ncells,)
                    if apply_correction_for_this_q:
                        epc_corr, tmp1_save = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=True)
                    else:
                        tmp1_save = np.einsum('jm,in -> jimn', np.conj(wave_kpq), wave_k)
                    for iband in enk_indices:
                        for jband in emkpq_indices:
                            tmp1 = tmp1_save[jband, iband]
                            trans_temp = 0
                            for ibranch in range(nbranches):
                                if match_table[iband, jband, ibranch]:
                                    epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                                tmp1, self.grad_mat, phase_k, phase_kpq,
                                                                dipole_correction=False)
                                    if apply_correction_for_this_q:
                                        epc = epc + epc_corr[iband, jband, ibranch]
                                        g2_tmp = np.abs(epc) * np.abs(epc)
                                        if is_rmp:                                            
                                            g2_tmp = g2_tmp - np.abs(epc_corr[iband, jband, ibranch]) * np.abs(epc_corr[iband, jband, ibranch])
                                    else:
                                        g2_tmp = np.abs(epc) * np.abs(epc)
                                    delta_f1 = w0gauss((eig_k[iband] - eig_kpq[jband] + freq[ibranch]) * self.inv_smearq, 0) * self.inv_smearq
                                    delta_f2 = w0gauss((eig_k[iband] - eig_kpq[jband] - freq[ibranch]) * self.inv_smearq, 0) * self.inv_smearq
                                    rate_tmp = g2_tmp * ((bose_qvs[ibranch] + fermi_kpqs[jband]) * delta_f1 +
                                                        (bose_qvs[ibranch] + 1.0 - fermi_kpqs[jband]) * delta_f2) * weights_q_local[iq]
                                    rate_all[iband, ik_all] += rate_tmp
                                    if mode_resolved:
                                        rate_all_resolved[iband, ik_all, ibranch] += rate_tmp
                                    if is_ibte:
                                        trans_temp += g2_tmp * ((bose_qvs[ibranch] + 1 - fermi_ks[iband]) * delta_f1 + 
                                                                        (bose_qvs[ibranch] + fermi_ks[iband]) * delta_f2) * weights_q_local[iq] 
                            if is_ibte:
                                trans_local.append((ik_all, ikpq_ibz, ikpq_fbz, iband, jband, trans_temp))
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
            self.comm.Barrier()
        if self.rank == 0 and mode_resolved:
            np.save('rate_mode_resolved.npy', rate_all_resolved)
        # sanity check!
        rate_all[np.abs(rate_all) < QTcts.TENPM160] = np.inf
        if not is_ibte:
            return rate_all, eigen, vnks
        dtype = [('ik', int), ('ikpq_ibz', int), ('ikpq_fbz', int), ('iband', int), ('jband', int), ('trans_prob', float)]
        scat_channel = np.array(trans_local, dtype=dtype)
        return rate_all, eigen, vnks, scat_channel

    def scattering_rate_cal_epc_sparse(
            self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float, mob_level='erta', is_metal:bool=False, is_rmp:bool=False):
        """
        Calculate EPC scattering rates for a non-collinear/non-magnetic channel using sparse gradients.

        This is the C-accelerated sparse-gradient path. It computes the total scattering rate
        for each selected band and k point. If ``mob_level == 'ibte'``, it also records
        transition probabilities for the IBTE solver. If ``self.mode_resolved`` is True and
        ``self.spinor_mode != 'nc'``, it saves per-phonon-branch rates to
        ``rate_mode_resolved.npy``.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape (nks_local, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the transport window, shape (nbands,).
            band_edge_energy (float): The energy of the band edge.
            mob_level (str): Transport level. ``'erta'`` returns only rates; ``'ibte'``
                also returns transition channels.
            is_metal (bool): If True, use the metallic energy-focus window.
            is_rmp (bool): If True, subtract the pure long-range correction contribution
                from |g|^2.

        Returns:
            tuple: For ``mob_level == 'erta'``, returns ``(rate_all, eigen, vnks)``.
                For ``mob_level == 'ibte'``, returns
                ``(rate_all, eigen, vnks, scat_channel)``.

            rate_all (np.ndarray): Total scattering rate, shape (nbands, nks_local).
            eigen (np.ndarray): Eigenvalues on the local k grid, shape (nbands, nks_local).
            vnks (np.ndarray): Band velocities, shape (nks_local, nbands, 3).
            scat_channel (np.ndarray): Only for IBTE. Structured array with fields
                ``ik``, ``ikpq_ibz``, ``ikpq_fbz``, ``iband``, ``jband``, and
                ``trans_prob``.

        Side Effects:
            If mode-resolved output is enabled, saves ``rate_mode_resolved.npy`` with shape
            (nbands, nks_local, nbranches).
        """

        if (self.cal_mode == 'conductivity' and self.spinor_mode == 'none'
                and not self.soc_switch and self.read_sparse_grad_mat):
            from qtrans.calculator.transport.mobility_sparse import scattering_rate_sparse_none_batched
            return scattering_rate_sparse_none_batched(
                self, k_grid, q_grid, band_indice, band_edge_energy,
                mob_level=mob_level, is_metal=is_metal, is_rmp=is_rmp)

        nbranches = self._get_nbranches()
        nbands = len(band_indice)
        nks = len(k_grid)
        mode_resolved = self.mode_resolved and self.spinor_mode != 'nc'

        if self.rank == 0 and mob_level == 'ibte':
            print('='*100)
            print(' '*16+'Calculating the scattering rate and transition probability for IBTE!')
            print('='*100 + "\n")

        rate_all = np.full((nbands, nks), np.inf)
        rate_all_resolved = np.full((nbands, nks, nbranches), np.inf) if mode_resolved else None
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal)
        
        # store info for ibte calculation!
        trans_local = []
        is_ibte = (mob_level == 'ibte')

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc_sparse')

        # Precompute q-dependent quantities outside the k-loop
        nqs_local = len(q_grid_local)
        # atomic_phase for all q: shape (natoms, nqs_local)
        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        # phvec_wap for all q: list of contiguous (nbranches, natoms, 3) arrays
        # atomic_phases_all[:, iq] shape (natoms,) -> [None, :, None] shape (1, natoms, 1) * phon_vecs[iq] (nbranches, natoms, 3)
        phvec_waps_all = [np.ascontiguousarray(atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq]) for iq in range(nqs_local)]
        # bose_weight for all q
        bose_grid = [bose_weight(freq_grid[iq], self.temperature) for iq in range(nqs_local)]
        # pre-allocate zero epc_corr for reuse
        epc_corr_zero = np.zeros((nbands, nbands, nbranches), dtype=complex)
        # Precompute phase_q for all q: exp(j2π R·q), shape (ncells, nqs_local)
        # Used to decompose phase_kpq = phase_k * phase_q via exp(j2π R·(k+q)) = exp(j2π R·k) * exp(j2π R·q)
        phase_q_all = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ q_grid_local.T))
        atomic_mass = np.ascontiguousarray(self.atomic_mass)
        # Pre-allocate output arrays for C helper (reused each (k,q) iteration)
        scatt_rate_k = np.zeros((nbands,), dtype=float, order='C')
        scatt_rate_k_resolved = np.zeros((nbands, nbranches), dtype=float, order='C') if mode_resolved else None

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k = np.ascontiguousarray(eigen_recv[:, ik])
                wave_k = np.ascontiguousarray(eigen_vecs_recv[ik])
                if is_metal:
                    enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                elif self.is_hole:
                    enk_inside_table = (eig_k >= efocus_threshold)
                else:        
                    enk_inside_table = (eig_k <= efocus_threshold)
                if not enk_inside_table.any():
                    continue
                rate_all[enk_inside_table, ik_all] = 0.0
                if mode_resolved:
                    rate_all_resolved[enk_inside_table, ik_all, :] = 0.0
                phase_k = np.ascontiguousarray(np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ k))) # shape: (ncells,)
                fermi_ks = np.ascontiguousarray(fermi_weight(eig_k - self.efermi, self.temperature))
                # Batch compute electronic structure for all k+q at once
                kpq_list = k + q_grid_local  # (nqs_local, 3)
                eig_kpq_all, wave_kpq_all = self._elec_cal_torch(kpq_list)
                # eig_kpq_all: (norbs or 2*norbs, nqs_local), wave_kpq_all: (nqs_local, norbs or 2*norbs, norbs)
                eig_kpq_all = eig_kpq_all[band_indice, :]    # (nbands, nqs_local)
                wave_kpq_all = wave_kpq_all[:, band_indice, :]  # (nqs_local, nbands, norbs)
                if mob_level == 'ibte':
                    BZ_index_list = self._get_kpq_BZ_index(kpq_list) # index in FBZ
                    IBZ_index_list = self.ir_indices[BZ_index_list] # index in IBZ
                for iq, q in enumerate(q_grid_local):
                    # retrieve pre-computed electronic info for k+q
                    eig_kpq = np.ascontiguousarray(eig_kpq_all[:, iq])
                    wave_kpq = np.ascontiguousarray(wave_kpq_all[iq])
                    if mob_level == 'ibte':
                        ikpq_ibz = IBZ_index_list[iq] # index in IBZ
                        ikpq_fbz = BZ_index_list[iq] # index in FBZ
                    if is_metal:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold[0]) & (eig_kpq <= efocus_threshold[1])
                    elif self.is_hole:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold)
                    else:        
                        emkpq_inside_table = (eig_kpq <= efocus_threshold)
                    if not emkpq_inside_table.any():
                        continue 
                    # phonon spectrum
                    apply_correction_for_this_q = apply_correction_for_qs[iq]
                    freq = np.ascontiguousarray(freq_grid[iq])
                    match_table = np.ascontiguousarray(self._get_match_table(eig_k, eig_kpq, freq))
                    if not match_table.any(): # skip the meaningless iterations
                        continue
                    phvec_wap = phvec_waps_all[iq]
                    bose_qvs = np.ascontiguousarray(bose_grid[iq])
                    fermi_kpqs = np.ascontiguousarray(fermi_weight(eig_kpq - self.efermi, self.temperature))
                    phase_kpq = np.ascontiguousarray(phase_k * phase_q_all[:, iq])  # exp(j2π R·k) * exp(j2π R·q) = exp(j2π R·(k+q))
                    if apply_correction_for_this_q:
                        epc_corr = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=False)
                    else:
                        epc_corr = epc_corr_zero
                    # reset pre-allocated output arrays
                    scatt_rate_k[:] = 0.0
                    if mode_resolved:
                        scatt_rate_k_resolved[:] = 0.0
                    trans_local_temp = []
                    if self.spinor_mode == 'nc':
                        scattering_rate_cal_EPC_helper_sparse_soc(freq, atomic_mass, phvec_wap,
                                                            phase_k, phase_kpq, self.grad_mat,
                                                            wave_k, wave_kpq,
                                                            enk_inside_table, emkpq_inside_table, match_table,
                                                            eig_k, eig_kpq, bose_qvs, fermi_kpqs,
                                                            epc_corr, scatt_rate_k,
                                                            trans_local_temp, fermi_ks,
                                                            self.inv_smearq, weights_q_local[iq], int(is_rmp), int(is_ibte))
                    else:
                        scattering_rate_cal_EPC_helper_sparse(freq, atomic_mass, phvec_wap,
                                                            phase_k, phase_kpq, self.grad_mat,
                                                            wave_k, wave_kpq,
                                                            enk_inside_table, emkpq_inside_table, match_table,
                                                            eig_k, eig_kpq, bose_qvs, fermi_kpqs,
                                                            epc_corr, scatt_rate_k, scatt_rate_k_resolved,
                                                            trans_local_temp, fermi_ks,
                                                            self.inv_smearq, weights_q_local[iq], int(is_rmp), int(is_ibte))
                    rate_all[:, ik_all] += scatt_rate_k
                    if mode_resolved:
                        rate_all_resolved[:, ik_all, :] += scatt_rate_k_resolved
                    if mob_level == 'ibte':
                        trans_local_temp = [(ik_all, ikpq_ibz, ikpq_fbz, *item) for item in trans_local_temp]
                        trans_local.extend(trans_local_temp)
            if self.rank == 0:
                logger.step(send_rank+1)
        rate_all *= QTcts.TWOPI
        # The rate_all of the whole q is obtained by allreducing the rate_all of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, rate_all, op=MPI.SUM)
            if mode_resolved:
                self.comm.Allreduce(MPI.IN_PLACE, rate_all_resolved, op=MPI.SUM)
            self.comm.Barrier()
        if self.rank == 0 and mode_resolved:
            np.save('rate_mode_resolved.npy', rate_all_resolved)
        # sanity check!
        # sanity check!
        rate_all[np.abs(rate_all) < QTcts.TENPM160] = np.inf
        if mob_level == 'erta':
            return rate_all, eigen, vnks
        elif mob_level == 'ibte':
            dtype = [('ik', int), ('ikpq_ibz', int), ('ikpq_fbz', int), ('iband', int), ('jband', int), ('trans_prob', float)]
            scat_channel = np.array(trans_local, dtype=dtype)
            return rate_all, eigen, vnks, scat_channel

    def scattering_rate_cal_epc_collinear_dense(
            self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float, mob_level='erta', is_metal:bool=False, is_rmp:bool=False):
        """
        Calculate EPC scattering rates for a collinear spin system using dense gradients.

        This routine evaluates the dense-gradient EPC matrix elements separately for the
        two collinear spin channels. If ``mob_level == 'ibte'``, transition probabilities
        are collected separately for each spin channel. If ``self.mode_resolved`` is True,
        the per-phonon-branch rate is saved to ``rate_mode_resolved.npy``.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape (nks_local, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the transport window, shape (nbands,).
            band_edge_energy (float): The energy of the band edge.
            mob_level (str): Transport level. ``'erta'`` returns only rates; ``'ibte'``
                also returns transition channels.
            is_metal (bool): If True, use the metallic energy-focus window.
            is_rmp (bool): If True, subtract the pure long-range correction contribution
                from |g|^2 when long-range corrections are applied.

        Returns:
            tuple: For ``mob_level == 'erta'``, returns ``(rate_all, eigen, vnks)``.
                For ``mob_level == 'ibte'``, returns
                ``(rate_all, eigen, vnks, scat_channel)``.

            rate_all (np.ndarray): Total scattering rate, shape (2, nbands, nks_local).
            eigen (np.ndarray): Eigenvalues on the local k grid, shape (2, nbands, nks_local).
            vnks (np.ndarray): Band velocities, shape (2, nks_local, nbands, 3).
            scat_channel (np.ndarray): Only for IBTE. Object array of length 2; each entry is
                a structured transition-channel array for one spin channel.

        Side Effects:
            If ``self.mode_resolved`` is True, saves ``rate_mode_resolved.npy`` with shape
            (2, nbands, nks_local, nbranches).
        """

        nbranches = self._get_nbranches()
        nbands = len(band_indice)
        nks = len(k_grid)
        mode_resolved = self.mode_resolved

        if self.rank == 0 and mob_level == 'ibte':
            print('='*100)
            print(' '*16+'Calculating the scattering rate and transition probability for IBTE!')
            print('='*100 + "\n")

        rate_all = np.full((2, nbands, nks), np.inf)
        rate_all_resolved = np.full((2, nbands, nks, nbranches), np.inf) if mode_resolved else None
        epc_corr_zero = np.zeros((nbands, nbands, nbranches), dtype=complex)
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal)
        
        is_ibte = (mob_level == 'ibte')
        if is_ibte:
            trans_local = [[], []] # two channels

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc_collinear')

        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq] for iq in range(len(q_grid_local))]
        bose_grid = [bose_weight(freq_grid[iq], self.temperature) for iq in range(len(q_grid_local))]

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                # eig_k: # shape (2, nbands)
                eig_k, wave_k = eigen_recv[:, :, ik], eigen_vecs_recv[:, ik]
                if is_metal:
                    enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                elif self.is_hole:
                    enk_inside_table = (eig_k >= efocus_threshold[:, None])
                else:        
                    enk_inside_table = (eig_k <= efocus_threshold[:, None])
                if not enk_inside_table.any():
                    continue
                rate_all[enk_inside_table, ik_all] = 0.0
                if mode_resolved:
                    for ispin in range(2):
                        rate_all_resolved[ispin, enk_inside_table[ispin], ik_all, :] = 0.0
                ibands_inside_by_spin = [np.where(enk_inside_table[ispin])[0] for ispin in range(2)]
                phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                if is_ibte:
                    # obtain the coordinates of kpq
                    kpq_list = k + q_grid_local # (nqs_local, 3)
                    BZ_index_list = self._get_kpq_BZ_index(kpq_list) # index in FBZ
                    IBZ_index_list = self.ir_indices[BZ_index_list] # index in IBZ
                for iq, q in enumerate(q_grid_local):
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                    eig_kpq = np.stack((eig_kpq[band_indice, 0], eig_kpq[band_indice+self.norbs, 0]), axis=0)
                    if is_ibte:
                        ikpq_ibz = IBZ_index_list[iq] # index in IBZ
                        ikpq_fbz = BZ_index_list[iq] # index in FBZ
                    if is_metal:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold[0]) & (eig_kpq <= efocus_threshold[1])
                    elif self.is_hole:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold[:, None])
                    else:        
                        emkpq_inside_table = (eig_kpq <= efocus_threshold[:, None])
                    if not emkpq_inside_table.any():
                        continue
                    jbands_inside_by_spin = [np.where(emkpq_inside_table[ispin])[0] for ispin in range(2)]
                    wave_kpq = np.stack((wave_kpq[0, band_indice], wave_kpq[0, band_indice+self.norbs]), axis=0)
                    phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    # phonon spectrum
                    apply_correction_for_this_q = apply_correction_for_qs[iq]
                    freq = freq_grid[iq]
                    phvec_wap = phvec_waps_all[iq]
                    bose_qvs = bose_grid[iq]
                    # cal epc
                    for ispin in range(2):
                        match_table = self._get_match_table(eig_k[ispin], eig_kpq[ispin], freq)
                        if not match_table.any():
                            continue
                        if is_metal:
                            fermi_kpqs = fermi_weight(eig_kpq[ispin] - self.efermi, self.temperature)
                            fermi_ks = fermi_weight(eig_k[ispin] - self.efermi, self.temperature)
                        else:
                            fermi_kpqs = fermi_weight(eig_kpq[ispin] - self.efermi[ispin], self.temperature)
                            fermi_ks = fermi_weight(eig_k[ispin] - self.efermi[ispin], self.temperature)
                        if apply_correction_for_this_q:
                            epc_corr, tmp1_save = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table[ispin], k, wave_k[ispin], wave_kpq[ispin], return_tmp1=True)
                        else:
                            epc_corr = epc_corr_zero
                            tmp1_save = np.einsum('jm,in -> jimn', np.conj(wave_kpq[ispin]), wave_k[ispin])
                        for iband in ibands_inside_by_spin[ispin]:
                            for jband in jbands_inside_by_spin[ispin]:
                                tmp1 = tmp1_save[jband, iband]
                                trans_temp = 0
                                # calculate epc
                                for ibranch in range(nbranches):
                                    if match_table[iband, jband, ibranch]:
                                        epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                        tmp1, self.grad_mat[ispin], phase_k, phase_kpq, dipole_correction=False)
                                        if apply_correction_for_this_q:
                                            epc = epc + epc_corr[iband, jband, ibranch]
                                            g2_tmp = np.abs(epc) * np.abs(epc)
                                            if is_rmp:                                            
                                                g2_tmp = g2_tmp - np.abs(epc_corr[iband, jband, ibranch]) * np.abs(epc_corr[iband, jband, ibranch])
                                        else:
                                            g2_tmp = np.abs(epc) * np.abs(epc)
                                        delta_f1 = w0gauss((eig_k[ispin,iband] - eig_kpq[ispin,jband] + freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                        delta_f2 = w0gauss((eig_k[ispin,iband] - eig_kpq[ispin,jband] - freq[ibranch]) * self.inv_smearq) * self.inv_smearq
                                        rate_tmp = g2_tmp * ((bose_qvs[ibranch] + fermi_kpqs[jband]) * delta_f1 +
                                                                        (bose_qvs[ibranch] + 1.0 - fermi_kpqs[jband]) * delta_f2) * weights_q_local[iq]
                                        rate_all[ispin,iband,ik_all] += rate_tmp
                                        if mode_resolved:
                                            rate_all_resolved[ispin, iband, ik_all, ibranch] += rate_tmp
                                        if is_ibte:
                                            trans_temp += g2_tmp * ((bose_qvs[ibranch] + 1 - fermi_ks[iband]) * delta_f1 + 
                                                                            (bose_qvs[ibranch] + fermi_ks[iband]) * delta_f2) * weights_q_local[iq] 
                                if is_ibte:
                                    trans_local[ispin].append((ik_all, ikpq_ibz, ikpq_fbz, iband, jband, trans_temp))
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
        if mob_level == 'erta':
            return rate_all, eigen, vnks
        elif is_ibte:
            dtype = [('ik', int), ('ikpq_ibz', int), ('ikpq_fbz', int), ('iband', int), ('jband', int), ('trans_prob', float)]
            scat_channel = np.empty(2, dtype=object)
            scat_channel[0] = np.array(trans_local[0], dtype=dtype)
            scat_channel[1] = np.array(trans_local[1], dtype=dtype) # combine two spin channels
            return rate_all, eigen, vnks, scat_channel

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

    def scattering_rate_cal_epc_collinear_sparse(
            self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float, mob_level='erta', is_metal:bool=False, is_rmp:bool=False):
        """
        Calculate EPC scattering rates for a collinear spin system using sparse gradients.

        This is the C-accelerated sparse-gradient version of
        ``scattering_rate_cal_epc_collinear_dense``. The calculation is performed
        independently for spin-up and spin-down channels. If ``self.mode_resolved`` is True,
        the sparse C helper also accumulates per-phonon-branch rates.

        Args:
            k_grid (np.ndarray): Local or filtered k sampling points, shape (nks_local, 3).
            q_grid (np.ndarray): q sampling points, shape (nqs, 3).
            band_indice (np.ndarray): Band indices included in the transport window, shape (nbands,).
            band_edge_energy (float): The energy of the band edge.
            mob_level (str): Transport level. ``'erta'`` returns only rates; ``'ibte'``
                also returns transition channels.
            is_metal (bool): If True, use the metallic energy-focus window.
            is_rmp (bool): If True, subtract the pure long-range correction contribution
                from |g|^2.

        Returns:
            tuple: For ``mob_level == 'erta'``, returns ``(rate_all, eigen, vnks)``.
                For ``mob_level == 'ibte'``, returns
                ``(rate_all, eigen, vnks, scat_channel)``.

            rate_all (np.ndarray): Total scattering rate, shape (2, nbands, nks_local).
            eigen (np.ndarray): Eigenvalues on the local k grid, shape (2, nbands, nks_local).
            vnks (np.ndarray): Band velocities, shape (2, nks_local, nbands, 3).
            scat_channel (np.ndarray): Only for IBTE. Object array of length 2; each entry is
                a structured transition-channel array for one spin channel.

        Side Effects:
            If ``self.mode_resolved`` is True, saves ``rate_mode_resolved.npy`` with shape
            (2, nbands, nks_local, nbranches).
        """

        nbranches = self._get_nbranches()
        nbands = len(band_indice)
        nks = len(k_grid)
        mode_resolved = self.mode_resolved

        if self.rank == 0 and mob_level == 'ibte':
            print('='*100)
            print(' '*16+'Calculating the scattering rate and transition probability for IBTE!')
            print('='*100 + "\n")

        rate_all = np.full((2, nbands, nks), np.inf)
        rate_all_resolved = np.full((2, nbands, nks, nbranches), np.inf) if mode_resolved else None
        atomic_mass = np.ascontiguousarray(self.atomic_mass)
        epc_corr_zero = np.zeros((nbands, nbands, nbranches), dtype=complex)
        scatt_rate_k = np.zeros((nbands,), dtype=float, order='C')
        scatt_rate_k_resolved = np.zeros((nbands, nbranches), dtype=float, order='C') if mode_resolved else None
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal)
        
        # store info for ibte calculation!
        trans_local = [[], []] # two channels
        is_ibte = (mob_level == 'ibte')

        if self.rank == 0:
            logger = time_logger(total_cycles=self.rank_size, routine_name='scattering_rate_cal_epc_collinear_sparse')

        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [np.ascontiguousarray(atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq]) for iq in range(len(q_grid_local))]
        bose_grid = [np.ascontiguousarray(bose_weight(freq_grid[iq], self.temperature)) for iq in range(len(q_grid_local))]

        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv = self.comm.bcast((eigen_vecs, eigen), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                # eig_k: # shape (2, nbands)
                eig_k, wave_k = eigen_recv[:, :, ik], eigen_vecs_recv[:, ik]
                if is_metal:
                    enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                elif self.is_hole:
                    enk_inside_table = (eig_k >= efocus_threshold[:, None])
                else:        
                    enk_inside_table = (eig_k <= efocus_threshold[:, None])
                if not enk_inside_table.any():
                    continue
                rate_all[enk_inside_table, ik_all] = 0.0
                if mode_resolved:
                    for ispin in range(2):
                        rate_all_resolved[ispin, enk_inside_table[ispin], ik_all, :] = 0.0
                phase_k = np.ascontiguousarray(np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1))) # shape: (ncells,)
                if is_ibte:
                    # obtain the coordinates of kpq
                    kpq_list = k + q_grid_local # (nqs_local, 3)
                    BZ_index_list = self._get_kpq_BZ_index(kpq_list) # index in FBZ
                    IBZ_index_list = self.ir_indices[BZ_index_list] # index in IBZ
                for iq, q in enumerate(q_grid_local):
                    kpq = k + q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                    eig_kpq = np.stack((eig_kpq[band_indice, 0], eig_kpq[band_indice+self.norbs, 0]), axis=0)
                    if is_ibte:
                        ikpq_ibz = IBZ_index_list[iq] # index in IBZ
                        ikpq_fbz = BZ_index_list[iq] # index in FBZ
                    if is_metal:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold[0]) & (eig_kpq <= efocus_threshold[1])
                    elif self.is_hole:
                        emkpq_inside_table = (eig_kpq >= efocus_threshold[:, None])
                    else:        
                        emkpq_inside_table = (eig_kpq <= efocus_threshold[:, None])
                    if not emkpq_inside_table.any():
                        continue 
                    wave_kpq = np.stack((wave_kpq[0, band_indice], wave_kpq[0, band_indice+self.norbs]), axis=0)
                    phase_kpq = np.ascontiguousarray(np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1))) # shape: (ncells,)
                    # phonon spectrum
                    apply_correction_for_this_q = apply_correction_for_qs[iq]
                    freq = np.ascontiguousarray(freq_grid[iq])
                    phvec_wap = phvec_waps_all[iq]
                    bose_qvs = bose_grid[iq]

                    for ispin in range(2):
                        match_table = np.ascontiguousarray(self._get_match_table(eig_k[ispin], eig_kpq[ispin], freq))
                        if not match_table.any():
                            continue
                        if is_metal:
                            fermi_kpqs = np.ascontiguousarray(fermi_weight(eig_kpq[ispin] - self.efermi, self.temperature))
                            fermi_ks = np.ascontiguousarray(fermi_weight(eig_k[ispin] - self.efermi, self.temperature))
                        else:
                            fermi_kpqs = np.ascontiguousarray(fermi_weight(eig_kpq[ispin] - self.efermi[ispin], self.temperature))
                            fermi_ks = np.ascontiguousarray(fermi_weight(eig_k[ispin] - self.efermi[ispin], self.temperature))
                        if apply_correction_for_this_q:
                            epc_corr = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table[ispin], k, wave_k[ispin], wave_kpq[ispin], return_tmp1=False)
                        else:
                            epc_corr = epc_corr_zero
                        # calculate epc using LRC correction as input
                        scatt_rate_k[:] = 0.0
                        if mode_resolved:
                            scatt_rate_k_resolved[:] = 0.0
                        trans_local_temp = []
                        scattering_rate_cal_EPC_helper_sparse(freq, atomic_mass,
                                                            np.ascontiguousarray(phvec_wap),
                                                            np.ascontiguousarray(phase_k), np.ascontiguousarray(phase_kpq), self.grad_mat[ispin],
                                                            np.ascontiguousarray(wave_k[ispin]), np.ascontiguousarray(wave_kpq[ispin]),
                                                            np.ascontiguousarray(enk_inside_table[ispin]), np.ascontiguousarray(emkpq_inside_table[ispin]),
                                                            np.ascontiguousarray(match_table),
                                                            np.ascontiguousarray(eig_k[ispin]), np.ascontiguousarray(eig_kpq[ispin]),
                                                            np.ascontiguousarray(bose_qvs), np.ascontiguousarray(fermi_kpqs),
                                                            np.ascontiguousarray(epc_corr), scatt_rate_k, scatt_rate_k_resolved,
                                                            trans_local_temp, np.ascontiguousarray(fermi_ks),
                                                            self.inv_smearq, weights_q_local[iq], int(is_rmp), int(is_ibte))
                        if is_ibte:
                            trans_local_temp = [(ik_all, ikpq_ibz, ikpq_fbz, *item) for item in trans_local_temp]
                            trans_local[ispin].extend(trans_local_temp)
                        rate_all[ispin, :, ik_all] += scatt_rate_k
                        if mode_resolved:
                            rate_all_resolved[ispin, :, ik_all, :] += scatt_rate_k_resolved
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
        if mob_level == 'erta':
            return rate_all, eigen, vnks
        elif is_ibte:
            dtype = [('ik', int), ('ikpq_ibz', int), ('ikpq_fbz', int), ('iband', int), ('jband', int), ('trans_prob', float)]
            scat_channel = np.empty(2, dtype=object)
            scat_channel[0] = np.array(trans_local[0], dtype=dtype)
            scat_channel[1] = np.array(trans_local[1], dtype=dtype) # combine two spin channels
            return rate_all, eigen, vnks, scat_channel
