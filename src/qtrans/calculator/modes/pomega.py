"""Mechanically extracted QTrans methods."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import print_peak_memeory
import numpy as np
import os


class _PomegaMixin:
    def eliashberg_spectrum_cal(self):
        if self.read_sparse_grad_mat:
            omegas, a2f = self._eliashberg_spectrum_cal_epc_sparse()
        else:
            omegas, a2f = self._eliashberg_spectrum_cal_epc_dense()
        return omegas, a2f

    def _Pomega_cal_epc(self):
        r"""
        Calculate pairing function \alpha^{2} F(\omega) under EPC theory.

        Args:
            k_grid (list or np.ndarray): (3,)
            q_grid (list or np.ndarray): (3,)
            bands_indices (np.ndarray): (list or np.ndarray)

        Returns:
            a2f: # shape: ()
        """
        pomega_restart_path = os.path.join(self.outdir, 'Pomega_restart.npz')
        if os.path.isfile(pomega_restart_path) and self.pomega_restart:
            if self.rank == 0:
                print("Restart file found: {}. Loading epc_strengths and skipping EPC main loop.".format(pomega_restart_path))
            restart_data = np.load(pomega_restart_path)
            epc_strengths = restart_data['epc_strengths']
            freq_grid_all = restart_data['freq_grid']
            dos_ef = float(restart_data['dos_ef'])
            self.efermi = float(restart_data['efermi'])
            self.weight_q = float(restart_data['weight_q'])

            # Reconstruct omegas_list
            if self.omega_range[0] < QTcts.TENPM10:
                omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
            else:
                omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)

            nomegas = len(omegas_list)
            nqs = epc_strengths.shape[-2]
            nbranches = epc_strengths.shape[-1]

            if self.spinor_mode == 'collinear':
                factor = self.weight_q / dos_ef
            else:
                factor = self.weight_q / (2.0 * dos_ef)
            delta_omegas = w0gauss((freq_grid_all[None, :, :] - omegas_list[:, None, None]) * self.inv_smearq) * self.inv_smearq
            delta_flat = delta_omegas.reshape(nomegas, -1)
            if self.spinor_mode == 'collinear':
                epc_flat = epc_strengths.reshape(epc_strengths.shape[0], -1)
                a2f = (epc_flat @ delta_flat.T) * factor
            else:
                epc_flat = epc_strengths.ravel()
                a2f = (delta_flat @ epc_flat) * factor
            return omegas_list, a2f

        # get the number of included electrons
        # When collinear case, only one electron in each band
        if not self.soc_switch:
            base_electron_count = len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) * 2
            if self.has_unpair_electron:
                num_electron_include = base_electron_count + 1 - self.doping_charge
            else:
                num_electron_include = base_electron_count - self.doping_charge
        else:
            base_electron_count = len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index])
            num_electron_include = base_electron_count - self.doping_charge
        if self.rank == 0:
            self._log(f"Number of electrons included in the calculation: {num_electron_include}")


        k_grid, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
        q_grid = self._get_monkhorst_pack(self.q_size, self.graph_data.latt, return_frac=True)

        nks_fbz = self.k_size[0] * self.k_size[1] * self.k_size[2]
        nks_ibz = len(k_grid)
        nqs = len(q_grid)
        nbands = len(self.bands_indices)

        if self.spinor_mode == 'nc':
            nbands_col = nbands
        elif self.spinor_mode == 'collinear':
            nbands_col = nbands * 2
        else:
            nbands_col = nbands
            self.weight_k *= 2.0

        self.weight_q = 1.0 / nqs

        k_grid_all, enks_local, _, q_grid_local, weights_q_local, freq_grid, _ = \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, None)
        # Reconstruct omegas_list
        if self.omega_range[0] < QTcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        nqs_local = len(q_grid_local)
        nks_local = len(k_grid_all[self.rank])
        nbranches = self._get_nbranches()

        if self.rank == 0:
            enks_all = [None] * self.rank_size
        else:
            enks_all = None
        self.comm.Barrier()
        enks_all = self.comm.gather(enks_local.reshape(nbands_col, nks_local), root=0)
        del enks_local

        if self.rank == 0:
            enks = np.concatenate(enks_all, axis=-1)    # (nbands, nks) or (2*nbands, nks) for collinear
            self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
            is_metal = self._get_is_metal(enks, self.efermi)
            if not is_metal:
                raise RuntimeError(f"Error(7001): Superconductivity can only be calculated for metallic systems.")
            dos_ef = self._get_ef_dos(enks, self.efermi, self.inv_smeark, self.gauss_type)
            if self.spinor_mode == 'collinear':
                if self.rank == 0:
                    print("Fermi energy = {:10.6f} eV\nDOS of Fermi level (sum of two spin channels) = {:10.6f}".format(self.efermi * QTcts.HARTREEtoEV, dos_ef / QTcts.HARTREEtoEV))
            else:
                # N(Ef) in lambda formula is the DOS per spin
                dos_ef = dos_ef / 2.0
                if self.rank == 0:
                    print("Fermi energy = {:10.6f} eV\nDOS of Fermi level (spin factor not included) = {:10.6f}".format(self.efermi * QTcts.HARTREEtoEV, dos_ef / QTcts.HARTREEtoEV))
            ele_data_tmp = [self.efermi, dos_ef]
            # resplit the k points
            mask = np.abs(enks - self.efermi) < self.e_thr
            _, kpt_indices_useful = np.where(mask)
            kpt_indices_useful = np.unique(kpt_indices_useful) # IBZ k-points really contributing
            nks_useful = kpt_indices_useful.shape[0] # total number of useful kpoints
            k_grid_useful = k_grid[kpt_indices_useful]                       # (nks_useful, 3)
            enks_all_useful = enks[:, kpt_indices_useful]                # (2*nbands, nks_useful)
            if self.spinor_mode == 'collinear':
                enks_all_useful = enks_all_useful.reshape(2, nbands, nks_useful) # (2, nbands, nks_useful)
            weights_k_useful = self.weight_k[kpt_indices_useful]             # (nks_useful,)
            counts = np.zeros(self.rank_size, dtype=int)
            for i in range(nks_useful):
                counts[i % self.rank_size] += 1
            split_sections = np.cumsum(counts)
            split_indices = split_sections[:-1]
            k_grid_useful_split = np.split(k_grid_useful, indices_or_sections=split_indices, axis=0)
            enks_all_useful_split = np.split(enks_all_useful, indices_or_sections=split_indices, axis=-1)
            weights_k_useful_split = np.split(weights_k_useful, indices_or_sections=split_indices, axis=0)
            self._print_mesh_mpi_check(nks_fbz, nks_ibz, nks_useful, nqs)
        else:
            ele_data_tmp = None
            k_grid_useful_split = None
            enks_all_useful_split = None
            weights_k_useful_split = None
        k_grid_useful_local = self.comm.scatter(k_grid_useful_split, root=0)
        enks_useful_local = self.comm.scatter(enks_all_useful_split, root=0)
        weights_k_useful_local = self.comm.scatter(weights_k_useful_split, root=0)
        ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
        self.efermi, dos_ef = ele_data_tmp[0], ele_data_tmp[1]
        if self.spinor_mode == 'collinear':
            epc_strengths = np.zeros((2, nqs_local, nbranches))
        else:
            epc_strengths = np.zeros((nqs_local, nbranches))
        # Precompute: filter valid q indices (nonzero q and at least one valid branch)
        branch_masks = freq_grid >= self.frequency_cutoff  # (nqs_local, nbranches)
        has_valid_branch = np.any(branch_masks, axis=1)  # (nqs_local,)
        valid_q_indices = np.where(has_valid_branch)[0]
        valid_q_grid = q_grid_local[valid_q_indices]  # (nqs_valid, 3)
        nqs_valid = len(valid_q_indices)
        # Cache band index arrays
        if self.spinor_mode == 'collinear':
            bands_up = self.bands_indices
            bands_down = self.bands_indices + self.norbs
        # Batch size for _elec_cal_torch (tune to control memory usage)
        batch_size = min(nqs_valid, 64)
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'Pomega_cal_epc')
        for send_rank in range(self.rank_size):
            eigen_recv, k_grid_recv, weights_k_recv = self.comm.bcast(
                (enks_useful_local, k_grid_useful_local, weights_k_useful_local), root=send_rank)
            for ik, k in enumerate(k_grid_recv):
                eig_k = eigen_recv[..., ik]  # (nbands, ) or (2, nbands)
                delta_nk_all = w0gauss((eig_k - self.efermi) * self.inv_smeark) * self.inv_smeark  # (nbands, ) or (2, nbands)
                enk_match = np.abs(eig_k - self.efermi) < self.e_thr  # (nbands, ) or (2, nbands)
                sum_delta_nk = np.sum(delta_nk_all * enk_match, axis=-1)  # float or (2,)
                wk = weights_k_recv[ik]
                # Process valid q points in batches
                for batch_start in range(0, nqs_valid, batch_size):
                    batch_end = min(batch_start + batch_size, nqs_valid)
                    q_batch = valid_q_grid[batch_start:batch_end]  # (batch_len, 3)
                    iq_batch = valid_q_indices[batch_start:batch_end]
                    kpq_batch = k + q_batch  # (batch_len, 3)
                    eig_kpq_batch, _ = self._elec_cal_torch(kpq_batch)  # (norbs, batch_len) or (2*norbs, batch_len)
                    if self.spinor_mode == 'collinear':
                        eig_kpq_up = eig_kpq_batch[bands_up]    # (nbands, batch_len)
                        eig_kpq_down = eig_kpq_batch[bands_down]  # (nbands, batch_len)
                        # delta and match for both spins: (nbands, batch_len)
                        delta_mkq_up = w0gauss((eig_kpq_up - self.efermi) * self.inv_smeark) * self.inv_smeark
                        delta_mkq_down = w0gauss((eig_kpq_down - self.efermi) * self.inv_smeark) * self.inv_smeark
                        match_up = np.abs(eig_kpq_up - self.efermi) < self.e_thr
                        match_down = np.abs(eig_kpq_down - self.efermi) < self.e_thr
                        # sum over bands: (batch_len,)
                        sum_delta_mkq_up = np.sum(delta_mkq_up * match_up, axis=0)
                        sum_delta_mkq_down = np.sum(delta_mkq_down * match_down, axis=0)
                        # ispin=0 pairs with jspin=1, ispin=1 pairs with jspin=0
                        factor_spin0 = sum_delta_nk[0] * sum_delta_mkq_up * wk  # (batch_len,)
                        factor_spin1 = sum_delta_nk[1] * sum_delta_mkq_down * wk    # (batch_len,)
                        for i, iq in enumerate(iq_batch):
                            bm = branch_masks[iq]
                            epc_strengths[0, iq, bm] += factor_spin0[i]
                            epc_strengths[1, iq, bm] += factor_spin1[i]
                    else:
                        eig_kpq = eig_kpq_batch[self.bands_indices]    # (nbands, batch_len)
                        # delta and match for both spins: (nbands, batch_len)
                        delta_mkq = w0gauss((eig_kpq - self.efermi) * self.inv_smeark) * self.inv_smeark
                        emkq_match = np.abs(eig_kpq - self.efermi) < self.e_thr
                        # sum over bands: (batch_len,)
                        sum_delta_mkq = np.sum(delta_mkq * emkq_match, axis=0)
                        # ispin=0 pairs with jspin=1, ispin=1 pairs with jspin=0
                        factor_delta = sum_delta_nk * sum_delta_mkq * wk  # (batch_len,)
                        for i, iq in enumerate(iq_batch):
                            bm = branch_masks[iq]
                            epc_strengths[iq, bm] += factor_delta[i]
            if self.rank == 0:
                logger.step(send_rank + 1)

        # === Save epc_strengths for restart ===
        # Gather local epc_strengths and freq_grid from all ranks to rank 0
        epc_strengths_gathered = self.comm.gather(epc_strengths, root=0)
        freq_grid_gathered = self.comm.gather(freq_grid, root=0)
        if self.rank == 0:
            if self.spinor_mode == 'collinear':
                epc_strengths_all = np.concatenate(epc_strengths_gathered, axis=1)  # (2, nqs, nbranches)
            else:
                epc_strengths_all = np.concatenate(epc_strengths_gathered, axis=0)  # (nqs, nbranches)
            freq_grid_all = np.concatenate(freq_grid_gathered, axis=0)  # (nqs, nbranches)
            np.savez(pomega_restart_path,
                     epc_strengths=epc_strengths_all,
                     freq_grid=freq_grid_all,
                     dos_ef=np.array(dos_ef),
                     efermi=np.array(self.efermi),
                     weight_q=np.array(self.weight_q))
            print("EPC strengths saved to {} for restart.".format(pomega_restart_path))

        if self.spinor_mode == 'collinear':
            factor = self.weight_q / dos_ef
        else:
            factor = self.weight_q / (2.0 * dos_ef)
        # delta_omegas: (nomegas, nqs_local, nbranches)
        delta_omegas = w0gauss((freq_grid[None, :, :] - omegas_list[:, None, None]) * self.inv_smearq) * self.inv_smearq
        delta_flat = delta_omegas.reshape(len(omegas_list), -1)  # (nomegas, nqs*nbranches)
        if self.spinor_mode == 'collinear':
            epc_flat = epc_strengths.reshape(epc_strengths.shape[0], -1)  # (nspins, nqs*nbranches)
            a2f = (epc_flat @ delta_flat.T) * factor  # (nspins, nomegas)
        else:
            epc_flat = epc_strengths.ravel()  # (nqs*nbranches,)
            a2f = (delta_flat @ epc_flat) * factor  # (nomegas,)

        # The a2f of the whole q is obtained by allreducing the a2f of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
        print_peak_memeory(self.comm, self.rank)
        return omegas_list, a2f
