"""Spectrum calculations."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.extensions import electron_self_energy_cal_EPC_helper_sparse
from qtrans.numerics.distributions import bose_weight
from qtrans.numerics.distributions import fermi_weight
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import gather_to_rank_root
from qtrans.parallel.mpi import split_into_rank
import numpy as np


def spectrum_function_cal(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    if QTc_obj.SE_type == 'electron':
        enks, omega_list, self_energies, spectrum_func = _spectrum_function_electron(QTc_obj, k_grid, q_grid, is_metal)
        return enks, omega_list, self_energies, spectrum_func
    elif QTc_obj.SE_type == 'phonon':
        freq_all, phon_vecs_all, omega_list, self_energies, spectrum_func = _spectrum_function_phonon(QTc_obj, k_grid, q_grid, is_metal)
        return freq_all, phon_vecs_all, omega_list, self_energies, spectrum_func

def _spectrum_function_phonon(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    if QTc_obj.read_sparse_grad_mat:
        pass
    else:
        freq_all, phon_vecs_all, omega_list, self_energies, spectrum_func = _spectrum_function_phonon_dense(QTc_obj, k_grid, q_grid, is_metal)
    return freq_all, phon_vecs_all, omega_list, self_energies, spectrum_func

def _spectrum_function_phonon_dense(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    efocus_threshold, k_grid_all, eigen, eigen_vecs, q_grid_local, _, freq_grid, phon_vecs, apply_correction_for_qs = \
        QTc_obj._transport_prepare(k_grid, q_grid, QTc_obj.bands_indices, band_edge_energy=QTc_obj.efermi, is_metal=is_metal)
    omega_list = np.arange(QTc_obj.SE_erange[0], QTc_obj.SE_erange[1]+QTc_obj.SE_erange[2]/10.0, QTc_obj.SE_erange[2])
    nomegas = len(omega_list)

    nbands = len(QTc_obj.bands_indices)
    nbranches = int(3) * QTc_obj.natoms
    nqs_local = len(q_grid_local)
    ieta = 1.0j*QTc_obj.smeark
    ipi = 1.0j*QTcts.PI
    self_energies_local = np.zeros((nqs_local, nbranches, nomegas), dtype=complex)
    if QTc_obj.rank == 0:
        logger = time_logger(QTc_obj.rank_size, 'spectrum_function_phonon_dense')

    ik_all = -1
    for send_rank in range(QTc_obj.rank_size):
        eigen_vecs_recv, eigen_recv = QTc_obj.comm.bcast((eigen_vecs, eigen), root=send_rank)
        for ik, k in enumerate(k_grid_all[send_rank]):
            ik_all += 1
            eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik]
            enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
            if not enk_inside_table.any():
                continue
            phase_k = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
            for iq, q in enumerate(q_grid_local):
                kpq = k + q
                eig_kpq, wave_kpq = QTc_obj._elec_cal_torch(kpq)
                eig_kpq = eig_kpq[QTc_obj.bands_indices, 0]
                wave_kpq = wave_kpq[0, QTc_obj.bands_indices, :]
                apply_correction_for_this_q = apply_correction_for_qs[iq]
                freq = freq_grid[iq]
                eigen_vec_phon = phon_vecs[iq]
                match_table = QTc_obj._get_match_table(eig_k, eig_kpq, freq)
                if not match_table.any():
                    continue
                atomic_phase = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                fermi_kpqs = fermi_weight(eig_kpq - QTc_obj.efermi, QTc_obj.temperature)
                fermi_ks = fermi_weight(eig_k - QTc_obj.efermi, QTc_obj.temperature)
                phase_kpq = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                if apply_correction_for_this_q:
                    epc_corr, tmp1_save = QTc_obj._get_LR_correction_kq(
                        q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=True)
                else:
                    tmp1_save = np.einsum('jm,in -> jimn', np.conj(wave_kpq), wave_k)
                for iband in (np.where(enk_inside_table)[0]):
                    for jband in range(nbands):
                        if np.abs(eig_kpq[jband] - eig_k[iband]) < QTcts.TENPM10:
                            continue
                        tmp1 = tmp1_save[jband, iband]
                        for ibranch in range(nbranches):
                            if match_table[iband, jband, ibranch]:
                                epc = QTc_obj._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                            tmp1, QTc_obj.grad_mat, phase_k, phase_kpq,
                                                            dipole_correction=False)
                                if apply_correction_for_this_q:
                                    epc = epc + epc_corr[iband, jband, ibranch]
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                else:
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                occ_term = fermi_kpqs[jband] - fermi_ks[iband]
                                resonant_term1 = eig_kpq[jband] - eig_k[iband] - omega_list - ieta
                                resonant_term2 = np.real(eig_kpq[jband] - eig_k[iband] - ieta)
                                se_factor = np.real(occ_term / resonant_term1 - occ_term / resonant_term2) * QTc_obj.weight_k[ik_all]
                                self_energies_local[iq, ibranch] +=  g2_tmp * se_factor
                                delta_fomega = w0gauss((eig_k[iband] - eig_kpq[jband] + omega_list) * QTc_obj.inv_smeark, 0) * QTc_obj.inv_smeark
                                se_factor = (fermi_ks[iband] - fermi_kpqs[jband]) * QTc_obj.weight_k[ik_all] * ipi
                                self_energies_local[iq, ibranch] += g2_tmp * se_factor * delta_fomega 
        if QTc_obj.rank == 0:
            logger.step(send_rank+1)

    self_energies_local = QTc_obj._self_energy_symmetrize(freq_grid, self_energies_local)
    normal_imag_part = 2.0 * freq_grid[:,:,None] * self_energies_local.imag
    normal_resonant_part = np.power(omega_list, 2)[None,None,:] - np.power(freq_grid, 2)[:,:,None] - 2.0 * freq_grid[:,:,None] * self_energies_local.real
    spectrum_func_local = np.sum(2.0 * freq_grid[:,:,None] * normal_imag_part / (np.power(normal_resonant_part, 2) + np.power(normal_imag_part, 2)), axis=1)
    spectrum_func_local = -1.0 * spectrum_func_local / QTcts.PI
    self_energies = gather_to_rank_root(self_energies_local, QTc_obj.comm)
    freq_all = gather_to_rank_root(freq_grid, QTc_obj.comm)
    phon_vecs_all = gather_to_rank_root(phon_vecs, QTc_obj.comm)
    spectrum_func = gather_to_rank_root(spectrum_func_local, QTc_obj.comm)
    if QTc_obj.rank == 0:
        self_energies = np.concatenate(self_energies, axis=0)
        freq_all = np.concatenate(freq_all, axis=0)
        phon_vecs_all = np.concatenate(phon_vecs_all, axis=0)
        spectrum_func = np.concatenate(spectrum_func, axis=0)
    return freq_all, phon_vecs_all, omega_list, self_energies, spectrum_func

def _spectrum_function_electron(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    if QTc_obj.spinor_mode == 'collinear':
        if QTc_obj.read_sparse_grad_mat:
            enks, omega_list, self_energies, spectrum_func = _spectrum_function_electron_epc_collinear_sparse(QTc_obj, k_grid, q_grid, is_metal=is_metal)
        else:
            enks, omega_list, self_energies, spectrum_func = _spectrum_function_electron_epc_collinear_dense(QTc_obj, k_grid, q_grid, is_metal=is_metal)
    else:
        if QTc_obj.read_sparse_grad_mat:
            enks, omega_list, self_energies, spectrum_func = _spectrum_function_electron_epc_sparse(QTc_obj, k_grid, q_grid, is_metal=is_metal)
        else:
            enks, omega_list, self_energies, spectrum_func = _spectrum_function_electron_epc_dense(QTc_obj, k_grid, q_grid, is_metal=is_metal)
    return enks, omega_list, self_energies, spectrum_func

def _prepare_electron_epc_self_energy_q_parallel(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    k_grid = QTc_obj._frac2car(k_grid)
    nbands = len(QTc_obj.bands_indices)
    efocus_threshold = [-QTc_obj.over_band_edge, QTc_obj.over_band_edge]

    q_grid_local_frac, q_split_indices = split_into_rank(
        q_grid, 0, QTc_obj.rank, QTc_obj.rank_size, return_indices=True)
    weights_q = np.asarray(QTc_obj.weight_q)
    if weights_q.ndim == 0:
        weights_q_local = np.full(len(q_grid_local_frac), weights_q.item())
    elif weights_q.ndim == 1:
        weights_q_local = split_into_rank(
            weights_q, 0, QTc_obj.rank, QTc_obj.rank_size, split_indices=q_split_indices)
    else:
        raise RuntimeError("Error(9999): q-parallel self-energy requires one-dimensional q weights.")

    nbranches = QTc_obj._get_nbranches()
    q_grid_local, freq_grid, phon_vecs, apply_correction_for_qs = QTc_obj._compute_phonon_spectrum(
        q_grid_local_frac, nbranches, is_metal)

    k_grid_local = split_into_rank(k_grid, 0, QTc_obj.rank, QTc_obj.rank_size)
    eigen_local, eigen_vecs_local, _, _ = QTc_obj._compute_electron_properties(
        k_grid_local, QTc_obj.bands_indices, QTc_obj.efermi, nbands)

    if QTc_obj.comm is None:
        eigen_chunks = [eigen_local]
        eigen_vecs_chunks = [eigen_vecs_local]
    else:
        eigen_chunks = QTc_obj.comm.allgather(eigen_local)
        eigen_vecs_chunks = QTc_obj.comm.allgather(eigen_vecs_local)

    if QTc_obj.spinor_mode == 'collinear':
        eigen = np.concatenate(eigen_chunks, axis=-1)
        eigen_vecs = np.concatenate(eigen_vecs_chunks, axis=1)
    else:
        eigen = np.concatenate(eigen_chunks, axis=-1)
        eigen_vecs = np.concatenate(eigen_vecs_chunks, axis=0)

    return efocus_threshold, k_grid, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs

def _spectrum_function_electron_epc_dense(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    k_grid = QTc_obj._frac2car(k_grid)
    efocus_threshold, k_grid_local, eigen_local, eigen_vecs_local, _, q_grid_all, freq_grid, phon_vecs, apply_correction_for_qs = \
        QTc_obj._transport_prepare(k_grid, q_grid, QTc_obj.bands_indices, band_edge_energy=QTc_obj.efermi, is_metal=is_metal, reduced_type='q')
    omega_list = np.arange(QTc_obj.SE_erange[0], QTc_obj.SE_erange[1]+QTc_obj.SE_erange[2]/10.0, QTc_obj.SE_erange[2])
    nomegas = len(omega_list)
    eigen_local = eigen_local - QTc_obj.efermi
    nqs = len(q_grid)
    nbands = len(QTc_obj.bands_indices)
    nbranches = int(3) * QTc_obj.natoms
    nks_local = len(k_grid_local)
    ieta = 1.0j*QTc_obj.smeark
    # eta_square = QTc_obj.smeark * QTc_obj.smeark
    self_energies_local = np.zeros((nks_local, nbands, nomegas), dtype=complex)
    # qp_strength_local = np.zeros((nks_local, nbands), dtype=float)
    if QTc_obj.rank == 0:
        logger = time_logger(QTc_obj.rank_size, 'spectrum_function_electron_epc_dense')

    iq_all = -1
    for send_rank in range(QTc_obj.rank_size):
        freq_grid_recv, phon_vecs_recv, apply_correction_for_qs_recv = QTc_obj.comm.bcast((freq_grid, phon_vecs, apply_correction_for_qs), root=send_rank)
        for iq, q in enumerate(q_grid_all[send_rank]):
            iq_all += 1
            freq, eigen_vec_phon, apply_correction_for_this_q = freq_grid_recv[iq], phon_vecs_recv[iq], apply_correction_for_qs_recv[iq]
            bose_qvs = bose_weight(freq, QTc_obj.temperature)
            atomic_phase = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
            phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
            for ik, k in enumerate(k_grid_local):
                eig_k, wave_k = eigen_local[:, ik], eigen_vecs_local[ik]
                enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                if not enk_inside_table.any():
                    continue
                phase_k = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                kpq = k + q
                eig_kpq, wave_kpq = QTc_obj._elec_cal_torch(kpq)
                eig_kpq = eig_kpq[QTc_obj.bands_indices, 0]
                wave_kpq = wave_kpq[0, QTc_obj.bands_indices, :]
                eig_kpq_shifted = eig_kpq - QTc_obj.efermi
                match_table = QTc_obj._get_match_table(omega_list, eig_kpq_shifted, freq, include_off_diag=True)
                if not match_table.any():
                    continue
                match_table = np.repeat(match_table[None, :, :], nbands, axis=0)
                fermi_kpqs = fermi_weight(eig_kpq_shifted, QTc_obj.temperature)
                phase_kpq = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                if apply_correction_for_this_q:
                    epc_corr, tmp1_save = QTc_obj._get_LR_correction_kq(
                        q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=True)
                else:
                    tmp1_save = np.einsum('jm,in -> jimn', np.conj(wave_kpq), wave_k)
                for iband in (np.where(enk_inside_table)[0]):
                    for jband in range(nbands):
                        tmp1 = tmp1_save[jband, iband]
                        for ibranch in range(nbranches):
                            if match_table[iband, jband, ibranch]:
                                epc = QTc_obj._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                            tmp1, QTc_obj.grad_mat, phase_k, phase_kpq,
                                                            dipole_correction=False)
                                if apply_correction_for_this_q:
                                    epc = epc + epc_corr[iband, jband, ibranch]
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                else:
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                occ_factor1 = fermi_kpqs[jband] + bose_qvs[ibranch]
                                occ_factor2 = 1.0 - fermi_kpqs[jband] + bose_qvs[ibranch]
                                e_pole1 = eig_kpq_shifted[jband] - freq[ibranch] - ieta
                                e_pole2 = eig_kpq_shifted[jband] + freq[ibranch] - ieta
                                resonant_term1 = omega_list - e_pole1
                                resonant_term2 = omega_list - e_pole2
                                se_factor = occ_factor1 / resonant_term1 + occ_factor2 / resonant_term2
                                se0_factor = -1.0 * np.real(occ_factor1 / e_pole1 + occ_factor2 / e_pole2)
                                self_energies_local[ik, iband] = self_energies_local[ik, iband] + g2_tmp * (se_factor - se0_factor) * QTc_obj.weight_q[iq_all]
        if QTc_obj.rank == 0:
            logger.step(send_rank+1)
    eigen_local = np.transpose(eigen_local)
    self_energies_local = QTc_obj._self_energy_symmetrize(eigen_local, self_energies_local)
    se_imag_local = self_energies_local.imag
    se_real_local = self_energies_local.real
    spectrum_func_local = np.sum(np.abs(se_imag_local) / (np.power(omega_list[None,None,:] - eigen_local[:,:,None] - se_real_local, 2) + np.power(se_imag_local, 2)), axis=1) / QTcts.PI
    self_energies = gather_to_rank_root(self_energies_local, QTc_obj.comm)
    spectrum_func = gather_to_rank_root(spectrum_func_local, QTc_obj.comm)
    eigen_all = gather_to_rank_root(eigen_local, QTc_obj.comm) # shape: (nks_local, nbands)
    if QTc_obj.rank == 0:
        eigen_all = np.concatenate(eigen_all, axis=0) # shape: (nks, nbands)
        self_energies = np.concatenate(self_energies, axis=0) # shape: (nks, nbands, nomegas)
        spectrum_func = np.concatenate(spectrum_func, axis=0) # shape: (nks, nomegas)
    else:
        spectrum_func = None
    return eigen_all, omega_list, self_energies, spectrum_func

def _spectrum_function_electron_epc_sparse(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    efocus_threshold, k_grid, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
        _prepare_electron_epc_self_energy_q_parallel(QTc_obj, k_grid, q_grid, is_metal=is_metal)
    omega_list = np.arange(QTc_obj.SE_erange[0], QTc_obj.SE_erange[1]+QTc_obj.SE_erange[2]/10.0, QTc_obj.SE_erange[2])
    eigen = eigen - QTc_obj.efermi
    nomegas = len(omega_list)
    nbands = len(QTc_obj.bands_indices)
    nbranches = int(3) * QTc_obj.natoms
    nks = len(k_grid)
    self_energies_local = np.zeros((nks, nbands, nomegas), dtype=complex)
    phase_k_all = np.exp(QTcts.JTWOPI * (QTc_obj.nbr_shift_of_cell_sc @ k_grid.T))
    atomic_mass = np.ascontiguousarray(QTc_obj.atomic_mass)
    if QTc_obj.rank == 0:
        logger = time_logger(max(len(q_grid_local), 1), 'spectrum_function_electron_epc_sparse')

    for iq, q in enumerate(q_grid_local):
        freq, eigen_vec_phon, apply_correction_for_this_q = freq_grid[iq], phon_vecs[iq], apply_correction_for_qs[iq]
        bose_qvs = bose_weight(freq, QTc_obj.temperature)
        atomic_phase = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
        phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
        kpqs = k_grid + q
        eig_kpqs, wave_kpqs = QTc_obj._elec_cal_torch(kpqs)
        eig_kpqs = eig_kpqs[QTc_obj.bands_indices]
        wave_kpqs = wave_kpqs[:, QTc_obj.bands_indices, :]
        phase_q = np.exp(QTcts.JTWOPI * (QTc_obj.nbr_shift_of_cell_sc @ q))
        phase_kpq_all = phase_k_all * phase_q[:, None]
        for ik, k in enumerate(k_grid):
            eig_k, wave_k = eigen[:, ik], eigen_vecs[ik]
            enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
            if not enk_inside_table.any():
                continue
            eig_kpq = eig_kpqs[:, ik]
            wave_kpq = wave_kpqs[ik]
            eig_kpq_shifted = eig_kpq - QTc_obj.efermi
            # match_table: shape (nbands, nbranches)
            match_table = QTc_obj._get_match_table(omega_list, eig_kpq_shifted, freq, include_off_diag=True)
            if not match_table.any():
                continue
            match_table = np.repeat(match_table[None, :, :], nbands, axis=0)
            fermi_kpqs = fermi_weight(eig_kpq_shifted, QTc_obj.temperature)
            if apply_correction_for_this_q:
                epc_corr = QTc_obj._get_LR_correction_kq(
                    q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=False)
            else:
                epc_corr = np.zeros((nbands, nbands, nbranches), dtype=complex)
            self_energy_single = np.zeros((nbands, nbranches, nomegas), dtype=complex, order='C')
            electron_self_energy_cal_EPC_helper_sparse(freq.copy(), atomic_mass.copy(), phvec_wap.copy(),
                                                        phase_k_all[:, ik].copy(), phase_kpq_all[:, ik].copy(), QTc_obj.grad_mat,
                                                        wave_k.copy(), wave_kpq.copy(),
                                                        enk_inside_table.copy(), match_table.copy(),
                                                        eig_kpq_shifted.copy(),
                                                        bose_qvs.copy(), fermi_kpqs.copy(),
                                                        epc_corr.copy(), self_energy_single,
                                                        omega_list.copy(), weights_q_local[iq], QTc_obj.smeark)
            self_energies_local[ik] += np.sum(self_energy_single, axis=1)
        if QTc_obj.rank == 0:
            logger.step(iq+1)

    if QTc_obj.comm is not None:
        QTc_obj.comm.Allreduce(MPI.IN_PLACE, self_energies_local, op=MPI.SUM)

    eigen_all = np.transpose(eigen)
    self_energies = QTc_obj._self_energy_symmetrize(eigen_all, self_energies_local)
    if QTc_obj.rank == 0:
        se_imag = self_energies.imag
        se_real = self_energies.real
        spectrum_func = np.sum(np.abs(se_imag) / (np.power(omega_list[None,None,:] - eigen_all[:,:,None] - se_real, 2) + np.power(se_imag, 2)), axis=1) / QTcts.PI
    else:
        spectrum_func = None
    return eigen_all, omega_list, self_energies, spectrum_func

def _spectrum_function_electron_epc_collinear_dense(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    k_grid = QTc_obj._frac2car(k_grid)
    efocus_threshold, k_grid_local, eigen_local, eigen_vecs_local, _, q_grid_all, freq_grid, phon_vecs, apply_correction_for_qs = \
        QTc_obj._transport_prepare(k_grid, q_grid, QTc_obj.bands_indices, band_edge_energy=QTc_obj.efermi, is_metal=is_metal, reduced_type='q')
    omega_list = np.arange(QTc_obj.SE_erange[0], QTc_obj.SE_erange[1]+QTc_obj.SE_erange[2]/10.0, QTc_obj.SE_erange[2])
    if is_metal:
        eigen_local = eigen_local - QTc_obj.efermi
    else:
        eigen_local = eigen_local - QTc_obj.efermi[:,None,None]
    nomegas = len(omega_list)
    nqs = len(q_grid)
    nbands = len(QTc_obj.bands_indices)
    nbranches = int(3) * QTc_obj.natoms
    nks_local = len(k_grid_local)
    ieta = 1.0j*QTc_obj.smeark
    # eta_square = QTc_obj.smeark * QTc_obj.smeark
    self_energies_local = np.zeros((2, nks_local, nbands, nomegas), dtype=complex)
    # qp_strength_local = np.zeros((nks_local, nbands), dtype=float)
    if QTc_obj.rank == 0:
        logger = time_logger(QTc_obj.rank_size, 'spectrum_function_electron_epc_collinear_dense')

    iq_all = -1
    for send_rank in range(QTc_obj.rank_size):
        freq_grid_recv, phon_vecs_recv, apply_correction_for_qs_recv = QTc_obj.comm.bcast((freq_grid, phon_vecs, apply_correction_for_qs), root=send_rank)
        for iq, q in enumerate(q_grid_all[send_rank]):
            iq_all += 1
            freq, eigen_vec_phon, apply_correction_for_this_q = freq_grid_recv[iq], phon_vecs_recv[iq], apply_correction_for_qs_recv[iq]
            bose_qvs = bose_weight(freq, QTc_obj.temperature)
            atomic_phase = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
            phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
            for ik, k in enumerate(k_grid_local):
                # eig_k: shape (2, nbands)
                eig_k, wave_k = eigen_local[:, :, ik], eigen_vecs_local[:, ik]
                enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                if not enk_inside_table.any():
                    continue
                phase_k = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                kpq = k + q
                eig_kpq, wave_kpq = QTc_obj._elec_cal_torch(kpq)
                eig_kpq = np.stack((eig_kpq[QTc_obj.bands_indices, 0], eig_kpq[QTc_obj.bands_indices+QTc_obj.norbs, 0]), axis=0)
                wave_kpq = np.stack((wave_kpq[0, QTc_obj.bands_indices], wave_kpq[0, QTc_obj.bands_indices+QTc_obj.norbs]), axis=0)
                phase_kpq = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                for ispin in range(2):
                    if not is_metal:
                        eig_kpq_shifted = eig_kpq[ispin] - QTc_obj.efermi[ispin]
                    else:
                        eig_kpq_shifted = eig_kpq[ispin] - QTc_obj.efermi
                    match_table = QTc_obj._get_match_table(omega_list, eig_kpq_shifted, freq, include_off_diag=True)
                    if not match_table.any():
                        continue
                    match_table = np.repeat(match_table[None, :, :], nbands, axis=0)
                    fermi_kpqs = fermi_weight(eig_kpq_shifted, QTc_obj.temperature)
                    if apply_correction_for_this_q:
                        epc_corr, tmp1_save = QTc_obj._get_LR_correction_kq(
                            q, freq, phvec_wap, match_table, enk_inside_table[ispin], k, wave_k[ispin], wave_kpq[ispin], return_tmp1=True)
                    else:
                        tmp1_save = np.einsum('jm,in -> jimn', np.conj(wave_kpq[ispin]), wave_k[ispin])
                    for iband in (np.where(enk_inside_table[ispin])[0]):
                        for jband in range(nbands):
                            tmp1 = tmp1_save[jband, iband]
                            for ibranch in range(nbranches):
                                if match_table[iband, jband, ibranch]:
                                    epc = QTc_obj._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                                tmp1, QTc_obj.grad_mat[ispin], phase_k, phase_kpq,
                                                                dipole_correction=False)
                                    if apply_correction_for_this_q:
                                        epc = epc + epc_corr[iband, jband, ibranch]
                                        g2_tmp = np.abs(epc) * np.abs(epc)
                                    else:
                                        g2_tmp = np.abs(epc) * np.abs(epc)
                                    occ_factor1 = fermi_kpqs[jband] + bose_qvs[ibranch]
                                    occ_factor2 = 1.0 - fermi_kpqs[jband] + bose_qvs[ibranch]
                                    e_pole1 = eig_kpq_shifted[jband] - freq[ibranch] - ieta
                                    e_pole2 = eig_kpq_shifted[jband] + freq[ibranch] - ieta
                                    resonant_term1 = omega_list - e_pole1
                                    resonant_term2 = omega_list - e_pole2
                                    se_factor = occ_factor1 / resonant_term1 + occ_factor2 / resonant_term2
                                    se0_factor = -1.0 * np.real(occ_factor1 / e_pole1 + occ_factor2 / e_pole2)
                                    self_energies_local[ispin, ik, iband] += g2_tmp * (se_factor - se0_factor) * QTc_obj.weight_q[iq_all]
        if QTc_obj.rank == 0:
            logger.step(send_rank+1)
    eigen_local = np.transpose(eigen_local, (0, 2, 1))
    spectrum_func_local = np.zeros((2, nks_local, nomegas))
    for ispin in range(2):
        self_energies_local[ispin] = QTc_obj._self_energy_symmetrize(eigen_local[ispin], self_energies_local[ispin])
        se_imag_local = self_energies_local[ispin].imag
        se_real_local = self_energies_local[ispin].real
        spectrum_func_local[ispin] = np.sum(np.abs(se_imag_local) / (np.power(omega_list[None,None,:] - eigen_local[ispin,:,:,None] - se_real_local, 2) + \
                                                                        np.power(se_imag_local, 2)), axis=1) / QTcts.PI
    self_energies = gather_to_rank_root(self_energies_local, QTc_obj.comm)
    spectrum_func = gather_to_rank_root(spectrum_func_local, QTc_obj.comm)
    eigen_all = gather_to_rank_root(eigen_local, QTc_obj.comm) # shape: (2, nks_local, nbands)
    if QTc_obj.rank == 0:
        eigen_all = np.concatenate(eigen_all, axis=1) # shape: (2, nks, nbands)
        self_energies = np.concatenate(self_energies, axis=1) # shape: (2, nks, nbands, nomegas)
        spectrum_func = np.concatenate(spectrum_func, axis=1) # shape: (2, nks, nomegas)
    else:
        spectrum_func = None
    return eigen_all, omega_list, self_energies, spectrum_func

def _spectrum_function_electron_epc_collinear_sparse(QTc_obj, k_grid:np.ndarray, q_grid:np.ndarray, is_metal:bool=False):
    efocus_threshold, k_grid, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, phon_vecs, apply_correction_for_qs = \
        _prepare_electron_epc_self_energy_q_parallel(QTc_obj, k_grid, q_grid, is_metal=is_metal)
    omega_list = np.arange(QTc_obj.SE_erange[0], QTc_obj.SE_erange[1]+QTc_obj.SE_erange[2]/10.0, QTc_obj.SE_erange[2])
    if is_metal:
        eigen = eigen - QTc_obj.efermi
    else:
        eigen = eigen - QTc_obj.efermi[:,None,None]
    nomegas = len(omega_list)
    nbands = len(QTc_obj.bands_indices)
    nbranches = int(3) * QTc_obj.natoms
    nks = len(k_grid)
    self_energies_local = np.zeros((2, nks, nbands, nomegas), dtype=complex)
    phase_k_all = np.exp(QTcts.JTWOPI * (QTc_obj.nbr_shift_of_cell_sc @ k_grid.T))
    atomic_mass = np.ascontiguousarray(QTc_obj.atomic_mass)
    if QTc_obj.rank == 0:
        logger = time_logger(max(len(q_grid_local), 1), 'spectrum_function_electron_epc_collinear_sparse')

    for iq, q in enumerate(q_grid_local):
        freq, eigen_vec_phon, apply_correction_for_this_q = freq_grid[iq], phon_vecs[iq], apply_correction_for_qs[iq]
        bose_qvs = bose_weight(freq, QTc_obj.temperature)
        atomic_phase = np.exp(QTcts.JTWOPI*np.sum(QTc_obj.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
        phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
        kpqs = k_grid + q
        eig_kpqs, wave_kpqs = QTc_obj._elec_cal_torch(kpqs)
        eig_kpqs = np.stack((eig_kpqs[QTc_obj.bands_indices], eig_kpqs[QTc_obj.bands_indices+QTc_obj.norbs]), axis=0)
        wave_kpqs = np.stack((wave_kpqs[:, QTc_obj.bands_indices], wave_kpqs[:, QTc_obj.bands_indices+QTc_obj.norbs]), axis=0)
        phase_q = np.exp(QTcts.JTWOPI * (QTc_obj.nbr_shift_of_cell_sc @ q))
        phase_kpq_all = phase_k_all * phase_q[:, None]
        for ik, k in enumerate(k_grid):
            # eig_k: shape (2, nbands)
            eig_k, wave_k = eigen[:, :, ik], eigen_vecs[:, ik]
            enk_inside_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
            if not enk_inside_table.any():
                continue
            eig_kpq = eig_kpqs[:, :, ik]
            wave_kpq = wave_kpqs[:, ik]
            for ispin in range(2):
                if not is_metal:
                    eig_kpq_shifted = eig_kpq[ispin] - QTc_obj.efermi[ispin]
                else:
                    eig_kpq_shifted = eig_kpq[ispin] - QTc_obj.efermi
                match_table = QTc_obj._get_match_table(omega_list, eig_kpq_shifted, freq, include_off_diag=True)
                if not match_table.any():
                    continue
                match_table = np.repeat(match_table[None, :, :], nbands, axis=0)
                fermi_kpqs = fermi_weight(eig_kpq_shifted, QTc_obj.temperature)
                if apply_correction_for_this_q:
                    epc_corr = QTc_obj._get_LR_correction_kq(
                        q, freq, phvec_wap, match_table, enk_inside_table[ispin], k, wave_k[ispin], wave_kpq[ispin], return_tmp1=False)
                else:
                    epc_corr = np.zeros((nbands, nbands, nbranches), dtype=complex)
                self_energy_single = np.zeros((nbands, nbranches, nomegas), dtype=complex, order='C')
                electron_self_energy_cal_EPC_helper_sparse(freq.copy(), atomic_mass.copy(), phvec_wap.copy(),
                                                            phase_k_all[:, ik].copy(), phase_kpq_all[:, ik].copy(), QTc_obj.grad_mat[ispin],
                                                            wave_k[ispin].copy(), wave_kpq[ispin].copy(),
                                                            enk_inside_table[ispin].copy(), match_table.copy(),
                                                            eig_kpq_shifted.copy(),
                                                            bose_qvs.copy(), fermi_kpqs.copy(),
                                                            epc_corr.copy(), self_energy_single,
                                                            omega_list.copy(), weights_q_local[iq], QTc_obj.smeark)
                self_energies_local[ispin, ik] += np.sum(self_energy_single, axis=1)
        if QTc_obj.rank == 0:
            logger.step(iq+1)

    if QTc_obj.comm is not None:
        QTc_obj.comm.Allreduce(MPI.IN_PLACE, self_energies_local, op=MPI.SUM)

    eigen_all = np.transpose(eigen, (0, 2, 1))
    self_energies = np.empty_like(self_energies_local)
    spectrum_func = None
    for ispin in range(2):
        self_energies[ispin] = QTc_obj._self_energy_symmetrize(eigen_all[ispin], self_energies_local[ispin])
    if QTc_obj.rank == 0:
        spectrum_func = np.zeros((2, nks, nomegas))
        for ispin in range(2):
            se_imag = self_energies[ispin].imag
            se_real = self_energies[ispin].real
            spectrum_func[ispin] = np.sum(np.abs(se_imag) / (np.power(omega_list[None,None,:] - eigen_all[ispin,:,:,None] - se_real, 2) + \
                                                              np.power(se_imag, 2)), axis=1) / QTcts.PI
    return eigen_all, omega_list, self_energies, spectrum_func
