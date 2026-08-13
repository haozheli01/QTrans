"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
from qtrans.extensions import epc_elements_cal_helper_sparse
import numpy as np


class _EPCMixin:
    def _epc_cal_element(self, k_cart:np.ndarray, q_cart:np.ndarray, 
                         wqv:float, phvec_wap:np.ndarray,
                         tmp1:np.ndarray, grad_mat:np.ndarray,
                         phase_k:np.ndarray, phase_kpq:np.ndarray,
                         dipole_correction:bool):
        factor = 1.0 / np.sqrt(2.0 * self.atomic_mass * wqv) # shape: (natoms, )
        tmp2 = np.einsum('ij,mn -> mnij', factor[:,None]*phvec_wap, tmp1)
        epc = 0.0
        for i_m, m in enumerate(self.cell_cut_list): # ncells
            for i_n, n in enumerate(self.cell_cut_list): # ncells 
                epc += np.conj(phase_kpq[m])*phase_k[n]*np.einsum('mnij,mnij', tmp2, grad_mat[i_m,i_n])
        if dipole_correction or self.quadrupole_correction:
            if self.system_2D:
                epc_corr = self._get_LR_correction_2D(tmp1, k_cart, q_cart, factor, phvec_wap)
            else:
                epc_corr = self._get_LR_correction_3D(tmp1, k_cart, q_cart, factor, phvec_wap, order=self.LRC_taylor_order)
        else:
            epc_corr = 0.0
        epc = epc + epc_corr
        return epc

    def epc_cal_path(self, k_fix:np.ndarray, q_path:np.ndarray, bands_indices_ini:np.ndarray, bands_indices_fin:np.ndarray, do_symm:bool=True, is_sparse:bool=False):
        """
        Calculate electron-phonon coupling along a q-path for a fixed initial k point.

        Args:
            k_fix (list or np.ndarray): Fixed initial k point in fractional coordinates, shape (3,).
            q_path (list or np.ndarray): q-point path in fractional coordinates, shape (nqs, 3).
            bands_indices_ini (np.ndarray): The band indices of initial state, begin from 0.
            bands_indices_fin (np.ndarray): The band indices of final state, begin from 0.
            do_symm (bool): If True, do the average over degenerate state.
            is_sparse (bool): If True, use the sparse grad_mat.

        Returns:
            epc_all (np.ndarray): Absolute EPC values in Hartree. For non-collinear or
                non-spin-polarized modes the shape is (nqs, nbands_ini, nbands_fin, nbranches).
                For collinear spin mode the shape is
                (2, nqs, nbands_ini, nbands_fin, nbranches), where the leading axis is spin.
        """
        nbands_ini = len(bands_indices_ini)
        nbands_fin = len(bands_indices_fin)
        nqs = len(q_path)
        nbranches = int(3) * self.natoms

        # calculate the phonon spectrum
        freq_grid, phon_vecs = self._phonon_cal(q_path)
        phon_vecs = phon_vecs.reshape(-1, nbranches, self.natoms, 3)
        if self.apply_correction or self.quadrupole_correction:
            apply_correction_for_qs = np.linalg.norm(q_path, axis=1) < self.q_cut
        else:
            apply_correction_for_qs = np.zeros(nqs, dtype=bool)
        k_fix = self._frac2car(k_fix)[0]
        q_path = self._frac2car(q_path)
        epc_all:np.ndarray = None
        eig_k, wave_k = self._elec_cal_torch(k_fix)
        phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k_fix[None,:], axis=-1)) # shape: (ncells, )
        if self.spinor_mode == 'collinear':
            eig_k = np.stack((eig_k[bands_indices_ini, 0], eig_k[bands_indices_ini+self.norbs, 0]), axis=0)
            wave_k = np.stack((wave_k[0, bands_indices_ini], wave_k[0, bands_indices_ini+self.norbs]), axis=0) # shape: (2, nbands_ini, norbs)
            epc_all = np.zeros((2, nqs, nbands_ini, nbands_fin, nbranches))
            for iq, q in enumerate(q_path):
                apply_correction_for_this_q = apply_correction_for_qs[iq]
                kpq = k_fix + q
                eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                eig_kpq = np.stack((eig_kpq[bands_indices_fin, 0], eig_kpq[bands_indices_fin+self.norbs, 0]), axis=0)
                wave_kpq = np.stack((wave_kpq[0, bands_indices_fin], wave_kpq[0, bands_indices_fin+self.norbs]), axis=0) # shape: (2, nbands_fin, norbs)
                freq = np.abs(freq_grid[iq])
                eigen_vec_phon = phon_vecs[iq]
                for ispin in range(2):
                    epc_all[ispin, iq] = self._epc_norm_cal_kq(k_fix, q, eig_k[ispin], wave_k[ispin], eig_kpq[ispin], wave_kpq[ispin], freq, eigen_vec_phon,
                                                            apply_correction_for_this_q, self.grad_mat[ispin],
                                                            phase_k, do_symm, is_sparse)
        else:
            eig_k = eig_k[bands_indices_ini, 0]
            wave_k = wave_k[0, bands_indices_ini] # shape: (nbands_ini, norbs)
            epc_all = np.zeros((nqs, nbands_ini, nbands_fin, nbranches))
            for iq, q in enumerate(q_path):
                apply_correction_for_this_q = apply_correction_for_qs[iq]
                kpq = k_fix + q
                eig_kpq, wave_kpq = self._elec_cal_torch(kpq)
                eig_kpq = eig_kpq[bands_indices_fin, 0]
                wave_kpq = wave_kpq[0, bands_indices_fin] # shape: (nbands_fin, norbs)
                freq = np.abs(freq_grid[iq])
                eigen_vec_phon = phon_vecs[iq] # shape (nbranches, natoms, 3)
                epc_all[iq] = self._epc_norm_cal_kq(k_fix, q, eig_k, wave_k, eig_kpq, wave_kpq, freq, eigen_vec_phon,
                                               apply_correction_for_this_q, self.grad_mat,
                                               phase_k, do_symm, is_sparse)
        return epc_all

    def _epc_norm_cal_kq(self, k_cart, q_cart,
                    eig_k:np.ndarray, wave_k:np.ndarray,
                    eig_kpq:np.ndarray, wave_kpq:np.ndarray,
                    freq:np.ndarray, phon_vec:np.ndarray,
                    dipole_correction:bool, grad_mat:np.ndarray,
                    phase_k:np.ndarray=None,
                    do_symm:bool=True,
                    is_sparse:bool=False):
        """
        Calculate absolute electron-phonon coupling matrix elements for one fixed (k, q) pair.

        Args:
            k_cart (np.ndarray): Cartesian k point, shape (3,).
            q_cart (np.ndarray): Cartesian q point, shape (3,).
            eig_k (np.ndarray): Initial-state eigenvalues at k, shape (nbands_ini,).
                The values are only used by the optional degeneracy symmetrization.
            wave_k (np.ndarray): Initial-state wave functions at k, shape (nbands_ini, norbs).
            eig_kpq (np.ndarray): Final-state eigenvalues at k+q, shape (nbands_fin,).
                The values are only used by the optional degeneracy symmetrization.
            wave_kpq (np.ndarray): Final-state wave functions at k+q, shape (nbands_fin, norbs).
            freq (np.ndarray): Phonon frequencies for q, shape (nbranches,).
            phon_vec (np.ndarray): Phonon eigenvectors for q, shape (nbranches, natoms, 3).
            dipole_correction (bool): Whether long-range dipole correction is enabled for this q.
            grad_mat: Electron-phonon gradient matrix. Use dense data when ``is_sparse`` is False
                and sparse CSR-like dictionaries when ``is_sparse`` is True.
            phase_k (np.ndarray or None): Precomputed electronic phase at k, shape (ncells,).
                If None, it is computed inside the function.
            do_symm (bool): If True, average EPC values over degenerate electronic states.
            is_sparse (bool): If True, call the sparse C helper; otherwise use the dense Python path.

        Returns:
            np.ndarray: Absolute EPC values in Hartree, shape
                (nbands_ini, nbands_fin, nbranches). Returns zeros when |q| is below the
                small-q cutoff used by this routine.
        """
        nbands_ini = len(eig_k)
        nbands_fin = len(eig_kpq)
        nbranches = len(freq)
        if phase_k is None:
            phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k_cart[None,:], axis=-1)) # shape: (ncells, )
        epcs_kq = np.zeros((nbands_ini, nbands_fin, nbranches))
        if np.linalg.norm(q_cart) < QTcts.TENPM6:
            return epcs_kq
        kpq_cart = k_cart + q_cart
        atomic_phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.pos*q_cart[None,:], axis=-1)) # shape (natoms, )
        phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*kpq_cart[None,:], axis=-1)) # shape (ncells, )
        # phon_vec_with_atomic_phase for short
        phon_vec_wap = atomic_phase[None, :, None] * phon_vec # shape (nbranches, natoms, 3)
        if is_sparse:
            ph_prefac = np.ascontiguousarray(1.0 / np.sqrt(2.0 * self.atomic_mass[None,:] * freq[:,None])) # shape (nbranches, natoms)
            # Zero out the prefactor for branches below the frequency cutoff so that
            # any small-frequency (acoustic-Gamma-like) contribution is excluded from g.
            # This mirrors the April-version behaviour in _epc_norm_cal_kq.
            mask = np.where(freq < self.phonon_frequency_cutoff)
            ph_prefac[mask] = 0.0
            phon_vec_wap = np.ascontiguousarray(phon_vec_wap)
            phase_k = np.ascontiguousarray(phase_k)
            phase_kpq = np.ascontiguousarray(phase_kpq)
            zero_epc_corr = np.zeros((nbranches, ), dtype=complex)
            for iband in range(nbands_ini):
                for jband in range(nbands_fin):
                    wave_coeff = np.ascontiguousarray(np.outer(np.conj(wave_kpq[jband]), wave_k[iband])) # shape (norbs, norbs)
                    epc_corr = zero_epc_corr
                    if dipole_correction or self.quadrupole_correction:
                        epc_corr = np.empty((nbranches, ), dtype=complex)
                        for ibranch in range(nbranches):
                            if self.system_2D:
                                # Use the full 2D long-range correction (handles dipole + quadrupole dispatch internally)
                                epc_corr[ibranch] = self._get_LR_correction_2D(wave_coeff, k_cart, q_cart, ph_prefac[ibranch], phon_vec_wap[ibranch])
                            else:
                                epc_corr[ibranch] = self._get_LR_correction_3D(wave_coeff, k_cart, q_cart, ph_prefac[ibranch], phon_vec_wap[ibranch], order=self.LRC_taylor_order)
                    g_abs = np.zeros((nbranches, ), dtype=float, order='C')
                    epc_elements_cal_helper_sparse(ph_prefac, phon_vec_wap,
                                                    phase_k, phase_kpq, grad_mat,
                                                    wave_coeff, epc_corr, g_abs)
                    epcs_kq[iband, jband] = g_abs
        else:
            for iband in range(nbands_ini):
                for jband in range(nbands_fin):
                    tmp1 = np.einsum('m,n -> mn', np.conj(wave_kpq[jband]), wave_k[iband])
                    for ibranch in range(nbranches):
                        if freq[ibranch] > self.phonon_frequency_cutoff:
                            epc = self._epc_cal_element(k_cart, q_cart, freq[ibranch], phon_vec_wap[ibranch], tmp1, grad_mat, phase_k, phase_kpq, dipole_correction)
                        else:
                            epc = 0.0
                        epcs_kq[iband, jband, ibranch] = np.abs(epc)
        if do_symm:
            epcs_kq = self._epc_symmetrize(epcs_kq, freq)
        return epcs_kq

    def _epc_symmetrize_dim(self, excs:np.ndarray, eigens:np.ndarray, tolerance:float=QTcts.TENPM6):
        excs = np.asarray(excs)
        eigens = np.asarray(eigens)
        diff = np.abs(eigens[:, None] - eigens[None, :])
        mask = diff < tolerance
        epc_sq = np.power(excs, 2)
        sum_sq:np.ndarray = np.matmul(mask, epc_sq)
        counts = mask.sum(axis=1)
        counts_reshaped = counts[(...,) + (None,) * (sum_sq.ndim - 1)]
        epc_sym = np.sqrt(sum_sq / counts_reshaped)
        return epc_sym

    def _epc_symmetrize(self, excs:np.ndarray, freq:np.ndarray, tolerance:float=QTcts.TENPM6):
        nbands_ini, nbands_fin, _ = excs.shape
        excs_symm = np.copy(excs)
        for iband in range(nbands_ini):
            for jband in range(nbands_fin):
                excs_symm[iband, jband] = self._epc_symmetrize_dim(excs_symm[iband, jband], freq, tolerance=tolerance)
        return excs_symm
