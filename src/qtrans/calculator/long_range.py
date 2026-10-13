"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
from qtrans.numerics.sparse import build_sparse_matrix
import numpy as np
import opt_einsum as oe


class _LongRangeMixin:
    def _Hv_cell_prepare(self,):
        if self.spinor_mode == 'collinear':
            Hv_cell = np.einsum('snij,nk->snijk', self.graph_data.H_cell, self.graph_data.nbr_shift_of_cell) # (2, ncells, norbs, norbs, 3)
        else:
            Hv_cell = np.einsum('nij,nk->nijk', self.graph_data.H_cell, self.graph_data.nbr_shift_of_cell) # (ncells, norbs, norbs, 3)
        return Hv_cell

    def _Sv_cell_prepare(self,):
        S_cell_full = self.graph_data.S_cell
        if self.spinor_mode == 'nc':
            ncells, norbs_half, _ = S_cell_full.shape
            norbs = norbs_half * 2
            new_S_cell = np.zeros((ncells, norbs, norbs), dtype=S_cell_full.dtype)
            new_S_cell[:, :norbs_half, :norbs_half] = S_cell_full
            new_S_cell[:, norbs_half:, norbs_half:] = S_cell_full
            S_cell_full = new_S_cell
        self.graph_data.S_cell_full = S_cell_full
        Sv_cell = np.einsum('nij,nk->nijk', S_cell_full, self.graph_data.nbr_shift_of_cell) # (ncells, norbs, norbs, 3)
        return Sv_cell

    def _vel_nk_cal_from_HS(self, eig, wfn, k_grid, collinear_part_index:int=0):
        """Calculate the band velocity of the electron

        Args:
            eig (np.ndarray): shape: (nbands, nks)
            wfn (np.ndarray): shape: (nks, nbands, norbs)
            k_grid (np.ndarray): shape: (nk, 3)

        Returns:
            elec_vel: # shape: (nk, nbands, 3)
        """
        k_vec = k_grid.reshape(-1, 3)
        nbr_shift = self.graph_data.nbr_shift_of_cell

        # Compute phase factor once: (nk, ncells)
        phase = 1j * np.exp(2j * np.pi * (k_vec @ nbr_shift.T))

        # Precompute conj(wfn) once, reused in both contractions
        conj_wfn = np.conj(wfn)

        # Build HK_v and contract with wfn
        if self.spinor_mode == 'collinear':
            Hv = self.graph_data.Hv_cell[collinear_part_index]
        else:
            Hv = self.graph_data.Hv_cell

        elec_vel_1 = oe.contract('nc,cijk,nbi,nbj->nbk', phase, Hv, conj_wfn, wfn).real  # (nk, nbands, 3)

        # Fused contraction for overlap part
        Sv = self.graph_data.Sv_cell
        # elec_vel_coor = oe.contract('nc,cijk,nbi,nbj,bn->nbk', phase, Sv, conj_wfn, wfn, eig).real  # (nk, nbands, 3)
        Sv = oe.contract('nc,cijk->nijk', phase, Sv) # (nk, norbs, norbs, 3)
        elec_vel_coor = oe.contract('nij, nik, in, njkm->nim', np.conj(wfn), wfn, eig, Sv).real  # (nk, nbands, 3)

        elec_vel = elec_vel_1 - elec_vel_coor
        return elec_vel

    def _P_cell_prepare(self,):
        P_cell = []
        for i in range(3):
            P_tmp, _, _, _, _ = build_sparse_matrix(self.graph_data.species, self.graph_data.cell_shift, self.nao_max, self.graph_data.Pon[:,:,:,i], 
                                                    self.graph_data.Poff[:,:,:,i], self.graph_data.edge_index, Ham_type=self.Ham_type)
            P_cell.append(P_tmp)
        P_cell = np.stack(P_cell, axis=-1)
        if self.spinor_mode == 'nc':
            ncells, norbs_half, _, ndirs = P_cell.shape
            norbs = norbs_half * 2
            new_P_cell = np.zeros((ncells, norbs, norbs, ndirs), dtype=P_cell.dtype)
            new_P_cell[:, :norbs_half, :norbs_half, :] = P_cell
            new_P_cell[:, norbs_half:, norbs_half:, :] = P_cell
            P_cell = new_P_cell
        return P_cell

    def _get_reciprocal_lattice_vectors(self, n1:int=3, n2:int=3, n3:int=3, exclude_gamma: bool=False, ix:int=0, iy:int=0, iz:int=0):
        X, Y, Z = np.mgrid[-n1+ix:n1+ix+1, -n2+iy:n2+iy+1, -n3+iz:n3+iz+1]
        g_grid = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=-1)
        if exclude_gamma:
            g_grid = np.delete(g_grid, [int((g_grid.shape[0]+1)/2)-1], axis=0)
        g_vec = np.tensordot(g_grid, self.graph_data.rec_latt, axes=1) # (ngs, 3)
        return g_vec

    def _get_LR_correction_kq(self, q:np.ndarray, freq:np.ndarray, phvec_wap:np.ndarray, match_table:np.ndarray, enk_inside_table:np.ndarray=None,
                                    k:np.ndarray=None, wave_k:np.ndarray=None, wave_kpq:np.ndarray=None,
                                    return_tmp1:bool=False):
        """
        Calculate the dipole corrections for electron-phonon coupling matrix under specific k-q pair.

        Args:
            q (np.ndarray): The q point coordinate in Cartesian. # shape (3, )
            freq (np.ndarray): The frequencies of the phonon at q. # shape (nbranches, )
            phvec_wap (np.ndarray): The eigen vectors of the phonon at q with atomic phase. # shape (nbranches, natoms, 3)
            match_table (np.ndarray): The match table. # shape (nbands, nbands, nbranches)
            enk_inside_table (np.ndarray): Flags whether e_nk falls in the interest energy range. (nbands, )
            k (np.ndarray): The k point coordinate in Cartesian. # shape (3, )
            wave_k (np.ndarray): The eigen vectors of the electron at k. # shape (nbands, norbs)
            wave_kpq (np.ndarray): The eigen vectors of the electron at k+q. # shape (nbands, norbs)
            save_tmp1 (bool): If True, return also the tmp1_save.

        Returns:
            epc_corr (np.ndarray): The dipole corrections for electron-phonon coupling matrix. # shape (nbands, nbands, nbranches)
            tmp1_save (np.ndarray): The inner production of two electron eigen vectors. # shape (nbands, nbands, norbs, norbs)
        """
        nbands, _, nbranches = match_table.shape
        epc_corr = np.zeros((nbands, nbands, nbranches), dtype=complex)
        if return_tmp1:
            tmp1_save = np.zeros((nbands, nbands, self.norbs, self.norbs), dtype=complex)
        # Precompute factor for all branches: shape (nbranches, natoms_uc)
        factors_all = 1.0 / np.sqrt(2.0 * self.atomic_mass[None, :] * np.abs(freq[:, None]))
        if self.system_2D:
            # Short-circuit when neither dipole nor quadrupole is requested.
            if not self.apply_correction and not self.quadrupole_correction:
                if return_tmp1:
                    return epc_corr, tmp1_save
                return epc_corr

            if np.abs(q[2]) > QTcts.TENPM6:
                raise ValueError("In 2D LR correction, q_z must be zero.")

            # Use zero BECs in the pure-quadrupole branch instead of mutating self.BECs.
            BECs_use = self.BECs if self.apply_correction else np.zeros((self.natoms, 3, 3))

            # ---- q-only precomputation ----
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G_2D(q)              # (ngs,3), (ngs,)
            qG_norm = np.linalg.norm(qG_vec_cart, axis=1)                           # (ngs,)
            factor1 = QTcts.PI / self.area_2D * self.range_separation_f(qG_norm) / qG_norm  # (ngs,)

            inplane_num = 2.0j * np.einsum('gb, kab->gka', qG_vec_cart, BECs_use)   # (ngs, natoms, 3)
            if self.quadrupole_correction:
                inplane_num = inplane_num + (
                    np.einsum('gb, gc, kabc->gka', qG_vec_cart, qG_vec_cart, self.Q_tensor)
                    - qG_norm[:, None, None] ** 2 * self.Q_tensor[:, :, 2, 2][None, :, :]
                )
            inplane_den = self.dielectric_in_plane(qG_vec_cart)                     # (ngs,)
            inplane_coef = inplane_num / inplane_den[:, None, None]                 # (ngs, natoms, 3)

            # Out-of-plane term is only present in the full (apply+quad) branch (matches April semantics).
            use_outplane = self.apply_correction and self.quadrupole_correction
            if use_outplane:
                outplane_num = 2.0 * np.einsum('g, ka->gka', qG_norm ** 2, BECs_use[:, :, 2])
                outplane_den = self.dielectric_out_plane(qG_vec_cart)
                outplane_coef = outplane_num / outplane_den[:, None, None]          # (ngs, natoms, 3)

            atomic_phase = np.exp(-1.0j * np.einsum('gx, kx->gk', qG_vec_cart, self.graph_data.pos))  # (ngs, natoms)

            # ---- k-only precomputation ----
            phase = np.exp(QTcts.JTWOPI * (self.graph_data.nbr_shift_of_cell @ k))   # (ncells,)
            phase_S = np.tensordot(phase, self.graph_data.S_cell_full,       axes=([0], [0]))  # (norbs, norbs)
            if use_outplane:
                phase_z = np.tensordot(phase, self.graph_data.P_cell[:, :, :, 2], axes=([0], [0]))  # (norbs, norbs)

            # ---- branch-resolved (factor * phvec) ----
            ph_pref_phvec = factors_all[:, :, None] * phvec_wap                     # (nbranches, natoms, 3)

            # ---- (iband, jband) loop with branch vectorisation ----
            for iband in np.where(enk_inside_table)[0]:
                for jband in range(nbands):
                    if not match_table[iband, jband, :].any():
                        continue
                    tmp1 = np.conj(wave_kpq[jband])[:, None] * wave_k[iband][None, :]  # (norbs, norbs)
                    if return_tmp1:
                        tmp1_save[jband, iband] = tmp1
                    S_mat_Bloch = np.sum(tmp1 * phase_S)
                    V_key_part = inplane_coef * S_mat_Bloch                          # (ngs, natoms, 3)
                    if use_outplane:
                        z_mat_Bloch = np.sum(tmp1 * phase_z)
                        V_key_part = V_key_part + outplane_coef * z_mat_Bloch
                    # branch-vectorised g_LR contraction
                    g_all = np.einsum('g, vka, gk, gka -> v',
                                       factor1, ph_pref_phvec, atomic_phase, V_key_part)  # (nbranches,)
                    match_mask = match_table[iband, jband, :]
                    epc_corr[iband, jband, match_mask] = g_all[match_mask]
        else:
            # Precompute q-dependent ewald quantities once (instead of per band/branch)
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q)
            atomic_phase_G = np.exp(-1.0j * (self.graph_data.pos @ qG_vec_cart.T))  # (natoms, ngs)
            gauss_filter = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term  # (ngs,)
            # Precompute k-dependent phase once
            phase = np.exp(QTcts.JTWOPI * (self.graph_data.nbr_shift_of_cell @ k))  # (ncells,)
            # Precompute order=1 specific quantities if needed
            order = self.LRC_taylor_order
            if order == 1:
                mat_r = 1.0j * np.einsum('nija, ga->gnij', self.graph_data.P_cell, qG_vec_cart)  # (ngs, ncells, norbs, norbs)
                mat_exp_taylor = self.graph_data.S_cell_full[None, :, :] + mat_r  # (ngs, ncells, norbs, norbs)
            # Precompute per-branch numerator and temp3
            use_dipole = self.apply_correction
            use_quadrupole = self.quadrupole_correction
            # dipole contribution per branch: (nbranches, natoms, ngs)
            if use_dipole:
                dipole_all = 1.0j * np.einsum('gi, kij, bkj->bkg', qG_vec_cart, self.BECs, phvec_wap)
            else:
                dipole_all = np.zeros((nbranches, self.natoms, len(exp_inner_term)), dtype=complex)
            # quadrupole contribution per branch: (nbranches, natoms, ngs)
            if use_quadrupole:
                quad_all = 0.5 * np.einsum('gb, kabc, gc, mka->mkg', qG_vec_cart, self.Q_tensor, qG_vec_cart, phvec_wap)
            else:
                quad_all = np.zeros_like(dipole_all)
            # Combined: numerator * gauss * atomic_phase, shape (nbranches, natoms, ngs)
            temp3_all = (dipole_all + quad_all) * gauss_filter[None, None, :] * atomic_phase_G[None, :, :]
            # Prefactor: FOURPI for all cases (1j is already in dipole_all)
            # dipole-only original: JFOURPI * einsum(BECs) = FOURPI * 1j * einsum(BECs) = FOURPI * dipole_all
            prefactor = QTcts.FOURPI
            inv_vol = 1.0 / self.volume_uc
            # For order==0: precompute branch-dependent scalar (independent of bands)
            if order == 0:
                # einsum('kg, k->') = sum over atoms of (temp3 * factors), then sum over G
                branch_scalar = prefactor * inv_vol * np.array(
                    [(temp3_all[ib] * factors_all[ib, :, None]).sum() for ib in range(nbranches)])  # (nbranches,)
                # Precompute phase-weighted S_cell: sum_n phase[n]*S_cell[n] -> (norbs, norbs)
                phase_S = np.tensordot(phase, self.graph_data.S_cell_full, axes=([0], [0]))  # (norbs, norbs)
            else:  # order == 1
                # Precompute phase-weighted mat_exp_taylor: sum_n phase[n]*mat_exp[g,n] -> (ngs, norbs, norbs)
                phase_mat = np.tensordot(mat_exp_taylor, phase, axes=([1], [0]))  # (ngs, norbs, norbs)
            for iband in (np.where(enk_inside_table)[0]):
                for jband in range(nbands):
                    if not match_table[iband, jband, :].any():
                        continue
                    tmp1 = np.conj(wave_kpq[jband])[:, None] * wave_k[iband][None, :]  # (norbs, norbs) outer product
                    if return_tmp1:
                        tmp1_save[jband, iband] = tmp1
                    # Compute sum_r (depends on wavefunctions)
                    if order == 0:
                        sum_r = np.sum(tmp1 * phase_S)  # scalar: Tr(tmp1 . phase_S^T)
                        # Vectorized assignment for all matching branches at once
                        match_mask = match_table[iband, jband, :]
                        epc_corr[iband, jband, match_mask] = branch_scalar[match_mask] * sum_r
                    else:  # order == 1
                        sum_r = np.sum(tmp1[None, :, :] * phase_mat, axis=(1, 2))  # (ngs,)
                        for ibranch in range(nbranches):
                            if match_table[iband, jband, ibranch]:
                                temp4 = (temp3_all[ibranch] * sum_r[None, :]).sum(-1)  # (natoms,)
                                epc_corr[iband, jband, ibranch] = prefactor * (temp4 * factors_all[ibranch]).sum() * inv_vol
        
        if return_tmp1:
            return epc_corr, tmp1_save
        else:
            return epc_corr

    def _get_LR_correction_kq_batch(self, q, freq, phvec_wap,
                                    match_batch, enk_batch, k_batch,
                                    wave_k_batch, wave_kpq_batch):
        """Calculate LR EPC corrections for a group of k points.

        The q-dependent Ewald and dielectric terms are built once and the
        k-dependent Bloch phases and wavefunction contractions are evaluated
        for the whole tile.  ``LRC_taylor_order == 1`` deliberately keeps the
        established scalar implementation because its first-order position
        expansion has a different contraction and has not been validated for
        the batched path.
        """
        k_batch = np.ascontiguousarray(k_batch)
        match_batch = np.ascontiguousarray(match_batch, dtype=bool)
        enk_batch = np.ascontiguousarray(enk_batch, dtype=bool)
        wave_k_batch = np.ascontiguousarray(wave_k_batch)
        wave_kpq_batch = np.ascontiguousarray(wave_kpq_batch)
        B = len(k_batch)
        nbands = match_batch.shape[1]
        mbands = match_batch.shape[2]
        nbranches = len(freq)
        corr = np.zeros(
            (B, nbands, mbands, nbranches), dtype=np.complex128)
        if B == 0:
            return corr

        def scalar_fallback():
            for row in range(B):
                corr[row] = self._get_LR_correction_kq(
                    q, freq, phvec_wap, match_batch[row], enk_batch[row],
                    k_batch[row], wave_k_batch[row], wave_kpq_batch[row],
                    return_tmp1=False)
            return corr

        # Keep the Taylor-order-one route unchanged.  This also covers future
        # combinations that do not yet have a validated vector contraction.
        if getattr(self, "LRC_taylor_order", 0) == 1:
            return scalar_fallback()

        # There is no LR contribution in this configuration.  Returning here
        # avoids constructing Ewald tables for the common disabled case.
        if not self.apply_correction and not self.quadrupole_correction:
            return corr

        if self.system_2D:
            if np.any(np.abs(np.asarray(q)[2]) > QTcts.TENPM6):
                raise ValueError("In 2D LR correction, q_z must be zero.")

            # The scalar implementation uses zero Born charges for a pure
            # quadrupole correction.  Keep that semantic without mutating the
            # calculator state shared by other q tiles.
            BECs_use = (self.BECs if self.apply_correction else
                        np.zeros((self.natoms, 3, 3)))
            factors_all = 1.0 / np.sqrt(
                2.0 * self.atomic_mass[None, :] * np.abs(freq[:, None]))
            qG_vec_cart, _ = self._get_LRC_ewald_G_2D(q)
            qG_norm = np.linalg.norm(qG_vec_cart, axis=1)
            factor1 = (QTcts.PI / self.area_2D *
                       self.range_separation_f(qG_norm) / qG_norm)
            inplane_num = 2.0j * np.einsum(
                'gb,kab->gka', qG_vec_cart, BECs_use)
            if self.quadrupole_correction:
                inplane_num = inplane_num + (
                    np.einsum('gb,gc,kabc->gka', qG_vec_cart,
                              qG_vec_cart, self.Q_tensor)
                    - qG_norm[:, None, None] ** 2 *
                    self.Q_tensor[:, :, 2, 2][None, :, :])
            inplane_den = self.dielectric_in_plane(qG_vec_cart)
            inplane_coef = inplane_num / inplane_den[:, None, None]

            use_outplane = self.apply_correction and self.quadrupole_correction
            if use_outplane:
                outplane_num = 2.0 * np.einsum(
                    'g,ka->gka', qG_norm ** 2, BECs_use[:, :, 2])
                outplane_den = self.dielectric_out_plane(qG_vec_cart)
                outplane_coef = outplane_num / outplane_den[:, None, None]

            atomic_phase = np.exp(
                -1.0j * np.einsum(
                    'gx,kx->gk', qG_vec_cart, self.graph_data.pos))
            ph_pref_phvec = factors_all[:, :, None] * phvec_wap
            phase = np.exp(
                QTcts.JTWOPI *
                (self.graph_data.nbr_shift_of_cell @ k_batch.T)).T
            phase_S = np.einsum(
                'bn,nij->bij', phase, self.graph_data.S_cell_full)
            if use_outplane:
                phase_z = np.einsum(
                    'bn,nij->bij', phase,
                    self.graph_data.P_cell[:, :, :, 2])

            for iband in range(nbands):
                active = enk_batch[:, iband]
                if not np.any(active):
                    continue
                for jband in range(mbands):
                    pair_match = match_batch[:, iband, jband, :]
                    valid = pair_match & active[:, None]
                    if not np.any(valid):
                        continue
                    tmp1 = (
                        np.conj(wave_kpq_batch[:, jband, :,])[:, :, None] *
                        wave_k_batch[:, iband, :][:, None, :])
                    s_mat = np.einsum('bij,bij->b', tmp1, phase_S)
                    v_key = inplane_coef[None, :, :, :] * s_mat[:, None, None, None]
                    if use_outplane:
                        z_mat = np.einsum('bij,bij->b', tmp1, phase_z)
                        v_key = v_key + outplane_coef[None, :, :, :] * z_mat[:, None, None, None]
                    g_all = np.einsum(
                        'g,vka,gk,bgka->bv', factor1, ph_pref_phvec,
                        atomic_phase, v_key)
                    corr[:, iband, jband, :][valid] = g_all[valid]
            return corr

        # 3D order-zero correction.  Dipole-only, quadrupole-only, and the
        # combined dipole+quadrupole variants share the same contraction; the
        # selected numerator terms determine the physical branch.
        factors_all = 1.0 / np.sqrt(
            2.0 * self.atomic_mass[None, :] * np.abs(freq[:, None]))
        qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q)
        atomic_phase_G = np.exp(
            -1.0j * (self.graph_data.pos @ qG_vec_cart.T))
        gauss_filter = np.exp(
            -exp_inner_term / self.ewald_param) / exp_inner_term
        use_dipole = self.apply_correction
        use_quadrupole = self.quadrupole_correction
        if use_dipole:
            dipole_all = 1.0j * np.einsum(
                'gi,kij,bkj->bkg', qG_vec_cart, self.BECs, phvec_wap)
        else:
            dipole_all = np.zeros(
                (nbranches, self.natoms, len(exp_inner_term)), dtype=complex)
        if use_quadrupole:
            quad_all = 0.5 * np.einsum(
                'gb,kabc,gc,mka->mkg', qG_vec_cart, self.Q_tensor,
                qG_vec_cart, phvec_wap)
        else:
            quad_all = np.zeros_like(dipole_all)
        temp3_all = ((dipole_all + quad_all) *
                     gauss_filter[None, None, :] *
                     atomic_phase_G[None, :, :])
        branch_scalar = QTcts.FOURPI / self.volume_uc * np.array(
            [(temp3_all[ib] * factors_all[ib, :, None]).sum()
             for ib in range(nbranches)])
        phase = np.exp(
            QTcts.JTWOPI *
            (self.graph_data.nbr_shift_of_cell @ k_batch.T)).T
        phase_S = np.einsum(
            'bn,nij->bij', phase, self.graph_data.S_cell_full)
        for iband in range(nbands):
            active = enk_batch[:, iband]
            if not np.any(active):
                continue
            for jband in range(mbands):
                pair_match = match_batch[:, iband, jband, :]
                valid = pair_match & active[:, None]
                if not np.any(valid):
                    continue
                tmp1 = (
                    np.conj(wave_kpq_batch[:, jband, :,])[:, :, None] *
                    wave_k_batch[:, iband, :][:, None, :])
                s_mat = np.einsum('bij,bij->b', tmp1, phase_S)
                g_all = s_mat[:, None] * branch_scalar[None, :]
                corr[:, iband, jband, :][valid] = g_all[valid]
        return corr

    def _dipole_correction_3D_taylor(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phvec_wap:np.ndarray, order:int=0):
        """
        Args:
            wave_coe_tp (np.array): (norbs, norbs)
            k_vec (np.array): (3,)
            q_vec (np.array): (3)
            freq (np.array): (nbranches,)
            ph_prefac (np.array): (natoms,)
            phvec_wap (np.array): (natoms, 3)

        Returns:
            ret: scalar
        """
        if order == 0:
            # qG_vec_cart in bohr^{-1}
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q_vec)
            phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1)) # shape: (ncells, )
            sum_r = np.einsum('n, ij, nij->', phase, wave_coe_tp, self.graph_data.S_cell_full)
            atomic_phase = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))  # shape: (natoms, ngs)
            temp1 = np.einsum('gi, kij, kj->kg', qG_vec_cart, self.BECs, phvec_wap) # shape: (natoms, ngs)
            temp2 = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term
            temp3 = temp1 * temp2[None,:] * atomic_phase # shape: (natoms, ngs)
            ret = QTcts.JFOURPI * np.einsum('kg, k->', temp3, ph_prefac) * sum_r / self.volume_uc
        elif order == 1:
            # qG_vec_cart in bohr^{-1}
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q_vec)
            phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1)) # shape: (ncells, )
            mat_r = 1.0j*np.einsum('nija, ga->gnij', self.graph_data.P_cell, qG_vec_cart)   # shape: (ngs, ncells, norbs, norbs)
            # mat_exp: shape (ngs, ncells, norbs, norbs)
            mat_exp_taylor = self.graph_data.S_cell_full[None, :, :] + mat_r
            sum_r = np.einsum('n, ij, gnij->g', phase, wave_coe_tp, mat_exp_taylor) # shape: (ngs, )
            atomic_phase = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))  # shape: (natoms, ngs)
            temp1 = np.einsum('gi, kij, kj->kg', qG_vec_cart, self.BECs, phvec_wap) # shape: (natoms, ngs)
            temp2 = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term
            temp3 = temp1 * temp2[None,:] * atomic_phase # shape: (natoms, ngs)
            temp4 = (temp3*sum_r[None,:]).sum(-1) # shape: (natoms, )
            ret = QTcts.JFOURPI * (temp4 * ph_prefac).sum() / self.volume_uc
        return ret

    def _dipole_and_quadrupole_correction_3D_taylor(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phvec_wap:np.ndarray, order:int=0):
        """
        ref from EPW2023 formula (72) and  DOI: 10.1103/PhysRevB.102.125203  formula (8).  
        by BJR.  
        Note that we omitted the V^{Hxc} term accordding to Ponce EPW2023.  
        
        Args:
            wave_coe_tp (np.array): (norbs, norbs)
            k_vec (np.array): (3,)
            q_vec (np.array): (3)
            freq (np.array): (nbranches,)
            phvec_wap (np.array): (natoms, 3)

        Returns:
            result: scalar
        """
        if order == 0:
            # qG_vec_cart in bohr^{-1}
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q_vec)
            phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1))
            sum_r = np.einsum('n, ij, nij->', phase, wave_coe_tp, self.graph_data.S_cell_full)    # shape: (0,)  # matrix element <Bloch | exp |Bloch >
            atomic_phase = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))  # shape: (natoms, ngs)
            # iZ   1/2 Q
            numerator_dipole = 1.0j * np.einsum("gi, kij, kj->kg", qG_vec_cart, self.BECs, phvec_wap)  # shape: (natoms, ngs)  
            numerator_quadrupole = 0.5 * np.einsum("gb, kabc, gc, ka->kg", qG_vec_cart, self.Q_tensor, qG_vec_cart, phvec_wap)  # shape: (natoms, ngs)
            denominator_and_filter = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term  # shape: (ngs, )  
            tmp3 = (numerator_dipole + numerator_quadrupole) * denominator_and_filter[None,:] * atomic_phase  # shape: (natoms, ngs)
            result = QTcts.FOURPI * np.einsum("kg, k->", tmp3, ph_prefac)  * sum_r / self.volume_uc  # shape: (0,)

        if order == 1:
            # qG_vec_cart in bohr^{-1}
            qG_vec_cart, exp_inner_term = self._get_LRC_ewald_G(q_vec)
            phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.nbr_shift_of_cell*k_vec[None,:], axis=-1)) # shape: (ncells, )
            mat_r = 1.0j*np.einsum('nija, ga->gnij', self.graph_data.P_cell, qG_vec_cart)   # shape: (ngs, ncells, norbs, norbs)
            mat_exp_taylor = self.graph_data.S_cell_full[None, :, :] + mat_r  # shape: (ngs, ncells, norbs, norbs)
            sum_r = np.einsum("n, ij, gnij->g", phase, wave_coe_tp, mat_exp_taylor)  # shape: (ngs, )  # matrix element <Bloch | exp |Bloch >
            atomic_phase = np.exp(-1.0j*np.einsum('ga, ka->kg', qG_vec_cart, self.graph_data.pos))  # shape: (natoms, ngs)
            # iZ   1/2 Q
            numerator_dipole = 1.0j * np.einsum("gi, kij, kj->kg", qG_vec_cart, self.BECs, phvec_wap)  # shape: (natoms, ngs)
            numerator_quadrupole = 0.5 * np.einsum("gb, kabc, gc, ka->kg", qG_vec_cart, self.Q_tensor, qG_vec_cart, phvec_wap)  # shape: (natoms, ngs)
            denominator_and_filter = np.exp(-exp_inner_term / self.ewald_param) / exp_inner_term  # shape: (ngs, )  
            temp3 = (numerator_dipole + numerator_quadrupole) * denominator_and_filter[None,:] * atomic_phase  # shape: (natoms, ngs)
            temp4 = (temp3 * sum_r[None,:]).sum(-1)  # shape: (natoms, )
            result = QTcts.FOURPI * ( temp4 * ph_prefac).sum()  / self.volume_uc
        
        return result

    def _only_quadrupole_correction_3D_taylor(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phvec_wap:np.ndarray, order:int=0):
        """
        for nonpolar materials, Born effective charge is zero. But there is still a quadrupole correction.
        to be done.
        or we can just set self.BECs to zero with certain shapes.  
        
        Args:
            wave_coe_tp (np.array): (norbs, norbs)
            k_vec (np.array): (3,)
            q_vec (np.array): (3)
            freq (np.array): (nbranches,)
            phvec_wap (np.array): (natoms, 3)

        Returns:
            result: scalar
        """

        pass
        # BJR:  to be checked with Si ......

        self.BECs = np.zeros((self.natoms, 3, 3))
        return self._dipole_and_quadrupole_correction_3D_taylor(wave_coe_tp, k_vec, q_vec, ph_prefac, phvec_wap, order=order)

    def _get_LR_correction_3D(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phvec_wap:np.ndarray, order:int=0):

        if self.apply_correction and (not self.quadrupole_correction):
            epc_corr = self._dipole_correction_3D_taylor(wave_coe_tp, k_vec, q_vec, ph_prefac, phvec_wap, order=order)
        elif self.apply_correction and self.quadrupole_correction:
            epc_corr = self._dipole_and_quadrupole_correction_3D_taylor(wave_coe_tp, k_vec, q_vec, ph_prefac, phvec_wap, order=order)
        elif (not self.apply_correction) and self.quadrupole_correction:
            epc_corr = self._only_quadrupole_correction_3D_taylor(wave_coe_tp, k_vec, q_vec, ph_prefac, phvec_wap, order=order)
        else:
            epc_corr = 0.0
        return epc_corr

    def _get_LRC_ewald_G(self, q_vec_cart:np.ndarray):
        # move q_vec to 1BZ
        q_vec = self._car2frac(q_vec_cart)
        q_vec = q_vec - np.floor(q_vec)
        # Transform q_vec to cartesian coordinates
        q_vec_cart = self._frac2car(q_vec)[0]
        if np.linalg.norm(q_vec_cart) < QTcts.TENPM6:
            G_vec_cart = self._get_reciprocal_lattice_vectors(5,5,5, True) # (ngs, 3)
        else:
            G_vec_cart = self._get_reciprocal_lattice_vectors(5,5,5, False) # (ngs, 3)
        qG_vec_cart = q_vec_cart[None, :] + G_vec_cart  # (ngs, 3)
        qG_vec_cart = qG_vec_cart * QTcts.TWOPI
        exp_inner_term = np.sum((qG_vec_cart @ self.DL) * qG_vec_cart, axis=-1)  # (ngs, )
        tmp_mask = exp_inner_term < (QTcts.EWALD_LN_CUTOFF * self.ewald_param)
        exp_inner_term = exp_inner_term[tmp_mask]  # (ngs_in, )
        qG_vec_cart = qG_vec_cart[tmp_mask] # (ngs_in, 3)
        return qG_vec_cart, exp_inner_term

    def _get_LRC_ewald_G_2D(self, q_vec_cart:np.ndarray):
        """q_vec_cart: (3,)  no 2pi factor."""
        q_vec = self._car2frac(q_vec_cart)
        q_vec = q_vec - np.floor(q_vec)
        q_vec_cart = self._frac2car(q_vec)[0]
        index_G_min_qplusG = self.find_min_qG(q_vec_cart)
        ix, iy, iz = index_G_min_qplusG

        Gsquare_max = QTcts.EWALD_LN_CUTOFF * self.ewald_param

        rec_latt_square = np.sum(self.graph_data.rec_latt ** 2, axis=1)
        nGx = 0 if self.q_size[0] == 1 else int(np.sqrt(Gsquare_max / (rec_latt_square[0] * QTcts.TWOPI_SQUARE))) + 1
        nGy = 0 if self.q_size[1] == 1 else int(np.sqrt(Gsquare_max / (rec_latt_square[1] * QTcts.TWOPI_SQUARE))) + 1
        nGz = 0 if self.q_size[2] == 1 else int(np.sqrt(Gsquare_max / (rec_latt_square[2] * QTcts.TWOPI_SQUARE))) + 1

        if np.linalg.norm(q_vec_cart) < QTcts.TENPM6:
            G_vec_cart = self._get_reciprocal_lattice_vectors(nGx, nGy, nGz, True,  ix=ix, iy=iy, iz=iz)
        else:
            G_vec_cart = self._get_reciprocal_lattice_vectors(nGx, nGy, nGz, False, ix=ix, iy=iy, iz=iz)

        qG_vec_cart = q_vec_cart[None, :] + G_vec_cart
        qG_vec_cart *= QTcts.TWOPI
        exp_inner_term = np.einsum('gi, ij, gj->g', qG_vec_cart, self.DL, qG_vec_cart)
        tmp_mask = exp_inner_term < (QTcts.EWALD_LN_CUTOFF * self.ewald_param)
        exp_inner_term = exp_inner_term[tmp_mask]
        qG_vec_cart = qG_vec_cart[tmp_mask]
        return qG_vec_cart, exp_inner_term

    def find_min_qG(self, q_vec_cart):
        """find G that minimises |q+G|.  q_vec_cart: (3,) no 2pi factor."""
        q_vec_frac = self._car2frac(q_vec_cart)
        q_vec_frac_wrapped = q_vec_frac - np.floor(q_vec_frac)
        q_cart_wrapped = self._frac2car(q_vec_frac_wrapped)[0]

        q_plus_G = q_cart_wrapped[None, :] + self.G_vectors_333
        norm_squares = np.einsum('gi, gi->g', q_plus_G, q_plus_G)
        min_index = np.argmin(norm_squares)

        index_G_min_qplusG = self.offsets_333[min_index].tolist()
        index_G_min_qplusG = np.array(index_G_min_qplusG, dtype=int)
        index_G_min_qplusG -= np.floor(q_vec_frac[0]).astype(int)
        return index_G_min_qplusG

    def range_separation_f(self, K):
        return 1 - np.tanh(K * self.range_separation_length_L / 2)

    def dielectric_in_plane(self, q_vec_cart:np.ndarray):
        """q_vec_cart already has the 2pi factor, in bohr^{-1}. Accepts (3,) or (nqs, 3)."""
        q_vec_cart = np.atleast_2d(q_vec_cart)
        q_norm = np.linalg.norm(q_vec_cart, axis=1)
        q_xy = q_vec_cart[:, 0:2]
        qaq = np.einsum('qi, ij, qj->q', q_xy, self.macro_polarizability_inplane, q_xy)
        result = 1 + QTcts.TWOPI * self.range_separation_f(q_norm) / q_norm * qaq
        if len(result.shape) >= 1:
            return result
        else:
            return result[0]

    def dielectric_out_plane(self, q_vec_cart:np.ndarray):
        """q_vec_cart already has the 2pi factor, in bohr^{-1}. Accepts (3,) or (nqs, 3)."""
        q_vec_cart = np.atleast_2d(q_vec_cart)
        q_norm = np.linalg.norm(q_vec_cart, axis=1)
        result = 1 - QTcts.TWOPI * q_norm * self.range_separation_f(q_norm) * self.macro_polarizability_outofplane
        if len(result.shape) >= 1:
            return result
        else:
            return result[0]

    def _LR_correction_2D(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phvec:np.ndarray, BECs:np.ndarray=None):
        """
        Full 2D long-range correction (dipole + quadrupole).  Ponce 2023 (PRB 107.155424), V^{Hxc} omitted.

        ph_prefac: sqrt(hbar / 2 m_kappa w_qv),  shape (natoms,)
        phvec:     phonon eigenvector,           shape (natoms, 3)
        BECs:      Born effective charges to use; None means self.BECs.
                   Pass a zero array for the pure-quadrupole branch instead of overwriting self.BECs.
        Returns scalar g_LR.
        """
        if np.abs(q_vec[2]) > QTcts.TENPM6:
            raise ValueError("In 2D LR correction, q_z must be zero.")

        BECs = self.BECs if BECs is None else BECs

        qplusG_vec_cart, exp_inner_term = self._get_LRC_ewald_G_2D(q_vec)
        qplusG_norm = np.linalg.norm(qplusG_vec_cart, axis=1)
        factor1 = QTcts.PI / self.area_2D * self.range_separation_f(qplusG_norm) / qplusG_norm

        inplane_numerator = (
            2.0j * np.einsum('gb, kab->gka', qplusG_vec_cart, BECs)
            + np.einsum('gb, gc, kabc->gka', qplusG_vec_cart, qplusG_vec_cart, self.Q_tensor)
            - qplusG_norm[:, None, None] ** 2 * self.Q_tensor[:, :, 2, 2][None, :, :]
        )
        inplane_denominator = self.dielectric_in_plane(qplusG_vec_cart)
        inplane_coef = inplane_numerator / inplane_denominator[:, None, None]

        outplane_numerator = 2.0 * np.einsum('g, ka->gka', qplusG_norm ** 2, BECs[:, :, 2])
        outplane_denominator = self.dielectric_out_plane(qplusG_vec_cart)
        outplane_coef = outplane_numerator / outplane_denominator[:, None, None]

        atomic_phase = np.exp(-1.0j * np.einsum('gx, kx->gk', qplusG_vec_cart, self.graph_data.pos))

        phase_neighbour = np.exp(QTcts.JTWOPI * np.sum(self.graph_data.nbr_shift_of_cell * k_vec[None, :], axis=-1))
        S_mat_Bloch = np.einsum('n, ij, nij->', phase_neighbour, wave_coe_tp, self.graph_data.S_cell_full)
        z_mat_Bloch = np.einsum('n, ij, nij->', phase_neighbour, wave_coe_tp, self.graph_data.P_cell[:, :, :, 2])
        V_key_part_Bloch = inplane_coef * S_mat_Bloch + outplane_coef * z_mat_Bloch
        g_LR_of_G = np.einsum('g, k, kx, gk, gkx->g', factor1, ph_prefac, phvec, atomic_phase, V_key_part_Bloch)
        return np.einsum('g->', g_LR_of_G)

    def _LR_correction_2D_only_dipole(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phvec:np.ndarray):
        """Dipole-only 2D long-range correction.  Ponce 2023, V^{Hxc} omitted."""
        if np.abs(q_vec[2]) > QTcts.TENPM6:
            raise ValueError("In 2D LR correction, q_z must be zero.")

        qplusG_vec_cart, exp_inner_term = self._get_LRC_ewald_G_2D(q_vec)
        qplusG_norm = np.linalg.norm(qplusG_vec_cart, axis=1)
        factor1 = QTcts.PI / self.area_2D * self.range_separation_f(qplusG_norm) / qplusG_norm
        inplane_numerator = 2.0j * np.einsum('gb, kab->gka', qplusG_vec_cart, self.BECs)
        inplane_denominator = self.dielectric_in_plane(qplusG_vec_cart)
        inplane_coef = inplane_numerator / inplane_denominator[:, None, None]

        atomic_phase = np.exp(-1.0j * np.einsum('gx, kx->gk', qplusG_vec_cart, self.graph_data.pos))

        phase_neighbour = np.exp(QTcts.JTWOPI * np.sum(self.graph_data.nbr_shift_of_cell * k_vec[None, :], axis=-1))
        S_mat_Bloch = np.einsum('n, ij, nij->', phase_neighbour, wave_coe_tp, self.graph_data.S_cell_full)
        V_key_part_Bloch = inplane_coef * S_mat_Bloch
        g_LR_of_G = np.einsum('g, k, kx, gk, gkx->g', factor1, ph_prefac, phvec, atomic_phase, V_key_part_Bloch)
        return np.einsum('g->', g_LR_of_G)

    def _get_LR_correction_2D(self, wave_coe_tp:np.ndarray, k_vec:np.ndarray, q_vec:np.ndarray, ph_prefac:np.ndarray, phvec:np.ndarray):
        if self.apply_correction and (not self.quadrupole_correction):
            return self._LR_correction_2D_only_dipole(wave_coe_tp, k_vec, q_vec, ph_prefac, phvec)
        elif self.apply_correction and self.quadrupole_correction:
            return self._LR_correction_2D(wave_coe_tp, k_vec, q_vec, ph_prefac, phvec)
        elif (not self.apply_correction) and self.quadrupole_correction:
            BECs_zero = np.zeros((self.natoms, 3, 3))
            return self._LR_correction_2D(wave_coe_tp, k_vec, q_vec, ph_prefac, phvec, BECs=BECs_zero)
        else:
            return 0.0
