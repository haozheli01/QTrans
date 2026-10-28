"""EPC pairing spectra and their spectral moments."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.errors import InputParameterError
from qtrans.core.logging import log
from qtrans.core.logging import time_logger
from qtrans.extensions import eliashberg_spectrum_cal_EPC_helper_sparse
from qtrans.extensions import eliashberg_spectrum_cal_EPC_helper_sparse_batch
from qtrans.extensions import prepare_sparse_grad_cache
from qtrans.extensions import NativeExtensionUnavailable
from qtrans.extensions import eliashberg_spectrum_cal_EPC_helper_sparse_soc
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import gather_to_rank_root
from qtrans.parallel.mpi import get_q_rank_indices
from qtrans.parallel.mpi import bcast_array_by_rank_chunks as _bcast_array_by_rank_chunks
from qtrans.parallel.mpi import print_peak_memeory
from qtrans.core.batch import get_k_batch_size
import numpy as np
import os


def _superconductivity_omega_grid(omega_range, omega_step):
    """Build the historical positive-frequency grid, with explicit validation."""
    bounds = np.asarray(omega_range, dtype=float)
    if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] < 0.0 or bounds[1] <= bounds[0]):
        raise InputParameterError(
            5242, 'superconductivity', 'omega_range',
            "omega_range must be finite [min, max] with 0 <= min < max."
        )
    if not np.isfinite(omega_step) or omega_step <= 0.0:
        raise InputParameterError(
            5243, 'superconductivity', 'omega_step',
            "omega_step must be finite and positive."
        )
    start = bounds[0] + omega_step if bounds[0] < QTcts.TENPM10 else bounds[0]
    omegas = np.arange(start, bounds[1] + omega_step / 10, omega_step)
    if omegas.size == 0:
        raise InputParameterError(
            5242, 'superconductivity', 'omega_range',
            "omega_range and omega_step produce an empty frequency grid."
        )
    return omegas


def _sparse_epc_q_none_batched(calculator, q, freq, phon_vec_q,
                               k_grid_useful, enks_useful, wave_ks_useful,
                               weights_k_useful, phase_k_all,
                               kpq_energy_table=None, grid2ir_idx=None,
                               batch_size=1024, grad_cache=None):
    """Process one non-spin q slice in bounded k batches.

    If ``grad_cache`` is a prepared ``SparseGradCache`` capsule, one native
    batch call handles all valid k points in each block.  The scalar helper is
    retained as a numerical fallback for an older native library (or for a
    path that needs a long-range correction).  If an IBZ energy table is
    supplied, it is used before calculating k+q wavefunctions; this is the
    commensurate-grid fast path.
    """
    nbranches = len(freq)
    norbs = int(getattr(calculator, 'norbs', 0))
    wave_bytes_per_k = max(1, norbs * norbs * np.dtype(np.complex128).itemsize)
    # Keep the requested 1024 upper bound, but avoid allocating multi-GiB
    # temporary eigenvector matrices with the current full-matrix solver.
    effective_batch = min(batch_size, get_k_batch_size(calculator, max_batch=batch_size))
    epc_q = np.zeros(nbranches, dtype=float)
    atomic_phase = np.exp(QTcts.JTWOPI * (calculator.graph_data.pos @ q))
    phvec_wap = np.ascontiguousarray(atomic_phase[None, :, None] * phon_vec_q)
    valid_branches = np.where(freq >= calculator.phonon_frequency_cutoff)[0]
    phase_q = np.exp(QTcts.JTWOPI * (calculator.nbr_shift_of_cell_sc @ q))
    for start in range(0, len(k_grid_useful), effective_batch):
        stop = min(start + effective_batch, len(k_grid_useful))
        k_block = k_grid_useful[start:stop]
        if kpq_energy_table is not None and grid2ir_idx is not None:
            bz = calculator._get_kpq_BZ_index(k_block + q)
            eig_block = np.ascontiguousarray(kpq_energy_table[grid2ir_idx[bz]])
            ematch_block = np.abs(eig_block - calculator.efermi) < calculator.e_thr
            valid = np.flatnonzero(np.any(ematch_block, axis=1))
            if valid.size == 0:
                continue
            solve_indices = valid
            eig_kpq = eig_block[solve_indices]
        else:
            kpqs = np.ascontiguousarray(k_block + q)
            eig_raw, wave_raw = calculator._elec_cal_torch(kpqs)
            eig_block = np.ascontiguousarray(eig_raw[calculator.bands_indices].T)
            ematch_block = np.abs(eig_block - calculator.efermi) < calculator.e_thr
            valid = np.flatnonzero(np.any(ematch_block, axis=1))
            if valid.size == 0:
                continue
            solve_indices = valid
            eig_kpq = eig_block[solve_indices]
            wave_raw = wave_raw[solve_indices]
        if kpq_energy_table is not None and grid2ir_idx is not None:
            kpqs = np.ascontiguousarray(k_block[solve_indices] + q)
            eig_raw, wave_raw = calculator._elec_cal_torch(kpqs)
        wave_kpq = np.ascontiguousarray(wave_raw[:, calculator.bands_indices, :])

        # The native batch kernel consumes a leading k axis.  Build all
        # per-k arrays once for this block, then accumulate its (B,branch)
        # output.  Keep the scalar loop below as a strict fallback.
        batch_used = False
        if grad_cache is not None:
            phase_k_batch = np.ascontiguousarray(
                phase_k_all[:, start:stop].T[solve_indices])
            phase_kpq_batch = np.ascontiguousarray(
                (phase_k_all[:, start:stop] * phase_q[:, None]).T[solve_indices])
            wave_k_batch = np.ascontiguousarray(wave_ks_useful[start:stop][solve_indices])
            eig_k_batch = np.ascontiguousarray(
                enks_useful[:, start:stop].T[solve_indices])
            enk_batch = np.ascontiguousarray(
                np.abs(eig_k_batch - calculator.efermi) < calculator.e_thr)
            ematch_batch = np.ascontiguousarray(ematch_block[solve_indices])
            eig_kpq_batch = np.ascontiguousarray(eig_kpq)
            weights_batch = np.ascontiguousarray(weights_k_useful[start:stop][solve_indices])
            epc_batch = np.zeros((len(solve_indices), nbranches), dtype=np.float64, order='C')
            try:
                eliashberg_spectrum_cal_EPC_helper_sparse_batch(
                    np.ascontiguousarray(freq), np.ascontiguousarray(calculator.atomic_mass),
                    phvec_wap, phase_k_batch, phase_kpq_batch, grad_cache,
                    wave_k_batch, wave_kpq, enk_batch, ematch_batch,
                    eig_k_batch, eig_kpq_batch, weights_batch, epc_batch,
                    calculator.phonon_frequency_cutoff, calculator.inv_smeark,
                    calculator.efermi, gauss_type=calculator.gauss_type)
                epc_q += np.sum(epc_batch, axis=0)
                batch_used = True
            except NativeExtensionUnavailable:
                # A pre-existing installation may not contain the new symbol;
                # retain the scalar implementation without changing results.
                batch_used = False
                grad_cache = None

        if batch_used:
            continue

        phase_kpq_all = phase_k_all[:, start:stop][:, solve_indices] * phase_q[:, None]
        for local_solve, local_idx in enumerate(solve_indices):
            global_idx = start + int(local_idx)
            inside = np.asarray(np.abs(enks_useful[:, global_idx] - calculator.efermi)
                                < calculator.e_thr, dtype=bool)
            if not inside.any():
                continue
            eig_k = enks_useful[:, global_idx]
            wave_k = wave_ks_useful[global_idx]
            phase_k = phase_k_all[:, global_idx]
            phase_kpq = phase_kpq_all[:, local_solve]
            for ibnd in np.where(inside)[0]:
                delta_nk = w0gauss((eig_k[ibnd] - calculator.efermi) * calculator.inv_smeark, calculator.gauss_type) * calculator.inv_smeark
                for jbnd in np.where(np.abs(eig_kpq[local_solve] - calculator.efermi) < calculator.e_thr)[0]:
                    delta_mkq = w0gauss((eig_kpq[local_solve, jbnd] - calculator.efermi) * calculator.inv_smeark, calculator.gauss_type) * calculator.inv_smeark
                    tmp1 = np.outer(np.conj(wave_kpq[local_solve, jbnd]), wave_k[ibnd])
                    weight_factor = delta_nk * delta_mkq * weights_k_useful[global_idx]
                    for ibranch in valid_branches:
                        epc = calculator._epc_cal_element(
                            k_grid_useful[global_idx], q, freq[ibranch],
                            phvec_wap[ibranch], tmp1, calculator.grad_mat,
                            phase_k, phase_kpq, dipole_correction=False)
                        epc_q[ibranch] += (epc * np.conj(epc)).real * weight_factor
    return epc_q


class _PairingSpectrumMixin:
    def eliashberg_spectrum_cal(self):
        """Calculate the superconducting EPC spectrum on the configured grids."""
        if self.read_sparse_grad_mat:
            return self._eliashberg_spectrum_cal_epc_sparse()
        return self._eliashberg_spectrum_cal_epc_dense()

    def _eliashberg_spectrum_cal_epc_dense(self):
        """
        Calculate the Eliashberg spectral function alpha^{2} F(omega) on dense k/q grids.

        This method computes the electron-phonon coupling (EPC) spectral function by
        iterating over q-points (parallelized via MPI) and k-points, evaluating the
        EPC matrix elements |g_{mn,v}(k, q)|^2 weighted by electronic delta functions
        at the Fermi level, and broadening the phonon frequencies onto a uniform omega grid.

        The function uses the following class attributes as input:
            self.k_size (list or np.ndarray): k-grid dimensions, shape (3,).
            self.q_size (list or np.ndarray): q-grid dimensions, shape (3,).
            self.bands_indices (np.ndarray): Indices of electronic bands to include.
            self.omega_range (list or tuple): [omega_min, omega_max] in Hartree.
            self.omega_step (float): Step size for the omega grid in Hartree.
            self.smeark (float): Smearing width for electronic delta functions.
            self.inv_smeark (float): Inverse of smeark.
            self.inv_smearq (float): Inverse of phonon smearing width.
            self.e_thr (float): Energy threshold for Fermi window filtering.
            self.phonon_frequency_cutoff (float): Minimum phonon frequency to include.
            self.spinor_mode (str): 'nc' (non-collinear), 'collinear', or other (non-spin-polarized).
            self.doping_charge (float): Doping charge for adjusting electron count.
            self.fermi_shift (float): Manual shift applied to the Fermi energy.
            self.gauss_type: Type of smearing function.
            self.graph_data: Object containing lattice info (latt, pos, num_electrons).
            self.grad_mat: Gradient matrix for EPC calculation.
            self.nbr_shift_of_cell_sc: Neighbor cell shift vectors for phase factors.
            self.comm: MPI communicator (or None for serial).
            self.rank (int): MPI rank of current process.
            self.rank_size (int): Total number of MPI processes.

        Returns:
            omegas_list (np.ndarray): Frequency grid, shape (nomegas,), in Hartree.
            a2f (np.ndarray): Eliashberg spectral function.
                - If spinor_mode == 'collinear': shape (2, nomegas), where index 0/1 = spin up/down.
                - Otherwise: shape (nomegas,).
        """

        # get the number of included electrons
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
            log(f"Number of electrons included in the calculation: {num_electron_include}", rank=self.rank)

        # generate the meshes for kpts and qpts
        k_grid, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
        q_grid = self._get_monkhorst_pack(self.q_size, return_frac=True)

        # initial the omega
        omegas_list = _superconductivity_omega_grid(self.omega_range, self.omega_step)

        nks_fbz = self.k_size[0] * self.k_size[1] * self.k_size[2]
        nks_ibz = len(k_grid)
        nqs = len(q_grid)

        self.weight_q = 1.0 / nqs

        k_grid_all, enks_local, wave_ks_local, q_grid_local, weights_q_local, freq_grid, phon_vecs= \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, None)

        nomegas = len(omegas_list)
        nbands = len(self.bands_indices)
        nqs_local = len(q_grid_local)
        nks_local = len(k_grid_all[self.rank])
        nbranches = self._get_nbranches()

        if self.spinor_mode == 'nc':
            nbands_col = nbands
        elif self.spinor_mode == 'collinear':
            nbands_col = nbands * 2
        else:
            nbands_col = nbands
            self.weight_k *= 2.0

        enks_local = enks_local.reshape(nbands_col, nks_local)
        enks_all = [enks_local] if self.comm is None else self.comm.allgather(enks_local)
        wave_ks_all = [wave_ks_local] if self.comm is None else self.comm.allgather(wave_ks_local)
        del enks_local, wave_ks_local

        enks = np.concatenate(enks_all, axis=-1)    # (nbands, nks) or (2*nbands, nks) for collinear
        wave_ks = np.concatenate(wave_ks_all, axis=-3) # (nks, nbands, norbs) or (2, nks, nbands, norbs) for collinear
        self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
        is_metal = self._get_is_metal(enks, self.efermi)
        if not is_metal:
            raise RuntimeError(f"Error(7001): Superconductivity can only be calculated for metallic systems.")
        dos_ef = self._get_ef_dos(enks, self.efermi, self.inv_smeark, self.gauss_type)
        if self.spinor_mode == 'collinear':
            log("Fermi energy = {:10.6f} eV\nDOS of Fermi level (sum of two spin channels) = {:10.6f}".format(self.efermi * QTcts.HARTREEtoEV, dos_ef / QTcts.HARTREEtoEV), rank=self.rank)
        else:
            # N(Ef) in lambda formula is the DOS per spin
            dos_ef = dos_ef / 2.0
            log("Fermi energy = {:10.6f} eV\nDOS of Fermi level (spin factor not included) = {:10.6f}".format(self.efermi * QTcts.HARTREEtoEV, dos_ef / QTcts.HARTREEtoEV), rank=self.rank)
        # resplit the k points
        mask = np.abs(enks - self.efermi) < self.e_thr
        _, kpt_indices_useful = np.where(mask)
        kpt_indices_useful = np.unique(kpt_indices_useful) # IBZ k-points really contributing
        nks_useful = kpt_indices_useful.shape[0] # total number of useful kpoints
        k_grid_useful = k_grid[kpt_indices_useful]                       # (nks_useful, 3)
        enks_useful = enks[:, kpt_indices_useful]                # (nbands, nks_useful) or (2*nbands, nks_useful) for collinear
        if self.spinor_mode == 'collinear':
            enks_useful = enks_useful.reshape(2, nbands, nks_useful) # (2, nbands, nks_useful)
            wave_ks_useful = wave_ks[:, kpt_indices_useful]    # (2, nks_useful, nbands, norbs)
        else:
            wave_ks_useful = wave_ks[kpt_indices_useful]    # (nks_useful, nbands, norbs)
        weights_k_useful = self.weight_k[kpt_indices_useful]             # (nks_useful,)
        self._print_mesh_mpi_check(nks_fbz, nks_ibz, nks_useful, nqs)

        if self.spinor_mode == 'collinear':
            epc_strengths = np.zeros((2, nqs_local, nbranches)) #[up, dn]
        else:
            epc_strengths = np.zeros((nqs_local, nbranches))

        enk_match_tables = np.abs(enks_useful - self.efermi) < self.e_thr  # (nbands, nks_useful_local) or (2, nbands, nks_useful_local)
        phase_k_all = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ k_grid_useful.T))  # (ncells, nks_useful_local)
        delta_nk_all = w0gauss((enks_useful - self.efermi) * self.inv_smeark, self.gauss_type) * self.inv_smeark

        if self.rank == 0:
            logger = time_logger(nqs_local, 'eliashberg_spectrum_cal_epc_dense')

        # calculate electron-phonon coupling strength
        for iq, q in enumerate(q_grid_local):
            # phonon spectrum
            freq = freq_grid[iq]
            eigen_vec_phon = phon_vecs[iq]
            # vectorized atomic_phase: matmul instead of element-wise multiply + sum
            atomic_phase = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q)) # shape (natoms, )
            phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
            # pre-compute valid branches mask (does not depend on k or bands)
            valid_branches = freq >= self.phonon_frequency_cutoff  # shape (nbranches,)
            valid_branch_indices = np.where(valid_branches)[0]
            if len(valid_branch_indices) == 0:
                if self.rank == 0:
                    logger.step(iq+1)
                continue
            for ik, k in enumerate(k_grid_useful):
                if self.spinor_mode == 'collinear':
                    enk_match_table = enk_match_tables[:, :, ik]
                else:
                    enk_match_table = enk_match_tables[:, ik]
                # skip the unmatch loops
                if not enk_match_table.any():
                    continue
                kpq = k+q
                # shape if collinear (2*norbs, 1), (1, 2*norbs, norbs), else (norbs, 1), (1, norbs, norbs)
                eig_kpqs, wave_kpqs = self._elec_cal_torch(kpq)
                if self.spinor_mode == 'collinear':
                    eig_kpqs = np.stack((eig_kpqs[self.bands_indices,0], eig_kpqs[self.bands_indices+self.norbs,0]), axis=0) # (2, nbands)
                    wave_kpqs = np.stack((wave_kpqs[0, self.bands_indices, :], wave_kpqs[0, self.bands_indices+self.norbs, :]), axis=0) # (2, nbands, norbs)
                else:
                    eig_kpqs = eig_kpqs[self.bands_indices, 0] # (nbands)
                    wave_kpqs = wave_kpqs[0, self.bands_indices, :] # (nbands, norbs)
                # skip the unmatch loops
                emkq_match_table = np.abs(eig_kpqs - self.efermi) < self.e_thr
                if not emkq_match_table.any():
                    continue
                # pre-compute delta_mkq for all bands at k+q
                delta_mkq_all_bands = w0gauss((eig_kpqs - self.efermi) * self.inv_smeark, self.gauss_type) * self.inv_smeark
                # use pre-computed phase_k; compute phase_kpq
                phase_k = phase_k_all[:, ik]
                phase_kpq = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ kpq)) # shape: (ncells,)
                weight_k_ik = weights_k_useful[ik]
                if self.spinor_mode == 'collinear':
                    wave_k = wave_ks_useful[:, ik]
                    for ispin in range(2):
                        if not enk_match_table[ispin].any():
                            continue
                        matched_ibnd = np.where(enk_match_table[ispin])[0]
                        matched_jbnd = np.where(emkq_match_table[ispin])[0]
                        if len(matched_ibnd) == 0 or len(matched_jbnd) == 0:
                            continue
                        for ibnd in matched_ibnd:
                            delta_nk = delta_nk_all[ispin, ibnd, ik]
                            wave_coe1 = wave_k[ispin, ibnd] # shape: (norbs,)
                            for jbnd in matched_jbnd:
                                delta_mkq = delta_mkq_all_bands[ispin, jbnd]
                                weight_factor = delta_nk * delta_mkq * weight_k_ik
                                wave_coe2 = wave_kpqs[ispin, jbnd] # shape: (norbs,)
                                tmp1 = np.outer(np.conj(wave_coe2), wave_coe1)
                                for ibranch in valid_branch_indices:
                                    epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                                tmp1, self.grad_mat[ispin], phase_k, phase_kpq, dipole_correction=False)
                                    g2_tmp = (epc * np.conj(epc)).real
                                    epc_strengths[ispin, iq, ibranch] += g2_tmp * weight_factor
                else:
                    wave_k = wave_ks_useful[ik]
                    matched_ibnd = np.where(enk_match_table)[0]
                    matched_jbnd = np.where(emkq_match_table)[0]
                    if len(matched_ibnd) == 0 or len(matched_jbnd) == 0:
                        continue
                    for ibnd in matched_ibnd:
                        delta_nk = delta_nk_all[ibnd, ik]
                        wave_coe1 = wave_k[ibnd] # shape: (norbs,)
                        for jbnd in matched_jbnd:
                            delta_mkq = delta_mkq_all_bands[jbnd]
                            weight_factor = delta_nk * delta_mkq * weight_k_ik
                            wave_coe2 = wave_kpqs[jbnd] # shape: (norbs,)
                            tmp1 = np.outer(np.conj(wave_coe2), wave_coe1)
                            for ibranch in valid_branch_indices:
                                epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                            tmp1, self.grad_mat, phase_k, phase_kpq, dipole_correction=False)
                                g2_tmp = (epc * np.conj(epc)).real
                                epc_strengths[iq, ibranch] += g2_tmp * weight_factor
            if self.rank==0:
                logger.step(iq+1)
        # The formula divide by an extra w_qv, but we do not do it here, because it will be eliminated in the a2f calculation.
        epc_strengths = epc_strengths / dos_ef # shape: (nqs[local], nmodes)
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        if self.spinor_mode == 'collinear':
            a2f = np.einsum('sqv, wqv->sw', epc_strengths, delta_omegas) * self.weight_q
        else:
            a2f = np.einsum('qv, wqv->w', epc_strengths, delta_omegas) / 2.0 * self.weight_q
        # The a2f of the whole q is obtained by allreducing the a2f of each process
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
        print_peak_memeory(self.comm, self.rank)
        return omegas_list, a2f

    def _eliashberg_spectrum_cal_epc_sparse(self):
        """
        Calculate the Eliashberg spectral function alpha^2 F(omega) using sparse
        electron-phonon coupling (EPC) matrix elements.

        This method computes alpha^2 F(omega) by:
        1. Generating k-point (IBZ) and q-point (full BZ) grids.
        2. Computing electronic structure (eigenvalues & wavefunctions) at all k-points,
           then filtering to retain only k-points near the Fermi surface (|E_nk - E_F| < e_thr).
        3. For each q-point, computing electronic structure at k+q, applying a second energy
           filter, and evaluating EPC matrix elements g_{mn}(k, q) for doubly-valid (k, k+q) pairs
           via the C extension `eliashberg_spectrum_cal_EPC_helper_sparse`.
        4. Accumulating |g|^2-weighted contributions into epc_strengths and convolving with
           Gaussian-broadened phonon delta functions to obtain alpha^2 F(omega).

        The calculation is MPI-parallelized: k-points are distributed across ranks for the
        initial electronic structure, then re-split by useful k-points; q-points are distributed
        across ranks for the main loop. The final alpha^2 F is obtained by Allreduce over all ranks.

        Supports spinor_mode = 'none' (spin-unpolarized, weight_k doubled for spin degeneracy),
        'collinear' (two independent spin channels), and 'nc' (non-collinear / spin-orbit).

        Uses the following instance attributes:
            self.k_size (tuple of int): (nk1, nk2, nk3) Monkhorst-Pack k-grid dimensions.
            self.q_size (tuple of int): (nq1, nq2, nq3) Monkhorst-Pack q-grid dimensions.
            self.bands_indices (np.ndarray): Indices of bands included in the EPC calculation.
            self.omega_range (tuple of float): (omega_min, omega_max) in Hartree.
            self.omega_step (float): Frequency grid step size in Hartree.
            self.e_thr (float): Energy threshold for Fermi surface filtering in Hartree.
            self.smeark (float): Electronic smearing width for Fermi level determination.
            self.inv_smeark (float): 1 / smeark, inverse electronic smearing.
            self.inv_smearq (float): Inverse phonon smearing width for delta function broadening.
            self.phonon_frequency_cutoff (float): Minimum phonon frequency to include (filters acoustic modes).
            self.gauss_type (int): Smearing function type (0=Gaussian, -1=cold, -99=Fermi-Dirac, etc.).
            self.spinor_mode (str): 'none', 'collinear', or 'nc'.
            self.graph_data: Object holding atomic structure (pos, latt, num_electrons, H_cell, S_cell, etc.).
            self.nbr_shift_of_cell_sc (np.ndarray): (ncells, 3) supercell lattice vectors for phase factors.
            self.grad_mat (dict or list of dict): Sparse gradient matrices {(icell, jcell, iatom, idir): (data, indices, indptr)}.
            self.atomic_mass (np.ndarray): (natoms,) atomic masses in a.m.u.
            self.doping_charge (float): Charge doping adjustment to electron count.
            self.fermi_shift (float): Manual shift applied to the computed Fermi level.
            self.comm: MPI communicator (or None for serial).
            self.rank (int): MPI rank index.
            self.rank_size (int): Total number of MPI ranks.

        Restart:
            After the EPC main loop completes, the full epc_strengths array (gathered from all
            MPI ranks) along with freq_grid, dos_ef, efermi, and weight_q are saved to
            ``{self.outdir}/epc_strengths_restart.npz`` by rank 0.

            On subsequent calls, if this file exists, the function loads the saved data, skips
            the entire EPC main loop (including electronic structure, Fermi surface filtering,
            and the q-point loop), and directly computes the a2f convolution. This allows
            re-computing a2f with different omega_range, omega_step, or inv_smearq without
            re-running the expensive EPC calculation.

            To force a full recalculation, delete the restart file before calling this method.

        Returns:
            omegas_list (np.ndarray): (nomegas,) frequency grid in Hartree.
            a2f (np.ndarray): Eliashberg spectral function.
                - spinor_mode='none' or 'nc': shape (nomegas,)
                - spinor_mode='collinear': shape (2, nomegas), [spin-up, spin-down].
        """

        # === Check for restart: load previously saved epc_strengths ===
        epc_restart_path = os.path.join(self.outdir, 'epc_strengths_restart.npz')
        if os.path.isfile(epc_restart_path) and self.epc_restart:
            log("Restart file found: {}. Loading epc_strengths and skipping EPC main loop.".format(epc_restart_path), rank=self.rank)
            restart_data = np.load(epc_restart_path)
            epc_strengths = restart_data['epc_strengths']
            freq_grid_all = restart_data['freq_grid']
            dos_ef = float(restart_data['dos_ef'])
            self.efermi = float(restart_data['efermi'])
            self.weight_q = float(restart_data['weight_q'])

            # Reconstruct omegas_list
            omegas_list = _superconductivity_omega_grid(self.omega_range, self.omega_step)

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
            log(f"Number of electrons included in the calculation: {num_electron_include}", rank=self.rank)

        # generate the meshes for kpts and qpts
        k_grid, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
        q_grid = self._get_monkhorst_pack(self.q_size, return_frac=True)

        # initial the omega
        omegas_list = _superconductivity_omega_grid(self.omega_range, self.omega_step)

        nks_fbz = self.k_size[0] * self.k_size[1] * self.k_size[2]
        nks_ibz = len(k_grid)
        nqs = len(q_grid)

        self.weight_q = 1.0 / nqs

        # ``_transport_prepare(..., reduced_type='k')`` historically performs
        # a contiguous q split.  Reorder the global q list before entering it
        # so the existing splitter produces deterministic hash-cyclic chunks.
        # This branch is deliberately restricted to the optimized scalar-spin
        # sparse path; all other modes retain the historical ordering.
        q_global_order = np.arange(nqs, dtype=int)
        if (self.spinor_mode == 'none' and not self.soc_switch and
                getattr(self, 'q_split_mode', 'contiguous') == 'hash-cyclic' and
                self.comm is not None):
            q_chunks = [get_q_rank_indices(
                q_grid, irank, self.rank_size, split_mode='hash-cyclic',
                q_size=self.q_size) for irank in range(self.rank_size)]
            q_global_order = np.concatenate(q_chunks)
            q_grid = q_grid[q_global_order]

        k_grid_all, enks_local, wave_ks_local, q_grid_local, weights_q_local, freq_grid, phon_vecs= \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, None)
        q_local_positions = np.array_split(np.arange(nqs, dtype=int), self.rank_size)[self.rank]
        q_global_indices_local = q_global_order[q_local_positions]

        nomegas = len(omegas_list)
        nbands = len(self.bands_indices)
        nqs_local = len(q_grid_local)
        nks_local = len(k_grid_all[self.rank])
        nbranches = self._get_nbranches()

        if self.spinor_mode == 'nc':
            nbands_col = nbands
        elif self.spinor_mode == 'collinear':
            nbands_col = nbands * 2
        else:
            nbands_col = nbands
            self.weight_k *= 2.0

        enks_local = enks_local.reshape(nbands_col, nks_local)
        enks_all = [enks_local] if self.comm is None else self.comm.allgather(enks_local)
        if (self.spinor_mode == 'none' and not self.soc_switch):
            # Avoid mpi4py pickle for the large initial wave table.  The
            # helper uses bounded typed broadcasts and preserves the
            # contiguous rank order used by k_grid_all.
            wave_ks = _bcast_array_by_rank_chunks(wave_ks_local, self.comm)
            wave_ks_all = None
        else:
            wave_ks_all = [wave_ks_local] if self.comm is None else self.comm.allgather(wave_ks_local)
        del enks_local, wave_ks_local

        enks = np.concatenate(enks_all, axis=-1)    # (nbands, nks) or (2*nbands, nks) for collinear
        if wave_ks_all is not None:
            wave_ks = np.concatenate(wave_ks_all, axis=-3) # (nks, nbands, norbs) or (2, nks, nbands, norbs) for collinear
        self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
        is_metal = self._get_is_metal(enks, self.efermi)
        if not is_metal:
            raise RuntimeError(f"Error(7001): Superconductivity can only be calculated for metallic systems.")
        dos_ef = self._get_ef_dos(enks, self.efermi, self.inv_smeark, self.gauss_type)
        if self.spinor_mode == 'collinear':
            log("Fermi energy = {:10.6f} eV\nDOS of Fermi level (sum of two spin channels) = {:10.6f}".format(self.efermi * QTcts.HARTREEtoEV, dos_ef / QTcts.HARTREEtoEV), rank=self.rank)
        else:
            # N(Ef) in lambda formula is the DOS per spin
            dos_ef = dos_ef / 2.0
            log("Fermi energy = {:10.6f} eV\nDOS of Fermi level (spin factor not included) = {:10.6f}".format(self.efermi * QTcts.HARTREEtoEV, dos_ef / QTcts.HARTREEtoEV), rank=self.rank)
        # resplit the k points
        mask = np.abs(enks - self.efermi) < self.e_thr
        _, kpt_indices_useful = np.where(mask)
        kpt_indices_useful = np.unique(kpt_indices_useful) # IBZ k-points really contributing
        nks_useful = kpt_indices_useful.shape[0] # total number of useful kpoints
        k_grid_useful = k_grid[kpt_indices_useful]                       # (nks_useful, 3)
        enks_useful = enks[:, kpt_indices_useful]                # (nbands, nks_useful) or (2*nbands, nks_useful) for collinear
        if self.spinor_mode == 'collinear':
            enks_useful = enks_useful.reshape(2, nbands, nks_useful) # (2, nbands, nks_useful)
            wave_ks_useful = wave_ks[:, kpt_indices_useful]    # (2, nks_useful, nbands, norbs)
        else:
            wave_ks_useful = wave_ks[kpt_indices_useful]    # (nks_useful, nbands, norbs)
        weights_k_useful = self.weight_k[kpt_indices_useful]             # (nks_useful,)
        self._print_mesh_mpi_check(nks_fbz, nks_ibz, nks_useful, nqs)

        if self.spinor_mode == 'collinear':
            epc_strengths = np.zeros((2, nqs_local, nbranches)) #[up, dn]
        else:
            epc_strengths = np.zeros((nqs_local, nbranches))

        enk_match_tables = np.abs(enks_useful - self.efermi) < self.e_thr  # (nbands, nks_useful_local) or (2, nbands, nks_useful_local)
        # === Pre-compute phase_k for all valid k-points (only depends on k, not q) ===
        # nbr_shift_of_cell_sc: (ncells, 3), k_grid_valid: (nks_useful_local, 3)
        phase_k_all = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ k_grid_useful.T))  # (ncells, nks_useful_local)
        atomic_mass = np.ascontiguousarray(self.atomic_mass)

        if self.rank == 0:
            logger = time_logger(nqs_local, 'eliashberg_spectrum_cal_epc_sparse')

        # Build FBZ->IBZ mapping once for the commensurate energy-table path.
        # The table stores only selected-band energies; wavefunctions remain on demand.
        grid2ir_idx = None
        kpq_energy_table = None
        if (self.spinor_mode == 'none' and not self.soc_switch and
                all(int(k) % int(q) == 0 for k, q in zip(self.k_size, self.q_size))):
            _, _, _, grid2ir_idx = self._get_ir_reciprocal_mesh(
                self.k_size, auxiliary_info=True, return_frac=True)
            kpq_energy_table = np.ascontiguousarray(enks.T)

        # Prepare CSR metadata once per rank.  The batch helper consumes this
        # capsule directly and therefore never reparses the Python gradient
        # dictionary for each k point or q point.  Long-range correction
        # paths are intentionally kept on the scalar implementation.
        grad_cache = None
        batch_allowed = (self.spinor_mode == 'none' and not self.soc_switch)
        for _correction_name in ('apply_correction', 'quadrupole_correction'):
            _correction_value = getattr(self, _correction_name, False)
            if isinstance(_correction_value, (bool, np.bool_)) and _correction_value:
                batch_allowed = False
        if batch_allowed:
            try:
                grad_cache = prepare_sparse_grad_cache(self.grad_mat)
            except NativeExtensionUnavailable:
                # Older installed native libraries have no cache entry point;
                # the scalar helper below remains numerically equivalent.
                grad_cache = None

        try:
            # Calculate one q at a time, but bound k+q electronic temporaries.
            for iq, q in enumerate(q_grid_local):
                epc_strengths[iq] = _sparse_epc_q_none_batched(
                    self, q, freq_grid[iq], phon_vecs[iq].reshape(-1, self.natoms, 3),
                    k_grid_useful, enks_useful, wave_ks_useful,
                    weights_k_useful, phase_k_all,
                    kpq_energy_table=kpq_energy_table, grid2ir_idx=grid2ir_idx,
                    batch_size=1024, grad_cache=grad_cache)
                if self.rank == 0:
                    logger.step(iq + 1)
        finally:
            # SparseGradCache owns references to every CSR ndarray; dropping
            # the capsule invokes its destructor and releases native metadata.
            grad_cache = None
        # The formula divide by an extra w_qv, but we do not do it here, because it will be eliminated in the a2f calculation.

        # === Save epc_strengths for restart ===
        # Gather local epc_strengths and freq_grid from all ranks to rank 0
        epc_strengths_gathered = gather_to_rank_root(epc_strengths, self.comm, root=0)
        freq_grid_gathered = gather_to_rank_root(freq_grid, self.comm, root=0)
        q_indices_gathered = gather_to_rank_root(q_global_indices_local, self.comm, root=0)
        if self.rank == 0:
            epc_strengths_all = np.concatenate(epc_strengths_gathered, axis=-2)  # (2, nqs, nbranches)
            freq_grid_all = np.concatenate(freq_grid_gathered, axis=0)  # (nqs, nbranches)
            q_indices_all = np.concatenate(q_indices_gathered, axis=0)
            q_order = np.argsort(q_indices_all, kind='stable')
            epc_strengths_all = epc_strengths_all[..., q_order, :]
            freq_grid_all = freq_grid_all[q_order]
            q_indices_all = q_indices_all[q_order]
            np.savez(epc_restart_path,
                     epc_strengths=epc_strengths_all,
                     freq_grid=freq_grid_all,
                     q_indices=q_indices_all,
                     dos_ef=np.array(dos_ef),
                     efermi=np.array(self.efermi),
                     weight_q=np.array(self.weight_q))
            print("EPC strengths saved to {} for restart.".format(epc_restart_path))

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

    def _lambda_cal_epc_sparse(self, k_grid, q_grid):
        """
        Calculate mode-resolved EPC coupling strengths for the phonon lambda integral.

        This sparse-gradient helper evaluates Fermi-surface-restricted EPC strengths
        on the supplied k/q meshes for non-magnetic systems. The returned strengths
        are divided by the phonon frequencies so that the caller can perform the
        final Brillouin-zone summation and normalization used in the lambda formula.

        Args:
            k_grid (np.ndarray): Full k sampling grid, shape (nks, 3).
            q_grid (np.ndarray): Full q sampling grid, shape (nqs, 3).

        Returns:
            tuple: ``(wqv_all, epc_strengths_all)`` on rank 0, and local values in
                serial execution.

            wqv_all (np.ndarray): Phonon frequencies for all q points and branches,
                shape (nqs, nbranches).
            epc_strengths_all (np.ndarray): EPC strengths divided by phonon
                frequency, shape (nqs, nbranches).

        Raises:
            NotImplementedError: If ``self.spinor_mode`` is not ``'none'``.
        """

        if self.spinor_mode != 'none':
            raise NotImplementedError("Lambda calculation for phonon only support non-magnetic systems.")
        nks_fbz = len(k_grid)
        nqs = len(q_grid)
        k_grid_all, enks_local, eigen_vecs, q_grid_local, weights_q_local, freq_grid, phon_vecs = \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, None)
        nqs_local = len(q_grid_local)
        nbranches = self._get_nbranches()
        enks_all = gather_to_rank_root(enks_local, self.comm)
        eigen_vecs_all = gather_to_rank_root(eigen_vecs, self.comm)
        del enks_local, eigen_vecs
        if self.rank == 0:
            enks_all = np.concatenate(enks_all, axis=-1)    # (nbands, nks)
            eigen_vecs_all = np.concatenate(eigen_vecs_all, axis=0) # (nks, nbands, norbs)
            # resplit the k points
            mask = np.abs(enks_all - self.efermi) < self.e_thr
            _, kpt_indices_useful = np.where(mask)
            kpt_indices_useful = np.unique(kpt_indices_useful) # IBZ k-points really contributing
            nks_useful = kpt_indices_useful.shape[0] # total number of useful kpoints
            k_grid_useful = k_grid[kpt_indices_useful]                       # (nks_useful, 3)
            enks_all_useful = enks_all[:, kpt_indices_useful]                # (nbands, nks_useful)
            eigen_vecs_all_useful = eigen_vecs_all[kpt_indices_useful]    # (nks_useful, nbands, norbs)
            weights_k_useful = self.weight_k[kpt_indices_useful]             # (nks_useful,)
            counts = np.zeros(self.rank_size, dtype=int)
            for i in range(nks_useful):
                counts[i % self.rank_size] += 1
            split_sections = np.cumsum(counts)
            split_indices = split_sections[:-1]
            k_grid_useful_split = np.split(k_grid_useful, indices_or_sections=split_indices, axis=0)
            enks_all_useful_split = np.split(enks_all_useful, indices_or_sections=split_indices, axis=1)
            eigen_vecs_all_useful_split = np.split(eigen_vecs_all_useful, indices_or_sections=split_indices, axis=0)
            weights_k_useful_split = np.split(weights_k_useful, indices_or_sections=split_indices, axis=0)
            self._print_mesh_mpi_check(nks_fbz, nks_fbz, nks_useful, nqs)
        else:
            k_grid_useful_split = None
            enks_all_useful_split = None
            eigen_vecs_all_useful_split = None
            weights_k_useful_split = None
        k_grid_useful_local = self.comm.scatter(k_grid_useful_split, root=0)
        enks_useful_local = self.comm.scatter(enks_all_useful_split, root=0)
        eigen_vecs_useful_local = self.comm.scatter(eigen_vecs_all_useful_split, root=0)
        weights_k_useful_local = self.comm.scatter(weights_k_useful_split, root=0)

        epc_strengths = np.zeros((nqs_local, nbranches))
        atomic_mass = np.ascontiguousarray(self.atomic_mass)
        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [
            np.ascontiguousarray(atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq])
            for iq in range(nqs_local)
        ]
        phase_q_all = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ q_grid_local.T))
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'lambda_cal_epc_sparse')
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv, k_grid_recv, weights_k_recv = self.comm.bcast(
                (eigen_vecs_useful_local, enks_useful_local, k_grid_useful_local, weights_k_useful_local), root=send_rank)
            for ik, k in enumerate(k_grid_recv):
                # eig_k: shape (nbands, ); wave_k: shape (nbands, norbs)
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik]
                enk_match_table = np.abs(eig_k - self.efermi) < self.e_thr
                if not np.any(enk_match_table):
                    continue
                phase_k = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ k)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    freq = freq_grid[iq]
                    kpq = k+q
                    eig_kpq, wave_kpq = self._elec_cal_torch(kpq) # (norbs, 1), (1, norbs, norbs)
                    eig_kpq = eig_kpq[self.bands_indices, 0] # (nbands, )
                    wave_kpq = wave_kpq[0, self.bands_indices] # (nbands, norbs)
                    emkq_match_table = np.abs(eig_kpq - self.efermi) < self.e_thr
                    if not np.any(emkq_match_table):
                        continue
                    phvec_wap = phvec_waps_all[iq]
                    phase_kpq = phase_k * phase_q_all[:, iq] # shape: (ncells,)
                    epc_strengths_q = np.zeros(nbranches, dtype=float, order='C')
                    eliashberg_spectrum_cal_EPC_helper_sparse(np.ascontiguousarray(freq), atomic_mass, phvec_wap,
                                                          np.ascontiguousarray(phase_k), np.ascontiguousarray(phase_kpq), self.grad_mat,
                                                          np.ascontiguousarray(wave_k), np.ascontiguousarray(wave_kpq),
                                                          np.ascontiguousarray(enk_match_table), np.ascontiguousarray(emkq_match_table),
                                                          np.ascontiguousarray(eig_k), np.ascontiguousarray(eig_kpq), epc_strengths_q,
                                                          self.phonon_frequency_cutoff, self.inv_smeark, self.efermi, weights_k_recv[ik])
                    epc_strengths[iq] += epc_strengths_q
            if self.rank==0:
                logger.step(send_rank+1)
        epc_strengths = epc_strengths / freq_grid[:,:]
        if self.comm is not None:
            wqv_all = gather_to_rank_root(freq_grid, self.comm)
            epc_strengths_all = gather_to_rank_root(epc_strengths, self.comm)
            if self.rank == 0:
                wqv_all = np.concatenate(wqv_all, axis=0)
                epc_strengths_all = np.concatenate(epc_strengths_all, axis=0)
        else:
            wqv_all = freq_grid
            epc_strengths_all = epc_strengths
        return wqv_all, epc_strengths_all

    def _epc_strength_cal(self, a2f, omegas):
        """
        Calculate the electron-pphonon coupling strength.

        args:
            a2f: The eliashberg_spectrum.
            omegas_list: A list of omegas corresponding to a2f.

        returns:
            epc_strength: The electron-phonon coupling strength.
        """
        mask = np.abs(omegas) > self.omega_step / 10.0
        if self.spinor_mode == 'collinear':
            epc_strength = np.array([0.0, 0.0])
            for ispin in range(2):
                epc_strength[ispin] = np.sum(a2f[ispin, mask] / omegas[mask]) * self.omega_step
        else:
            epc_strength = np.sum(a2f[mask] / omegas[mask]) * self.omega_step
        epc_strength *= 2.0
        return epc_strength

    def _logave_freq_cal(self, epc_strength, a2f, omegas):
        """
        Calculate the logarithmic average of the phonon frequencies.

        args:
            epc_strength: The lambda.
            a2f: The eliashberg_spectrum.
            omegas_list: A list of omegas corresponding to a2f.

        returns:
            logave_freq: The logarithmic average of the phonon frequencies in Hartree. 
        """
        omegas_abs = np.abs(omegas)
        mask = omegas_abs > QTcts.TENPM40
        inner_term = np.sum(a2f[mask] / omegas[mask] * np.log(omegas_abs[mask])) * self.omega_step / epc_strength 
        inner_term *= 2.0
        return np.exp(inner_term)
