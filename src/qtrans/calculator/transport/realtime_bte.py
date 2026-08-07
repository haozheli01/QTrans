"""Mechanically extracted QTrans methods."""

from collections import defaultdict
from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.extensions import dynamics_setup_helper_sparse
from qtrans.numerics.distributions import bose_weight
from qtrans.numerics.distributions import fermi_weight
from qtrans.numerics.gaussian import w0gauss
import numpy as np
import os


class _RealtimeBTEMixin:
    def dynamics_setup(self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float, is_rmp:bool=False):
        """
        Setup the initial electron distribution f_{nk} and scattering channels using the dense grad_mat.

        Args:
            k_grid (np.ndarray): The k sampling points. # shape (nks, 3)
            q_grid (np.ndarray): The q sampling points. # shape (nqs, 3)
            bands_indice (np.ndarray): The bands indices. # shape (nbands, )
            band_edge_energy (float): The energy of band edge.
            is_rmp (bool): Set to True, if you want to calculate the remainder part of scattering rate.
        Returns:
            eigen (np.ndarray): The E_nk for electrons. # shape (nbands, nks)
            dist_ele_0 (np.ndarray): The initial electron distribution f_{nk}. # shape (nbands, nks)
            scat_channel (np.ndarray): Structured array with fields: ik, ikpq, iband, jband, freq, g2, de
        """
        nbranches = int(3) * self.natoms
        if self.rank == 0:
            print('='*100)
            print(' '*30+'Setup the initial distribution for rtBTE!')
            print('='*100 + "\n")

        efocus_threshold, _, eigen, eigen_vecs, q_grid_local, _, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal=False)
        # self.comm.Barrier()
        # gather all the info about electrons on each pool 
        eigen = self.comm.allgather(eigen) # (nbands, nks_local)
        eigen = np.concatenate(eigen, axis=-1) # (nbands, nks)
        eigen_vecs = self.comm.allgather(eigen_vecs) # (nks_local, nbands, norbs) 
        eigen_vecs = np.concatenate(eigen_vecs, axis=0) # (nks, nbands, norbs)
        dist_ele_0 = self.init_dist_ele(eigen) # (nbands, nks)

        # screen the contributing scattering channels
        scat_data_list = []
        for ik, k in enumerate(k_grid):
            eig_k, wave_k = eigen[:, ik], eigen_vecs[ik]
            if self.is_hole:
                enk_inside_table = (eig_k >= efocus_threshold)
            else:        
                enk_inside_table = (eig_k <= efocus_threshold)
            if not enk_inside_table.any():
                continue
            phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
            # obtain the coordinates of kpq
            kpq_list = k + q_grid_local # (nqs_local, 3)
            BZ_index_list = self._get_kpq_BZ_index(kpq_list) # index in FBZ
            IBZ_index_list = self.ir_indices[BZ_index_list] # index in IBZ
            for iq, q in enumerate(q_grid_local):
                kpq = k + q
                ikpq = IBZ_index_list[iq]
                eig_kpq = eigen[:, ikpq]
                wave_kpq = eigen_vecs[ikpq]
                if self.adv_cal:
                    ikpq = BZ_index_list[iq] # if efield exists, save FBZ_index instead
                if self.is_hole:
                    emkpq_inside_table = (eig_kpq >= efocus_threshold)
                else:        
                    emkpq_inside_table = (eig_kpq <= efocus_threshold)
                if not emkpq_inside_table.any():
                    continue 
                # phonon spectrum
                freq = freq_grid[iq]
                apply_correction_for_this_q = apply_correction_for_qs[iq]
                eigen_vec_phon = phon_vecs[iq]
                match_table = self._get_match_table(eig_k, eig_kpq, freq)
                if not match_table.any(): # skip the meaningless iterations
                    continue
                atomic_phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                # cal epc
                phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                if apply_correction_for_this_q:
                    epc_corr, tmp1_save = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=True)
                else:
                    tmp1_save = np.einsum('jm,in -> jimn', np.conj(wave_kpq), wave_k)
                for iband in (np.where(enk_inside_table)[0]):
                    for jband in (np.where(emkpq_inside_table)[0]):
                        tmp1 = tmp1_save[jband, iband]
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
                                scat_data_list.append((ik, ikpq, iband, jband, freq[ibranch], g2_tmp, (eig_k[iband] - eig_kpq[jband])))
        dtype = [('ik', int), ('ikpq', int), ('iband', int), ('jband', int), ('freq', float), ('g2', float), ('de', float)]
        scat_channel = np.array(scat_data_list, dtype=dtype)
        if self.adv_cal:
            eigen = eigen[:, self.ir_indices] # (nbands, nks_fbz)
            dist_ele_0 = dist_ele_0[:, self.ir_indices] # (nbands, nks_fbz)
            scat_channel = self.expand_scat_channel_to_fbz(scat_channel, self.ir_indices) # expand scat_channel to FBZ
        return eigen, dist_ele_0, scat_channel

    def dynamics_setup_sparse(self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray, band_edge_energy:float, is_rmp:bool=False):
        """
        Setup the initial electron distribution f_{nk} and scattering channels using the sparse grad_mat.

        Args:
            k_grid (np.ndarray): The k sampling points. # shape (nks, 3)
            q_grid (np.ndarray): The q sampling points. # shape (nqs, 3)
            bands_indice (np.ndarray): The bands indices. # shape (nbands, )
            band_edge_energy (float): The energy of band edge.
            is_rmp (bool): Set to True, if you want to calculate the remainder part of scattering rate.
        Returns:
            eigen (np.ndarray): The E_nk for electrons. # shape (nbands, nks)
            dist_ele_0 (np.ndarray): The initial electron distribution f_{nk}. # shape (nbands, nks)
            scat_channel (np.ndarray): Structured array with fields: ik, ikpq, iband, jband, freq, g2, de
        """
        nbranches = int(3) * self.natoms
        nbands = len(band_indice)
        if self.rank == 0:
            print('='*100)
            print(' '*30+'Setup the initial distribution for rtBTE!')
            print('='*100 + "\n")

        efocus_threshold, _, eigen, eigen_vecs, q_grid_local, _, freq_grid, phon_vecs, apply_correction_for_qs = \
            self._transport_prepare(k_grid, q_grid, band_indice, band_edge_energy, is_metal=False)
        self.comm.Barrier()
        # gather all the info about electrons on each pool 
        eigen = self.comm.allgather(eigen) # (nbands, nks_local)
        eigen = np.concatenate(eigen, axis=-1) # (nbands, nks)
        eigen_vecs = self.comm.allgather(eigen_vecs) # (nks_local, nbands, norbs) 
        eigen_vecs = np.concatenate(eigen_vecs, axis=0) # (nks, nbands, norbs)
        dist_ele_0 = self.init_dist_ele(eigen) # (nbands, nks)

        # screen the contributing scattering channels
        scat_data_list = []
        for ik, k in enumerate(k_grid):
            eig_k, wave_k = eigen[:, ik], eigen_vecs[ik]
            if self.is_hole:
                enk_inside_table = (eig_k >= efocus_threshold)
            else:        
                enk_inside_table = (eig_k <= efocus_threshold)
            if not enk_inside_table.any():
                continue
            phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
            # obtain the coordinates of kpq
            kpq_list = k + q_grid_local # (nqs_local, 3)
            BZ_index_list = self._get_kpq_BZ_index(kpq_list) # index in FBZ
            IBZ_index_list = self.ir_indices[BZ_index_list] # index in IBZ
            for iq, q in enumerate(q_grid_local):
                kpq = k + q
                ikpq = IBZ_index_list[iq]
                eig_kpq = eigen[:, ikpq]
                wave_kpq = eigen_vecs[ikpq]
                if self.adv_cal:
                    ikpq = BZ_index_list[iq] # if efield exists, save FBZ_index instead
                if self.is_hole:
                    emkpq_inside_table = (eig_kpq >= efocus_threshold)
                else:        
                    emkpq_inside_table = (eig_kpq <= efocus_threshold)
                if not emkpq_inside_table.any():
                    continue 
                # phonon spectrum
                freq = freq_grid[iq]
                apply_correction_for_this_q = apply_correction_for_qs[iq]
                eigen_vec_phon = phon_vecs[iq]
                match_table = self._get_match_table(eig_k, eig_kpq, freq)
                if not match_table.any(): # skip the meaningless iterations
                    continue
                atomic_phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                # cal epc
                phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                if apply_correction_for_this_q:
                    epc_corr = self._get_LR_correction_kq(q, freq, phvec_wap, match_table, enk_inside_table, k, wave_k, wave_kpq, return_tmp1=False)
                else:
                    epc_corr = np.zeros((nbands, nbands, nbranches), dtype=complex)
                scat_data = []
                dynamics_setup_helper_sparse(freq.copy(), self.atomic_mass.copy(),phvec_wap.copy(), 
                                                    phase_k.copy(),phase_kpq.copy(),self.grad_mat,
                                                    wave_k.copy(),wave_kpq.copy(),
                                                    enk_inside_table.copy(), emkpq_inside_table.copy(), match_table.copy(),
                                                    eig_k.copy(),eig_kpq.copy(),
                                                    epc_corr.copy(), scat_data, ik, ikpq, int(is_rmp))
                scat_data_list.extend(scat_data)
        dtype = [('ik', int), ('ikpq', int), ('iband', int), ('jband', int), ('freq', float), ('g2', float), ('de', float)]
        scat_channel = np.array(scat_data_list, dtype=dtype)
        if self.adv_cal:
            eigen = eigen[:, self.ir_indices] # (nbands, nks_fbz)
            dist_ele_0 = dist_ele_0[:, self.ir_indices] # (nbands, nks_fbz)
            scat_channel = self.expand_scat_channel_to_fbz(scat_channel, self.ir_indices) # expand scat_channel to FBZ
        return eigen, dist_ele_0, scat_channel

    def init_dist_ele(self, eigen:np.ndarray, init_dist_typ:str='fermi_dirac'):
        """
        Initialize the electronic distribution function for real-time Boltzman transport equation (rt-BTE).

        Args:
            eigen (np.ndarray): Eigenvalue array of shape (nbands, nks)
            init_dist_typ (str): Initialization type ('fermi_dirac' supported)

        Returns:
            np.ndarray: Initialized electron distribution (same shape as eigen)
        """
        if init_dist_typ == 'fermi_dirac':
            dist_ele = fermi_weight(eigen - self.efermi, self.temperature) # FD distribution
        else:
            pass # Loretnzian/Gaussian distribution
        return dist_ele

    def get_collision_term(self, dist_ele:np.ndarray, scat_channel:np.ndarray):
        """
        Calculate the electron-phonon collision term.

        Args:
            dist_ele (np.ndarray): Electron distribution array (nband, nk)
            scat_channel (np.ndarray): Structured array with fields: ik, ikpq, iband, jband, freq, g2, de

        Returns:
            np.ndarray: Collision term (same shape as dist_ele)
        """
        # unpack info
        coll = np.zeros_like(dist_ele)
        iks = scat_channel['ik']
        ikpqs = scat_channel['ikpq']
        ibs = scat_channel['iband']
        jbs = scat_channel['jband']
        freqs = scat_channel['freq']
        g2s = scat_channel['g2']
        des = scat_channel['de']
        # obtain weight functions
        bose_qvs = bose_weight(freqs, self.temperature_lat)
        delta_f1s = w0gauss((des + freqs) * self.inv_smearq, 0) * self.inv_smearq
        delta_f2s = w0gauss((des - freqs) * self.inv_smearq, 0) * self.inv_smearq
        # distributions
        fermi_k = dist_ele[ibs, iks]
        fermi_kpq = dist_ele[jbs, ikpqs]
        # scattering process
        abs_terms = fermi_kpq * (bose_qvs + 1) - fermi_k * (fermi_kpq + bose_qvs)
        em_terms = fermi_kpq * (fermi_k + bose_qvs) - fermi_k * (1 + bose_qvs)
        temps = delta_f1s * abs_terms + delta_f2s * em_terms
        prefactor = QTcts.TWOPI * self.weight_of_q
        temps *= prefactor * g2s
        np.add.at(coll, (ibs, iks), temps)
        # gather the total result from each pool
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, coll, op=MPI.SUM)
        # considering the efield term
        if self.adv_cal:
            drift_efield_term = self.get_efield_term(dist_ele, self.efield, self.final_bvec, self.final_bweight, self.neighbors_table)
            coll = coll + drift_efield_term
        return coll

    def rtBTE_solver(self, time_step:float, dist_ele:np.ndarray, scat_channel:np.ndarray, solver:str='euler'):
        """
        Solve the real-time BTE using either Euler or RK4 method.

        Args:
            time_step (float): Time step in femtoseconds
            dist_ele (np.ndarray): Electron distribution (nband, nk)
            scat_channel (np.ndarray): Structured scattering data
            solver (str): Solver method ('euler' or 'rk4')

        Returns:
            np.ndarray: Updated electron distribution
        """
        if solver == 'euler':
            collision_interg = self.get_collision_term(dist_ele, scat_channel)
            dist_ele_new = dist_ele + collision_interg * time_step
        elif solver == 'rk4':
            k1 = self.get_collision_term(dist_ele, scat_channel)
            k2 = self.get_collision_term(dist_ele + 0.5 * time_step * k1, scat_channel)
            k3 = self.get_collision_term(dist_ele + 0.5 * time_step * k2, scat_channel)
            k4 = self.get_collision_term(dist_ele + time_step * k3, scat_channel)
            dist_ele_new = dist_ele + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise NotImplementedError("Unsupported solver for rtBTE!")
        return dist_ele_new

    def save_energy_resolved_dist(self, step_index:int, eigen:np.ndarray, dist_ele:np.ndarray, energy_step:float, band_edge_energy:float,
                             E_min:float = None, E_max:float = None, output_dir: str = "rt_results"):
        """
        Compute and save energy-resolved electron distribution p(E, t) from f_{nk}(t) and E_{nk}.

        Args:
            step_index: current time step index
            eigen: (nbands, nks_ibz) eigen energies E_{nk}
            dist_ele: (nbands, nks_ibz) electron distribution f_{nk}(t)
            energy_step: resolution of energy grid
            band_edge_energy: reference energy to shift energy grid (e.g. CBM or VBM)
            E_min / E_max: optional energy range for grid (in Hartree)
        Returns:
            None; saves results to file
        """
        # folder for save energy-resolved distributions
        # output_dir = "rt_results"
        if self.rank == 0 and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Define energy grid
        nbnds, nks = eigen.shape
        small_energy = self.e_thr * 3 # for better resolution
        E_min = eigen.min() if E_min is None else E_min
        E_max = eigen.max() if E_max is None else E_max
        E_grid = np.arange(E_min - small_energy, E_max + energy_step + small_energy, energy_step)
        # when considering hole, dist. is 1 - fnk
        if self.is_hole:
            dist_ele = 1 - dist_ele
        # Initialize result
        all_eps = eigen.flatten() #  (nbnds * nks,)
        all_f = dist_ele.flatten() #  (nbnds * nks,)
        if self.adv_cal:
            k_weights_flat = np.ones_like(all_eps) / nks
            if self.spinor_mode == 'none':
                k_weights_flat *= 2.0
        else:
            k_weights_flat = np.tile(self.weight_k, nbnds) # (nbnds * nks,)
        all_coeffs_flat = all_f * k_weights_flat * self.inv_smeark #  (nbnds * nks,)
        p_E = np.zeros(E_grid.shape[0]) # (n_E_grid,)
        for i, E in enumerate(E_grid):
            diff = all_eps - E
            weights = w0gauss(diff * self.inv_smeark)
            p_E[i] = np.dot(weights, all_coeffs_flat)

        if self.rank == 0:
            if step_index == 0:
                # shift energy related to CBM or VBM and convert to meV
                E_grid_shifted = (E_grid - band_edge_energy) * QTcts.HARTREEtoMEV
                np.savetxt(os.path.join(output_dir, "E_grid.dat"), E_grid_shifted, fmt='%-12.1f', header="Energy[meV]")
            # save individual distribution file
            filename = os.path.join(output_dir, f"p_E_{step_index:05d}.dat")
            np.savetxt(filename, p_E, fmt='%-15.8f')

    def combine_dists_to_final_file(self, total_steps: int, output_step: int, output_dir: str = "rt_results", final_filename: str = "P_E.dat"):
        """
        (Postprocessing) Combine all individual p_E files into a final P_E.dat file.
        This function reads the energy grid from 'E_grid.dat' and appends the p_E distributions
        from each time step (that is a multiple of `output_step`) into a single file.
        Args:
            total_steps (int): Total number of time steps in the simulation.
            output_step (int): Interval between outputs (e.g., output every `output_step` fs).
            output_dir (str): Directory where individual p_E files and E_grid.dat are stored.
            final_filename (str): Name of the final output file to write (default: 'P_E.dat').
        Returns:
            None. The final combined file is written to disk.
        """
        # === Load the energy grid ===
        egrid_path = os.path.join(output_dir, "E_grid.dat")
        try:
            E_grid = np.loadtxt(egrid_path, skiprows=1).reshape(-1, 1)
        except FileNotFoundError:
            if self.rank ==0:
                print(f"[Error] Energy grid file not found: {egrid_path}")
            return
        # Initialize data containers
        all_columns = [E_grid]
        header_parts = ["Energy[meV]".ljust(12)]
        # Determine which steps to include
        time_steps = [i for i in range(0, total_steps + 1, output_step)]
        # === Loop through each time step and collect p_E data ===
        for t in time_steps:
            filename = f"p_E_{t:05d}.dat"
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    p_E_column = np.loadtxt(filepath).reshape(-1, 1)
                    all_columns.append(p_E_column)
                    header_parts.append(f"P_E({t:05d}fs)".ljust(15))
                except Exception as e:
                    if self.rank ==0:
                        print(f"[Warning] Failed to read {filename}: {e}")
            else:
                if self.rank ==0:
                    print(f"[Warning] Missing file: {filename}")
        # === Stack data and save ===
        final_data = np.hstack(all_columns)
        header = "".join(header_parts)
        fmt = ['%-12.1f'] + ['%-15.8f'] * (final_data.shape[1] - 1)
        # Save to disk
        np.savetxt(final_filename, final_data, fmt=fmt, header=header, comments='', delimiter='')

    def expand_scat_channel_to_fbz(self, scat_channel:np.ndarray, ir_indices:np.ndarray):
        # Create an array of FBZ k-point indices: [0, 1, 2, ..., N-1]
        fbz_indices = np.arange(len(ir_indices))

        # Step 1: Build a reverse map from each IBZ index to all equivalent FBZ indices
        ibz_to_fbz = defaultdict(list)
        for fbz, ibz in zip(fbz_indices, ir_indices):
            ibz_to_fbz[ibz].append(fbz)

        # Step 2: For each scattering channel in IBZ, get how many FBZ k-points it maps to
        ik_ibz = scat_channel['ik']  # shape: (n_channels,)
        repeat_counts = np.array([len(ibz_to_fbz[ik]) for ik in ik_ibz])  # shape: (n_channels,)
        total = repeat_counts.sum()  # total number of expanded FBZ channels

        # Step 3: Allocate output array with expanded size, same dtype
        sc_expanded = np.empty(total, dtype=scat_channel.dtype)

        # Step 4: Copy all fields except 'ik' by repeating original values according to repeat_counts
        for field in scat_channel.dtype.names:
            if field != 'ik':
                sc_expanded[field] = np.repeat(scat_channel[field], repeat_counts)

        # Step 5: Replace 'ik' field with the list of FBZ k-points each IBZ k-point maps to
        sc_expanded['ik'] = np.concatenate([ibz_to_fbz[ik] for ik in ik_ibz])

        return sc_expanded

    def check_parallel(self, existing_shells:list, new_shell_vectors:np.ndarray, tol=1e-5):
        # ref: boltz_grid_neighbors.f90 of Perturbo
        """
        Check if any vector in the new shell is (nearly) parallel to existing shell vectors.
        
        Args:
        - existing_shells: list of arrays, each containing vectors from an already accepted shell
        - new_shell_vectors: array of vectors in the candidate shell
        - tol: tolerance for detecting near-parallelism via cosine similarity
        
        Returns:
        - True if a parallel vector is found, otherwise False
        """
        for vec_new in new_shell_vectors:
            norm_new = np.linalg.norm(vec_new)
            for shell in existing_shells:
                for vec_old in shell:
                    norm_old = np.linalg.norm(vec_old)
                    cos_theta_abs = np.abs(np.dot(vec_new, vec_old) / (norm_new * norm_old))
                    if np.abs(cos_theta_abs - 1.0) < tol:
                        return True  # Found a nearly parallel vector
        return False

    def get_optimal_neighbors_and_weights(self, k_dims:list, bg_matrix:np.ndarray, kcutoff=5, shell_tol=1e-3, b1_tol=1e-5):
        # ref: boltz_grid_neighbors.f90 of Perturbo
        """
        Generate optimal finite difference vectors and weights for computing k-space derivatives.
        This selects neighboring k-points in reciprocal space and computes weights that satisfy 
        symmetry constraints (the "B1 condition", i.e., Eq. (B1) of  N. Marzari, D. Vanderbilt, Phys. Rev. B 56 (1997) 12847.).

        Args:
            k_dims: (3,) integer array of k-grid dimensions, e.g., [20, 20, 20]
            bg_matrix: (3, 3) matrix of reciprocal lattice vectors (in Cartesian coordinates)
            kcutoff: int, how many shells (+/-kcutoff) to search for candidate finite difference vectors
            shell_tol: float, tolerance for grouping vectors into the same shell (relative distance)
            b1_tol: float, tolerance to accept solution as satisfying the B1 condition

        Returns:
            final_bvec: (Nb, 3) array of selected finite difference vectors in Cartesian coordinates
            final_bweight:(Nb,) corresponding weights for each vector
        """

        # === Step 1: Generate candidate neighbor vectors in integer lattice coordinates ===
        r = np.arange(-kcutoff, kcutoff + 1)
        # Create a full 3D grid of vectors: shape = ((2*kcutoff+1)^3, 3)
        grid = np.stack(np.meshgrid(r, r, r), -1).reshape(-1, 3)
        # Remove the zero vector (no self-difference)
        candidate_b_cryst = np.delete(grid, len(grid) // 2, axis=0)
        # Convert lattice coordinates to fractional coordinates (w.r.t k-grid size)
        candidate_b_frac = candidate_b_cryst / np.array(k_dims)

        # === Step 2: Convert to Cartesian coordinates and compute Euclidean distance ===
        candidate_b_cart = candidate_b_frac @ bg_matrix  # Convert to Cartesian
        distances = np.linalg.norm(candidate_b_cart, axis=1)

        # === Step 3: Sort by distance and group into shells (shelling) ===
        sorted_indices = np.argsort(distances)
        sorted_b_cart = candidate_b_cart[sorted_indices]
        sorted_distances = distances[sorted_indices]

        all_shells = []
        current_shell_dist = sorted_distances[0]
        current_shell_vectors = []
        for i in range(len(sorted_distances)):
            # Group vectors into the same shell based on relative distance tolerance
            if np.abs(sorted_distances[i] - current_shell_dist) < shell_tol * sorted_distances[0]:
                current_shell_vectors.append(sorted_b_cart[i])
            else:
                all_shells.append(np.array(current_shell_vectors))  # finalize current shell
                current_shell_dist = sorted_distances[i]
                current_shell_vectors = [sorted_b_cart[i]]
        all_shells.append(np.array(current_shell_vectors))  # add last shell

        # === Step 4: Iteratively add shells and solve for finite difference weights ===
        # Target constraint vector for symmetric gradient operator: diag(1,1,1), off-diag(0)
        q_target = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        selected_shells = []
        for i in range(len(all_shells)):
            candidate_shell = all_shells[i]
            # Check for linear dependence (parallel vectors) to improve numerical stability
            if self.check_parallel(selected_shells, candidate_shell):
                if self.rank ==0:
                    print(f"Shell {i} is parallel to existing shells. Skipping.")
                continue
            # Try adding the shell to the current set
            current_shells_try = selected_shells + [candidate_shell]
            n_shells = len(current_shells_try)

            # Build the A matrix for symmetry constraints (6 rows: xx, yy, zz, xy, yz, zx)
            amat = np.zeros((6, n_shells))
            for j in range(n_shells):
                shell_vectors = current_shells_try[j]
                bx, by, bz = shell_vectors[:, 0], shell_vectors[:, 1], shell_vectors[:, 2]
                amat[0, j] = np.sum(bx * bx)
                amat[1, j] = np.sum(by * by)
                amat[2, j] = np.sum(bz * bz)
                amat[3, j] = np.sum(bx * by)
                amat[4, j] = np.sum(by * bz)
                amat[5, j] = np.sum(bz * bx)
            # Solve amat * weights = q_target using SVD
            try:
                U, s, Vh = np.linalg.svd(amat, full_matrices=False)
            except np.linalg.LinAlgError:
                if self.rank ==0:
                    print(f"SVD failed for {n_shells} shells. Skipping.")
                continue
            # Reject if any singular value is too small (ill-conditioned)
            if np.any(s < 1e-6):
                if self.rank ==0:
                    print(f"Found small singular value for {n_shells} shells. Rejecting and trying more.")
                continue
            # Compute weights: w = V S^(-1) U^(T) q_target
            s_inv = np.diag(1.0 / s)
            weights = Vh.T @ s_inv @ U.T @ q_target  # Moore-Penrose pseudo-inverse solution
            # Check B1 constraint: amat * weights ≈ q_target
            q_check = amat @ weights
            if np.all(np.abs(q_check - q_target) < b1_tol):
                if self.rank ==0:
                    print(f"B1 condition satisfied with {n_shells} shells.")
                # Flatten vectors and assign weight to each vector in each shell
                final_bvec = np.vstack(all_shells[:n_shells])  # all vectors used
                final_bweight = np.repeat(weights, [len(s) for s in all_shells[:n_shells]])
                return final_bvec, final_bweight
            else:
                # Accept the shell anyway and keep adding more in next loop
                selected_shells.append(candidate_shell)
                if self.rank ==0:
                    print(f"B1 not satisfied with {n_shells} shells. Accepted shell and trying to add more.")

        raise RuntimeError("Could not satisfy B1 condition. Increase kcutoff(Default: 5).")

    def get_neighbor_index(self, final_bvec: np.ndarray, kpoints_cart: np.ndarray):
        """
        Get k-space index for neighbor kpts using pre-calculated bvec.

        Args:
            final_bvec (np.ndarray): Neighbor vectors in Cartesian coordinates, with shape (nbvecs, 3).
            kpoints_cart (np.ndarray): Cartesian coordinates of kpts, with shape (nks_fbz,3).
        Returns:
            neighbors_table (np.ndarray): Index of the ib-th neighbor of the ik-th k-point. Shape: (nks_fbz, nbvecs)
        """
        # Get dimensions
        nks_fbz = kpoints_cart.shape[0]
        nbvecs = final_bvec.shape[0]
        # --- Find all neighbor k-points and their indices ---
        # Calculate Cartesian coordinates of all neighbors for all k-points:
        #  (nks_fbz, 1, 3) + (1, nbvecs, 3) -> (nks_fbz, nbvecs, 3)
        all_neighbors_cart = kpoints_cart[:, np.newaxis, :] + final_bvec[np.newaxis, :, :]
        # Reshape for the vectorized index function: (nks_fbz, nbvecs, 3) -> (nks_fbz * nbvecs, 3)
        reshaped_neighbors = all_neighbors_cart.reshape(-1, 3)
        # Get the flattened 1D index for every neighbor of every k-point
        all_neighbor_indices = self._get_kpq_BZ_index(reshaped_neighbors)
        # Reshape indices into a lookup table where neighbors_table[ik, ib] is the
        # index of the ib-th neighbor of the ik-th k-point. Shape: (nks_fbz, nbvecs)
        neighbors_table = all_neighbor_indices.reshape(nks_fbz, nbvecs)

        return neighbors_table

    def get_kspace_gradient(self, dist_ele: np.ndarray, final_bvec: np.ndarray, final_bweight: np.ndarray, neighbors_table: np.ndarray):
        """
        Compute k-space gradient for electron distribution using pre-calculated bvec and bweight.

        Args:
            dist_ele (np.ndarray): Electron distribution f_{nk}, with shape (nbands, nks_fbz).
            final_bvec (np.ndarray): Neighbor vectors in Cartesian coordinates, with shape (nbvecs, 3).
            final_bweight (np.ndarray): Corresponding weights for each neighbor vector, with shape (nbvecs,).
            neighbors_table (np.ndarray): Index of the ib-th neighbor of the ik-th k-point, with shape: (nks_fbz, nbvecs)
        Returns:
            np.ndarray: The k-space gradient for each band and k-point, with shape (nbands, nks_fbz, 3).
        """
        # --- Calculate the gradient using vectorized operations ---
        # Use advanced indexing to get the distribution values at all neighbor points
        # for all bands. The result has shape (nbands, nks_fbz, nbvecs).
        dist_at_neighbors = dist_ele[:, neighbors_table]
        # Calculate the difference: f(k+b) - f(k).
        # We reshape dist_ele to (nbands, nks_fbz, 1) to enable broadcasting.
        delta_dist = dist_at_neighbors - dist_ele[:, :, np.newaxis]

        # Calculate the final gradient sum: sum_i { w_i * [f(k+b_i) - f(k)] * b_i }
        # We use broadcasting to multiply arrays of shape:
        # (nbands, nks_fbz, nbvecs) * (nbvecs,) * (nbvecs, 3)
        # The result is summed over the nbvecs axis (axis=2).
        gradient = np.einsum(
            'ijk,k,kl->ijl',
            delta_dist, # (nbands, nks_fbz, nbvecs)
            final_bweight, # (nbvecs,)
            final_bvec # (nbvecs, 3)
        )
        # Because the gradient term should have the unit of Bohr and bvecs already have the unit of Bohr^-1 * (2*Pi),
        # so w_i should have the unit of Bohr^2, and we should devide the factor 2*Pi!
        gradient = gradient / QTcts.TWOPI
        return gradient

    def f2g(self, f:np.ndarray, maxg: float = 35.0):
        """
        Vectorized conversion from Fermi-Dirac distribution f to auxiliary function g,
        as used in Perturbo for more numerically stable gradient computation.
        The relation is:
            g = ln((1 - f) / f) = ln(1 - f) - ln(f)
        To avoid numerical overflow/underflow at f → 0 or f → 1, we clip f and apply
        saturation consistent with the Fortran routine.

        Args:
            f (np.ndarray): Electron distribution, expected to be in (0, 1).
            maxg (float): Saturation threshold for extreme values.

        Returns:
            np.ndarray: g values, same shape as f.
        """
        # Avoid log(0) by bounding f away from 0 and 1
        f = np.clip(f, 1e-20, 1 - 1e-20)

        # Compute log safely
        logf = np.log(f)
        log1mf = np.log1p(-f)  # More stable than log(1 - f) near f ≈ 0

        # Main formula: g = log((1 - f) / f)
        g = log1mf - logf

        # Saturate g if f is too close to 0 or 1
        threshold = np.exp(-maxg)
        g = np.where(f < threshold, maxg, g)
        g = np.where((1 - f) < threshold, -maxg, g)

        return g

    def get_efield_term(self, dist_ele:np.ndarray, efield:np.ndarray, final_bvec:np.ndarray, final_bweight:np.ndarray, neighbors_table:np.ndarray):
        """
        Compute the eE·df/dk term using :
        eE* df/dk = eE* (f-1)*f * dg/dk = eE* (f-1)*f * sum_{b}{ w_b * bvec * [g(k+b) - g(b)]}
        Args:
            dist_ele (np.ndarray): Fermi-Dirac distribution f_{nk}, shape (nbands, nks_fbz)
            efield (np.ndarray): Electric field vector, shape (3,)
            final_bvec: (Nb, 3) array of selected finite difference vectors in Cartesian coordinates
            final_bweight:(Nb,) corresponding weights for each vector
            neighbors_table (np.ndarray): Index of the ib-th neighbor of the ik-th k-point, with shape: (nks_fbz, nbvecs)
        Returns:
            np.ndarray: The electric-field driving term, shape (nbands, nks_fbz)
        """
        # Convert f -> g to improve stability of gradient
        dist_ele_g = self.f2g(dist_ele) # shape: (nbands, nks_fbz)
        # Compute gradient of g using finite differences
        g_gradient = self.get_kspace_gradient(dist_ele_g, final_bvec, final_bweight, neighbors_table) # shape: (nbands, nks_fbz, 3)
        # Compute dot product with electric field
        efield_term = np.sum(g_gradient * efield, axis=-1)
        # Combine with f(f - 1)
        efield_term = dist_ele * (dist_ele - 1) * efield_term
        return efield_term

    def get_total_carrier_density(self, dist_ele: np.ndarray) -> float:
        """
        Compute the total carrier density by performing a weighted summation 
        over the electronic distribution function in k-space.
        Args:
            dist_ele (np.ndarray): Electronic distribution function f_{nk}, shaped (nbands, nks).
        Returns:
            float: Total carrier density.
        """
        nbnds, nks = dist_ele.shape
        # Determine the population based on whether it's hole or electron transport
        if self.is_hole:
            population = 1 - dist_ele
        else:
            population = dist_ele
        # Determine the k-point weights depending on calculation mode
        # If adv_cal is True, we are using the full Brillouin zone (FBZ), 
        # where each k-point has equal weight
        if self.adv_cal:
            k_weights = np.ones(nks) / nks
            # For non-spinor calculations, multiply weights by 2 to account for spin degeneracy
            if self.spinor_mode == 'none':
                k_weights *= 2.0
        else:
            # In the IBZ mode, use the precomputed k-point weights
            k_weights = self.weight_k
        # Broadcast the weights to all bands and sum over bands and k-points
        # k_weights[np.newaxis, :] has shape (1, nks), broadcastable with (nbands, nks)
        carrier_density = np.sum(population * k_weights[np.newaxis, :])
        
        return carrier_density

    def get_drift_velocity(self, band_velocity:np.ndarray, dist_ele:np.ndarray, efield:np.ndarray, carrier_concentration:float, volume_uc:float):
        """
        Compute the drift velocity along the direction of the electric field.

        Args:
            band_velocity (np.ndarray): Array of shape (nks, nbands, 3), the group velocity vectors for each band and k-point in Cartesian coordinates.
            dist_ele (np.ndarray): Array of shape (nbands, nks), representing the electron distribution function f_{nk}(t).
            efield (np.ndarray): Array of shape (3,), the external electric field vector.
            carrier_concentration (float): Carrier concentration n_c (units: number per unit volume).
            volume_uc (float): Unit cell volume (must be in consistent units with concentration).

        Returns:
            float: Drift velocity component along the electric field direction ⟨v_parallel(t)⟩.
        """
        # Normalize electric field vector
        efield_norm = efield / np.linalg.norm(efield)  # shape (3,)
        # Project velocity onto field direction: v_parallel = v \cdot E
        v_parallel = np.einsum("ijk,k->ji", band_velocity, efield_norm)  # shape (nbands, nks)
        # Sum f * v_parallel over all bands and k-points
        numerator = np.sum(dist_ele * v_parallel)
        # Total number of k-points
        nks = band_velocity.shape[1]
        # Compute drift velocity
        drift_velocity = numerator / (carrier_concentration * volume_uc * nks)
        return drift_velocity

    def get_velocity_only(self, kpts:np.ndarray):
        
        nkpts = kpts.shape[0]
        nbands = len(self.bands_indices)
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nkpts):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_split = np.split(kpts, indices_or_sections=split_sections, axis=0)
        k_grid_local = k_grid_split[self.rank]
        if k_grid_local.size > 0:
            # enks_local: # shape if collinear (2*norbs, nks_local), else (norbs, nks_local)
            enks_local, eigen_vecs_local = self._elec_cal_torch(k_grid_local)
            if self.spinor_mode == 'collinear':
                # eigen: (2, nbands, nks_local)
                enks_local = np.stack((enks_local[self.bands_indices], enks_local[self.bands_indices+self.norbs]), axis=0)
                # eigen_vecs: (2, nks_local, nbands, norbs)
                eigen_vecs_local = np.stack((eigen_vecs_local[:, self.bands_indices], eigen_vecs_local[:, self.bands_indices+self.norbs]), axis=0)
                nks_local = k_grid_local.size
                vnks = np.zeros((2, nks_local, nbands, 3))
                for ispin in range(2):
                    vnks[ispin] = self._vel_nk_cal_from_HS(enks_local[ispin], eigen_vecs_local[ispin], k_grid_local, collinear_part_index=ispin) # (nks_local, nbands, 3)
            else:
                enks_local = enks_local[self.bands_indices, :]
                eigen_vecs_local = eigen_vecs_local[:, self.bands_indices, :]
                vnks = self._vel_nk_cal_from_HS(enks_local, eigen_vecs_local, k_grid_local) # (nks_local, nbands, 3)
        else:
            if self.spinor_mode == 'collinear':
                enks_local, eigen_vecs_local = np.empty((2, nbands, 0)), np.empty((2, 0, nbands, self.norbs))
                vnks = np.empty((2, 0, nbands, 3))
            else:
                enks_local, eigen_vecs_local = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))
                vnks = np.empty((0, nbands, 3))
        
        vnks_all = self.comm.gather(vnks, root=0)
        if self.rank == 0:
            # combine vnks
            if self.spinor_mode == 'collinear':
                # shape: list of (2, nks_local, nbands, 3)
                vnks_merged = np.concatenate(vnks_all, axis=1)  # shape: (2, nks, nbands, 3)
            else:
                # shape: list of (nks_local, nbands, 3)
                vnks_merged = np.concatenate(vnks_all, axis=0) # shape: (nks, nbands, 3)
            return vnks_merged
        else:
            return None
