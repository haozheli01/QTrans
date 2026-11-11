"""Mechanically extracted QTrans methods."""

from collections import defaultdict
from mpi4py import MPI
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from qtrans.core.constants import QTcts
from qtrans.core.batch import get_k_batch_size
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import split_into_rank, get_q_rank_indices
from typing import Union
import math
import os
import numpy as np
import spglib


class _TransportCommonMixin:
    def _get_kpq_BZ_index(self, kpqs_cart):
        # kpqs_cart (n,3)
        kpqs_arr = np.atleast_2d(kpqs_cart) # shape -> (n,3)
        nk1mul2 = self.k_size[0] * self.k_size[1]
        # Add 10^(-10) to safely tranform float to int

        frac = self._car2frac(kpqs_arr)
        mask_neg = frac < -QTcts.TENPM10
        frac[mask_neg] += 1.0
        scaled = np.round(frac * np.array(self.k_size)[None, :]) + QTcts.TENPM10
        ikpqs = scaled.astype(int)

        for dim in range(3):
            over = ikpqs[:, dim] >= self.k_size[dim]
            ikpqs[over, dim] -= self.k_size[dim]

        BZ_indices  = (
                ikpqs[:, 0]
                + ikpqs[:, 1] * self.k_size[0]
                + ikpqs[:, 2] * nk1mul2
                )
        
        return BZ_indices

    def _build_kpoint_mappings(self, mesh, shift=[0,0,0], return_index=False):
        """
        Calculates k-point mappings (FBZ -> IBZ) using spglib 
        Returns: 
        - ir_indices: FBZ index -> IBZ index (np.ndarray, shape (nks_tot,))
        - rot_indices: FBZ index -> list of rotation indices (list, shape (nks_tot, num_rot))
        - all_rots_cart: All symmetry rotations in Cartesian coordinates (ndarray (N_sym, 3, 3))
        - ibz_to_fbz: IBZ index -> list of equivalent (FBZ index, rot_index)
        """
        # --- Setup 1: Convert to pymatgen structure for spglib ---
        struct = Structure(lattice=self.graph_data.latt*QTcts.BOHRtoANG,
                            species=[Element.from_Z(k).symbol for k in self.graph_data.species],
                            coords=self.graph_data.pos*QTcts.BOHRtoANG, coords_are_cartesian=True)
        positions = np.round(struct.frac_coords,7)
        # Determine cell data and symmetry operations based on spinor mode
        if self.soc_switch:
            if self.magnetic_type == 'none': # Nonmagnetic system with SOC, the symmetry is the same as the non-SOC case.
                cell = (self.graph_data.latt*QTcts.BOHRtoANG, positions, self.graph_data.species)
                # 0) Get k-points and irreducible map (with Time Reversal Symmetry)
                raw_map, addrs = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift, is_dense=True, is_time_reversal=True)
                # 1) Gather all symmetry rotations (Rotations + Time Reversal)
                sym_data = spglib.get_symmetry(cell)
                all_rots = np.array(sym_data['rotations'], dtype=int)   # shape (N_rots,3,3)
                TRS_rots = -1 * all_rots
                all_rots = np.concatenate([all_rots, TRS_rots], axis=0)
            else:
                # For magnetic systems with SOC, we need to consider the effect of SOC on symmetry.
                # --- Non-collinear mode (no symmetry) ---
                # need to optimize!
                k_vec = self._get_monkhorst_pack(mesh, shift=shift)
                ir_indices = np.arange(len(k_vec))
                rot_indices = [[0]for _ in range(len(k_vec))]
                all_rots = np.eye(3)[None, :, :]
                ibz_to_fbz = {i: [[0]] for i in range(len(k_vec))}
                if return_index:
                    ibz_to_fbz_index = {i: [i] for i in range(len(k_vec))}
                    return ir_indices, rot_indices, all_rots, ibz_to_fbz, ibz_to_fbz_index
                else:
                    return ir_indices, rot_indices, all_rots, ibz_to_fbz
        else:
            if self.spinor_mode == 'none': # Non-magnetic system with no SOC
                cell = (self.graph_data.latt*QTcts.BOHRtoANG, positions, self.graph_data.species)
                # 0) Get k-points and irreducible map (with Time Reversal Symmetry)
                raw_map, addrs = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift, is_dense=True, is_time_reversal=True)
                # 1) Gather all symmetry rotations (Rotations + Time Reversal)
                sym_data = spglib.get_symmetry(cell)
                all_rots = np.array(sym_data['rotations'], dtype=int)   # shape (N_rots,3,3)
                TRS_rots = -1 * all_rots
                all_rots = np.concatenate([all_rots, TRS_rots], axis=0)
            elif self.spinor_mode == 'collinear': # Collinear magnetic system with no SOC, the symmetry is reduced compared to non-magnetic case, but still can be handled by spglib with magmom tags.
                # Create numbers_data with magmom tags for collinear symmetry
                magtag_numbers = []
                magmom_tag_map = {}
                new_tag = 1000
                magmoms = self.graph_data.spin_length * self.graph_data.moment_direct
                prec_order = -math.floor(np.log10(np.abs(self.moment_similar_tolerence))) + 1
                numbers_data = self.graph_data.species.copy()
                for i, (number, magmom) in enumerate(zip(numbers_data[self.graph_data.mag_atom_index], magmoms)):
                    key = (number, round(magmom, prec_order))
                    if key not in magmom_tag_map:
                        magmom_tag_map[key] = new_tag
                        new_tag += 1
                    magtag_numbers.append(magmom_tag_map[key])
                numbers_data[self.graph_data.mag_atom_index] = magtag_numbers
                cell = (self.graph_data.latt*QTcts.BOHRtoANG, positions, numbers_data)
                # 0) Get k-points and irreducible map (No Time Reversal Symmetry)
                raw_map, addrs = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift, is_dense=True, is_time_reversal=False)
                # 1) Gather all symmetry rotations (Rotations only)
                sym_data = spglib.get_symmetry(cell)
                all_rots = np.array(sym_data['rotations'], dtype=int)   # shape (N_rots,3,3)
            elif self.spinor_mode == 'nc':
                k_vec = self._get_monkhorst_pack(mesh, shift=shift)
                ir_indices = np.arange(len(k_vec))
                rot_indices = [[0] for _ in range(len(k_vec))]
                all_rots = np.eye(3)[None, :, :]
                ibz_to_fbz = {i: [[0]] for i in range(len(k_vec))}
                if return_index:
                    ibz_to_fbz_index = {i: [i] for i in range(len(k_vec))}
                    return ir_indices, rot_indices, all_rots, ibz_to_fbz, ibz_to_fbz_index
                return ir_indices, rot_indices, all_rots, ibz_to_fbz
        
        if self.rank == 0:
            print('The symmetry for building kpoint mapping is:', spglib.get_spacegroup(cell, symprec=1e-5),'\n')
        # Convert fractional rotation matrices to Cartesian
        G     = self.graph_data.rec_latt               # shape (3,3)
        G_inv = np.linalg.inv(G)                       # shape (3,3)
        all_rots_cart = np.empty_like(all_rots, dtype=float)  # (N_rots,3,3)
        for i, R_frac in enumerate(all_rots):
            all_rots_cart[i] = G_inv @ R_frac @ G # work out!

        # 2) K-point mapping using Integer Coordinates
        kpts_int = addrs                             # (N_kpts, 3)
        mesh_array = np.array(mesh, dtype=int)       # (3,)
        
        ir_ids = np.unique(raw_map)                  # raw irreducible labels
        ir_kpts_int = kpts_int[ir_ids]               # (N_ir, 3)

        # Define robust integer wrap (Pure integer logic for [-M/2+1, M/2] range)
        def wrap_k_int(k_int):
            k_mod = k_int % mesh_array
            M_half = mesh_array // 2
            # Shift points k > M/2 (which are on the negative side of the BZ) by -M
            k_wrapped = np.where(k_mod > M_half, k_mod - mesh_array, k_mod)
            return k_wrapped

        def hash_k_int(k_int_wrapped):
            return tuple(k_int_wrapped)

        # 3) Build lookup table: Rotated IBZ coords → (ir_index, R_index)
        fbz_lookup = defaultdict(list)
        for ir_idx, k_ir in enumerate(ir_kpts_int):
            for R_idx, R in enumerate(all_rots): 
                # Crucial: Exact integer rotation
                k_rot = np.dot(k_ir, R)
                # Wrap back to the spglib integer range
                k_rot_wrapped = wrap_k_int(k_rot)
                fbz_lookup[hash_k_int(k_rot_wrapped)].append((ir_idx, R_idx))

        # 4) Allocate and fill output arrays
        N = len(kpts_int)
        ir_indices = np.empty(N, dtype=int)
        rot_indices = []
        ibz_to_fbz = defaultdict(list)
        ibz_to_fbz_index = defaultdict(list)

        for i, k in enumerate(kpts_int):
            # Look up the k-point's equivalent in the rotated IBZ set
            hk = hash_k_int(k)
            if hk not in fbz_lookup:
                raise RuntimeError(f"FBZ k-point (int) {k} not matched to any IBZ point. Check symmetry operations.")
            # Map FBZ point 'i' to its IBZ index
            ir_i, _ = fbz_lookup[hk][0]
            ir_indices[i] = ir_i
            # Store all possible rotation indices that map this point
            temp = [r_i_alt for _, r_i_alt in fbz_lookup[hk]]
            ibz_to_fbz[ir_i].append(temp)
            rot_indices.append(temp)
            ibz_to_fbz_index[ir_i].append(i)

        if return_index:
            return ir_indices, rot_indices, all_rots_cart, ibz_to_fbz, ibz_to_fbz_index
        else:
            return ir_indices, rot_indices, all_rots_cart, ibz_to_fbz

    def _get_match_table(self, eigs_k, eigs_kpq, freqs, include_off_diag:bool=False):
        """
        Get the match table to speed up scattering rate calculations.
        The current method follows the implementation in pert_utils.f90 from the package Perturbo.

        Args:
            eigs_k (np.ndarray): # shape (nbands, )
            eigs_kpq (np.ndarray): # shape (mbands, )
            freqs (np.ndarray): # shape (nbranches, )

        Returns:
            match_table: # shape: (nbands, mbands, nbranches)
        """
        if include_off_diag:
            match_table = np.zeros((len(eigs_kpq), len(freqs)), dtype=bool)
            emax = np.max(eigs_k)
            emin = np.min(eigs_k)
            for ibranch, wqv in enumerate(freqs):
                if wqv < self.phonon_frequency_cutoff:
                    continue
                match_table[:, ibranch] = (eigs_kpq - wqv - self.e_thr < emax) & (eigs_kpq + wqv + self.e_thr > emin)
        else:
            match_table = np.zeros((len(eigs_k), len(eigs_kpq), len(freqs)), dtype=bool)
            for ibranch, wqv in enumerate(freqs):
                if wqv < self.phonon_frequency_cutoff:
                    continue
                for iband, enk in enumerate(eigs_k):
                    for jband, emkq in enumerate(eigs_kpq):
                        if np.abs(np.abs(enk - emkq) - wqv) < self.e_thr:
                            match_table[iband, jband, ibranch] = True
        return match_table

    def _get_nbranches(self):
        nbranches = int(3) * self.natoms
        return nbranches

    def _compute_phonon_spectrum(self, q_grid_local, nbranches, is_metal):
        """Compute phonon spectrum and eigenvectors for local q points."""
        apply_correction_for_qs = None
        if q_grid_local.size > 0:
            use_q_batch = (
                self.cal_mode == 'conductivity' and not is_metal
                and self.spinor_mode == 'none' and not self.soc_switch
                and self.magnetic_type == 'none' and self.read_sparse_grad_mat
                and self.LRC_taylor_order == 0)
            if use_q_batch:
                freq_grid, boson_vecs = self._phonon_cal_q_batch(q_grid_local)
            else:
                freq_grid, boson_vecs = self._phonon_cal(q_grid_local)
            nqs_local = len(freq_grid)
            boson_vecs = boson_vecs.reshape(nqs_local, nbranches, self.natoms, 3)
            # Phonopy uses fractional q; compare the cutoff in Cartesian units.
            q_grid_local = self._frac2car(q_grid_local)
            q_cut_sq = self.q_cut * self.q_cut
            if (not is_metal) and (self.apply_correction or self.quadrupole_correction):
                apply_correction_for_qs = np.sum(np.square(q_grid_local), axis=1) < q_cut_sq
            else:
                apply_correction_for_qs = np.zeros(len(q_grid_local), dtype=bool)
        else:
            q_grid_local = np.empty((0, 3))
            freq_grid = np.empty((0, nbranches))
            boson_vecs = np.empty((0, nbranches, self.natoms, 3))
            apply_correction_for_qs = np.empty((0,))
        return q_grid_local, freq_grid, boson_vecs, apply_correction_for_qs

    def _compute_electron_properties(self, k_grid_local, band_indice, band_edge_energy, nbands):
        """Compute eigenvalues, eigenvectors, velocities, and fermi speed for local k points."""
        if (self.cal_mode == 'conductivity' and self.spinor_mode == 'none'
                and not self.soc_switch and self.read_sparse_grad_mat):
            nlocal = len(k_grid_local)
            eigen = np.empty((nbands, nlocal))
            eigen_vecs = np.empty((nlocal, nbands, self.norbs), dtype=np.complex128)
            vnks = np.empty((nlocal, nbands, 3))
            block = get_k_batch_size(self)
            for start in range(0, nlocal, block):
                stop = min(start + block, nlocal)
                eig_raw, wave_raw = self._elec_cal_torch(k_grid_local[start:stop])
                eig_sel = np.ascontiguousarray(eig_raw[band_indice, :])
                wave_sel = np.ascontiguousarray(wave_raw[:, band_indice, :])
                eigen[:, start:stop] = eig_sel
                eigen_vecs[start:stop] = wave_sel
                vnks[start:stop] = self._vel_nk_cal_from_HS(
                    eig_sel, wave_sel, k_grid_local[start:stop])
                del eig_raw, wave_raw, eig_sel, wave_sel
            return eigen, eigen_vecs, vnks, None
        vnks = None
        fermi_speed = None
        if k_grid_local.size > 0:
            # eigen: (norbs, nk_local) eigen_vecs: (nk_local, norbs, norbs)
            eigen, eigen_vecs = self._elec_cal_torch(k_grid_local)
            if self.spinor_mode == 'collinear':
                # eigen: (2, nbands, nks_local)
                nks_local = eigen.shape[1]
                eigen_col = np.empty((2, nbands, nks_local), dtype=eigen.dtype)
                eigen_col[0] = eigen[band_indice]
                eigen_col[1] = eigen[band_indice+self.norbs]
                eigen = eigen_col
                # eigen_vecs: (2, nks_local, nbands, norbs)
                eigen_vecs_col = np.empty((2, nks_local, nbands, self.norbs), dtype=eigen_vecs.dtype)
                eigen_vecs_col[0] = eigen_vecs[:, band_indice]
                eigen_vecs_col[1] = eigen_vecs[:, band_indice+self.norbs]
                eigen_vecs = eigen_vecs_col
                if self.cal_mode in ('resistivity', 'conductivity'):
                    nks_local = len(k_grid_local)
                    vnks = np.zeros((2, nks_local, nbands, 3))
                    for ispin in range(2):
                        vnks[ispin] = self._vel_nk_cal_from_HS(eigen[ispin], eigen_vecs[ispin], k_grid_local, collinear_part_index=ispin)
                    if self.cal_mode == 'resistivity':
                        delta_nks = w0gauss((eigen - band_edge_energy) * self.inv_smeark) * self.inv_smeark
                        vnks_sq = np.square(vnks)
                        fermi_speed = np.empty((2, 3))
                        for s in range(2):
                            fermi_speed[s] = delta_nks[s].T.ravel() @ vnks_sq[s].reshape(-1, 3)
            else:
                eigen = eigen[band_indice, :]
                eigen_vecs = eigen_vecs[:, band_indice, :]
                if self.cal_mode in ('resistivity', 'conductivity'):
                    vnks = self._vel_nk_cal_from_HS(eigen, eigen_vecs, k_grid_local)
                    if self.cal_mode == 'resistivity':
                        delta_nks = w0gauss((eigen - band_edge_energy) * self.inv_smeark) * self.inv_smeark
                        fermi_speed = delta_nks.T.ravel() @ np.square(vnks).reshape(-1, 3)
        else:
            if self.spinor_mode == 'collinear':
                eigen, eigen_vecs = np.empty((2, nbands, 0)), np.empty((2, 0, nbands, self.norbs))
                if self.cal_mode in ('resistivity', 'conductivity'):
                    vnks = np.empty((2, 0, nbands, 3))
                if self.cal_mode == 'resistivity':
                    fermi_speed = np.zeros((2, 3))
            else:
                eigen, eigen_vecs = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))
                if self.cal_mode in ('resistivity', 'conductivity'):
                    vnks = np.empty((0, nbands, 3))
                if self.cal_mode == 'resistivity':
                    fermi_speed = np.zeros(3)
        return eigen, eigen_vecs, vnks, fermi_speed

    def _transport_prepare(self, k_grid:np.ndarray, q_grid:np.ndarray, band_indice:np.ndarray,
                           band_edge_energy:Union[float,np.ndarray], is_metal:bool=False, reduced_type:str='k', sm_gauge:bool=False):
        # k_grid is cartesian and q_grid is fractional
        nks = len(k_grid)
        nbands = len(band_indice)

        if self.cal_mode == 'conductivity':
            if is_metal:
                # here band_edge_energy is the fermi level of the system.
                efocus_threshold = [band_edge_energy - self.over_band_edge, band_edge_energy + self.over_band_edge]
            elif self.is_hole:
                efocus_threshold = band_edge_energy - self.over_band_edge
            else:
                efocus_threshold = band_edge_energy + self.over_band_edge
        elif self.cal_mode == 'breakdown':
            if self.is_hole:
                efocus_threshold = [band_edge_energy - self.fbd_erange[1] - self.e_thr, band_edge_energy - self.fbd_erange[0] + self.e_thr]
            else:
                efocus_threshold = [band_edge_energy + self.fbd_erange[0] - self.e_thr, band_edge_energy + self.fbd_erange[1] + self.e_thr]
        elif self.cal_mode == 'resistivity':
            # here band_edge_energy is the fermi level of the system.
            efocus_threshold = [band_edge_energy - self.e_thr, band_edge_energy + self.e_thr]
        elif self.cal_mode == 'boltzmann':
            # here band_edge_energy is the fermi level not the VBM/CBM of the system.
            if self.is_hole:
                efocus_threshold = band_edge_energy - self.over_band_edge
            else:
                efocus_threshold = band_edge_energy + self.over_band_edge
        elif self.cal_mode == 'self-energy':
            efocus_threshold = [-self.over_band_edge, self.over_band_edge]

        if reduced_type == 'k':
            # q points are parallelized and q grid is split
            trace_startup = (os.environ.get(
                'QTRANS_TRACE_MOBILITY_STARTUP') == '1')
            sparse_mobility = (self.cal_mode == 'conductivity' and
                               self.spinor_mode == 'none' and not self.soc_switch
                               and self.read_sparse_grad_mat)
            if sparse_mobility and getattr(self, '_mobility_q_prepartitioned', False):
                # The conductivity workflow has already scattered q and
                # weights from rank 0.  Do not recreate a global q index array
                # or run the splitter a second time.
                q_indices = np.asarray(self._mobility_q_global_indices,
                                       dtype=int)
                q_grid_local = np.ascontiguousarray(q_grid)
                weights_q_local = np.ascontiguousarray(self.weight_q)
            elif sparse_mobility:
                # BEGIN QTRANS_STARTUP_TRACE (remove with tests/test_mobility_startup_trace.py)
                trace_q0 = MPI.Wtime() if trace_startup else None
                q_indices = get_q_rank_indices(
                    q_grid, self.rank, self.rank_size,
                    split_mode=getattr(self, 'q_split_mode', 'contiguous'),
                    q_size=self.q_size)
                if trace_startup:
                    self._mobility_trace_qsplit_s = MPI.Wtime() - trace_q0
                self._mobility_q_global_indices = q_indices
                q_grid_local = np.asarray(q_grid)[q_indices]
                weights_q_local = np.asarray(self.weight_q)[q_indices]
            else:
                q_grid_local, split_indices = split_into_rank(q_grid, 0, self.rank, self.rank_size, return_indices=True)
                weights_q = np.asarray(self.weight_q)
                if len(weights_q.shape) != 1:
                    weights_q_local = None
                else:
                    weights_q_local = split_into_rank(weights_q, 0, self.rank, self.rank_size, split_indices=split_indices)
                    self.weight_q = None
            del q_grid
            nbranches = self._get_nbranches()

            if trace_startup:
                trace_phonon0 = MPI.Wtime()
            # compute boson spectrum
            q_grid_local, freq_grid, boson_vecs, apply_correction_for_qs = self._compute_phonon_spectrum(q_grid_local, nbranches, is_metal)
            if trace_startup:
                self._mobility_trace_phonon_s = MPI.Wtime() - trace_phonon0
            # END QTRANS_STARTUP_TRACE

            # k grid is split
            split_sections = np.zeros(self.rank_size, dtype=int)
            for i in range(nks):
                split_sections[i%self.rank_size] += 1
            split_sections = np.cumsum(split_sections, axis=0)
            k_grid_all = np.split(k_grid, indices_or_sections=split_sections, axis=0)

            # compute electron properties
            eigen, eigen_vecs, vnks, fermi_speed = self._compute_electron_properties(
                k_grid_all[self.rank], band_indice, band_edge_energy, nbands)

            if self.cal_mode == 'resistivity':
                if self.comm is not None:
                    self.comm.Allreduce(MPI.IN_PLACE, fermi_speed, op=MPI.SUM)

            if self.cal_mode == 'resistivity':
                return efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, boson_vecs, fermi_speed
            elif self.cal_mode == 'breakdown':
                return efocus_threshold, k_grid_all, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, boson_vecs, apply_correction_for_qs
            elif self.cal_mode == 'conductivity':
                return efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, boson_vecs, apply_correction_for_qs
            elif self.cal_mode == 'boltzmann':
                return efocus_threshold, k_grid_all, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, boson_vecs, apply_correction_for_qs
            elif self.cal_mode == 'self-energy':
                return efocus_threshold, k_grid_all, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, boson_vecs, apply_correction_for_qs
            else:
                return k_grid_all, eigen, eigen_vecs, q_grid_local, weights_q_local, freq_grid, boson_vecs
        elif reduced_type == 'q':
            # q points are parallelized
            q_grid_local = split_into_rank(q_grid, 0, self.rank, self.rank_size)
            nbranches = self._get_nbranches()

            # compute boson spectrum
            q_grid_local, freq_grid, boson_vecs, apply_correction_for_qs = self._compute_phonon_spectrum(q_grid_local, nbranches, is_metal)

            q_grid_all = self.comm.allgather(q_grid_local)

            # k points are parallelized
            k_grid_local, split_indices = split_into_rank(k_grid, 0, self.rank, self.rank_size, return_indices=True)
            weights_k = np.asarray(self.weight_k)
            if len(weights_k.shape) != 1:
                weights_k_local = None
            else:
                weights_k_local = split_into_rank(weights_k, 0, self.rank, self.rank_size, split_indices=split_indices)
                self.weight_k = None

            # compute electron properties
            eigen, eigen_vecs, vnks, fermi_speed = self._compute_electron_properties(
                k_grid_local, band_indice, band_edge_energy, nbands)

            if self.cal_mode == 'resistivity':
                if self.comm is not None:
                    self.comm.Allreduce(MPI.IN_PLACE, fermi_speed, op=MPI.SUM)

            if self.cal_mode == 'resistivity':
                return efocus_threshold, k_grid_local, eigen, eigen_vecs, weights_k_local, vnks, q_grid_all, freq_grid, boson_vecs, fermi_speed
            elif self.cal_mode == 'breakdown':
                return efocus_threshold, k_grid_local, eigen, eigen_vecs, weights_k_local, q_grid_all, freq_grid, boson_vecs, apply_correction_for_qs
            elif self.cal_mode == 'conductivity':
                return efocus_threshold, k_grid_local, eigen, eigen_vecs, weights_k_local, vnks, q_grid_all, freq_grid, boson_vecs, apply_correction_for_qs
            elif self.cal_mode == 'self-energy':
                return efocus_threshold, k_grid_local, eigen, eigen_vecs, weights_k_local, q_grid_all, freq_grid, boson_vecs, apply_correction_for_qs
            else:
                return k_grid_local, eigen, eigen_vecs, weights_k_local, q_grid_all, freq_grid, boson_vecs
