"""Electron-phonon coupling calculation and plotting workflows."""

from qtrans.core.constants import QTcts
from qtrans.parallel.mpi import gather_to_rank_root
import numpy as np
import os


class _EPCPlotMixin:
    def plot_epc(self):
        """
        Calculate the EPC for a special q path.

        Args:

        Returns:
        """
        if self.epc_grid_mode:
            return self._epc_cal_on_grid()
        nqs = len(self.high_symmetry_k_vecs)
        nbranches = self.natoms * 3
        nbands_ini = len(self.dispersion_select_index[0])
        nbands_fin = len(self.dispersion_select_index[1])
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        hsk_path_vecs_all = np.split(self.high_symmetry_k_vecs, indices_or_sections=split_sections, axis=0)
        if hsk_path_vecs_all[self.rank].size > 0:
            hsk_path_vecs = hsk_path_vecs_all[self.rank]
            # shape: if collinear (2, nqs_local, nbands_ini, nbands_fin, nbranches), else (nqs_local, nbands_ini, nbands_fin, nbranches)
            if self.read_sparse_grad_mat:
                epcs_local = self.epc_cal_path(self.epc_path_fix_k, hsk_path_vecs, self.dispersion_select_index[0], self.dispersion_select_index[1], do_symm=True, is_sparse=True)
            else:
                epcs_local = self.epc_cal_path(self.epc_path_fix_k, hsk_path_vecs, self.dispersion_select_index[0], self.dispersion_select_index[1], do_symm=True)
        else:
            hsk_path_vecs = np.empty((0, 3))
            if self.spinor_mode == 'collinear':
                epcs_local = np.empty((2, 0, nbands_ini, nbands_fin, nbranches))
            else:
                epcs_local = np.empty((0, nbands_ini, nbands_fin, nbranches))
        epcs_all = gather_to_rank_root(epcs_local, self.comm)
        if self.rank == 0:
            epcs = np.concatenate(epcs_all, axis=-4)
            self._output_dispersion(epcs, 'epc')

    def _epc_cal_on_grid(self):
        """Calculate EPC |g(k,q)| with k on IBZ and q on full FBZ grid.

        Parallelization: k loop is serial across ranks; q_FBZ is split across
        all ranks within each k iteration. This gives up to nq_FBZ-way
        parallelism (much larger than nks_ibz) and dramatically reduces
        per-rank memory.

        Storing g(k_ibz, q_FBZ) (no q-side IBZ reduction) is required for
        physical correctness: |g(k, q)| only has the symmetry of the
        stabilizer of k as a function of q, not the full point group.
        Reducing q by the full point group is lossy.
        """
        k_mesh = np.array(self.epc_k_mesh, dtype=int)
        q_mesh = np.array(self.epc_q_mesh, dtype=int)

        # k on IBZ; q on full FBZ (no symmetry reduction)
        k_ibz, weight_k = self._get_ir_reciprocal_mesh(k_mesh, return_frac=True)
        _, _, q_fbz, _ = self._get_ir_reciprocal_mesh(q_mesh, auxiliary_info=True, return_frac=True)

        nks = len(k_ibz)
        nqs = len(q_fbz)
        nbranches = self.natoms * 3
        nbands_ini = len(self.dispersion_select_index[0])
        nbands_fin = len(self.dispersion_select_index[1])

        # MPI: distribute q_FBZ across ranks (k loop is shared)
        q_split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            q_split_sections[i % self.rank_size] += 1
        q_split_sections = np.cumsum(q_split_sections, axis=0)
        q_fbz_all = np.split(q_fbz, indices_or_sections=q_split_sections, axis=0)
        q_local = q_fbz_all[self.rank]
        nq_local = len(q_local)
        ranks_with_work = sum(1 for arr in q_fbz_all if len(arr) > 0)

        if self.rank == 0:
            print(f"EPC grid mode: k_mesh={list(k_mesh)}, q_mesh={list(q_mesh)}")
            print(f"  nk_ibz={nks}, nq_FBZ={nqs}, nbands_ini={nbands_ini}, "
                  f"nbands_fin={nbands_fin}, nbranches={nbranches}", flush=True)
            spin_factor = 2 if self.spinor_mode == 'collinear' else 1
            mem_GB = spin_factor * nks * nqs * nbands_ini * nbands_fin * nbranches * 8 / 1024**3
            print(f"  estimated epc memory on rank 0: {mem_GB:.2f} GB (float64)", flush=True)
            print(f"  MPI: rank_size={self.rank_size}, q parallelism={ranks_with_work} ranks doing work", flush=True)
            if self.rank_size > nqs:
                idle = self.rank_size - nqs
                print(f"  WARNING: rank_size > nq_FBZ={nqs}; {idle} ranks will be IDLE.")
                print(f"           Consider reducing rank_size to <= {nqs} to save resources.", flush=True)
            elif self.rank_size > max(1, nqs // 4):
                print(f"  Note: per-rank q load is small (~{nqs // max(1, self.rank_size)} q/rank); "
                      f"gather overhead may become significant.", flush=True)

        # Pre-allocate full epc array on rank 0 only
        if self.rank == 0:
            if self.spinor_mode == 'collinear':
                epcs = np.zeros((2, nks, nqs, nbands_ini, nbands_fin, nbranches))
            else:
                epcs = np.zeros((nks, nqs, nbands_ini, nbands_fin, nbranches))
        else:
            epcs = None

        is_sparse = self.read_sparse_grad_mat
        print_every = max(1, nks // 20)

        for ik in range(nks):
            if nq_local > 0:
                epc_local = self.epc_cal_path(k_ibz[ik], q_local,
                                self.dispersion_select_index[0],
                                self.dispersion_select_index[1],
                                do_symm=True, is_sparse=is_sparse)
                # epc_local shape:
                #   none/nc:   (nq_local, nb_ini, nb_fin, nbranches)
                #   collinear: (2, nq_local, nb_ini, nb_fin, nbranches)
            else:
                if self.spinor_mode == 'collinear':
                    epc_local = np.empty((2, 0, nbands_ini, nbands_fin, nbranches))
                else:
                    epc_local = np.empty((0, nbands_ini, nbands_fin, nbranches))

            # Gather along q axis to root for this k
            epc_gathered = gather_to_rank_root(epc_local, self.comm)

            if self.rank == 0:
                if self.spinor_mode == 'collinear':
                    epcs[:, ik] = np.concatenate(epc_gathered, axis=1)
                else:
                    epcs[ik] = np.concatenate(epc_gathered, axis=0)

                if (ik + 1) % print_every == 0 or ik == nks - 1:
                    print(f"  k {ik+1}/{nks} done.", flush=True)

        if self.rank == 0:
            # Phonon frequencies on FBZ q grid
            freq_grid, _ = self._phonon_cal(q_fbz)  # (nqs_FBZ, nbranches)

            # Frequency cutoff: branches with freq < phonon_frequency_cutoff are
            # already zeroed inside epc_cal_path (see e.g. l.2534 sparse,
            # l.2562 dense). Save the cutoff so downstream code knows the
            # threshold; the filtered (q, branch) positions can be re-derived
            # from frequency < phonon_frequency_cutoff_Ha.
            n_filtered = int(np.sum(freq_grid < self.phonon_frequency_cutoff))
            n_total = freq_grid.size

            # k-side symmetry info (used to fold k_full -> k_ibz downstream)
            _, _, k_all_rots_cart, _ = self._build_kpoint_mappings(k_mesh)
            _, _, k_fbz_frac, k_grid2ir = self._get_ir_reciprocal_mesh(
                k_mesh, auxiliary_info=True, return_frac=True)

            outfile = os.path.join(self.outdir, "epc_grid.npz")
            np.savez(outfile,
                     epc=epcs,
                     frequency=freq_grid,
                     phonon_frequency_cutoff_Ha=self.phonon_frequency_cutoff,
                     k_ibz_frac=k_ibz,
                     weight_k=weight_k,
                     k_fbz_frac=k_fbz_frac,
                     k_grid2ir=k_grid2ir,
                     k_symmetry_rotations=k_all_rots_cart,
                     q_fbz_frac=q_fbz,
                     k_mesh=k_mesh,
                     q_mesh=q_mesh,
                     bands_ini=self.dispersion_select_index[0],
                     bands_fin=self.dispersion_select_index[1])
            print(f"EPC grid data saved to {outfile}")
            print(f"  epc shape: {epcs.shape}, unit: Hartree, mem: {epcs.nbytes/1024**3:.2f} GB")
            print(f"  frequency shape: {freq_grid.shape}, unit: Hartree")
            print(f"  phonon_frequency_cutoff = {self.phonon_frequency_cutoff / QTcts.MEVtoHARTREE:.4f} meV"
                  f" ({self.phonon_frequency_cutoff:.6e} Hartree); "
                  f"filtered {n_filtered}/{n_total} (q,branch) modes "
                  f"(|g| zeroed inside epc_cal_path).")
            print(f"  k_ibz: {nks} points, k_fbz: {len(k_fbz_frac)} points")
            print(f"  q_fbz: {nqs} points (no q-IBZ reduction; q is full FBZ)")
