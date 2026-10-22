"""Electrical and thermal conductivity calculation workflow."""
from mpi4py import MPI
"""Conductivity and mobility calculation-mode entry point."""

from qtrans.core.constants import QTcts
from qtrans.core.logging import log
from qtrans.numerics.random import random_cauchy
from qtrans.numerics.random import random_uniform
from qtrans.parallel.mpi import print_peak_memeory
from qtrans.parallel.mpi import gather_array_to_root_chunks
from qtrans.parallel.mpi import bcast_array_from_root_chunks
from qtrans.parallel.mpi import get_q_rank_index_parts
from qtrans.calculator.transport.mobility_sparse import validate_commensurate_lookup
from qtrans.core.batch import get_k_batch_size
import numpy as np
import os


class _ConductivityMixin:
    def conductivity_cal(self):
        """
        Calculate the electrical conductivity (metal) or mobility (insulator) using ERTA or IBTE.

        The workflow:
        1. Generate IBZ/FBZ k-grids and q-grid, compute electronic eigenvalues and Fermi level.
        2. Screen IBZ k-points by energy focus threshold to identify contributing k-points.
        3. Compute (or read from file) scattering rates, eigenvalues, and band velocities.
        4. Calculate conductivity/mobility via ERTA (zero-order solution).
        5. If mob_level='ibte', iteratively solve the Boltzmann transport equation until convergence.
        6. Print conductivity/mobility tensors and Seebeck coefficients.
        """

        # define basic dimensions of k/q-grid
        k_grid_ibz, self.weight_k, k_grid_fbz, grid2ir_idx = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=True)
        self.ir_indices, rot_indices, all_rots_cart, ibz_to_fbz, ibz_to_fbz_index = self._build_kpoint_mappings(self.k_size, return_index = True)
        sparse_none = (self.cal_mode == 'conductivity' and self.spinor_mode == 'none'
                       and not self.soc_switch and self.read_sparse_grad_mat
                       and not self.epc_rate_path)
        self._mobility_kpq_energy_ibz = None
        self._mobility_grid2ir_idx = None

        # BEGIN QTRANS_LOCAL_Q_PARTITION (remove with the q-owner optimization)
        local_q_partition = (sparse_none and self.comm is not None and
                             self.MC_sampling == 'none')
        if local_q_partition and self.rank != 0:
            q_grid_fbz = None
            self.weight_q = None
        elif self.MC_sampling == 'cauchy':
            q_grid_fbz, self.weight_q = random_cauchy(self.nsamples, cauchy_scale=self.cauchy_scale, random_seed=self.sampling_seed)
        elif self.MC_sampling == 'uniform':
            q_grid_fbz, self.weight_q = random_uniform(self.nsamples, random_seed=self.sampling_seed)
        else:
            q_grid_fbz = self._get_monkhorst_pack(self.q_size, return_frac=True)
            self.weight_q = np.ones(len(q_grid_fbz)) / len(q_grid_fbz)
        if local_q_partition:
            n_q_fbz = self.comm.bcast(
                len(q_grid_fbz) if self.rank == 0 else None, root=0)
        else:
            n_q_fbz = len(q_grid_fbz)
        # END QTRANS_LOCAL_Q_PARTITION
        nks_fbz = len(k_grid_fbz)
        nks_ibz = len(k_grid_ibz)
        nbands = len(self.bands_indices)
        if self.spinor_mode == 'none':
            self.weight_k *= 2.0

        # check the system is metal or insulator
        split_sections = np.full(self.rank_size, nks_ibz // self.rank_size, dtype=int)
        split_sections[:nks_ibz % self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_split = np.split(k_grid_ibz, indices_or_sections=split_sections, axis=0)
        k_grid_local = k_grid_split[self.rank]
        if sparse_none:
            # The Fermi-level and metal test still need all bands.  Discard
            # full eigenvectors after each bounded solve, retaining energies.
            enks_local = np.empty((self.norbs, len(k_grid_local)))
            block = get_k_batch_size(self)
            for start in range(0, len(k_grid_local), block):
                stop = min(start + block, len(k_grid_local))
                eig_block, wave_block = self._elec_cal_torch(k_grid_local[start:stop])
                enks_local[:, start:stop] = eig_block
                del eig_block, wave_block
        elif k_grid_local.size > 0:
            # enks_local: # shape if collinear (2*norbs, nks_local), else (norbs, nks_local)
            enks_local, _ = self._elec_cal_torch(k_grid_local)
        else:
            if self.spinor_mode == 'collinear':
                enks_local = np.empty((2*self.norbs, 0))
            else:
                enks_local = np.empty((self.norbs, 0))
        if self.rank == 0:
            enks_all = [None] * self.rank_size
        else:
            enks_all = None
        if sparse_none:
            enks_all = gather_array_to_root_chunks(enks_local, self.comm, root=0)
        else:
            self.comm.Barrier()
            enks_all = self.comm.gather(enks_local, root=0)
        if self.rank == 0:
            # enks: # shape if collinear (2*norbs, nks), else (norbs, nks)
            enks = np.concatenate(enks_all, axis=-1)
            num_electron_include = self.graph_data.num_electrons - self.doping_charge
            self.efermi = self._get_fermi_level_metal(enks, self.temperature, self.gauss_type, num_electron_include) + self.fermi_shift
            is_metal = self._get_is_metal(enks, self.efermi)
            if is_metal:
                print("The system is a metal, we calculate conductivity.")
            else:
                print("The system is a insulator, we calculate mobility.")
            ele_data_tmp = [self.efermi, is_metal]
        else:
            ele_data_tmp = None
        del enks_local, enks_all, split_sections, k_grid_local # clean 
        ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
        self.efermi, is_metal = ele_data_tmp[0], ele_data_tmp[1]

        # screen out IBZ k-points and corresponding bands really contributing!
        if is_metal:
            if self.rank == 0:
                # here band_edge_energy is the fermi level of the system.
                print("Fermi energy = {:10.6f} eV.".format(self.efermi * QTcts.HARTREEtoEV))
                efocus_threshold = [self.efermi - self.over_band_edge, self.efermi + self.over_band_edge]
                if self.spinor_mode == 'collinear':
                    # enks: shape (2, nbands, nks_ibz)
                    enks = np.stack((enks[self.bands_indices], enks[self.bands_indices+self.norbs]), axis=0)
                else:
                    # enks: shape (nbands, nks_ibz)
                    enks = enks[self.bands_indices]
                mask = (enks >= efocus_threshold[0]) & (enks <= efocus_threshold[1])

                if self.spinor_mode == 'collinear':
                    _, _, kpt_indices_usefull = np.where(mask) # we calculate kpts having contribution from any spin
                else:
                    _, kpt_indices_usefull = np.where(mask)
                unique_kpts = np.unique(kpt_indices_usefull) # IBZ k-points really contributing
            else:
                unique_kpts = None
            unique_kpts = self.comm.bcast(unique_kpts, root=0)

        else:
            if self.is_hole:
                band_edge_index = self.VBM_band_index
            else:
                band_edge_index = self.CBM_band_index
            if self.spinor_mode == 'collinear':
                iband_edge = np.where(np.array(self.bands_indices)==band_edge_index)[0][0]
            else:
                iband_edge = np.where(np.array(self.bands_indices)==band_edge_index)[0][0]
            if self.rank ==0:
                if self.spinor_mode == 'collinear':
                    # However, when we focus on the phonon-limited conductivity of a collinear system,
                    # we consider both spin-up and spin-down.
                    enks = np.stack((enks[self.bands_indices], enks[self.bands_indices+self.norbs]), axis=0) # enks: shape (2, nbands, nks_ibz)
                else:
                    # If the system is not collinear, wo only focus on the spin-up states of all |nk> states
                    enks = enks[self.bands_indices] # enks: shape (nbands, nks_ibz)
                if self.spinor_mode == 'collinear':
                    self.efermi = np.zeros(2, dtype=float)
                    self.carrier_density = np.zeros(2, dtype=float)
                    band_edge_energy = np.zeros(2, dtype=float)
                    for ispin in range(2):
                        self.efermi[ispin], self.carrier_density[ispin], _ = self._get_fermi_level_insulator(enks[ispin], iband_edge)
                        log("=" * 100, rank=self.rank)
                        if self.system_2D:
                            print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-2) for spin {}.".format(
                                self.efermi[ispin] * QTcts.HARTREEtoEV, self.carrier_density[ispin] * self.inv_cell / (QTcts.BOHRtoCM ** 2), ispin+1
                                ))
                        else:
                            print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-3) for spin {}.".format(
                                self.efermi[ispin] * QTcts.HARTREEtoEV, self.carrier_density[ispin] * self.inv_cell / (QTcts.BOHRtoCM ** 3), ispin+1
                                ))
                        print("=" * 100,'\n')
                        if self.is_hole:
                            band_edge_energy[ispin] = self._get_evbm(enks[ispin], iband_edge)
                        else:
                            band_edge_energy[ispin] = self._get_ecbm(enks[ispin], iband_edge)
                    ele_data_tmp = [*self.efermi, *self.carrier_density, *band_edge_energy]
                else:
                    self.efermi, self.carrier_density, _ = self._get_fermi_level_insulator(enks[:nbands], iband_edge)
                    print("=" * 100)
                    if self.system_2D:
                        print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-2).".format(
                            self.efermi * QTcts.HARTREEtoEV, self.carrier_density * self.inv_cell / (QTcts.BOHRtoCM ** 2)
                            ))
                    else:
                        print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-3).".format(
                            self.efermi * QTcts.HARTREEtoEV, self.carrier_density * self.inv_cell / (QTcts.BOHRtoCM ** 3)
                            ))
                    print("=" * 100)
                    if self.is_hole:
                        band_edge_energy = self._get_evbm(enks, iband_edge)
                    else:
                        band_edge_energy = self._get_ecbm(enks, iband_edge)
                    ele_data_tmp = [self.efermi, self.carrier_density, band_edge_energy]
            else:
                ele_data_tmp = None
            ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
            if self.rank ==0:
                if self.spinor_mode == 'collinear':
                    if self.is_hole:
                        efocus_threshold = band_edge_energy - self.over_band_edge
                        mask = (enks >= efocus_threshold[:, None, None]) # enks: shape (2, nbands, nks_ibz)
                    else:
                        efocus_threshold = band_edge_energy + self.over_band_edge
                        mask = (enks <= efocus_threshold[:, None, None])
                    _, _, kpt_indices_usefull = np.where(mask)
                else:
                    if self.is_hole:
                        efocus_threshold = band_edge_energy - self.over_band_edge
                        mask = (enks >= efocus_threshold)
                    else:
                        efocus_threshold = band_edge_energy + self.over_band_edge
                        mask = (enks <= efocus_threshold)
                    _, kpt_indices_usefull = np.where(mask)
                unique_kpts = np.unique(kpt_indices_usefull) # IBZ k-points really contributing
            else:
                unique_kpts = None
            unique_kpts = self.comm.bcast(unique_kpts, root=0)

        if sparse_none:
            if self.rank == 0:
                reason = validate_commensurate_lookup(
                    self, k_grid_ibz, k_grid_fbz, grid2ir_idx, q_grid_fbz)
            else:
                reason = None
            reason = self.comm.bcast(reason, root=0)
            if reason is None:
                table_root = np.ascontiguousarray(enks.T) if self.rank == 0 else None
                self._mobility_kpq_energy_ibz = bcast_array_from_root_chunks(
                    table_root, self.comm, root=0)
                self._mobility_grid2ir_idx = np.asarray(grid2ir_idx, dtype=int)
                if self.rank == 0:
                    log('Sparse mobility k+q selected-band IBZ lookup enabled.\n', rank=self.rank)
            elif self.rank == 0:
                log(f'Sparse mobility k+q lookup disabled: {reason}; using direct bounded solves.\n', rank=self.rank)

        # distribute kpoints really contributing to each pool
        nks_ibz_usefull = unique_kpts.shape[0] # total number of usefull kpoints
        k_grid_ibz_usefull = k_grid_ibz[unique_kpts] # Card. of usefull kpoints. shape: (nks_ibz_usefull, 3)
        split_sections = np.full(self.rank_size, nks_ibz_usefull // self.rank_size, dtype=int)
        split_sections[:nks_ibz_usefull % self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        # Pre-compute slice boundaries for this rank (used in IBTE iterations)
        _sl_start = int(split_sections[self.rank - 1]) if self.rank > 0 else 0
        _sl_end = int(split_sections[self.rank])
        k_grid_local = k_grid_ibz_usefull[_sl_start:_sl_end]
        k_index_local = unique_kpts[_sl_start:_sl_end]
        if local_q_partition:
            trace_q_partition0 = MPI.Wtime() if os.environ.get(
                'QTRANS_TRACE_MOBILITY_STARTUP') == '1' else None
            if self.rank == 0:
                q_parts = get_q_rank_index_parts(
                    q_grid_fbz, self.rank_size,
                    split_mode=getattr(self, 'q_split_mode', 'contiguous'),
                    q_size=self.q_size)
                q_coord_parts = [q_grid_fbz[idx] for idx in q_parts]
                q_weight_parts = [self.weight_q[idx] for idx in q_parts]
            else:
                q_parts = q_coord_parts = q_weight_parts = None
            q_grid_fbz = self.comm.scatter(q_coord_parts, root=0)
            self.weight_q = self.comm.scatter(q_weight_parts, root=0)
            self._mobility_q_global_indices = self.comm.scatter(q_parts, root=0)
            del q_coord_parts, q_weight_parts, q_parts
            self._mobility_q_prepartitioned = True
            if trace_q_partition0 is not None:
                self._mobility_trace_qsplit_s = (
                    MPI.Wtime() - trace_q_partition0)
        self._print_mesh_mpi_check(nks_fbz, nks_ibz, nks_ibz_usefull, n_q_fbz)
        # BEGIN QTRANS_STARTUP_TRACE (remove with tests/test_mobility_startup_trace.py)
        # This is a synchronized reference immediately after the mesh/rank
        # report.  MPI.Wtime values are only compared as local elapsed times.
        if os.environ.get('QTRANS_TRACE_MOBILITY_STARTUP') == '1':
            self.comm.Barrier()
            self._mobility_trace_t0 = MPI.Wtime()
        # END QTRANS_STARTUP_TRACE
        # Pre-compute unit conversion factors used repeatedly in ERTA/IBTE
        _sigma_unit = 1.0 / (self.volume_uc * nks_fbz) / (QTcts.HARTREEtoOHM * QTcts.BOHRtoM)
        _mob_unit_base = QTcts.EV * self.inv_cell / nks_fbz / (QTcts.HBAR_EV * QTcts.BOHRtoCM)

        if is_metal:
            # if the system is a metal, calculate conductivity here.
            rate_ir = None
            if self.epc_rate_path:
                rate_ir = self._read_and_broadcast_rate(self.epc_rate_path, 'epc', nbands, nks_ibz)
                # we need to split the rate file to useful kpts first.
                if self.spinor_mode == 'collinear':
                    rate_useful = rate_ir[:, :, unique_kpts]
                else:
                    rate_useful = rate_ir[:, unique_kpts]
                eigen_local, vnks_local = self._compute_eigen_velocity_local(k_grid_local, nbands)
                if self.mob_level == 'ibte':
                    if self.rank == 0:
                        print(f"Reading epc scattering channels from {self.scattering_channel_dir}")
                    scat_channel_local = self._read_scattering_channel(self.scattering_channel_dir)
            else:
                rate_useful, eigen_local, vnks_local, fout_name, scat_channel_local = \
                    self._dispatch_scattering_rate(k_grid_ibz_usefull, q_grid_fbz, self.bands_indices,
                                                   self.efermi, is_metal=True)
                if self.rank == 0: # here saves only the rate file.
                    # fold rate of usefull IBZ kpoints back to the full IBZ kpoint
                    if self.spinor_mode == 'collinear':
                        rate_ir = np.full((2, nbands, nks_ibz), np.inf)
                        rate_ir[:, :, unique_kpts] = rate_useful
                    else:
                        rate_ir = np.full((nbands, nks_ibz), np.inf)
                        rate_ir[:, unique_kpts] = rate_useful
                    filepath = os.path.join(self.outdir, fout_name)
                    self._write_rate_file(filepath, self.weight_k, enks, rate_ir, is_metal=True)

            # split the rate to each pool!
            rate_local = rate_useful[..., _sl_start:_sl_end]

            F_0_local, sigma_serta, Seebeck_serta, _ = self._compute_erta(
                eigen_local, vnks_local, rate_local, k_grid_local, nbands,
                self.efermi, all_rots_cart, ibz_to_fbz, k_index_local,
                _sigma_unit, _mob_unit_base, is_metal=True)

            if self.mob_level == 'ibte':
                scat_kwargs = dict(scat_channel_local=scat_channel_local)
                self._run_ibte_iterations(
                    F_0_local, eigen_local, vnks_local, rate_useful,
                    k_grid_local, nbands, nks_ibz_usefull, self.efermi,
                    all_rots_cart, ibz_to_fbz, k_index_local,
                    rot_indices, unique_kpts, _sl_start, _sl_end,
                    _sigma_unit, _mob_unit_base,
                    is_metal=True, sigma_serta=sigma_serta,
                    **scat_kwargs)
        else:
            # if the system is a insulator, calculate mobility here.
            if self.spinor_mode == 'collinear':
                self.efermi = np.array(ele_data_tmp[:2], dtype=float)
                self.carrier_density = np.array(ele_data_tmp[2:4], dtype=float)
                band_edge_energy = np.array(ele_data_tmp[4:], dtype=float)
            else:
                self.efermi, self.carrier_density, band_edge_energy = ele_data_tmp
            rate_ir = None
            if self.epc_rate_path:
                rate_ir = self._read_and_broadcast_rate(self.epc_rate_path, 'total epc', nbands, nks_ibz)
            elif self.polar_rate_path or self.rmp_rate_path:
                if self.polar_rate_path and (not self.rmp_rate_path):
                    rate_ir = self._read_and_broadcast_rate(self.polar_rate_path, 'polar epc', nbands, nks_ibz)
                elif self.rmp_rate_path and (not self.polar_rate_path):
                    rate_ir = self._read_and_broadcast_rate(self.rmp_rate_path, 'rmp epc', nbands, nks_ibz)
                elif self.polar_rate_path and self.rmp_rate_path:
                    rate_ir = self._merge_rate_files(self.polar_rate_path, self.rmp_rate_path, 'polar epc', 'rmp epc', nbands, nks_ibz)
            else:
                rate_useful, eigen_local, vnks_local, fout_name, scat_channel_local = \
                    self._dispatch_scattering_rate(k_grid_ibz_usefull, q_grid_fbz, self.bands_indices,
                                                   band_edge_energy, is_metal=False, polar_split=self.polar_split)
                if self.rank == 0: # here saves only the rate file.
                    # fold rate of usefull IBZ kpoints back to the full IBZ kpoint
                    if self.spinor_mode == 'collinear':
                        rate_ir = np.full((2, nbands, nks_ibz), np.inf)
                        rate_ir[:, :, unique_kpts] = rate_useful
                    else:
                        rate_ir = np.full((nbands, nks_ibz), np.inf)
                        rate_ir[:, unique_kpts] = rate_useful
                    filepath = os.path.join(self.outdir, fout_name)
                    self._write_rate_file(filepath, self.weight_k, enks, rate_ir)

            if self.epc_rate_path or self.polar_rate_path or self.rmp_rate_path:
                # we need to split the rate file to useful kpts first.
                if self.spinor_mode == 'collinear':
                    rate_useful = rate_ir[:, :, unique_kpts]
                else:
                    rate_useful = rate_ir[:, unique_kpts]
                eigen_local, vnks_local = self._compute_eigen_velocity_local(k_grid_local, nbands)
                if self.mob_level == 'ibte':
                    # load scattering channel for each rank
                    if self.rank == 0:
                        print(f"Reading epc scattering channels from {self.scattering_channel_dir}")
                    scat_channel_local = self._read_scattering_channel(self.scattering_channel_dir)
            # split the rate to each pool!
            rate_local = rate_useful[..., _sl_start:_sl_end]

            F_0_local, sigma_serta, Seebeck_serta, mobility_serta = self._compute_erta(
                eigen_local, vnks_local, rate_local, k_grid_local, nbands,
                self.efermi, all_rots_cart, ibz_to_fbz, k_index_local,
                _sigma_unit, _mob_unit_base, is_metal=False,
                carrier_density=self.carrier_density, inv_cell=self.inv_cell)

            if self.mob_level == 'ibte':
                scat_kwargs = dict(scat_channel_local=scat_channel_local)
                self._run_ibte_iterations(
                    F_0_local, eigen_local, vnks_local, rate_useful,
                    k_grid_local, nbands, nks_ibz_usefull, self.efermi,
                    all_rots_cart, ibz_to_fbz, k_index_local,
                    rot_indices, unique_kpts, _sl_start, _sl_end,
                    _sigma_unit, _mob_unit_base,
                    is_metal=False, sigma_serta=sigma_serta,
                    mobility_serta=mobility_serta,
                    carrier_density=self.carrier_density, inv_cell=self.inv_cell,
                    **scat_kwargs)
        print_peak_memeory(self.comm, self.rank)
