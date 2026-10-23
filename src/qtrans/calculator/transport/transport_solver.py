"""Shared ERTA and IBTE transport-solver services."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.numerics.distributions import minus_dfermi
from typing import List
import numpy as np
import opt_einsum as oe
import os
import sys


class _TransportSolverMixin:
    def _read_rate_file(self, filepath:str, shape:List[int]):
        """Read internal-unit rates, preserving missing states as infinity."""
        if len(shape) != 2 or any(int(size) <= 0 for size in shape):
            raise ValueError(f"Invalid rate array shape {shape}; expected [nbands, nks] with positive sizes.")
        nbands, nks = (int(size) for size in shape)

        if self.cal_mode == 'conductivity':
            if self.magnetic_type == 'collinear':
                rate_shape = (2, nbands, nks)
                expected_columns = 7
                rate_columns = (4, 6)
            else:
                rate_shape = (nbands, nks)
                expected_columns = 5
                rate_columns = (4,)
        elif self.cal_mode == 'breakdown':
            rate_shape = (2, nbands, nks)
            expected_columns = 6
            rate_columns = (4, 5)
        else:
            raise ValueError(f"Cannot read rate file for unsupported calculation mode '{self.cal_mode}'.")

        # Rate files can have one, two, or no comment header lines.  Inspect
        # the leading lines instead of unconditionally discarding two rows.
        header_lines = []
        skiprows = 0
        has_data = False
        with open(filepath, 'r') as fin:
            for iline, line in enumerate(fin):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    header_lines.append(stripped[1:].strip())
                    continue
                first_token = stripped.split(maxsplit=1)[0]
                try:
                    float(first_token)
                except ValueError:
                    header_lines.append(stripped)
                    skiprows = iline + 1
                    continue
                has_data = True
                break

        normalized_header = ''.join(' '.join(header_lines).lower().split())
        physical_unit_markers = (
            'enk(ev)', 'scattering_rate(1/ps)', 'energy_loss_rate(ev/ps)',
            'enk[up](ev)', 'enk[down](ev)',
            'scatt_rate[up](1/ps)', 'scatt_rate[down](1/ps)',
        )
        if any(marker in normalized_header for marker in physical_unit_markers):
            raise ValueError(
                f"Rate file '{filepath}' uses eV/ps output units, but QTrans expects its "
                "current internal Hartree-unit format. Convert the file before reading it."
            )

        rate_ir = np.full(rate_shape, np.inf)
        if not has_data:
            return rate_ir

        try:
            data = np.loadtxt(filepath, comments='#', skiprows=skiprows, ndmin=2)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse rate file '{filepath}' for '{self.cal_mode}': {exc}"
            ) from exc
        if data.shape[1] != expected_columns:
            raise ValueError(
                f"Rate file '{filepath}' has {data.shape[1]} columns for '{self.cal_mode}', "
                f"but {expected_columns} columns are required."
            )

        index_data = data[:, :2]
        if not np.all(np.isfinite(index_data)) or not np.all(index_data == np.floor(index_data)):
            raise ValueError(f"Rate file '{filepath}' contains non-integer ik or ibnd indices.")
        ik_arr = index_data[:, 0]
        ibnd_arr = index_data[:, 1]
        invalid_index = (ik_arr < 0) | (ik_arr >= nks) | (ibnd_arr < 0) | (ibnd_arr >= nbands)
        if np.any(invalid_index):
            irow = int(np.flatnonzero(invalid_index)[0])
            raise ValueError(
                f"Rate file '{filepath}' data row {irow + 1} has ibnd={ibnd_arr[irow]:g}, "
                f"ik={ik_arr[irow]:g}; expected 0 <= ibnd < {nbands} and 0 <= ik < {nks}."
            )
        ik_arr = ik_arr.astype(int)
        ibnd_arr = ibnd_arr.astype(int)
        flat_indices = ibnd_arr * nks + ik_arr
        if np.unique(flat_indices).size != flat_indices.size:
            raise ValueError(f"Rate file '{filepath}' contains duplicate (ibnd, ik) entries.")

        if len(rate_columns) == 1:
            rate_ir[ibnd_arr, ik_arr] = data[:, rate_columns[0]]
        else:
            rate_ir[0, ibnd_arr, ik_arr] = data[:, rate_columns[0]]
            rate_ir[1, ibnd_arr, ik_arr] = data[:, rate_columns[1]]

        return rate_ir

    def _write_rate_file(self, filepath:str, weight_k:np.ndarray, enks:np.ndarray, rate:np.ndarray, with_header:bool=True, is_metal:bool=False):
        enks_shape = enks.shape
        nks = enks_shape[-1]
        nbands = enks_shape[-2]
        rate_shape = rate.shape
        out_case = len(enks_shape) + len(rate_shape)

        # Build (ibnd, ik) index grid: ibnd varies fastest (inner loop), ik slowest (outer loop)
        ik_all, ibnd_all = np.meshgrid(np.arange(nks), np.arange(nbands), indexing='ij')
        ik_flat = ik_all.ravel()
        ibnd_flat = ibnd_all.ravel()

        with open(filepath, 'w') as fout:
            if out_case == 4:
                if with_header:
                    if is_metal:
                        fout.write(f"#Fermi level = {self.efermi:.8f} Hartree\n")
                    else:
                        fout.write(f"#Fermi level = {self.efermi:.8f} Hartree    Density of carrier = {self.carrier_density:.8e} \n")
                    fout.write(f"#{'ik_ibz':<8}{'ibnd':<6}{'weight_k':<18}{'enk(Hartree)':<18}{'scattering_rate(Hartree)':<24}\n")
                mask = ~np.isinf(rate[ibnd_flat, ik_flat])
                iks, ibnds = ik_flat[mask], ibnd_flat[mask]
                lines = [f"{ik:<9}{ibnd:<6}{weight_k[ik]:<18.12f}{enks[ibnd, ik]:<18.12f}{rate[ibnd, ik]:<24.16e}"
                         for ik, ibnd in zip(iks, ibnds)]
                fout.write('\n'.join(lines))
                if lines:
                    fout.write('\n')
            elif out_case == 5:
                if with_header:
                    fout.write(
                        "# ik_ibz  ibnd  weight_k  enk(Hartree)  scattering_rate(Hartree)  "
                        "energy_loss_rate(Hartree^2)\n"
                    )
                mask = ~np.isinf(rate[0, ibnd_flat, ik_flat])
                iks, ibnds = ik_flat[mask], ibnd_flat[mask]
                lines = [f"{ik:<9}{ibnd:<6}{weight_k[ik]:<18.12f}{enks[ibnd, ik]:<18.12f}{rate[0, ibnd, ik]:<24.16e}{rate[1, ibnd, ik]:<24.16e}"
                         for ik, ibnd in zip(iks, ibnds)]
                fout.write('\n'.join(lines))
                if lines:
                    fout.write('\n')
            elif out_case == 6:
                if with_header:
                    if is_metal:
                        fout.write(f"#Fermi level = {self.efermi:.8f} Hartree\n")
                    else:
                        fout.write(f"#Fermi level = {self.efermi[0]:.8f} Hartree and {self.efermi[1]:.8f} Hartree   Density of carrier = {self.carrier_density[0]:.8e} and {self.carrier_density[1]:.8e}\n")
                    fout.write(
                        "# ik_ibz  ibnd  weight_k  enk[up](Hartree)  scatt_rate[up](Hartree)  "
                        "enk[down](Hartree)  scatt_rate[down](Hartree)\n"
                    )
                mask = ~np.isinf(rate[:, ibnd_flat, ik_flat]).all(axis=0)
                iks, ibnds = ik_flat[mask], ibnd_flat[mask]
                lines = [f"{ik:<9}{ibnd:<6}{weight_k[ik]:<18.12f}"
                         f"{enks[0, ibnd, ik]:<18.12f}"
                         f"{rate[0, ibnd, ik]:<24.16e}"
                         f"{enks[1, ibnd, ik]:<18.12f}"
                         f"{rate[1, ibnd, ik]:<24.16e}"
                         for ik, ibnd in zip(iks, ibnds)]
                fout.write('\n'.join(lines))
                if lines:
                    fout.write('\n')
            else:
                raise NotImplementedError("Cannot output the rate file.", '7101')

    def _write_scattering_channel(self, scat_channel:np.ndarray, output_dir:str="scattering_channels"):
        output_dir = os.path.join(self.outdir, output_dir)
        if self.rank == 0 and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.comm.Barrier()
        filename = f"epc_scat_channel_rank{self.rank}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, scat_channel)

    def _read_scattering_channel(self, folder_path):
        filename = f"epc_scat_channel_rank{self.rank}.npy"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[Rank {self.rank}] Cannot find scat_channel file: {filepath}")
        scat_channel = np.load(filepath, allow_pickle=True)
        return scat_channel

    def _get_epc_func(self):
        """Return the standard EPC scattering-rate method for the current spin mode."""
        if self.spinor_mode == 'collinear':
            return self.scattering_rate_cal_epc_collinear_sparse if self.read_sparse_grad_mat \
                else self.scattering_rate_cal_epc_collinear_dense
        else:
            return self.scattering_rate_cal_epc_sparse if self.read_sparse_grad_mat \
                else self.scattering_rate_cal_epc_dense

    def _get_scattering_rate_func(self, is_metal:bool, polar_split:str=None):
        """Select the scattering-rate kernel for the requested transport channel.

        Standard EPC kernels handle ``none`` and ``rmp``.  The pure polar channel
        uses a dedicated kernel because it contains only the long-range correction
        and has no short-range gradient contribution.
        """
        if not is_metal and polar_split == 'polar':
            if self.spinor_mode == 'collinear':
                return self.scattering_rate_cal_epc_polar_collinear
            return self.scattering_rate_cal_epc_polar
        return self._get_epc_func()

    def _dispatch_scattering_rate(self, k_grid:np.ndarray, q_grid:np.ndarray, bands_indices:np.ndarray,
                                   energy_ref, is_metal:bool, polar_split:str=None):
        """
        Dispatch the appropriate scattering rate calculation method.

        Selects the kernel based on spin mode, gradient representation, and polar
        channel. The pure polar channel is ERTA-only, while ``rmp`` uses the standard
        EPC kernel with ``is_rmp=True``. The standard IBTE path also writes its
        transition channels to disk.

        Args:
            k_grid (np.ndarray): useful IBZ k-points.
            q_grid (np.ndarray): full BZ q-points.
            bands_indices (np.ndarray): band index array.
            energy_ref: reference energy (Fermi level for metal, band edge for insulator).
            is_metal (bool): True for metal, False for insulator.
            polar_split (str or None): polar splitting mode (insulator phonon only).

        Returns:
            rate_useful (np.ndarray): scattering rates on useful k-points.
            eigen_local (np.ndarray): local eigenvalues.
            vnks_local (np.ndarray): local band velocities.
            fout_name (str): output filename for the rate file.
            scat_channel_local: scattering channel data (only for IBTE), or None.
        """
        col_suffix = '_collinear' if self.spinor_mode == 'collinear' else ''

        is_pure_polar = (not is_metal and polar_split == 'polar')
        is_rmp = (not is_metal and polar_split == 'rmp')
        if is_pure_polar and self.mob_level != 'erta':
            raise NotImplementedError(
                "The pure polar scattering channel currently supports only ERTA; "
                "IBTE requires transition channels that are not produced by the polar kernel."
            )

        rate_func = self._get_scattering_rate_func(is_metal=is_metal, polar_split=polar_split)
        if is_pure_polar:
            result = rate_func(k_grid, q_grid, bands_indices, energy_ref)
            rate_useful, eigen_local, vnks_local = result
            scat_channel_local = None
            fout_name = f'rate_nk_polar{col_suffix}.dat'
        else:
            result = rate_func(
                k_grid,
                q_grid,
                bands_indices,
                energy_ref,
                mob_level=self.mob_level,
                is_metal=is_metal,
                is_rmp=is_rmp,
            )
            if self.mob_level == 'ibte':
                rate_useful, eigen_local, vnks_local, scat_channel_local = result
                self._write_scattering_channel(scat_channel_local)
            else:
                rate_useful, eigen_local, vnks_local = result
                scat_channel_local = None
            if is_rmp:
                fout_name = f'rate_nk_rmp{col_suffix}.dat'
            else:
                fout_name = f'rate_nk{col_suffix}.dat'

        return rate_useful, eigen_local, vnks_local, fout_name, scat_channel_local

    def _read_and_broadcast_rate(self, path:str, label:str, nbands:int, nks_ibz:int):
        """
        Read a single rate file on rank 0 and broadcast to all ranks.

        Args:
            path (str): path to rate file.
            label (str): human-readable label for log messages.
            nbands (int): number of bands.
            nks_ibz (int): number of IBZ k-points.

        Returns:
            rate_ir (np.ndarray): rate array, broadcast to all ranks.
        """
        rate_ir = None
        if self.rank == 0:
            print(f"Reading {label} scattering rate from {path}")
            rate_ir = self._read_rate_file(path, [nbands, nks_ibz])
        rate_ir = self.comm.bcast(rate_ir, root=0)
        return rate_ir

    def _merge_rate_files(self, path_a:str, path_b:str, label_a:str, label_b:str, nbands:int, nks_ibz:int):
        """
        Read two rate files, check inf-mask consistency, merge by addition, and broadcast.

        Args:
            path_a (str): path to first rate file.
            path_b (str): path to second rate file.
            label_a (str): human-readable label for first file.
            label_b (str): human-readable label for second file.
            nbands (int): number of bands.
            nks_ibz (int): number of IBZ k-points.

        Returns:
            rate_ir (np.ndarray): merged rate array, broadcast to all ranks.
        """
        rate_ir = None
        if self.rank == 0:
            print(f"Reading {label_a} scattering rate from {path_a}")
            print(f"Reading {label_b} scattering rate from {path_b}")
            rate_a = self._read_rate_file(path_a, [nbands, nks_ibz])
            rate_b = self._read_rate_file(path_b, [nbands, nks_ibz])
            if not np.array_equal(np.isinf(rate_a), np.isinf(rate_b)):
                print(f"Warning: Please ensure the {label_a} scattering rate and {label_b} "
                      f"scattering rate are calculated using the same parameters, except for sampling.")
            rate_a[np.isinf(rate_a)] = 0.0
            rate_b[np.isinf(rate_b)] = 0.0
            rate_ir = rate_a + rate_b
            del rate_a, rate_b
            rate_ir[np.abs(rate_ir) < QTcts.TENPM160] = np.inf
        rate_ir = self.comm.bcast(rate_ir, root=0)
        return rate_ir

    def _compute_erta(self, eigen_local, vnks_local, rate_local, k_grid_local, nbands,
                      efermi, all_rots_cart, ibz_to_fbz, k_index_local,
                      sigma_unit, mob_unit_base, is_metal,
                      carrier_density=None, inv_cell=None):
        """
        Compute the ERTA (zero-order BTE) solution and print results.

        Args:
            eigen_local: local eigenvalues.
            vnks_local: local band velocities.
            rate_local: local scattering rates.
            k_grid_local: local k-points.
            nbands: number of bands.
            efermi: Fermi level (scalar for metal/non-collinear insulator, array(2) for collinear insulator).
            all_rots_cart, ibz_to_fbz, k_index_local: symmetry data.
            sigma_unit: conductivity unit conversion factor.
            mob_unit_base: mobility unit base factor.
            is_metal: True for conductivity, False for mobility.
            carrier_density: carrier density (insulator only).
            inv_cell: inverse cell volume (insulator only).

        Returns:
            F_0_local: local F_0 distribution function.
            sigma_serta: ERTA conductivity tensor.
            Seebeck_serta: Seebeck coefficient tensor.
            mobility_serta: ERTA mobility tensor (None for metals).
        """
        is_collinear = (self.spinor_mode == 'collinear')
        shape_33 = (2, 3, 3) if is_collinear else (3, 3)

        if k_grid_local.size > 0:
            nks_local = k_grid_local.shape[0]
            if is_collinear:
                F_0_local = np.zeros((2, nks_local, nbands, 3))
                sigma_temp = np.zeros((2, 3, 3))
                sigma_See_temp = np.zeros((2, 3, 3))
                for ispin in range(2):
                    ef = efermi if is_metal else efermi[ispin]
                    mdf = minus_dfermi(eigen_local[ispin] - ef, self.temperature)
                    F_0_local[ispin] = oe.contract('nk, kni, nk->kni', mdf, vnks_local[ispin], 1.0 / rate_local[ispin])
                    sigma_temp[ispin], sigma_See_temp[ispin] = self.get_conductivity_symm(
                        eigen_local[ispin] - ef, vnks_local[ispin], F_0_local[ispin], all_rots_cart, ibz_to_fbz, k_index_local)
            else:
                F_0_local = np.zeros((nks_local, nbands, 3))
                mdf = minus_dfermi(eigen_local - efermi, self.temperature)
                F_0_local = oe.contract('nk, kni, nk->kni', mdf, vnks_local, 1.0 / rate_local)
                sigma_temp, sigma_See_temp = self.get_conductivity_symm(
                    eigen_local - efermi, vnks_local, F_0_local, all_rots_cart, ibz_to_fbz, k_index_local)
        else:
            if is_collinear:
                F_0_local = np.zeros((2, 0, nbands, 3))
            else:
                F_0_local = np.zeros((0, nbands, 3))
            sigma_temp = np.zeros(shape_33)
            sigma_See_temp = np.zeros(shape_33)

        mobility_serta = None
        if is_metal:
            sigma_temp = sigma_temp * sigma_unit
            sigma_See_temp = sigma_See_temp * sigma_unit
        else:
            mobility_temp = sigma_temp * mob_unit_base
            if is_collinear:
                for ispin in range(2):
                    mobility_temp[ispin] = mobility_temp[ispin] / (QTcts.EV * carrier_density[ispin] * inv_cell) * (QTcts.BOHRtoCM ** 3)
            else:
                mobility_temp = mobility_temp / (QTcts.EV * carrier_density * inv_cell) * (QTcts.BOHRtoCM ** 3)

        sigma_serta = np.zeros(shape_33)
        sigma_See_serta = np.zeros(shape_33)
        if self.comm is not None:
            self.comm.Allreduce(sigma_temp, sigma_serta, op=MPI.SUM)
            self.comm.Allreduce(sigma_See_temp, sigma_See_serta, op=MPI.SUM)
        if not is_metal:
            mobility_serta = np.zeros(shape_33)
            if self.comm is not None:
                self.comm.Allreduce(mobility_temp, mobility_serta, op=MPI.SUM)
        Seebeck_serta = np.linalg.pinv(sigma_serta) @ sigma_See_serta / self.temperature / QTcts.HARTREEtoKELVIN * QTcts.HARTREEtoEV

        if not is_collinear and self.spinor_mode == 'none':
            if is_metal:
                sigma_serta = sigma_serta * 2
            else:
                mobility_serta = mobility_serta * 2

        if self.rank == 0:
            print('=' * 100)
            print(' ' * 32 + 'Energy Relaxation Time Approximation')
            print('=' * 100)
            primary = sigma_serta if is_metal else mobility_serta
            primary_label = 'Conductivity(Siemens/m)' if is_metal else 'Mobility(cm^2/V/s)'
            seebeck_label = 'Seebeck coefficient(V/K)' if is_metal else 'Seebeck(Volt/Kelvin)'
            if is_collinear:
                for ispin in range(2):
                    p_label = f"Conductivity[spin-{ispin+1}](Siemens/m)" if is_metal else f"Mobility[spin-{ispin+1}](cm^2/V/s)"
                    s_label = f"Seebeck coefficient[spin-{ispin+1}](V/K)" if is_metal else f"Seebeck[spin-{ispin+1}](Volt/Kelvin)"
                    print(f"{p_label}      x                y                z")
                    print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][0]))
                    print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][1]))
                    print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][2]))
                    print(f"{s_label}      x                y                z")
                    print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_serta[ispin][0]))
                    print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_serta[ispin][1]))
                    print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_serta[ispin][2]))
            else:
                print(f"{primary_label}           x                y                z")
                print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[0]))
                print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[1]))
                print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[2]))
                print(f"{seebeck_label}           x                y                z")
                print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_serta[0]))
                print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_serta[1]))
                print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_serta[2]))
            print('\n')
        sys.stdout.flush()
        return F_0_local, sigma_serta, Seebeck_serta, mobility_serta

    def _run_ibte_iterations(self, F_0_local, eigen_local, vnks_local, rate_useful,
                             k_grid_local, nbands, nks_ibz_usefull, efermi,
                             all_rots_cart, ibz_to_fbz, k_index_local,
                             rot_indices, unique_kpts, sl_start, sl_end,
                             sigma_unit, mob_unit_base,
                             is_metal, sigma_serta, mobility_serta=None,
                             scat_channel_local=None,
                             carrier_density=None, inv_cell=None):
        is_collinear = (self.spinor_mode == 'collinear')
        shape_33 = (2, 3, 3) if is_collinear else (3, 3)

        F_0 = self.comm.allgather(F_0_local)
        F_0 = np.concatenate(F_0, axis=1 if is_collinear else 0)
        F_in = F_0
        conv_in = sigma_serta.copy() if is_metal else mobility_serta.copy()

        cycle = 0
        error = QTcts.TENPP6
        while error > self.ibte_error and cycle < self.ibte_cycle:
            cycle += 1
            sigma_temp = np.zeros(shape_33)
            sigma_See_temp = np.zeros(shape_33)

            if is_collinear:
                if k_grid_local.size > 0:
                    F_out = np.zeros((2, nks_ibz_usefull, nbands, 3))
                    for ispin in range(2):
                        jspin = int(1) - ispin
                        F_out[ispin] = self.get_F_nk(F_in[ispin], rate_useful[ispin], scat_channel_local[ispin],
                                                        rot_indices, all_rots_cart, unique_kpts)
                else:
                    F_out = np.zeros((2, nks_ibz_usefull, nbands, 3))
            else:
                if k_grid_local.size > 0:
                    F_out = self.get_F_nk(F_in, rate_useful, scat_channel_local, rot_indices, all_rots_cart, unique_kpts)
                else:
                    F_out = np.zeros((nks_ibz_usefull, nbands, 3))

            if self.comm is not None:
                self.comm.Allreduce(MPI.IN_PLACE, F_out, op=MPI.SUM)
            F_out += F_0
            F_out_local = F_out[:, sl_start:sl_end] if is_collinear else F_out[sl_start:sl_end]

            if k_grid_local.size > 0:
                if is_collinear:
                    for ispin in range(2):
                        ef = efermi if is_metal else efermi[ispin]
                        sigma_temp[ispin], sigma_See_temp[ispin] = self.get_conductivity_symm(
                            eigen_local[ispin] - ef, vnks_local[ispin], F_out_local[ispin],
                            all_rots_cart, ibz_to_fbz, k_index_local)
                        if is_metal:
                            sigma_temp[ispin] *= sigma_unit
                            sigma_See_temp[ispin] *= sigma_unit
                else:
                    sigma_temp, sigma_See_temp = self.get_conductivity_symm(
                        eigen_local - efermi, vnks_local, F_out_local,
                        all_rots_cart, ibz_to_fbz, k_index_local)
                    if is_metal:
                        sigma_temp *= sigma_unit
                        sigma_See_temp *= sigma_unit

            if not is_metal:
                mobility_temp = sigma_temp * mob_unit_base
                if is_collinear:
                    for ispin in range(2):
                        mobility_temp[ispin] = mobility_temp[ispin] / (QTcts.EV * carrier_density[ispin] * inv_cell) * (QTcts.BOHRtoCM ** 3)
                else:
                    mobility_temp = mobility_temp / (QTcts.EV * carrier_density * inv_cell) * (QTcts.BOHRtoCM ** 3)

            sigma_out = np.zeros(shape_33)
            sigma_See_out = np.zeros(shape_33)
            if self.comm is not None:
                self.comm.Allreduce(sigma_temp, sigma_out, op=MPI.SUM)
                self.comm.Allreduce(sigma_See_temp, sigma_See_out, op=MPI.SUM)
            if not is_metal:
                mobility_out = np.zeros(shape_33)
                if self.comm is not None:
                    self.comm.Allreduce(mobility_temp, mobility_out, op=MPI.SUM)

            Seebeck_out = np.linalg.pinv(sigma_out) @ sigma_See_out / self.temperature / QTcts.HARTREEtoKELVIN * QTcts.HARTREEtoEV

            if not is_collinear and self.spinor_mode == 'none':
                if is_metal:
                    sigma_out = sigma_out * 2
                else:
                    mobility_out = mobility_out * 2

            conv_out = sigma_out if is_metal else mobility_out
            error = np.max(np.abs(conv_out - conv_in))

            if self.rank == 0:
                print('=' * 100)
                print('Iteration number: {:>4d}'.format(cycle))
                print('=' * 100)
                primary = sigma_out if is_metal else mobility_out
                primary_label = 'Conductivity(Siemens/m)' if is_metal else 'Mobility(cm^2/V/s)'
                seebeck_label = 'Seebeck coefficient(V/K)' if is_metal else 'Seebeck(Volt/Kelvin)'
                if is_collinear:
                    for ispin in range(2):
                        p_label = f"Conductivity[spin-{ispin+1}](Siemens/m)" if is_metal else f"Mobility[spin-{ispin+1}](cm^2/V/s)"
                        s_label = f"Seebeck coefficient[spin-{ispin+1}](V/K)" if is_metal else f"Seebeck[spin-{ispin+1}](Volt/Kelvin)"
                        print(f"{p_label}      x                y                z")
                        print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][0]))
                        print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][1]))
                        print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*primary[ispin][2]))
                        print(f"{s_label}      x                y                z")
                        print('            x            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_out[ispin][0]))
                        print('            y            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_out[ispin][1]))
                        print('            z            {:12.4e}      {:12.4e}      {:12.4e}'.format(*Seebeck_out[ispin][2]))
                else:
                    print(f"{primary_label}           x                y                z")
                    print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[0]))
                    print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[1]))
                    print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*primary[2]))
                    print(f"{seebeck_label}           x                y                z")
                    print('       x              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_out[0]))
                    print('       y              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_out[1]))
                    print('       z              {:12.4e}     {:12.4e}     {:12.4e}'.format(*Seebeck_out[2]))
                print(f"Error of this iteration: {error:.4e}\n")
            sys.stdout.flush()
            F_in = F_out
            conv_in = conv_out

        if self.rank == 0:
            print('=' * 100)
            if error <= self.ibte_error:
                print(' ' * 27 + f"Converged in {cycle} cycles with error = {error:.4e}")
            else:
                print(' ' * 18 + f"Warning: reached max iteration {self.ibte_cycle} with error = {error:.4e}")
            print('=' * 100)

    def _compute_eigen_velocity_local(self, k_grid_local:np.ndarray, nbands:int):
        """
        Compute local eigenvalues and band velocities for k-points assigned to this MPI rank.

        Args:
            k_grid_local (np.ndarray): k-points owned by this rank, shape (nk_local, 3). May have size 0.
            nbands (int): number of bands in self.bands_indices.

        Returns:
            eigen_local (np.ndarray): eigenvalues.
                Collinear: shape (2, nbands, nk_local). Non-collinear: shape (nbands, nk_local).
            vnks_local (np.ndarray): band velocities.
                Collinear: shape (2, nk_local, nbands, 3). Non-collinear: shape (nk_local, nbands, 3).
        """
        if k_grid_local.size > 0:
            eigen_local, eigen_vecs_local = self._elec_cal_torch(k_grid_local)
            if self.spinor_mode == 'collinear':
                eigen_local = np.stack((eigen_local[self.bands_indices], eigen_local[self.bands_indices+self.norbs]), axis=0)
                eigen_vecs_local = np.stack((eigen_vecs_local[:, self.bands_indices], eigen_vecs_local[:, self.bands_indices+self.norbs]), axis=0)
            else:
                eigen_local = eigen_local[self.bands_indices]
                eigen_vecs_local = eigen_vecs_local[:, self.bands_indices]
            if self.spinor_mode == 'collinear':
                vnks_local = np.zeros((2, len(k_grid_local), nbands, 3))
                vnks_local[0] = self._vel_nk_cal_from_HS(eigen_local[0], eigen_vecs_local[0], k_grid_local, collinear_part_index=0)
                vnks_local[1] = self._vel_nk_cal_from_HS(eigen_local[1], eigen_vecs_local[1], k_grid_local, collinear_part_index=1)
            else:
                vnks_local = self._vel_nk_cal_from_HS(eigen_local, eigen_vecs_local, k_grid_local)
        else:
            if self.spinor_mode == 'collinear':
                eigen_local = np.empty((2, nbands, 0))
                vnks_local = np.empty((2, 0, nbands, 3))
            else:
                eigen_local = np.empty((nbands, 0))
                vnks_local = np.empty((0, nbands, 3))
        return eigen_local, vnks_local
