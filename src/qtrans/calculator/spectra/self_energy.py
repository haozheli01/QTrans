"""Self-energy data processing and output helpers."""

from pathlib import Path
from qtrans.core.constants import QTcts
from qtrans.io.filesystem import check_path_exist
from qtrans.numerics.gaussian import w0gauss
import numpy as np
import os


class _SelfEnergyMixin:
    def _self_energy_symmetrize(self, eigen_values:np.ndarray, self_energies:np.ndarray, tolerance:float=QTcts.TENPM6):
        """
        Symmetrize self-energies at each k-point by averaging over nearly-degenerate energies.
        
        Parameters:
        -----------
        eigen_values : ndarray of shape (nks, nbands)
            Eigen energy at each k-point.
        self_energies : ndarray of shape (nks, nbands, nomegas)
            Self energies.
        tolerance : float
            Threshold for determining degeneracy.
        
        Returns:
        --------
        self_energies_symm : ndarray of same shape as self_energies
            Symmetrized self energies.
        """

        se_shape = self_energies.shape
        ndim = len(se_shape)
        nks = se_shape[0]
        nbands = se_shape[1]
        self_energies_symm = np.empty_like(self_energies)

        if ndim == 2:
            for ik in range(nks):
                eigen_k = eigen_values[ik]
                used = np.zeros(nbands, dtype=bool)
                for iband in range(nbands):
                    if used[iband]:
                        continue
                    degenerate = np.abs(eigen_k - eigen_k[iband]) < tolerance
                    used[degenerate] = True
                    count = np.count_nonzero(degenerate)
                    if count > 0:
                        avg = np.mean(self_energies[ik, degenerate], axis=0)
                        self_energies_symm[ik, degenerate] = avg
                    else:
                        self_energies_symm[ik, iband] = self_energies[ik, iband]
        elif ndim == 3:
            for ik in range(nks):
                eigen_k = eigen_values[ik]
                used = np.zeros(nbands, dtype=bool)
                for iband in range(nbands):
                    if used[iband]:
                        continue
                    degenerate = np.abs(eigen_k - eigen_k[iband]) < tolerance
                    used[degenerate] = True
                    count = np.count_nonzero(degenerate)
                    if count > 0:
                        avg = np.mean(self_energies[ik, degenerate], axis=0)
                        self_energies_symm[ik, degenerate] = avg[None, :]
                    else:
                        self_energies_symm[ik, iband] = self_energies[ik, iband]
        return self_energies_symm

    def _output_spectrum_func(self, type_prefix:str, xp_dist:np.ndarray, omega_list:np.ndarray, spectrum_func:np.ndarray):
        sf_shape = spectrum_func.shape
        nps, nomegas = sf_shape[-2], sf_shape[-1]
        ip_col = np.repeat(np.arange(nps), nomegas)
        xp_col = np.repeat(xp_dist, nomegas)
        omega_eV_col = np.tile(omega_list * QTcts.HARTREEtoEV, nps)
        if type_prefix == 'phonon':
            spectrum_val_col = spectrum_func.flatten() / QTcts.HARTREEtoEV
            data_to_save = np.column_stack((ip_col, xp_col, omega_eV_col, spectrum_val_col))
            header = 'ik          xk      omega(eV)      spectrum(1/eV)'
            fmt_str = '%d    %12.6f    %12.6f    %12.6f'
        if type_prefix == 'electron':
            if self.spinor_mode == 'collinear':
                spectrum_val_col = spectrum_func.reshape(2, -1) / QTcts.HARTREEtoEV
                data_to_save = np.column_stack((ip_col, xp_col, omega_eV_col, spectrum_val_col[0], spectrum_val_col[1]))
                header = 'ik          xk      omega(eV)      spectrum-up(1/eV)      spectrum-down(1/eV)'
                fmt_str = '%d    %12.6f    %12.6f    %12.6f    %12.6f'
            else:
                spectrum_val_col = spectrum_func.flatten() / QTcts.HARTREEtoEV
                data_to_save = np.column_stack((ip_col, xp_col, omega_eV_col, spectrum_val_col))
                header = 'ik          xk      omega(eV)      spectrum(1/eV)'
                fmt_str = '%d    %12.6f    %12.6f    %12.6f'
        filepath = os.path.join(self.outdir, f"{type_prefix}.spectrum")
        try:
            np.savetxt(filepath, data_to_save, fmt=fmt_str, header=header, comments='')
        except Exception as e:
            raise IOError(f"Error(6011): Failed to write {filepath}: {e}")

    def _read_self_energy_file(self, folder_path:str, type_prefix:str='electron'):
        nks = len(self.high_symmetry_k_vecs)
        is_exist = check_path_exist(folder_path, 'dir')
        if not is_exist:
            raise IOError(f"Error(6014): Failed to read self energy from {folder_path} : Folder not exiss.")
        nbands = len(self.bands_indices)
        omega_list = np.arange(self.SE_erange[0], self.SE_erange[1]+self.SE_erange[2]/10.0, self.SE_erange[2])
        nomegas = len(omega_list)
        # self_energy: (2, nks, nbands, nomegas)
        self_energy = np.zeros((2, nks, nbands, nomegas))
        filepath = ""
        spin_str = ['up', 'down']
        try:
            for ik in range(nks):
                for ispin in range(2):
                    filepath = os.path.join(folder_path, f"{type_prefix}-se-{ik+1}-{spin_str[ispin]}.dat")
                    data = np.loadtxt(filepath)
                    nomegas_data, nbands_data = data.shape
                    se_real = np.transpose(data[:, 1::2])
                    se_imag = np.transpose(data[:, 2::2])
                    nbands_data = (nbands_data - 1) // 2
                    if nbands != nbands_data:
                        raise IOError(f"Different number of bands.")
                    if nomegas != nomegas_data:
                        raise IOError(f"Different number of energy points.")
                    self_energy[ispin, ik] = se_real + 1.0j*se_imag
        except Exception as e:
            raise IOError(f"Error(6014): Failed to read self energy from {filepath} : {e}.")
            
        return omega_list, self_energy

    def _write_self_energy_file(self, filepath:str, header:str, fmt:str, self_energy:np.ndarray, omega_list:np.ndarray=None, k_dist_list:np.ndarray=None):
        if omega_list is None:
            # self_energy: shape: (nks, nbands)
            nks, nbands = self_energy.shape
            data_to_save = np.empty((nks, 2 * nbands+1))
            data_to_save[:, 0] = k_dist_list
            for iband in range(nbands):
                data_to_save[:, 2*iband+1]   = self_energy[:, iband].real
                data_to_save[:, 2*iband+2] = self_energy[:, iband].imag
        else:
            nomegas = self_energy.shape[-1]
            real_parts = self_energy.real
            imag_parts = self_energy.imag
            se_re_im_interleaved = np.stack((real_parts, imag_parts), axis=-1).transpose(1, 0, 2).reshape(nomegas, -1)
            data_to_save = np.column_stack((omega_list*QTcts.HARTREEtoEV, se_re_im_interleaved))
        try:
            np.savetxt(filepath, data_to_save, fmt=fmt, header=header)
        except Exception as e:
            raise IOError(f"Error(6012): Failed to write {filepath}: {e}")

    def _output_self_energy(self, type_prefix:str, p_grid:np.ndarray, self_energies:np.ndarray, omega_list:np.ndarray=None):
        if omega_list is None:
            self._output_self_energy_point(type_prefix, p_grid, self_energies)
        else:
            self._output_self_energy_all(type_prefix, p_grid, omega_list, self_energies)

    def _output_self_energy_point(self, type_prefix:str, p_grid:np.ndarray, self_energies:np.ndarray):
        #  self_energies: shape: (2, nks, nbands)
        se_shape = self_energies.shape
        _, nbands = se_shape[-2], se_shape[-1]
        # Dynamically build header and format string
        if type_prefix == 'phonon':
            se_symbol = 'Pi'
        elif type_prefix == 'electron':
            se_symbol = 'Sigma'
            spin_dict = {
                0: 'up',
                1: 'down'
            }
        elif type_prefix == 'electron-Z':
            se_symbol = 'Z'
            spin_dict = {
                0: 'up',
                1: 'down'
            }
        band_headers = []
        band_fmts = []
        for iband in range(nbands):
            band_headers.append(f'{se_symbol}_{iband+1}_real(eV)')
            band_headers.append(f'{se_symbol}_{iband+1}_imag(eV)')
            band_fmts.append('%12.6f')
            band_fmts.append('%12.6f')
        header = 'xk    ' + '    '.join(band_headers)
        fmt_str = '%12.6f    ' + '    '.join(band_fmts)
        self_energies = self_energies * QTcts.HARTREEtoEV
        if type_prefix == 'phonon':
            filepath = f"{type_prefix}-self-energy.dat"
            self._write_self_energy_file(filepath, header, fmt_str, self_energies, k_dist_list=self.high_symmetry_k_dist)
        elif type_prefix == 'electron':
            if self.spinor_mode == 'collinear':
                for ispin in range(2):
                    filepath = f"{type_prefix}-self-energy-{spin_dict[ispin]}.dat"
                    self._write_self_energy_file(filepath, header, fmt_str, self_energies[ispin], k_dist_list=self.high_symmetry_k_dist)
            else:
                filepath = f"{type_prefix}-self-energy.dat"
                self._write_self_energy_file(filepath, header, fmt_str, self_energies, k_dist_list=self.high_symmetry_k_dist)
        elif type_prefix == 'electron-Z':
            if self.spinor_mode == 'collinear':
                for ispin in range(2):
                    filepath = f"{type_prefix[:-2]}-renorm-factor-{spin_dict[ispin]}.dat"
                    self._write_self_energy_file(filepath, header, fmt_str, self_energies[ispin], k_dist_list=self.high_symmetry_k_dist)
            else:
                filepath = f"{type_prefix[:-2]}-renorm-factor.dat"
                self._write_self_energy_file(filepath, header, fmt_str, self_energies, k_dist_list=self.high_symmetry_k_dist)

    def _output_self_energy_all(self, type_prefix:str, p_grid:np.ndarray, omega_list:np.ndarray, self_energies:np.ndarray):
        # p_grid is fractional
        se_shape = self_energies.shape
        nps, nbands = se_shape[-3], se_shape[-2]
        se_folder_path = Path(self.outdir) / 'self_energy'
        try:
            se_folder_path.mkdir(parents=False, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error(7002): Failed to create directory {se_folder_path}: {e}")

        # Dynamically build header and format string
        if type_prefix == 'phonon':
            se_symbol = 'Pi'
        elif type_prefix == 'electron':
            se_symbol = 'Sigma'
            spin_dict = {
                0: 'up',
                1: 'down'
            }
        band_headers = []
        band_fmts = []
        for iband in range(nbands):
            band_headers.append(f'{se_symbol}_{iband+1}_real(eV)')
            band_headers.append(f'{se_symbol}_{iband+1}_imag(eV)')
            band_fmts.append('%12.6f')
            band_fmts.append('%12.6f')
        header = 'omega(eV)    ' + '    '.join(band_headers)
        fmt_str = '%12.6f    ' + '    '.join(band_fmts)

        for ip in range(nps):
            if type_prefix == 'phonon':
                se_this = self_energies[ip] * QTcts.HARTREEtoEV
                filepath = se_folder_path / f"{type_prefix}-se-{ip+1}.dat"
                header_this = "point = {:10.5f} {:10.5f} {:10.5f}\n".format(*p_grid[ip]) + header
                self._write_self_energy_file(filepath, header_this, fmt_str, se_this, omega_list=omega_list)
            elif type_prefix == 'electron':
                if self.spinor_mode == 'collinear':
                    for ispin in range(2):
                        se_this = self_energies[ispin, ip] * QTcts.HARTREEtoEV
                        filepath = se_folder_path / f"{type_prefix}-se-{ip+1}-{spin_dict[ispin]}.dat"
                        header_this = "point = {:10.5f} {:10.5f} {:10.5f}    {}\n".format(*p_grid[ip], f"spin-{spin_dict[ispin]}") + header
                        self._write_self_energy_file(filepath, header_this, fmt_str, se_this, omega_list=omega_list)
                else:
                    se_this = self_energies[ip] * QTcts.HARTREEtoEV
                    filepath = se_folder_path / f"{type_prefix}-se-{ip+1}.dat"
                    header_this = "point = {:10.5f} {:10.5f} {:10.5f}\n".format(*p_grid[ip]) + header
                    self._write_self_energy_file(filepath, header_this, fmt_str, se_this, omega_list=omega_list)
