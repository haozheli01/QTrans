"""Mechanically extracted QTrans methods."""

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from qtrans.core.constants import QTcts
from qtrans.core.logging import log
from qtrans.io.filesystem import make_directory
from qtrans.numerics.kpoints import kpoints_generator
from typing import List
import io
import numpy as np
import os


class _OutputMixin:
    def _get_hsk_path(self, nks_path, hsk_points:List[List[float]]=None, hsk_labels:List[str]=None):
        """
        Get the q points path.

        Args:
            hsk_points (list or np.ndarray): List of special q points in fractional coordinate.
            nks_path (list or int): Number of q points between. 
            hsk_labels (list): The name of special q points.
            automatic (bool): If true, automatically generate the high symmetry points.
        Returns:
            q_path (np.ndarray): List of q points path.
        """
        kpts=kpoints_generator(dim_k=3, lat=self.graph_data.latt)
        if self.high_symmetry_auto:
            struct = Structure(lattice=self.graph_data.latt*QTcts.BOHRtoANG,
                species=[Element.from_Z(k).symbol for k in self.graph_data.species],
                coords=self.graph_data.pos*QTcts.BOHRtoANG, coords_are_cartesian=True)
            try:
                kpath_seek = KPathSeek(structure=struct)
            except:
                raise RuntimeError("Cannot automatically generate k path for this structure.", '1011')
            hsk_labels = []
            for lbs in kpath_seek.kpath['path']:
                hsk_labels += lbs
            log('Automatically generating high-symmetry points..', rank=self.rank)

            # remove adjacent duplicates   
            res = [hsk_labels[0]]
            [res.append(x) for x in hsk_labels[1:] if x != res[-1]]
            hsk_labels = res

            hsk_points = [kpath_seek.kpath['kpoints'][k] for k in hsk_labels]
            hsk_labels_plot = [rf'${lb}$' for lb in hsk_labels]

        try:
            hsk_labels_plot = [rf'${lb}$' for lb in hsk_labels]
            k_vec, k_dist, k_node = kpts.k_path(hsk_points, nks_path, report=False)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
        except:
            raise RuntimeError("kpoints_generator failed.", '1012')
        info_str = '\n'.join(["{:<6s}: {:6.4f}    {:6.4f}    {:6.4f}".format(hsk_labels[i], *hsk_points[i]) for i in range(len(hsk_labels))])
        log("High-Symmetry Points:", rank=self.rank)
        log(info_str, rank=self.rank)
        return k_vec, k_dist, k_node, hsk_labels_plot

    def _read_dispersion(self, filepath:str):
        nks = len(self.high_symmetry_k_vecs)
        try:
            data = np.loadtxt(filepath)
        except Exception as e:
            raise IOError(f"Error(6013): Failed to read band dispersion from {filepath} : e")
        nks_data = np.unique(data[:,0])
        if nks != nks_data:
            raise RuntimeError("Error(7005): The number of k-points in the provided band dispersion does not match nks_path in the dispersion block.")
        enks:np.ndarray = data[:,1:]
        ndim = data.shape[1]
        if self.spinor_mode == 'none' and ndim == 1:
            enks = enks.reshape(-1, nks) # (nbands, nks)
            if self.rank == 0:
                print(f"Reading band dispersion for non-magnetic system from {filepath}")
                print(f"{enks.shape[0]} bands are readed.")
        elif self.spinor_mode == 'collinear' and ndim == 2:
            enks = enks.reshape(-1, nks, 2) # (nbands, nks, 2)
            if self.rank == 0:
                print(f"Reading band dispersion for collinear system from {filepath}")
                print(f"{enks.shape[0]} bands are readed.")
            enks = np.transpose(enks, (2, 0, 1)) # (2, nbands, nks)
        else:
            raise NotImplementedError(f"The function _read_dispersion is not implemented for spinor_mode {self.spinor_mode}")
        return enks

    def _output_write_band_format(self, fout:io.TextIOWrapper, data:np.ndarray):
        # data: shape (nbands, ncols-1, nks) or (nbands, nks)
        data_shape = data.shape
        nbands, nks = data_shape[0], data_shape[-1]
        data = data.reshape(nbands, -1, nks)
        ncols = data.shape[1] + 1
        fmt='    '.join(["%.10f"]*ncols)
        for iband in range(nbands):
            output_data = np.column_stack((self.high_symmetry_k_dist, data[iband].T))
            np.savetxt(fout, output_data, fmt=fmt)
            fout.write("\n")

    def _output_dispersion(self, data:np.ndarray, mode:str, infos=np.ndarray):
        if mode == 'band':
            # data: shape (2, nbands, nks) or (nbands, nks)
            data = data * QTcts.HARTREEtoEV
            data_shape = data.shape
            nbands = data_shape[-2]
            if len(data_shape) == 3:
                data = np.transpose(data, (1,0,2))
            filename = os.path.join(self.outdir, "bands.dispersion")
            with open(filename, 'w') as fout:
                fout.write(f"# k_label: {' '.join(self.high_symmetry_labels)}\n")
                fout.write(f"# k_node: {'  '.join(f'{k:.10f}' for k in self.high_symmetry_k_nodes)}\n")
                self._output_write_band_format(fout, data)
        elif mode == 'phonon':
            # data: shape (nbranches, nqs)
            data = data.T * QTcts.HARTREEtoMEV
            filename = os.path.join(self.outdir, "phbands.dispersion")
            with open(filename, 'w') as fout:
                fout.write(f"# q_label: {' '.join(self.high_symmetry_labels)}\n")
                fout.write(f"# q_node: {'  '.join(f'{round(k, 10)}' for k in self.high_symmetry_k_nodes)}\n")
                self._output_write_band_format(fout, data)
        elif mode == 'epc':
            freq_grid, _ = self._phonon_cal(self.high_symmetry_k_vecs)  # (nqs, nbranches)
            freq_grid = np.where(freq_grid > 0.0, freq_grid, 0.0) # to avoid imaginary frequency
            # maybe phonon calculations are repeated ...........
            factor = np.sqrt(2*self.atomic_mass.sum() * freq_grid)  # shape (nqs, nbranches)

            if self.spinor_mode == 'collinear':
                _, _, nbands_ini, nbands_fin, nbranches = data.shape
            else:
                _, nbands_ini, nbands_fin, nbranches = data.shape
            # data: shape (nbranches, 2, nqs, nbands_ini, nbands_fin) or (nbranches, nqs, nbands_ini, nbands_fin)
            data = np.moveaxis(data, -1, 0)
            # data_ave: shape (nbranches, 2, nqs) or (nbranches, nqs)
            data_ave = np.sqrt(np.sum(np.sum(np.power(data, 2), axis=-1), axis=-1) / nbands_ini / nbands_fin)  
            
            if self.spinor_mode == 'collinear':
                def_pot = np.einsum("qv, vsq -> sqv", factor, data_ave)  # shape (spins, nqs, nbranches)
            else:
                def_pot = np.einsum("qv, vq -> qv", factor, data_ave)    # shape (nqs, nbranches)

            data_ave = data_ave * QTcts.HARTREEtoMEV
            def_pot = def_pot * QTcts.HARTREEtoEV / QTcts.BOHRtoANG

            filename_ave = os.path.join(self.outdir, "epc-ave.dispersion")
            with open(filename_ave, 'w') as fout:
                fout.write(f"# q_label: {' '.join(self.high_symmetry_labels)}\n")
                fout.write(f"# q_node: {'  '.join(f'{round(k, 10)}' for k in self.high_symmetry_k_nodes)}\n")
                # self._output_write_band_format(fout, data_ave)
                nqs = def_pot.shape[-2]
                nbranches = def_pot.shape[-1]
                
                if self.spinor_mode == "collinear":
                    fout.write("#  iq    q_path  ibranch   frequency(meV)   |g|_up(meV)    |g|_down(meV)    D_up(eV/Ang)     D_down(eV/Ang) \n")
                    output_data = np.zeros((nqs*nbranches, 8))
                    idx = 0
                    for iq in range(nqs):
                        q_path_val = self.high_symmetry_k_dist[iq]
                        for ibranch in range(nbranches):
                            freq_val = freq_grid[iq, ibranch] * QTcts.HARTREEtoMEV
                            g_up = data_ave[ibranch, 0, iq] 
                            g_down = data_ave[ibranch, 1, iq]
                            D_up = def_pot[0, iq, ibranch]
                            D_down = def_pot[1, iq, ibranch]
                            output_data[idx] = [iq, q_path_val, ibranch+1, freq_val, g_up, g_down, D_up, D_down]
                            idx += 1
                    
                    np.savetxt(fout, output_data, fmt="%5d  %10.5f  %5d    %15.10f    %15.10f    %15.10f    %15.10f    %15.10f")
                    fout.write("\n")

                else:
                    fout.write("#  iq    q_path  ibranch   frequency(meV)    |g|(meV)     Deform. pot.(eV/Ang) \n")
                    output_data = np.zeros((nqs*nbranches, 6))
                    idx = 0
                    for iq in range(nqs):
                        q_path_val = self.high_symmetry_k_dist[iq]
                        for ibranch in range(nbranches):
                            freq_val = freq_grid[iq, ibranch] * QTcts.HARTREEtoMEV
                            g_val = data_ave[ibranch, iq] 
                            D_val = def_pot[iq, ibranch]
                            output_data[idx] = [iq, q_path_val, ibranch+1, freq_val, g_val, D_val]
                            idx += 1
                    
                    np.savetxt(fout, output_data, fmt="%5d  %10.5f  %5d    %15.10f    %15.10f    %15.10f")
                    fout.write("\n")
            
            output_folder = os.path.join(self.outdir, 'dispersions')
            make_directory(output_folder)
            for iband, band_ini in enumerate(self.dispersion_select_index[0]):
                for jband, band_fin in enumerate(self.dispersion_select_index[1]):
                    filename = os.path.join(output_folder, f"epc-{band_ini+1}-{band_fin+1}.dispersion")    
                    with open(filename, 'w') as fout:
                        fout.write(f"# q_label: {' '.join(self.high_symmetry_labels)}\n")
                        fout.write(f"# q_node: {'  '.join(f'{round(k, 10)}' for k in self.high_symmetry_k_nodes)}\n")
                        self._output_write_band_format(fout, data[..., iband, jband])
        elif mode == 'nesting':
            # data: shape: if collinear (2, nqs), else (nqs, )
            data = data[None,:] / QTcts.HARTREEtoEV / QTcts.HARTREEtoEV
            filename = os.path.join(self.outdir, "nesting.dispersion")
            with open(filename, 'w') as fout:
                fout.write(f"# q_label: {' '.join(self.high_symmetry_labels)}\n")
                fout.write(f"# q_node: {'  '.join(f'{round(k, 10)}' for k in self.high_symmetry_k_nodes)}\n")
                self._output_write_band_format(fout, data)

    def _output_lambda(self, data:np.ndarray):
        # data: shape: (2, nqs, nbranches)
        filename = "lambda_qv.dat"
        filepath = os.path.join(self.outdir, filename)
        with open(filepath, 'w') as fout:
            self._output_write_band_format(fout, data.transpose(2, 0, 1))
