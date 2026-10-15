"""Electronic and phonon dispersion calculation workflows."""

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from qtrans.calculator.spectra.spectrum import spectrum_function_cal
from qtrans.calculator.spectra.modes import preflight_mode_output, write_mode_output
from qtrans.core.constants import QTcts
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import gather_to_rank_root
from qtrans.parallel.mpi import split_into_rank
import numpy as np
import os


class _DispersionMixin:
    def _cal_nesting_function(self, k_grid:np.ndarray, q_grid:np.ndarray, enks:np.ndarray):
        # q: frac -> cart
        q_grid = self._frac2car(q_grid)
        nqs = len(q_grid)

        if self.spinor_mode == 'collinear':
            enks = np.stack((enks[self.bands_indices], enks[self.bands_indices+self.norbs]), axis=0) # (2, nbands, nks)
            nesting_func = np.zeros((2, nqs))
            delta_nks = w0gauss((enks - self.efermi) * self.inv_smeark) * self.inv_smeark
            delta_sum_nks = np.sum(delta_nks, axis=1)   # shape: (2, nks)
            for iq, q in enumerate(q_grid):
                kpq_grid = k_grid + q
                eig_kpq_all, _ = self._elec_cal_torch(kpq_grid)
                eig_kpq_all = np.stack((eig_kpq_all[self.bands_indices], eig_kpq_all[self.bands_indices+self.norbs]), axis=0)  # shape: (2, nbands, nks)
                delta_mkq = w0gauss((eig_kpq_all - self.efermi) * self.inv_smeark) * self.inv_smeark
                delta_sum_mkqs = np.sum(delta_mkq, axis=1)   # shape: (2, nks)
                nesting_func[:, iq] = np.einsum('sk, sk, k->s', delta_sum_nks, delta_sum_mkqs, self.weight_k)
        else:
            enks = enks[self.bands_indices]
            nesting_func = np.zeros(nqs)
            delta_nks = w0gauss((enks - self.efermi) * self.inv_smeark) * self.inv_smeark
            delta_sum_nks = np.sum(delta_nks, axis=0)   # shape: (nks, )
            for iq, q in enumerate(q_grid):
                kpq_grid = k_grid + q
                eig_kpq_all, _ = self._elec_cal_torch(kpq_grid)
                eig_kpq_all = eig_kpq_all[self.bands_indices]  # shape: (nbands, nks)
                delta_mkq = w0gauss((eig_kpq_all - self.efermi) * self.inv_smeark) * self.inv_smeark
                delta_sum_mkqs = np.sum(delta_mkq, axis=0)   # shape: (nks, )
                nesting_func[iq] = np.sum(delta_sum_nks * delta_sum_mkqs * self.weight_k)
        return nesting_func

    def plot_band(self):
        """
        Calculate the electron energy for a special k path.

        Args:

        Returns:
        """
        if self.band_grid_mode:
            return self._band_cal_on_grid()
        if self.disp_renormalize:
            mode_result = {} if getattr(self, 'SE_mode_resolved', False) else None
            if mode_result is not None:
                preflight_mode_output(self)
            self.cal_mode = 'self-energy'
            # q_grid_ibz, self.weight_q = self._get_ir_reciprocal_mesh(self.q_size, auxiliary_info=False, return_frac=True)
            q_grid_fbz = self._get_monkhorst_pack(self.q_size, return_frac=True)
            self.weight_q = np.ones(len(q_grid_fbz)) / len(q_grid_fbz)
            # for solving fermi level only
            k_grid_ibz, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
            if self.spinor_mode == 'none':
                self.weight_k *= 2.0
            nqs_fbz = self.q_size[0] * self.q_size[1] * self.q_size[2]
            # nqs_ibz = len(q_grid_ibz)
            nks = len(self.high_symmetry_k_vecs)
            nbands = len(self.bands_indices)
            
            # check the system is metal or insulator
            k_grid_local = split_into_rank(k_grid_ibz, 0, self.rank, self.rank_size)
            if k_grid_local.size > 0:
                # enks_local: # shape if collinear (2*norbs, nks_local), else (norbs, nks_local)
                enks_local, _ = self._elec_cal_torch(k_grid_local)
            else:
                if self.spinor_mode == 'collinear':
                    enks_local = np.empty((2*self.norbs, 0))
                else:
                    enks_local = np.empty((self.norbs, 0))
            enks_all = gather_to_rank_root(enks_local, self.comm)      
            if self.rank == 0:
                # enks: # shape if collinear (2*norbs, nks), else (norbs, nks)
                enks = np.concatenate(enks_all, axis=-1)
                num_electron_include = self.graph_data.num_electrons - self.doping_charge
                self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
                is_metal = self._get_is_metal(enks, self.efermi)
                if is_metal:
                    print("The system is a metal.")
                else:
                    print("The system is a insulator.")
                ele_data_tmp = [self.efermi, is_metal]
            else:
                ele_data_tmp = None
            del enks_local, enks_all, k_grid_local
            ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
            self.efermi, is_metal = ele_data_tmp[0], ele_data_tmp[1]
            self.comm.Barrier()

            if is_metal:
                if self.rank == 0:
                    # here band_edge_energy is the fermi level of the system.
                    print("Fermi energy = {:10.6f} eV.".format(self.efermi * QTcts.HARTREEtoEV))
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
                        # If the system is collinear, wo only focus on the spin-up states of all |nk> states
                        enks = enks[self.bands_indices] # enks: shape (nbands, nks_ibz)
                    if self.spinor_mode == 'collinear':
                        self.efermi = np.zeros(2, dtype=float)
                        self.carrier_density = np.zeros(2, dtype=float)
                        for ispin in range(2):
                            self.efermi[ispin], self.carrier_density[ispin], _ = self._get_fermi_level_insulator(enks[ispin], iband_edge)
                            print("=" * 100)
                            if self.system_2D:
                                print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-2) for spin {}.".format(
                                    self.efermi[ispin] * QTcts.HARTREEtoEV, self.carrier_density[ispin] * self.inv_cell / (QTcts.BOHRtoCM ** 2), ispin+1
                                    ))
                            else:
                                print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-3) for spin {}.".format(
                                    self.efermi[ispin] * QTcts.HARTREEtoEV, self.carrier_density[ispin] * self.inv_cell / (QTcts.BOHRtoCM ** 3), ispin+1
                                    ))
                            print("=" * 100,'\n')
                    else:
                        self.efermi, self.carrier_density, _ = self._get_fermi_level_insulator(enks, iband_edge)
                        del enks
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
                    ele_data_tmp = [self.efermi, self.carrier_density]
                else:
                    ele_data_tmp = None
                ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
                self.efermi, self.carrier_density = ele_data_tmp[0], ele_data_tmp[1]
            self._print_mesh_mpi_check(nqs_fbz, nqs_fbz, nqs_fbz, nks, reduced_type='q')
            enks, omega_list, self_energies, spectrum_func = spectrum_function_cal(
                self, self.high_symmetry_k_vecs, q_grid_fbz, is_metal=is_metal,
                mode_result=mode_result)
            if mode_result is not None:
                write_mode_output(self, self.high_symmetry_k_vecs, q_grid_fbz,
                                  enks, omega_list, self_energies, mode_result)
            if self.rank == 0:
                if self.spinor_mode == 'collinear':
                    enks = np.transpose(enks, (0, 2, 1))
                else:
                    enks = np.transpose(enks)
                print("Outputing electron spectrum...\n")
                print("The electron energies are referenced to the Fermi level.")
                self._output_spectrum_func('electron', self.high_symmetry_k_dist, omega_list, spectrum_func)
                self._output_self_energy('electron', self.high_symmetry_k_vecs, self_energies, omega_list=omega_list)
        elif self.disp_nesting:
            k_grid_ibz, self.weight_k, k_grid_fbz, k_grid_maps = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=True)
            nks_fbz = len(k_grid_fbz)
            if self.spinor_mode == 'none':
                self.weight_k *= 2.0
            # get fermi energy
            k_grid_ibz_local = split_into_rank(k_grid_ibz, 0, self.rank, self.rank_size)
            k_grid_fbz_local, split_indices_fbz = split_into_rank(k_grid_fbz, 0, self.rank, self.rank_size, return_indices=True)
            if k_grid_ibz_local.size > 0:
                # enks_local: # shape if collinear (2*norbs, nks_local), else (norbs, nks_local)
                enks_local, _ = self._elec_cal_torch(k_grid_ibz_local)
            else:
                if self.spinor_mode == 'collinear':
                    enks_local = np.empty((2*self.norbs, 0))
                else:
                    enks_local = np.empty((self.norbs, 0))
            enks_all = gather_to_rank_root(enks_local, self.comm)      
            if self.rank == 0:
                # enks: # shape if collinear (2*norbs, nks_ibz), else (norbs, nks_ibz)
                enks = np.concatenate(enks_all, axis=-1)
                num_electron_include = self.graph_data.num_electrons - self.doping_charge
                self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
                is_metal = self._get_is_metal(enks, self.efermi)
                if is_metal:
                    print("Fermi energy = {:10.6f} eV.".format(self.efermi * QTcts.HARTREEtoEV))
                else:
                    raise RuntimeError("Error(7006): The nesting function can only be calculated for metallic systems.")
                ele_data_tmp = [self.efermi]
                enks = enks[:, k_grid_maps]
                enks_all = np.split(enks, split_indices_fbz, axis=1)
            else:
                ele_data_tmp = None
                enks_all = None
            del enks_local, k_grid_ibz_local
            enks_local = self.comm.scatter(enks_all, root=0)
            ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
            self.efermi = ele_data_tmp[0]
            self.comm.Barrier()

            # calculate nesting function
            nks_local = len(k_grid_fbz_local)
            self.weight_k = np.ones(nks_local) / nks_fbz
            if self.spinor_mode == 'none':
                self.weight_k *= 2.0
            nesting_func = self._cal_nesting_function(k_grid_fbz_local, self.high_symmetry_k_vecs, enks_local) 
        else:
            hsk_path_vecs = self._frac2car(self.high_symmetry_k_vecs)
            nks = len(hsk_path_vecs)
            nbands = len(self.dispersion_select_index)
            hsk_path_vecs_local = split_into_rank(hsk_path_vecs, 0, self.rank, self.rank_size)
            if hsk_path_vecs_local.size > 0:            
                # enks_local: shape none->(norbs, nks_local); col->(2*norbs, nks_local);  nc->(2*norbs, nks_local)
                enks_local, _ = self._elec_cal_torch(hsk_path_vecs_local)
                if self.spinor_mode in ['none', 'nc']:
                    # enks_local: (nbands, nks_local)
                    enks_local = enks_local[self.dispersion_select_index, :]
                elif self.spinor_mode == 'collinear':
                    # enks_local: (2, nbands, nks_local)
                    enks_local = np.stack((enks_local[self.dispersion_select_index], enks_local[self.dispersion_select_index+self.norbs]), axis=0)
            else:
                if self.spinor_mode in ['none', 'nc']:
                    enks_local = np.empty((nbands, 0))
                elif self.spinor_mode == 'collinear':
                    enks_local = np.empty((2, nbands, 0))
            enks_all = gather_to_rank_root(enks_local, self.comm)
            if self.rank == 0:
                # enks: # shape none->(nbands, nks); col->(2, nbands, nks);  nc->(nbands, nks)
                enks = np.concatenate(enks_all, axis=-1)
        if self.disp_nesting:
            if self.rank == 0:
                self._output_dispersion(nesting_func, 'nesting')
        else:
            if self.rank == 0:
                self._output_dispersion(enks, 'band')

    def _band_cal_on_grid(self):
        """Calculate E_nk and v_nk on IBZ k grid and save with symmetry info."""
        k_mesh = np.array(self.band_k_mesh, dtype=int)
        band_indices = self.dispersion_select_index
        nbands = len(band_indices)

        # IBZ k points (fractional) and weights
        k_ibz_frac, weight_k = self._get_ir_reciprocal_mesh(k_mesh, return_frac=True)
        k_ibz_cart = self._frac2car(k_ibz_frac)
        nks = len(k_ibz_cart)

        if self.rank == 0:
            print(f"Band grid mode: k_mesh={list(k_mesh)}, nk_ibz={nks}, nbands={nbands}", flush=True)

        # MPI split over k points
        k_local = split_into_rank(k_ibz_cart, 0, self.rank, self.rank_size)
        nk_local = len(k_local)

        if nk_local > 0:
            # enks_raw: (norbs, nk_local) or (2*norbs, nk_local); wfn_raw: (nk_local, norbs, norbs)
            enks_raw, wfn_raw = self._elec_cal_torch(k_local)
            if self.spinor_mode == 'collinear':
                # enks_local: (2, nbands, nk_local)
                enks_local = np.stack((enks_raw[band_indices], enks_raw[band_indices+self.norbs]), axis=0)
                # wfn_local: (2, nk_local, nbands, norbs)
                wfn_local = np.stack((wfn_raw[:, band_indices], wfn_raw[:, band_indices+self.norbs]), axis=0)
                vnks_local = np.zeros((2, nk_local, nbands, 3))
                for ispin in range(2):
                    vnks_local[ispin] = self._vel_nk_cal_from_HS(
                        enks_local[ispin], wfn_local[ispin], k_local, collinear_part_index=ispin)
            elif self.spinor_mode == 'nc':
                enks_local = enks_raw[band_indices, :]
                wfn_local = wfn_raw[:, band_indices, :]
                vnks_local = self._vel_nk_cal_from_HS(enks_local, wfn_local, k_local)
            else:
                # spinor_mode == 'none'
                enks_local = enks_raw[band_indices, :]
                wfn_local = wfn_raw[:, band_indices, :]
                vnks_local = self._vel_nk_cal_from_HS(enks_local, wfn_local, k_local)
        else:
            if self.spinor_mode == 'collinear':
                enks_local = np.empty((2, nbands, 0))
                vnks_local = np.empty((2, 0, nbands, 3))
            else:
                enks_local = np.empty((nbands, 0))
                vnks_local = np.empty((0, nbands, 3))

        # Gather to root
        # enks_local: last axis is k; vnks_local: axis 0 (or 1 for collinear) is k
        enks_all = gather_to_rank_root(enks_local, self.comm)
        vnks_all = gather_to_rank_root(vnks_local, self.comm)

        if self.rank == 0:
            if self.spinor_mode == 'collinear':
                enks = np.concatenate(enks_all, axis=-1)    # (2, nbands, nks_ibz)
                vnks = np.concatenate(vnks_all, axis=1)     # (2, nks_ibz, nbands, 3)
            else:
                enks = np.concatenate(enks_all, axis=-1)    # (nbands, nks_ibz)
                vnks = np.concatenate(vnks_all, axis=0)     # (nks_ibz, nbands, 3)

            # Symmetry mapping info
            _, _, all_rots_cart, _ = self._build_kpoint_mappings(k_mesh)
            _, _, k_fbz_frac, k_grid2ir = self._get_ir_reciprocal_mesh(
                k_mesh, auxiliary_info=True, return_frac=True)

            outfile = os.path.join(self.outdir, "band_grid.npz")
            np.savez(outfile,
                     energy=enks,
                     velocity=vnks,
                     k_ibz_frac=k_ibz_frac,
                     weight_k=weight_k,
                     k_fbz_frac=k_fbz_frac,
                     k_grid2ir=k_grid2ir,
                     symmetry_rotations=all_rots_cart,
                     k_mesh=k_mesh,
                     bands_indices=band_indices)
            print(f"Band grid data saved to {outfile}")
            print(f"  energy shape: {enks.shape}, unit: Hartree")
            print(f"  velocity shape: {vnks.shape}, unit: atomic units")
            print(f"  k_ibz: {nks} points, k_fbz: {len(k_fbz_frac)} points")

    def plot_fermi_surface(self):
        data_file = np.loadtxt(self.mesh_file)
        kpt_frac = data_file[:,1:4]
        kpt_cart = self._frac2car(kpt_frac)
        kpt_cart_local = split_into_rank(kpt_cart, 0, self.rank, self.rank_size)
        self.dispersion_select_index = np.arange(0, self.norbs, dtype="int64")
        if kpt_cart_local.size > 0:            
            enks_local, _ = self._elec_cal_torch(kpt_cart_local)
            if self.spinor_mode in ['none', 'nc']:
                enks_local = enks_local[self.dispersion_select_index, :]
            elif self.spinor_mode == 'collinear':
                enks_local = np.stack((enks_local[self.dispersion_select_index], enks_local[self.dispersion_select_index+self.norbs]), axis=0)
        else:
            if self.spinor_mode in ['none', 'nc']:
                enks_local = np.empty((len(self.dispersion_select_index), 0))
            elif self.spinor_mode == 'collinear':
                enks_local = np.empty((2, len(self.dispersion_select_index), 0))
        enks_all = gather_to_rank_root(enks_local, self.comm)
        if self.rank == 0:
            # enks: # shape none->(nbands, nks); col->(2, nbands, nks);  nc->(nbands, nks)
            enks = np.concatenate(enks_all, axis=-1)
            if self.spinor_mode == 'collinear':
                enks = np.transpose(enks, (0, 2, 1))
            else:
                enks = np.transpose(enks)
            filepath = os.path.join(self.outdir, 'fermi_surface_enks.npy')
            np.save(filepath, enks)

    def plot_phonon(self):
        """ 
        Calculate the phonon dispersion for a special q path.

        Args:

        Returns:
        """
        nbranches = len(self.dispersion_select_index)
        nqs = len(self.high_symmetry_k_vecs)
        if self.disp_renormalize:
            self.cal_mode = 'self-energy'
            k_grid_ibz, self.weight_k = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=False)
            nks_fbz = self.k_size[0] * self.k_size[1] * self.k_size[2]
            
            if self.spinor_mode == 'none':
                self.weight_k *= 2.0

            # check the system is metal or insulator
            k_grid_ibz_local = split_into_rank(k_grid_ibz, 0, self.rank, self.rank_size)
            if k_grid_ibz_local.size > 0:
                # enks_local: # shape if collinear (2*norbs, nks_local), else (norbs, nks_local)
                enks_local, _ = self._elec_cal_torch(k_grid_ibz_local)
            else:
                if self.spinor_mode == 'collinear':
                    enks_local = np.empty((2*self.norbs, 0))
                else:
                    enks_local = np.empty((self.norbs, 0))
            enks_all = gather_to_rank_root(enks_local, self.comm)
            if self.rank == 0:
                # enks: # shape if collinear (2*norbs, nks), else (norbs, nks)
                enks = np.concatenate(enks_all, axis=-1)
                num_electron_include = self.graph_data.num_electrons - self.doping_charge
                self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
                is_metal = self._get_is_metal(enks, self.efermi)
                if is_metal:
                    print("The system is a metal.")
                else:
                    print("The system is a insulator.")
                ele_data_tmp = [self.efermi, is_metal]
            else:
                ele_data_tmp = None
            del enks_local, enks_all, k_grid_ibz_local
            ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
            self.efermi, is_metal = ele_data_tmp[0], ele_data_tmp[1]

            if is_metal:
                if self.rank == 0:
                    # here band_edge_energy is the fermi level of the system.
                    print("Fermi energy = {:10.6f} eV.".format(self.efermi * QTcts.HARTREEtoEV))
            else:
                if self.is_hole:
                    band_edge_index = self.VBM_band_index
                else:
                    band_edge_index = self.CBM_band_index
                if self.spinor_mode == 'collinear':
                    iband_edge = np.where(np.array(self.bands_indices)==band_edge_index)[0][0]
                else:
                    iband_edge = np.where(np.array(self.bands_indices)==band_edge_index)[0][0]
                if self.rank == 0:
                    if self.spinor_mode == 'collinear':
                        # However, when we focus on the phonon-limited conductivity of a collinear system,
                        # we consider both spin-up and spin-down.
                        enks = np.stack((enks[self.bands_indices], enks[self.bands_indices+self.norbs]), axis=0) # enks: shape (2, nbands, nks_ibz)
                    else:
                        # If the system is collinear, wo only focus on the spin-up states of all |nk> states
                        enks = enks[self.bands_indices] # enks: shape (nbands, nks_ibz)
                    if self.spinor_mode == 'collinear':
                        self.efermi = np.zeros(2, dtype=float)
                        self.carrier_density = np.zeros(2, dtype=float)
                        for ispin in range(2):
                            self.efermi[ispin], self.carrier_density[ispin], _ = self._get_fermi_level_insulator(enks[ispin], iband_edge)
                            print("=" * 100)
                            if self.system_2D:
                                print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-2) for spin {}.".format(
                                    self.efermi[ispin] * QTcts.HARTREEtoEV, self.carrier_density[ispin] * self.inv_cell / (QTcts.BOHRtoCM ** 2), ispin+1
                                    ))
                            else:
                                print("Fermi energy = {:10.6f} eV, Carrier density = {:12.4e} cm^(-3) for spin {}.".format(
                                    self.efermi[ispin] * QTcts.HARTREEtoEV, self.carrier_density[ispin] * self.inv_cell / (QTcts.BOHRtoCM ** 3), ispin+1
                                    ))
                            print("=" * 100,'\n')
                    else:
                        self.efermi, self.carrier_density, _ = self._get_fermi_level_insulator(enks, iband_edge)
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
            
            k_grid_fbz = self._get_monkhorst_pack(self.k_size)
            nks_fbz = len(k_grid_fbz)
            if self.spinor_mode == 'none':
                self.weight_k = np.ones(nks_fbz) / nks_fbz * 2.0
            else:
                self.weight_k = np.ones(nks_fbz) / nks_fbz
            self._print_mesh_mpi_check(nks_fbz, nks_fbz, nks_fbz, nqs)
            wqvs, phon_vecs, omega_list, self_energies, spectrum_func = spectrum_function_cal(self, k_grid_fbz, self.high_symmetry_k_vecs, is_metal=is_metal)
            if self.rank == 0:
                print("Outputing phonon spectrum...\n")
                self._output_spectrum_func('phonon', self.high_symmetry_k_dist, omega_list, spectrum_func)
                self._output_self_energy_all('phonon', self.high_symmetry_k_vecs, self_energies, omega_list=omega_list)            
        elif self.disp_lambda:
            k_grid_ibz, self.weight_k, k_grid_fbz, _ = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=True)
            nks_fbz = len(k_grid_fbz)
            if self.spinor_mode == 'none':
                self.weight_k *= 2.0
            # check the system is metal or insulator
            k_grid_ibz_local = split_into_rank(k_grid_ibz, 0, self.rank, self.rank_size)
            if k_grid_ibz_local.size > 0:
                # enks_local: # shape if collinear (2*norbs, nks_local), else (norbs, nks_local)
                enks_local, _ = self._elec_cal_torch(k_grid_ibz_local)
            else:
                if self.spinor_mode == 'collinear':
                    enks_local = np.empty((2*self.norbs, 0))
                else:
                    enks_local = np.empty((self.norbs, 0))
            enks_all = gather_to_rank_root(enks_local, self.comm)
            if self.rank == 0:
                # enks: # shape if collinear (2*norbs, nks), else (norbs, nks)
                enks = np.concatenate(enks_all, axis=-1)
                num_electron_include = self.graph_data.num_electrons - self.doping_charge
                self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
                is_metal = self._get_is_metal(enks, self.efermi)
                if is_metal:
                    print("The system is a metal.")
                    print("Fermi energy = {:10.6f} eV.".format(self.efermi * QTcts.HARTREEtoEV))
                else:
                    raise RuntimeError("Error(7001): Lambda can only be calculated for metallic systems.")
                ele_data_tmp = [self.efermi]
            else:
                ele_data_tmp = None
            del enks_local, enks_all, k_grid_ibz_local
            ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
            self.efermi = ele_data_tmp[0]
            if self.spinor_mode == 'none':
                self.weight_k = np.ones(nks_fbz) / nks_fbz * 2.0
            else:
                self.weight_k = np.ones(nks_fbz) / nks_fbz
            wqvs, epc_strengths = self._lambda_cal_epc_sparse(k_grid_fbz, self.high_symmetry_k_vecs)
            if self.rank == 0:
                print("Outputing mode-resolved electron phonon coupling strengths...\n")
                self._output_lambda(epc_strengths[np.newaxis, :, :])
        else:
            split_sections = np.zeros(self.rank_size, dtype=int)
            for i in range(nqs):
                split_sections[i%self.rank_size] += 1
            split_sections = np.cumsum(split_sections, axis=0)
            hsk_path_vecs_all = np.split(self.high_symmetry_k_vecs, indices_or_sections=split_sections, axis=0)
            if hsk_path_vecs_all[self.rank].size > 0:
                wqvs_local, phon_vecs_local = self._phonon_cal(hsk_path_vecs_all[self.rank])
                wqvs_local = wqvs_local[:, self.dispersion_select_index]
                phon_vecs_local = phon_vecs_local[:, self.dispersion_select_index]
            else:
                wqvs_local, phon_vecs_local = np.empty((0, nbranches)), np.empty((0, nbranches, self.natoms*3))
            if self.rank == 0:
                wqvs_all = [None] * self.rank_size
            else:
                wqvs_all = None
            self.comm.Barrier()
            wqvs_all = self.comm.gather(wqvs_local, root=0)
            if self.rank == 0:
                # wqvs: shape (nqs, nbranches)
                wqvs = np.concatenate(wqvs_all, axis=0)
        if self.save_eigen_vectors == 'ascii':
            if self.rank == 0:
                phon_vecs_all = [None] * self.rank_size
            else:
                phon_vecs_all = None
            phon_vecs_local = phon_vecs_local.reshape(-1, nbranches, self.natoms, 3)
            self.comm.Barrier()
            phon_vecs_all = self.comm.gather(phon_vecs_local, root=0)
            if self.rank == 0:
                # phon_vecs: shape (nqs, nbranches, natoms_uc, 3)
                phon_vecs = np.concatenate(phon_vecs_all, axis=0)
                atoms_symbols = [Element.from_Z(k).symbol for k in self.graph_data.species]
                equal_positions = self.graph_data.pos * QTcts.BOHRtoANG
                struct = Structure(lattice=self.graph_data.latt*QTcts.BOHRtoANG, species=atoms_symbols,
                                    coords=equal_positions, coords_are_cartesian=True)
                latt_a, latt_b, latt_c = struct.lattice.abc
                latt_alpha, latt_beta, latt_gamma = np.array(struct.lattice.angles) / 180 * QTcts.PI
                reduced_latt = np.zeros(6)
                reduced_latt[0] = latt_a
                reduced_latt[1] = latt_b * np.cos(latt_gamma)
                reduced_latt[2] = latt_b * np.sin(latt_gamma)
                reduced_latt[3] = latt_c * np.cos(latt_beta)
                reduced_latt[4] = latt_c * (np.cos(latt_alpha) - np.cos(latt_beta) * \
                                    np.cos(latt_gamma)) / np.sin(latt_gamma)
                reduced_latt[5] = latt_c * np.sqrt(1.0 + 2.0 * np.cos(latt_alpha) * np.cos(latt_beta) * np.cos(latt_gamma) - \
                                    np.power(np.cos(latt_alpha), 2) - np.power(np.cos(latt_beta), 2) - np.power(np.cos(latt_gamma), 2)) / np.sin(latt_gamma)
                if self.chiral_projection:
                    # mask out those extreme small value in phonon vectors
                    phon_vecs_abs = np.abs(phon_vecs)
                    mask = phon_vecs_abs > QTcts.TENPM5
                    phon_vecs = phon_vecs * mask
                    phon_vecs_abs = np.abs(phon_vecs)
                    phon_vecs_abs[~mask] = np.inf
                    x_vecs = phon_vecs[:,:,:,0]
                    y_vecs = phon_vecs[:,:,:,1]
                    # phonon circular polarization # shape (nqs, nbranches, natoms)
                    polarization = (np.conj(x_vecs) * y_vecs - x_vecs * np.conj(y_vecs)) / (phon_vecs_abs[:,:,:,0]*phon_vecs_abs[:,:,:,0] + phon_vecs_abs[:,:,:,1]*phon_vecs_abs[:,:,:,1])
                    polarization = polarization.imag
                    with open(os.path.join(self.outdir, "phonon_polarization.dat"), 'w') as fout:
                        for iq in range(nqs):
                            fout.write(f"# q-point : {iq+1}    "+"  ".join(atoms_symbols)+'\n')
                            for ibranch in range(nbranches):
                                outstr_tmp = "  ".join(["{:12.8f}".format(each) for each in polarization[iq, ibranch]])
                                fout.write(f"Branch {self.dispersion_select_index[ibranch]+1} : "+outstr_tmp+'\n')
                            fout.write('\n')
                for iq in range(nqs):
                    fout_name = os.path.join(self.outdir, f"phvecs-{iq+1}.ascii")
                    freq = wqvs[iq]
                    eigen_vec = phon_vecs[iq]   # shape (nbranches, natoms, 3)
                    if (freq < 0.0).any():
                        print(f"Warnning: phvecs-{iq+1}.ascii contains a imaginary frequency.")
                    with open(fout_name, 'w') as fout:
                        fout.write(f"# Phonon eigen vectors for iq = {iq+1}. Use V_sim to load this file.\n")
                        fout.write("{:15.9f}  {:15.9f}  {:15.9f}\n".format(*reduced_latt[:3]))
                        fout.write("{:15.9f}  {:15.9f}  {:15.9f}\n".format(*reduced_latt[3:]))
                        for iatom in range(self.natoms):
                            fout.write("{:15.9f}  {:15.9f}  {:15.9f}  {}\n".format(*equal_positions[iatom], atoms_symbols[iatom]))
                        for ibranch in range(nbranches):
                            fout.write("#metaData: qpt=[{:10.6f};{:10.6f};{:10.6f};{:10.6f} \\ \n".format(*self.high_symmetry_k_vecs[iq], freq[ibranch]*QTcts.HARTREEtoMEV))
                            for iatom in range(self.natoms):
                                fout.write("#; {:15.9f}; {:15.9f}; {:15.9f}; {:15.9f}; {:15.9f}; {:15.9f} \\ \n".format(*(eigen_vec[ibranch, iatom].real), *(eigen_vec[ibranch, iatom].imag)))
                            fout.write("# ]\n")
        if self.plot_2D_axes:
            if self.rank == 0:
                plot_2D_inv_smear = 1.0 / self.plot_2D_smear
                energy_min, energy_max = np.min(wqvs), np.max(wqvs)
                energies = np.linspace(energy_min, energy_max, self.plot_2D_ncuts)
                ncuts = len(energies)
                # cut_intensity: shape (ncuts, nqs, nbranches)
                cut_intensity = w0gauss((wqvs[None,:,:] - energies[:,None,None]) * plot_2D_inv_smear) * plot_2D_inv_smear
                cut_intensity = cut_intensity * QTcts.HARTREEtoEV
                # total_intensity: shape (ncuts, nqs)
                total_intensity = np.sum(cut_intensity, axis=2)
                fout = open(os.path.join(self.outdir, "phbands.dispersion"), 'w')
                fout.write("#      x      y      z      ibranch      intensity(1/eV)")
                fout_sum = open(os.path.join(self.outdir, "phbands-sum-branch.dispersion"), 'w')
                fout_sum.write("#      x      y      z      intensity(1/eV)")
                for icut in range(ncuts):
                    fout.write("# cut{:d} : {:12.8e} eV\n".format(icut, energies[icut]*QTcts.HARTREEtoEV))
                    fout_sum.write("# cut{:d} : {:12.8e} eV\n".format(icut, energies[icut]*QTcts.HARTREEtoEV))
                    for iq in range(nqs):
                        fout_sum.write("{:10.6f}  {:10.6f}  {:10.6f}    {:12.8e}\n".format(*self.high_symmetry_k_vecs[iq], total_intensity[icut, iq]))
                        for ibranch in range(nbranches):
                            fout.write("{:10.6f}  {:10.6f}  {:10.6f}    {:d}      {:12.8e}\n".format(*self.high_symmetry_k_vecs[iq], \
                                self.dispersion_select_index[ibranch]+1, cut_intensity[icut, iq, ibranch]))
                    fout.write('\n')
                    fout_sum.write('\n')
                fout.close()
                fout_sum.close()
        else:
            if self.rank == 0:
                self._output_dispersion(wqvs, 'phonon')
