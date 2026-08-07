"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from typing import Dict
from typing import List
import numpy as np


class _IterativeBTEMixin:
    def get_F_nk(self, F_in_all, rate_useful, scat_channel, rot_indices, all_rots_cart, unique_kpts):
        r"""
        Calculate the F_nk using F_in calculated in last iteration.
        Args:
            F_in_all (np.ndarray): F_nk in full usefull IBZ kpoints. # shape (nks_ibz_usefull, nbands, 3)
            rate_useful (np.ndarray): Scattering rate in full usefull IBZ kpoints. # shape (nbands, nks_ibz_usefull)
            scat_channel (np.ndarray): Structured array with [k_idx_useful_ibz, kpq_idx_ibz, kpq_idx_fbz, iband, jband, transition prob]
            rot_indices (list): FBZ-->rotation index (into all_rots) # shape (nks_tot,num_rot)
            all_rots_cart (np.ndarray): Matrix of every symmetry rotation  # shape (N_sym, 3, 3)
            unique_kpts (np.ndarray): The IBZ kpoints really contributing # shape (nks_ibz_usefull,)
        Returns:
            F_out (np.ndarray): new F_nk in local kpoints # shape (nks_ibz_usefull, nbands, 3)
        """

        nbands, nks_ibz_usefull = rate_useful.shape
        F_out = np.zeros((nks_ibz_usefull, nbands, 3))
        rate_inv_local = 1.0 / rate_useful

        # Precompute mapping from IBZ_index -> unique_kpts index
        ibz_to_unique_idx = {ibz: i for i, ibz in enumerate(unique_kpts)}

        if len(scat_channel) == 0:
            return F_out

        iks = scat_channel['ik']
        ikpqs_ibz = scat_channel['ikpq_ibz']
        ikpqs_fbz = scat_channel['ikpq_fbz']
        ibs = scat_channel['iband']
        jbs = scat_channel['jband']
        trans_probs = scat_channel['trans_prob']
        nchannel = len(iks)

        for idx in range(nchannel):
            kpt_index = iks[idx]
            BZ_index = ikpqs_fbz[idx]
            IBZ_index = ikpqs_ibz[idx]
            iband_index = ibs[idx]
            jband_index = jbs[idx]
            trans_prob = trans_probs[idx]
            # Skip if IBZ_index not in unique_kpts
            index = ibz_to_unique_idx.get(IBZ_index)
            if index is None:
                continue
            # Get all rotation matrices for this BZ point
            rot_index_list = rot_indices[BZ_index]
            counter = len(rot_index_list)
            rot_mats = all_rots_cart[rot_index_list]  # shape (n_rot, 3, 3)
            # F_in vector for j-band at IBZ point
            F_in_vec = F_in_all[index, jband_index]  # shape (3,)
            # Compute rotated contributions: shape (n_rot, 3)
            rotated_vecs = np.einsum("i,rij->rj", F_in_vec, rot_mats)
            # Average the results corresponding degeneracy to keep the symmetry! Important for symmetry!
            F_out[kpt_index, iband_index] += rotated_vecs.sum(axis=0) * (trans_prob * rate_inv_local[iband_index, kpt_index] / counter)
        
        F_out *= QTcts.TWOPI
        return F_out # shape (nks_ibz_usefull, nbands, 3)

    def get_conductivity_symm(self,
                        elec_energies: np.ndarray,    # (nbands, nk_loc)
                        elec_velocities: np.ndarray,   # (nk_loc, nbands, 3)
                        F_in:            np.ndarray,   # (nk_loc, nbands, 3)
                        all_rots:        np.ndarray,   # (N_rots, 3, 3)
                        ibz_to_fbz:      Dict[int, List[List]],
                        k_index_local:   np.ndarray    # (nk_loc,)
                        ):
        r"""
        Get the conductivity (metal) or the mobility (insulator) using k-points symmetries.

        Args: 
            elec_energies (np.ndarray): The energies matrix, ref from efermi. # shape (nk_local, nbands)
            elec_velocities (np.ndarray): The velocities matrix. # shape (nk_local, nbands, 3)
            F_in (np.ndarray): \partial_{E_{\beta}} f_{nk} # shape (nk_local, nbands, 3)
            all_rots (np.ndarray): Matrix of every symmetry rotation  # shape (N_sym, 3, 3)
            ibz_to_fbz (dict): IBZ index → list of (rot_index_list)
            k_index_local (np.ndarray): index of local kpoints # shape (nk_local,)
        Returns:
        """
        cond_mat = np.zeros((3,3))
        cond_See_mat = np.zeros((3,3))
        if k_index_local.size >0:
            for local_idx, ir_idx in enumerate(k_index_local):
                v_ir = elec_velocities[local_idx]   # shape (nbands,3)
                E_ir = elec_energies[:,local_idx]   # shape (nbands,)
                F_ir = F_in[local_idx]              # shape (nbands,3)
                # for each symmetry mapping of this irreducible point
                ir_idx = int(ir_idx)
                for rot_idx_list in ibz_to_fbz[ir_idx]:
                    counter = len(rot_idx_list) # degeneracy of rot_mat for a FBZ-->IBZ mapping
                    v_full = np.zeros_like(v_ir)
                    F_full = np.zeros_like(F_ir)
                    for rot_idx in rot_idx_list:
                        rot_mat = all_rots[rot_idx]          # shape (3,3)
                        # Average the results corresponding degeneracy to keep the symmetry! Important for symmetry!
                        v_full += v_ir @ rot_mat / counter       # shape (nbands,3)
                        F_full += F_ir @ rot_mat / counter       # shape (nbands,3)
                    cond_mat += np.einsum('ni,nj->ij', v_full, F_full) # shape: (3, 3) 
                    cond_See_mat += np.einsum('n,ni,nj->ij', E_ir, v_full, F_full) # shape: (3, 3)
        else:
            cond_mat = np.zeros((3,3))
            cond_See_mat = np.zeros((3,3))
        return cond_mat, cond_See_mat

    def boltzmann_cal(self):
        """
        Calculate the the time-dependent electronic occupation using the real-time Boltzman transport equation (rt-BTE).

        Args:

        Returns:
        """
        # define basic dimensions of k/q-grid
        k_grid_ibz, self.weight_k, k_grid_fbz, self.ir_indices = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=True)
        # self.ir_indices, rot_indices, all_rots_cart, ibz_to_fbz = self._build_kpoint_mappings(self.k_size)
        q_grid_fbz = self._get_monkhorst_pack(self.q_size, self.graph_data.latt, return_frac=True)
        self.weight_q = np.ones(len(q_grid_fbz)) / len(q_grid_fbz)
        self.weight_of_q = 1 / len(q_grid_fbz)
        nks_ibz = len(k_grid_ibz)
        nbands = len(self.bands_indices)
        if self.spinor_mode == 'none':
            self.weight_k *= 2.0
        if self.adv_cal:
            # Get finite-difference b-vectors and weights
            self.final_bvec, self.final_bweight = self.get_optimal_neighbors_and_weights(self.k_size, self.graph_data.rec_latt) # shape: (nbvecs, 3) and (nbvecs,)
            # Find all neighbor k-points and their indices
            self.neighbors_table = self.get_neighbor_index(self.final_bvec, k_grid_fbz) # shape: (nks_fbz, nbvecs)

        # check the system is metal or insulator
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nks_ibz):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_split = np.split(k_grid_ibz, indices_or_sections=split_sections, axis=0)
        k_grid_local = k_grid_split[self.rank]
        if k_grid_local.size > 0:
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
        self.comm.Barrier()
        enks_all = self.comm.gather(enks_local, root=0)
        if self.rank == 0:
            # enks: # shape if collinear (2*norbs, nks), else (norbs, nks)
            enks = np.concatenate(enks_all, axis=-1)
            num_electron_include = self.graph_data.num_electrons - self.doping_charge
            self.efermi = self._get_fermi_level_metal(enks, self.temperature, self.gauss_type, num_electron_include) + self.fermi_shift
            is_metal = self._get_is_metal(enks, self.efermi)
            if is_metal:
                print("Dealing with metallic systems!")
            else:
                print("Dealing with insulating systems!")
            ele_data_tmp = [self.efermi, is_metal]
        else:
            ele_data_tmp = None
        del enks_local, enks_all, split_sections, k_grid_local # clean 
        ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
        self.efermi, is_metal = ele_data_tmp[0], ele_data_tmp[1]

        if is_metal:
            pass
        else:
            if self.is_hole:
                band_edge_index = self.VBM_band_index
            else:
                band_edge_index = self.CBM_band_index
            if self.soc_switch:
                raise NotImplementedError("SOC mobility is not supported!")
            # elif self.spinor_mode == 'collinear':
            #     iband_edge = np.where(np.array(self.bands_indices)==band_edge_index)[0][0] #assuming the fermi levels for both spin are the same?
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
                        self.efermi[ispin], self.carrier_density[ispin] = self._get_fermi_level_insulator(enks[ispin], iband_edge)
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
                        if self.is_hole:
                            band_edge_energy[ispin] = self._get_evbm(enks[ispin], iband_edge)
                        else:
                            band_edge_energy[ispin] = self._get_ecbm(enks[ispin], iband_edge)
                        del enks
                    ele_data_tmp = [*self.efermi, *self.carrier_density, *band_edge_energy]
                else:
                    self.efermi, self.carrier_density = self._get_fermi_level_insulator(enks[:nbands], iband_edge)
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
                    del enks
                    ele_data_tmp = [self.efermi, self.carrier_density, band_edge_energy]
            else:
                ele_data_tmp = None
            ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
            if self.spinor_mode == 'collinear':
                self.efermi = np.array(ele_data_tmp[:2], dtype=float)
                self.carrier_density = np.array(ele_data_tmp[2:4], dtype=float)
                band_edge_energy = np.array(ele_data_tmp[4:], dtype=float)
            else:
                self.efermi, self.carrier_density, band_edge_energy = ele_data_tmp

            # setup the initial distribution and scattering channels
            if self.read_sparse_grad_mat:
                eigen, dist_ele_0, scat_channel = self.dynamics_setup_sparse(k_grid_ibz, q_grid_fbz, self.bands_indices, band_edge_energy)
            else:
                eigen, dist_ele_0, scat_channel = self.dynamics_setup(k_grid_ibz, q_grid_fbz, self.bands_indices, band_edge_energy)
            # if self.rank != 0:
            #     del eigen # non-io node do not need eigen

            # For adv_cal, we need v_nk in FBZ
            if self.adv_cal:
                # for rank0: shape: (nks, nbands, 3); for else rank: None
                band_velocity = self.get_velocity_only(k_grid_fbz)

            # save the initial distribution to file
            if self.rank == 0:
                if self.is_hole:
                    self.save_energy_resolved_dist(0, eigen, dist_ele_0, self.energy_step, band_edge_energy,
                                                   band_edge_energy - self.over_band_edge, band_edge_energy)
                else:
                    self.save_energy_resolved_dist(0, eigen, dist_ele_0, self.energy_step, band_edge_energy,
                                                   band_edge_energy, band_edge_energy + self.over_band_edge)
                init_nc = self.get_total_carrier_density(dist_ele_0)
            else:
                init_nc = None
            init_nc = self.comm.bcast(init_nc, root=0) # initial carrier density

            # Enter the main loop
            time_box = []
            vel_box  = []

            dist_ele_old = dist_ele_0
            if self.rank == 0:
                logger = time_logger(total_cycles=self.num_step, routine_name='rtBTE_main_loop')
            for istep in range(self.num_step):
                dist_ele_new = self.rtBTE_solver(self.time_step, dist_ele_old, scat_channel, self.boltz_solver)
                # normalize the carrier distribution using carrier density
                if self.adv_cal or self.norm_dist:
                    if self.is_hole:
                        now_nc = self.get_total_carrier_density(dist_ele_new)
                    else:
                        now_nc = self.get_total_carrier_density(dist_ele_new)
                    dist_ele_new = dist_ele_new / now_nc * init_nc
                dist_ele_old = dist_ele_new
                if self.rank == 0:
                    logger.step(istep+1)
                    # save the distribution every output_step
                    if (istep + 1) % self.output_step == 0:
                        if self.is_hole:
                            self.save_energy_resolved_dist(istep + 1, eigen, dist_ele_new, self.energy_step, band_edge_energy,
                                                           band_edge_energy - self.over_band_edge, band_edge_energy)
                        else:
                            self.save_energy_resolved_dist(istep + 1, eigen, dist_ele_new, self.energy_step, band_edge_energy,
                                                           band_edge_energy, band_edge_energy + self.over_band_edge)
                        if self.adv_cal:
                            drift_vel = self.get_drift_velocity(band_velocity, dist_ele_new, self.efield, self.carrier_density, self.volume_uc)
                            # print(istep + 1,'fs:',drift_vel*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)
                            time_box.append(int(istep + 1))
                            vel_box.append(drift_vel*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)
            if self.rank ==0:
                self.combine_dists_to_final_file(self.num_step, self.output_step)
                # print(time_box)
                # print(vel_box)
