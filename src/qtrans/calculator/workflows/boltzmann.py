"""Real-time Boltzmann calculation workflow."""

from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
import numpy as np


class _BoltzmannMixin:
    def boltzmann_cal(self):
        """
        Calculate the the time-dependent electronic occupation using the real-time Boltzman transport equation (rt-BTE).

        Args:

        Returns:
        """
        # define basic dimensions of k/q-grid
        k_grid_ibz, self.weight_k, k_grid_fbz, self.ir_indices = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=True)
        # self.ir_indices, rot_indices, all_rots_cart, ibz_to_fbz = self._build_kpoint_mappings(self.k_size)
        q_grid_fbz = self._get_monkhorst_pack(self.q_size, return_frac=True)
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
