"""Phonon-limited resistivity calculation workflow."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.numerics.distributions import bose_weight
import numpy as np


class _ResistivityMixin:
    def _get_is_fermi_crossing(self, enks:np.ndarray):
        """
        Identify bands that cross the Fermi level.

        Args:
            enks (np.ndarray): Band energies on a k grid, shape (nbands, nks).

        Returns:
            np.ndarray: Boolean mask of Fermi-crossing bands, shape (nbands,).
        """
        emin = np.min(enks, axis=1)
        emax = np.max(enks, axis=1)
        is_fermi_crossing = (emin < self.efermi) & (emax > self.efermi)
        return is_fermi_crossing

    def resistivity_cal(self):
        """
        Calculate phonon-limited resistivity from transport Eliashberg spectra.

        The method builds full k/q meshes, determines the Fermi level and Fermi-level
        DOS, evaluates ordinary and transport Eliashberg functions using either dense
        or sparse EPC gradients, writes ``a2f.dat`` on rank 0, and prints Ziman-like
        and Allen-like resistivity estimates in micro-Ohm cm.

        Side Effects:
            Updates ``self.weight_k``, ``self.weight_q``, ``self.efermi``, and
            possibly ``self.ziman_nc``. Rank 0 writes ``a2f.dat``.
        """
        # Accroding to the formula, we need the full grid for k and q.
        irr_k_grid, self.weight_k, k_grid, _ = self._get_ir_reciprocal_mesh(self.k_size, auxiliary_info=True)
        q_grid = self._get_monkhorst_pack(self.q_size, return_frac=True)
        self.weight_q = np.ones(len(q_grid)) / len(q_grid)
        nks = len(k_grid)
        nqs = len(q_grid)
        nbands = len(self.bands_indices)

        if self.spinor_mode == 'none' or self.spinor_mode == 'collinear':
            if self.has_unpair_electron:
                num_electron_include = \
                    len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) * 2 + 1 - \
                    self.doping_charge
            else:
                num_electron_include = \
                    len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) * 2 - \
                    self.doping_charge
        elif self.spinor_mode == 'nc':
            num_electron_include = \
                    len(np.array(self.bands_indices)[self.bands_indices < self.CBM_band_index]) - self.doping_charge
        
        if self.spinor_mode == 'none':
            self.weight_k *= 2.0
        else:
            self.weight_k *= 1.0

        # get fermi energy and fermi dos
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(len(irr_k_grid)):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        irr_k_grid_all = np.split(irr_k_grid, indices_or_sections=split_sections, axis=0)
        if irr_k_grid_all[self.rank].size > 0:            
            # eigen: (norbs, nk_local) eigen_vecs: (nk_local, norbs, norbs)
            enks_local, _ = self._elec_cal_torch(irr_k_grid_all[self.rank])
            if self.spinor_mode == 'collinear':
                enks_local = np.concatenate((enks_local[self.bands_indices], enks_local[self.bands_indices+self.norbs]), axis=0)
            else:
                enks_local = enks_local[self.bands_indices]
        else:
            if self.spinor_mode == 'collinear':
                enks_local = np.empty((2*nbands, 0))
            else:
                enks_local = np.empty((nbands, 0))
        if self.rank == 0:
            enks_all = [None] * self.rank_size
        else:
            enks_all = None
        self.comm.Barrier()
        enks_all = self.comm.gather(enks_local, root=0)
        if self.rank == 0:
            enks = np.concatenate(enks_all, axis=-1) # (2*nbands, nks) for collinear case, else (nbands, nks)
            # get the fermi energy and dos of fermi level (self.efermi and dos_ef will bcast to all ranks).
            self.efermi = self._get_fermi_level_metal(enks, self.smeark, self.gauss_type, num_electron_include) + self.fermi_shift
            # N(Ef) in formula is the DOS per spin
            if self.spinor_mode == 'collinear':
                # if collinear, dos_ef is a two elements list.
                dos_ef = [self._get_ef_dos(enks[:nbands], self.efermi, self.inv_smeark, self.gauss_type), self._get_ef_dos(enks[nbands:], self.efermi, self.inv_smeark, self.gauss_type)]
                if self.rank == 0:
                    print("Calculation conduct on a collinear system.")
                    print("Fermi energy = {} eV\nDOS of Fermi level (spin up) = {}\nDOS of Fermi level (spin down) = {}".format(
                        self.efermi * QTcts.HARTREEtoEV, dos_ef[0] / QTcts.HARTREEtoEV, dos_ef[1] / QTcts.HARTREEtoEV))
                    print('-'*100)
            else:
                dos_ef = self._get_ef_dos(enks, self.efermi, self.inv_smeark, self.gauss_type) / 2.0
                if self.rank == 0:
                    print("Fermi energy = {} eV\nDOS of Fermi level (spin factor not included) = {}".format(self.efermi * QTcts.HARTREEtoEV, dos_ef / QTcts.HARTREEtoEV))
                    print('-'*100)
            ele_data_tmp = [self.efermi, dos_ef]
        else:
            ele_data_tmp = None
        ele_data_tmp = self.comm.bcast(ele_data_tmp, root=0)
        self.efermi, dos_ef = ele_data_tmp
        # if ziman_nc < 0.0, set it as the number of bands crossing the fermi level.
        if self.ziman_nc < 0.0:
            if irr_k_grid_all[self.rank].size > 0:
                if self.spinor_mode == 'collinear':
                    is_fermi_crossing = np.vstack([self._get_is_fermi_crossing(enks_local[:nbands]), self._get_is_fermi_crossing(enks_local[nbands:])])
                else:
                    is_fermi_crossing = self._get_is_fermi_crossing(enks_local)
            else:
                if self.spinor_mode == 'collinear':
                    is_fermi_crossing =np.zeros((2,nbands), dtype=bool)
                else:
                    is_fermi_crossing = np.zeros(nbands, dtype=bool)
            if self.comm is not None:
                self.comm.Allreduce(MPI.IN_PLACE, is_fermi_crossing, op=MPI.LOR)
            self.ziman_nc = np.sum(is_fermi_crossing, axis=-1)
            if self.rank == 0:
                if self.spinor_mode == 'collinear':
                    for ispin in range(2):
                        print(f"The number of spin-{ispin+1} carriers per unit cell is automatically set to be {self.ziman_nc[ispin]:.4f}.")
                else:
                    print("The number of carriers per unit cell is automatically set to be {:.4f}.".format(self.ziman_nc))
            self.ziman_nc = self.ziman_nc / self.volume_uc        
        if self.spinor_mode == 'none':
            self.weight_k = 2.0 / nks
        else:
            self.weight_k = 1.0 / nks

        if self.spinor_mode == 'collinear':
            # collinear system
            # a2f, a2f_tr: shape (nomegas,2)
            if self.read_sparse_grad_mat:
                omegas, a2f, a2f_tr, fermi_speed = self.transport_eliashberg_function_cal_epc_iso_collinear_sparse(k_grid, q_grid, dos_ef)
            else:
                omegas, a2f, a2f_tr, fermi_speed = self.transport_eliashberg_function_cal_epc_iso_collinear_dense(k_grid, q_grid, dos_ef)
            if self.rank == 0:
                a2f_io = np.hstack((omegas[:, np.newaxis]*QTcts.HARTREEtoMEV, a2f_tr, a2f))
                np.savetxt("a2f.dat", a2f_io, fmt="%.10f", delimiter="    ",
                    header="omega[meV]    a2F_tr_spin1    a2F_tr_spin2    a2F_spin1       a2F_spin2")
            if self.rank == 0:
                dist_factor = bose_weight(omegas, self.temperature)
                for ispin in range(2):
                    resist_strength = np.sum(omegas * a2f_tr[:,ispin] * dist_factor * (1.0 + dist_factor)) * self.omega_step
                    ## Way 1: similar with EPW Computer Physics Communications 209 (2016) 116–133
                    resistance_Ziman = resist_strength * QTcts.FOURPI / self.ziman_nc[ispin] / self.temperature
                    resistance_Ziman = resistance_Ziman * QTcts.HARTREEtoOHM * QTcts.TENPP6 * QTcts.BOHRtoCM

                    ## Way 2: similar with PHYSICAL REVIEW B 92, 075405 (2015)
                    resistance_Allen = resist_strength * QTcts.TWOPI * self.volume_uc / self.temperature / dos_ef[ispin] / np.mean(fermi_speed[ispin])**2
                    resistance_Allen = resistance_Allen * QTcts.HARTREEtoOHM * QTcts.TENPP6 * QTcts.BOHRtoCM

                    print('-'*100)
                    print(f"Ziman's resistance[spin-{ispin+1}](micro-Ohm*cm) = {resistance_Ziman:12.4e}")
                    print(f"Allen's resistance[spin-{ispin+1}](micro-Ohm*cm) = {resistance_Allen:12.4e}")
                print('-'*100)
            # raise NotImplementedError("Collinear system is not supported now.")
        else:
            # non-magnetic or nc system
            # a2f, a2f_tr: shape (nomegas, )
            if self.read_sparse_grad_mat:
                omegas, a2f, a2f_tr, fermi_speed = self.transport_eliashberg_function_cal_epc_iso_sparse(k_grid, q_grid, dos_ef)
            else:
                omegas, a2f, a2f_tr, fermi_speed = self.transport_eliashberg_function_cal_epc_iso_dense(k_grid, q_grid, dos_ef)
            if self.rank == 0:
                a2f_io = np.hstack((omegas[:, np.newaxis]*QTcts.HARTREEtoMEV, a2f_tr[:, np.newaxis], a2f[:, np.newaxis]))
                np.savetxt("a2f.dat", a2f_io, fmt="%.10f", delimiter="    ",
                    header="omega[meV]    a2F_tr          a2F")
            if self.rank == 0:
                dist_factor = bose_weight(omegas, self.temperature)
                resist_strength = np.sum(omegas * a2f_tr * dist_factor * (1.0 + dist_factor)) * self.omega_step
                ## Way 1: similar with EPW Computer Physics Communications 209 (2016) 116–133
                resistance_Ziman = resist_strength * QTcts.FOURPI / self.ziman_nc / self.temperature
                resistance_Ziman = resistance_Ziman * QTcts.HARTREEtoOHM * QTcts.TENPP6 * QTcts.BOHRtoCM

                ## Way 2: similar with PHYSICAL REVIEW B 92, 075405 (2015)
                resistance_Allen = resist_strength * QTcts.TWOPI * self.volume_uc / self.temperature / dos_ef / np.mean(fermi_speed)**2
                resistance_Allen = resistance_Allen * QTcts.HARTREEtoOHM * QTcts.TENPP6 * QTcts.BOHRtoCM

                print('-'*100)
                print("Ziman's resistance(micro-Ohm*cm) = {:12.4e}".format(resistance_Ziman))
                print("Allen's resistance(micro-Ohm*cm) = {:12.4e}".format(resistance_Allen))
                print('-'*100)
