"""Mechanically extracted QTrans methods."""

from mpi4py import MPI
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.extensions import transport_eliashberg_function_cal_EPC_iso_helper_sparse
from qtrans.numerics.gaussian import w0gauss
from qtrans.parallel.mpi import print_peak_memeory
import numpy as np


class _TransportEliashbergMixin:
    def transport_eliashberg_function_cal_epc(self, k_grid, q_grid, dos_ef):
        """
        Calculate the tensor transport Eliashberg function with dense EPC gradients.

        This routine evaluates the velocity-weighted electron-phonon spectral
        function on full k/q grids. It is intended for metallic systems and uses
        Fermi-surface delta functions, band velocities, phonon frequencies, and dense
        EPC matrix elements to build a Cartesian tensor spectrum.

        Args:
            k_grid (np.ndarray): Full k-point grid in fractional coordinates,
                shape (nks, 3).
            q_grid (np.ndarray): Full q-point grid in fractional coordinates,
                shape (nqs, 3).
            dos_ef (float): Density of states at the Fermi level.

        Returns:
            tuple: ``(omegas_list, a2f)`` where ``omegas_list`` has shape
                (nomegas,) and ``a2f`` has shape (nomegas, 3, 3).
        """

        nks = len(k_grid)
        nqs = len(q_grid)
        nbranches = int(3) * self.natoms
        nbands = len(self.bands_indices)

        # initial omegas
        if self.omega_range[0] < QTcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros(nomegas)
        
        # q points are parallelized and q grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        q_grid = np.split(q_grid, indices_or_sections=split_sections, axis=0)
        if q_grid[self.rank].size>0:
            # calculate the phonon spectrum in parallel
            q_grid = q_grid[self.rank]
            freq_grid, phon_vecs = self._phonon_cal(q_grid)
            nqs_local = len(q_grid)
            phon_vecs = phon_vecs.reshape(nqs_local, nbranches, self.natoms, 3)
            # change fractional coordinates to cartesian coordinates
            q_grid = self._frac2car(q_grid)
        else:
            nqs_local = 0
            q_grid = np.empty((0, 3))
            freq_grid = np.empty((0, nbranches))
            phon_vecs = np.empty((0, nbranches, self.natoms, 3))
        
        # k points are parallelized and k grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nks):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_all = np.split(k_grid, indices_or_sections=split_sections, axis=0)
        if k_grid_all[self.rank].size > 0:            
            # eigen: (norbs, nk_local) eigen_vecs: (nk_local, norbs, norbs)
            eigen, eigen_vecs = self._elec_cal_torch(k_grid_all[self.rank])
            eigen = eigen[self.bands_indices,:]
            eigen_vecs = eigen_vecs[:, self.bands_indices, :]
            vnks = self._vel_nk_cal_from_HS(eigen, eigen_vecs, k_grid_all[self.rank]) # (nk_split, nbands, 3)
            delta_nks = w0gauss((eigen - self.efermi) * self.inv_smeark) * self.inv_smeark
            fermi_speed = np.einsum('knl, nk->l', vnks * vnks, delta_nks)
        else:
            eigen, eigen_vecs = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))
            vnks = np.empty((0, nbands, 3))
            fermi_speed = np.zeros(3)
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, fermi_speed, op=MPI.SUM)
        fermi_speed = np.sqrt(fermi_speed * self.weight_k / dos_ef)
        
        if self.rank == 0:
            print("Fermi speed(m/s)      x                y                z")
            print('               {:12.4e}      {:12.4e}      {:12.4e}'.format(*(fermi_speed*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)))

        epc_strengths = np.zeros((nqs_local, nbranches, 3, 3))
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'transport_eliashberg_function_cal_epc')
        # calculate electron-phonon coupling strength
        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv, vnks_recv = self.comm.bcast((eigen_vecs, eigen, vnks), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik]
                enk_match_table = np.abs(eig_k - self.efermi) < self.e_thr
                if not enk_match_table.any():
                    continue
                vnk = vnks_recv[ik]
                phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for iq, q in enumerate(q_grid):
                    kpq = k+q
                    eig_kpq, eigen_vecs_kpq = self._elec_cal_torch(kpq) # (norbs, 1), (1, norbs, norbs)
                    eig_kpq = eig_kpq[self.bands_indices, 0] # (nbands, )
                    wave_kpq = eigen_vecs_kpq[0, self.bands_indices, :] # (nbands, norbs)
                    emkq_match_table = np.abs(eig_kpq - self.efermi) < self.e_thr
                    if not emkq_match_table.any():
                        continue
                    atomic_phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                    # vmkq
                    vmkq = self._vel_nk_cal_from_HS(eig_kpq.reshape(nbands,1), wave_kpq.reshape(1,nbands,self.norbs), kpq) # (1, nbands, 3)
                    vmkq = vmkq[0]
                    # phonon spectrum
                    freq = freq_grid[iq]
                    eigen_vec_phon = phon_vecs[iq]
                    phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                    # cal epc
                    phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    for ibnd in range(nbands):
                        if not enk_match_table[ibnd]:
                            continue
                        for jbnd in range(nbands):
                            if not emkq_match_table[jbnd]:
                                continue
                            wave_coe1 = wave_k[ibnd] # shape: (norbs,)
                            wave_coe2 = wave_kpq[jbnd] # shape: (norbs,)
                            tmp1 = np.einsum('m,n -> mn', np.conj(wave_coe2), wave_coe1)
                            # velocity_factor: shape (3, 3)
                            velocity_factor = np.outer(vnk[ibnd], vnk[ibnd] - vmkq[jbnd])
                            for ibranch in range(nbranches):
                                if freq[ibranch] < self.phonon_frequency_cutoff:
                                    continue
                                epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                            tmp1, self.grad_mat, phase_k, phase_kpq,
                                                            dipole_correction=False)
                                g2_tmp = np.abs(epc) * np.abs(epc)
                                delta_nk = w0gauss((eig_k[ibnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                                delta_mkq = w0gauss((eig_kpq[jbnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                                epc_strengths[iq, ibranch] += g2_tmp * delta_nk * delta_mkq * velocity_factor
            if self.rank==0:
                logger.step(send_rank+1)
        epc_strengths = epc_strengths * self.weight_k / dos_ef # shape: (nqs[local], nmodes, 3, 3)
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        # a2f: shape (nomegas, 3, 3)
        a2f = np.einsum('qvab, wqv->wab', epc_strengths, delta_omegas) /nqs / QTcts.TWOPI
        a2f = a2f / np.outer(fermi_speed, fermi_speed)[None,:,:]
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
        print_peak_memeory(self.comm, self.rank)
        return omegas_list, a2f

    def transport_eliashberg_function_cal_epc_new(self, k_grid, q_grid, dos_ef):
        """
        Calculate an alternative tensor transport Eliashberg function.

        This dense-gradient variant uses a dimensionless Cartesian transport factor
        ``I - v(k) v(k+q) / (v(k) v(k))``. It otherwise follows the same
        Fermi-surface filtering and EPC accumulation workflow as
        ``transport_eliashberg_function_cal_epc``.

        Args:
            k_grid (np.ndarray): Full k-point grid in fractional coordinates,
                shape (nks, 3).
            q_grid (np.ndarray): Full q-point grid in fractional coordinates,
                shape (nqs, 3).
            dos_ef (float): Density of states at the Fermi level.

        Returns:
            tuple: ``(omegas_list, a2f)`` where ``omegas_list`` has shape
                (nomegas,) and ``a2f`` has shape (nomegas, 3, 3).
        """

        nks = len(k_grid)
        nqs = len(q_grid)
        nmodes = int(3) * self.natoms
        nbands = len(self.bands_indices)

        # initial omegas
        if self.omega_range[0] < QTcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros(nomegas)
        
        # q points are parallelized and q grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nqs):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        q_grid = np.split(q_grid, indices_or_sections=split_sections, axis=0)
        if q_grid[self.rank].size>0:
            # calculate the phonon spectrum in parallel
            q_grid = q_grid[self.rank]
            freq_grid, phon_vecs = self._phonon_cal(q_grid)
            nqs_local = len(q_grid)
            phon_vecs = phon_vecs.reshape(nqs_local, nmodes, self.natoms, 3)
            # change fractional coordinates to cartesian coordinates
            q_grid = self._frac2car(q_grid)
        else:
            nqs_local = 0
            q_grid = np.empty((0, 3))
            freq_grid = np.empty((0, nmodes))
            phon_vecs = np.empty((0, nmodes, self.natoms, 3))
        
        # k points are parallelized and k grid is split
        split_sections = np.zeros(self.rank_size, dtype=int)
        for i in range(nks):
            split_sections[i%self.rank_size] += 1
        split_sections = np.cumsum(split_sections, axis=0)
        k_grid_all = np.split(k_grid, indices_or_sections=split_sections, axis=0)
        if k_grid_all[self.rank].size > 0:            
            # eigen: (norbs, nk_local) eigen_vecs: (nk_local, norbs, norbs)
            eigen, eigen_vecs = self._elec_cal_torch(k_grid_all[self.rank])
            eigen = eigen[self.bands_indices,:]
            eigen_vecs = eigen_vecs[:, self.bands_indices, :]
            vnks = self._vel_nk_cal_from_HS(eigen, eigen_vecs, k_grid_all[self.rank]) # (nk_split, nbands, 3)
            delta_nks = w0gauss((eigen - self.efermi) * self.inv_smeark) * self.inv_smeark
            fermi_speed = np.einsum('knl, nk->l', vnks * vnks, delta_nks)
        else:
            eigen, eigen_vecs = np.empty((nbands, 0)), np.empty((0, nbands, self.norbs))
            vnks = np.empty((0, nbands, 3))
            fermi_speed = np.zeros(3)
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, fermi_speed, op=MPI.SUM)
        fermi_speed = np.sqrt(fermi_speed * self.weight_k / dos_ef)
        
        if self.rank == 0:
            print("Fermi speed(m/s)      x                y                z")
            print('               {:12.4e}      {:12.4e}      {:12.4e}'.format(*(fermi_speed*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)))

        epc_strengths = np.zeros((nqs_local, nmodes, 3, 3))
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'transport_eliashberg_function_cal_epc')
        # calculate electron-phonon coupling strength
        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv, vnks_recv = self.comm.bcast((eigen_vecs, eigen, vnks), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik]
                enk_match_table = np.abs(eig_k - self.efermi) < self.e_thr
                if not enk_match_table.any():
                    continue
                vnk = vnks_recv[ik]
                phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for iq, q in enumerate(q_grid):
                    kpq = k+q
                    eig_kpq, eigen_vecs_kpq = self._elec_cal_torch(kpq) # (norbs, 1), (1, norbs, norbs)
                    eig_kpq = eig_kpq[self.bands_indices, 0] # (nbands, )
                    wave_kpq = eigen_vecs_kpq[0, self.bands_indices, :] # (nbands, norbs)
                    emkq_match_table = np.abs(eig_kpq - self.efermi) < self.e_thr
                    if not emkq_match_table.any():
                        continue
                    atomic_phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                    # vmkq
                    vmkq = self._vel_nk_cal_from_HS(eig_kpq.reshape(nbands,1), wave_kpq.reshape(1,nbands,self.norbs), kpq) # (1, nbands, 3)
                    vmkq = vmkq[0]
                    # phonon spectrum
                    freq = freq_grid[iq]
                    eigen_vec_phon = phon_vecs[iq]
                    phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                    # cal epc
                    phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    for ibnd in range(nbands):
                        if not enk_match_table[ibnd]:
                            continue
                        for jbnd in range(nbands):
                            if not emkq_match_table[jbnd]:
                                continue
                            wave_coe1 = wave_k[ibnd] # shape: (norbs,)
                            wave_coe2 = wave_kpq[jbnd] # shape: (norbs,)
                            tmp1 = np.einsum('m,n -> mn', np.conj(wave_coe2), wave_coe1)
                            # velocity_factor: shape (3, 3)
                            velocity_factor = np.eye(3) - np.outer(vnk[ibnd], vmkq[jbnd])/np.outer(vnk[ibnd], vnk[ibnd])
                            #velocity_factor = (1 - np.dot(vnk[ibnd], vmkq[jbnd]) / np.linalg.norm(vnk[ibnd])**2 ) * np.eye(3)
                            for ibranch in range(nmodes):
                                if freq[ibranch] < self.phonon_frequency_cutoff:
                                    continue
                                epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                            tmp1, self.grad_mat, phase_k, phase_kpq,
                                                            dipole_correction=False)
                                g2_tmp = np.abs(epc) * np.abs(epc)
                                delta_nk = w0gauss((eig_k[ibnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                                delta_mkq = w0gauss((eig_kpq[jbnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                                epc_strengths[iq, ibranch] += g2_tmp * delta_nk * delta_mkq * velocity_factor
            if self.rank==0:
                logger.step(send_rank+1)
        epc_strengths = epc_strengths * self.weight_k / dos_ef # shape: (nqs[local], nmodes, 3, 3)
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        # a2f: shape (nomegas, 3, 3)
        a2f = np.einsum('qvab, wqv->wab', epc_strengths, delta_omegas) /nqs / 2.0
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
        return omegas_list, a2f

    def transport_eliashberg_function_cal_epc_iso_dense(self, k_grid, q_grid, dos_ef):
        """
        Calculate isotropic Eliashberg and transport Eliashberg functions using dense gradients.

        The routine computes both the ordinary Eliashberg spectral function ``a2f``
        and the transport-weighted spectrum ``a2f_tr`` for a non-magnetic metallic
        system. The transport weighting uses the MRTA velocity factor.

        Args:
            k_grid (np.ndarray): Full k-point grid in fractional coordinates,
                shape (nks, 3).
            q_grid (np.ndarray): Full q-point grid in fractional coordinates,
                shape (nqs, 3).
            dos_ef (float): Density of states at the Fermi level per spin.

        Returns:
            tuple: ``(omegas_list, a2f, a2f_tr, fermi_speed)``. ``omegas_list``,
                ``a2f``, and ``a2f_tr`` have shape (nomegas,), and
                ``fermi_speed`` has shape (3,).
        """
        # ref: PHYSICAL REVIEW B 92, 075405 (2015) & Computer Physics Communications 209 (2016) 116–133

        nks = len(k_grid)
        nqs = len(q_grid)
        nbranches = int(3) * self.natoms
        nbands = len(self.bands_indices)

        # initial omegas
        if self.omega_range[0] < QTcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros(nomegas)
        a2f_tr = np.zeros(nomegas)
        # k_grid_all: shape LIST[(nks_local, 3)]
        # eigen: shape (nbands, nks_local)
        # eigen_vecs: shape (nks_local, nbands, norbs)
        # vnks: shape (nks_local, nbands, 3)
        # q_grid_local: shape (nqs_local, 3)
        # weights_q_local: shape (nqs_local, )
        # freq_grid: shape (nqs_local, nbranches)
        # boson_vecs: shape (nqs_local, nbranches, natoms_uc, 3)
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, _, freq_grid, phon_vecs, fermi_speed = \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, self.efermi, is_metal=True)
        
        fermi_speed = np.sqrt(fermi_speed * (self.weight_k / 2) / dos_ef) # Because of dos_ef = dos per spin, thus weight_k should not count twice!!!

        if self.rank == 0:
            print("Fermi speed(m/s)      x                y                z")
            print('               {:12.4e}      {:12.4e}      {:12.4e}'.format(*(fermi_speed*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)))
        
        nqs_local = len(q_grid_local)
        epc_strengths = np.zeros((nqs_local, nbranches))
        epc_strengths_tr = np.zeros((nqs_local, nbranches))
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'transport_eliashberg_function_cal_epc_iso_dense')

        # calculate electron-phonon coupling strength
        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv, vnks_recv = self.comm.bcast((eigen_vecs, eigen, vnks), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik]
                enk_match_table = np.abs(eig_k - self.efermi) < self.e_thr
                if not enk_match_table.any():
                    continue
                vnk = vnks_recv[ik]
                phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    kpq = k+q
                    eig_kpq, eigen_vecs_kpq = self._elec_cal_torch(kpq) # (norbs, 1), (1, norbs, norbs)
                    eig_kpq = eig_kpq[self.bands_indices, 0] # (nbands, )
                    wave_kpq = eigen_vecs_kpq[0, self.bands_indices, :] # (nbands, norbs)
                    emkq_match_table = np.abs(eig_kpq - self.efermi) < self.e_thr
                    if not emkq_match_table.any():
                        continue
                    atomic_phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                    # vmkq
                    vmkq = self._vel_nk_cal_from_HS(eig_kpq.reshape(nbands,1), wave_kpq.reshape(1,nbands,self.norbs), kpq) # (1, nbands, 3)
                    vmkq = vmkq[0]
                    # phonon spectrum
                    freq = freq_grid[iq]
                    eigen_vec_phon = phon_vecs[iq]
                    phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                    # cal epc
                    phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    for ibnd in range(nbands):
                        if not enk_match_table[ibnd]:
                            continue
                        for jbnd in range(nbands):
                            if not emkq_match_table[jbnd]:
                                continue
                            wave_coe1 = wave_k[ibnd] # shape: (norbs,)
                            wave_coe2 = wave_kpq[jbnd] # shape: (norbs,)
                            tmp1 = np.einsum('m,n -> mn', np.conj(wave_coe2), wave_coe1)
                            # MRTA
                            if np.linalg.norm(vnk[ibnd])**2 > QTcts.TENPM4: # sanity check
                                velocity_factor = 1.0 - np.dot(vnk[ibnd], vmkq[jbnd]) / np.linalg.norm(vnk[ibnd])**2  
                                #velocity_factor = 1.0 - np.dot(vnk[ibnd], vmkq[jbnd]) / np.linalg.norm(vnk[ibnd]) / np.linalg.norm(vmkq[jbnd])
                            else:
                                velocity_factor = 1.0
                            # SERTA
                            #velocity_factor = 1.0

                            for ibranch in range(nbranches):
                                if freq[ibranch] < self.phonon_frequency_cutoff:
                                    continue
                                epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                            tmp1, self.grad_mat, phase_k, phase_kpq,
                                                            dipole_correction=False)
                                g2_tmp = np.abs(epc) * np.abs(epc)
                                delta_nk = w0gauss((eig_k[ibnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                                delta_mkq = w0gauss((eig_kpq[jbnd] - self.efermi) * self.inv_smeark) * self.inv_smeark

                                ## LHZ: there are some problems within adaptive smearing!
                                # inv_smeark_nk = 1.0 / self._adapt_smearing(self.k_size, vnk[ibnd])
                                # delta_nk = w0gauss((eig_k[ibnd] - self.efermi) * inv_smeark_nk) * inv_smeark_nk
                                # if self.rank == 0:
                                #     print('smearing:', self._adapt_smearing(self.k_size, vnk[ibnd]) * QTcts.HARTREEtoMEV)
                                # inv_smeark_mkq = 1.0 / self._adapt_smearing(self.k_size, vmkq[jbnd])
                                # delta_mkq = w0gauss((eig_kpq[jbnd] - self.efermi) * inv_smeark_mkq) * inv_smeark_mkq

                                epc_strengths[iq, ibranch] = epc_strengths[iq, ibranch] + g2_tmp * delta_nk * delta_mkq
                                epc_strengths_tr[iq, ibranch] = epc_strengths_tr[iq, ibranch] + g2_tmp * delta_nk * delta_mkq * velocity_factor
            if self.rank==0:
                logger.step(send_rank+1)
        # epc_strengths / freq_qv is the lambda
        epc_strengths = epc_strengths * self.weight_k / dos_ef
        epc_strengths_tr = epc_strengths_tr * self.weight_k / dos_ef
        epc_strengths_tr[epc_strengths_tr < 0] = 0
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        # a2f, a2f_tr: shape (nomegas)
        a2f = np.einsum('qv, wqv->w', epc_strengths, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        a2f_tr = np.einsum('qv, wqv->w', epc_strengths_tr, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        # The factor 2 in PRB 92, 075405 (2015) formula (26) has been considered in weight_k.
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
            self.comm.Allreduce(MPI.IN_PLACE, a2f_tr, op=MPI.SUM)
        return omegas_list, a2f, a2f_tr, fermi_speed

    def transport_eliashberg_function_cal_epc_iso_sparse(self, k_grid, q_grid, dos_ef):
        """
        Calculate isotropic Eliashberg and transport spectra using sparse gradients.

        This is the C-accelerated sparse-gradient counterpart of
        ``transport_eliashberg_function_cal_epc_iso_dense``. It computes ``a2f`` and
        ``a2f_tr`` for non-magnetic metallic systems and accumulates mode-resolved EPC
        strengths on each local q point.

        Args:
            k_grid (np.ndarray): Full k-point grid in fractional coordinates,
                shape (nks, 3).
            q_grid (np.ndarray): Full q-point grid in fractional coordinates,
                shape (nqs, 3).
            dos_ef (float): Density of states at the Fermi level per spin.

        Returns:
            tuple: ``(omegas_list, a2f, a2f_tr, fermi_speed)``. ``omegas_list``,
                ``a2f``, and ``a2f_tr`` have shape (nomegas,), and
                ``fermi_speed`` has shape (3,).
        """
        # ref: PHYSICAL REVIEW B 92, 075405 (2015) & Computer Physics Communications 209 (2016) 116–133

        nks = len(k_grid)
        nqs = len(q_grid)
        nmodes = int(3) * self.natoms
        nbands = len(self.bands_indices)

        # initial omegas
        if self.omega_range[0] < QTcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros(nomegas)
        a2f_tr = np.zeros(nomegas)
        
        # k_grid_all: shape LIST[(nks_local, 3)]
        # eigen: shape (nbands, nks_local)
        # eigen_vecs: shape (nks_local, nbands, norbs)
        # vnks: shape (nks_local, nbands, 3)
        # q_grid_local: shape (nqs_local, 3)
        # weights_q_local: shape (nqs_local, )
        # freq_grid: shape (nqs_local, nbranches)
        # boson_vecs: shape (nqs_local, nbranches, natoms_uc, 3)
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, _, freq_grid, phon_vecs, fermi_speed = \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, self.efermi, is_metal=True)
        fermi_speed = np.sqrt(fermi_speed * (self.weight_k / 2) / dos_ef) # Because of dos_ef = dos per spin, thus weight_k should not count twice!!!

        if self.rank == 0:
            print("Fermi speed(m/s)      x                y                z")
            print('               {:12.4e}      {:12.4e}      {:12.4e}'.format(*(fermi_speed*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)))
        nqs_local = len(q_grid_local)
        epc_strengths = np.zeros((nqs_local, nmodes))
        epc_strengths_tr = np.zeros((nqs_local, nmodes))
        atomic_mass = np.ascontiguousarray(self.atomic_mass)
        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [
            np.ascontiguousarray(atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq])
            for iq in range(nqs_local)
        ]
        phase_q_all = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ q_grid_local.T))
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'transport_eliashberg_function_cal_epc_iso_sparse')

        # calculate electron-phonon coupling strength
        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv, vnks_recv = self.comm.bcast((eigen_vecs, eigen, vnks), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:, ik], eigen_vecs_recv[ik]
                enk_match_table = np.abs(eig_k - self.efermi) < self.e_thr
                if not enk_match_table.any():
                    continue
                vnk = vnks_recv[ik]
                # vnk = vnk / np.linalg.norm(vnk, axis=1)[:,None]
                phase_k = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ k)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    kpq = k+q
                    eig_kpq, eigen_vecs_kpq = self._elec_cal_torch(kpq) # (norbs, 1), (1, norbs, norbs)
                    eig_kpq = eig_kpq[self.bands_indices, 0] # (nbands, )
                    wave_kpq = eigen_vecs_kpq[0, self.bands_indices, :] # (nbands, norbs)
                    emkq_match_table = np.abs(eig_kpq - self.efermi) < self.e_thr
                    if not emkq_match_table.any():
                        continue
                    # vmkq
                    vmkq = self._vel_nk_cal_from_HS(eig_kpq.reshape(nbands,1), wave_kpq.reshape(1,nbands,self.norbs), kpq) # (1, nbands, 3)
                    vmkq = vmkq[0]
                    # vmkq = vmkq / np.linalg.norm(vmkq, axis=1)[:,None]
                    epc_strengths_q = np.zeros((self.natoms*3,), dtype=float, order='C')
                    epc_strengths_tr_q = np.zeros((self.natoms*3,), dtype=float, order='C')
                    # phonon spectrum
                    freq = freq_grid[iq]
                    phvec_wap = phvec_waps_all[iq] # shape (nbranches, natoms, 3)
                    # cal epc
                    phase_kpq = phase_k * phase_q_all[:, iq] # shape: (ncells,)
                    transport_eliashberg_function_cal_EPC_iso_helper_sparse(
                        np.ascontiguousarray(freq), atomic_mass, phvec_wap,
                        np.ascontiguousarray(phase_k), np.ascontiguousarray(phase_kpq), self.grad_mat, np.ascontiguousarray(wave_k),
                        np.ascontiguousarray(wave_kpq), np.ascontiguousarray(enk_match_table), np.ascontiguousarray(emkq_match_table), np.ascontiguousarray(eig_k),
                        np.ascontiguousarray(eig_kpq), np.ascontiguousarray(vnk), np.ascontiguousarray(vmkq), epc_strengths_q, epc_strengths_tr_q,
                        self.phonon_frequency_cutoff, self.inv_smeark, self.efermi)
                    epc_strengths[iq] = epc_strengths[iq] + epc_strengths_q
                    epc_strengths_tr[iq] = epc_strengths_tr[iq] + epc_strengths_tr_q
            if self.rank==0:
                logger.step(send_rank+1)
        # epc_strengths / freq_qv is the lambda
        epc_strengths = epc_strengths * self.weight_k / dos_ef
        epc_strengths_tr = epc_strengths_tr * self.weight_k / dos_ef
        epc_strengths_tr[epc_strengths_tr < 0] = 0
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        # a2f, a2f_tr: shape (nomegas)
        a2f = np.einsum('qv, wqv->w', epc_strengths, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        a2f_tr = np.einsum('qv, wqv->w', epc_strengths_tr, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        # The factor 2 in PRB 92, 075405 (2015) formula (26) has been considered in weight_k.
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
            self.comm.Allreduce(MPI.IN_PLACE, a2f_tr, op=MPI.SUM)
        print_peak_memeory(self.comm, self.rank)
        return omegas_list, a2f, a2f_tr, fermi_speed

    def transport_eliashberg_function_cal_epc_iso_collinear_dense(self, k_grid, q_grid, dos_ef):
        """
        Calculate spin-resolved isotropic Eliashberg spectra with dense gradients.

        This collinear-spin dense path evaluates ``a2f`` and ``a2f_tr`` separately
        for spin-up and spin-down channels. Each spin channel uses its own DOS,
        velocity, and dense gradient matrix.

        Args:
            k_grid (np.ndarray): Full k-point grid in fractional coordinates,
                shape (nks, 3).
            q_grid (np.ndarray): Full q-point grid in fractional coordinates,
                shape (nqs, 3).
            dos_ef (Sequence[float]): Spin-resolved density of states at the Fermi
                level, length 2.

        Returns:
            tuple: ``(omegas_list, a2f, a2f_tr, fermi_speed)``. ``a2f`` and
                ``a2f_tr`` have shape (nomegas, 2), and ``fermi_speed`` has shape
                (2, 3).
        """
        # ref: PHYSICAL REVIEW B 92, 075405 (2015) & Computer Physics Communications 209 (2016) 116–133

        nks = len(k_grid)
        nqs = len(q_grid)
        nbranches = int(3) * self.natoms
        nbands = len(self.bands_indices)

        # initial omegas
        if self.omega_range[0] < QTcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros((nomegas,2))
        a2f_tr = np.zeros((nomegas,2))
        # k_grid_all: shape LIST[(nks_local, 3)]
        # eigen: shape (2, nbands, nks_local)
        # eigen_vecs: shape (2, nks_local, nbands, norbs)
        # vnks: shape (2, nks_local, nbands, 3)
        # q_grid_local: shape (nqs_local, 3)
        # weights_q_local: shape (nqs_local, )
        # freq_grid: shape (nqs_local, nbranches)
        # boson_vecs: shape (nqs_local, nbranches, natoms_uc, 3)
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, fermi_speed = \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, self.efermi, is_metal=True)
        
        fermi_speed = np.sqrt(fermi_speed * self.weight_k / np.array(dos_ef)[:, None]) #(2,3)

        if self.rank == 0:
            for ispin in range(2):
                print(f"Fermi speed[spin-{ispin+1}](m/s)  x                y                z")
                print('                   {:12.4e}      {:12.4e}      {:12.4e}'.format(*(fermi_speed[ispin]*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)))
                print('-'*100)
        nqs_local = len(q_grid_local)
        epc_strengths = np.zeros((2, nqs_local, nbranches))
        epc_strengths_tr = np.zeros((2, nqs_local, nbranches))
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'transport_eliashberg_function_cal_epc_iso_collinear')

        # calculate electron-phonon coupling strength
        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv, vnks_recv = self.comm.bcast((eigen_vecs, eigen, vnks), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:,:,ik], eigen_vecs_recv[:,ik]
                enk_match_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                if not enk_match_table.any():
                    continue
                vnk = vnks_recv[:,ik]
                phase_k = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*k[None,:], axis=-1)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    kpq = k+q
                    freq = freq_grid[iq]
                    eigen_vec_phon = phon_vecs[iq]
                    eig_kpq, eigen_vecs_kpq = self._elec_cal_torch(kpq) # (2*norbs, 1), (1, 2*norbs, norbs)
                    eig_kpq = np.stack((eig_kpq[self.bands_indices, 0], eig_kpq[self.bands_indices+self.norbs, 0]), axis=0)
                    wave_kpq = np.stack((eigen_vecs_kpq[0, self.bands_indices], eigen_vecs_kpq[0, self.bands_indices+self.norbs]), axis=0)
                    emkq_match_table = (eig_kpq >= efocus_threshold[0]) & (eig_kpq <= efocus_threshold[1])
                    if not emkq_match_table.any():
                        continue
                    atomic_phase = np.exp(QTcts.JTWOPI*np.sum(self.graph_data.pos*q[None,:], axis=-1)) # shape (natoms, )
                    phvec_wap = atomic_phase[None,:,None] * eigen_vec_phon # shape (nbranches, natoms, 3)
                    phase_kpq = np.exp(QTcts.JTWOPI*np.sum(self.nbr_shift_of_cell_sc*(kpq)[None,:], axis=-1)) # shape: (ncells,)
                    for ispin in range(2):
                        vmkq = self._vel_nk_cal_from_HS(eig_kpq[ispin].reshape(nbands,1), wave_kpq[ispin].reshape(1,nbands,self.norbs), kpq, collinear_part_index = ispin) # (1, nbands, 3)
                        vmkq = vmkq[0]
                        # cal epc
                        for ibnd in (np.where(enk_match_table[ispin])[0]):
                            for jbnd in (np.where(emkq_match_table[ispin])[0]):
                                wave_coe1 = wave_k[ispin,ibnd] # shape: (norbs,)
                                wave_coe2 = wave_kpq[ispin,jbnd] # shape: (norbs,)
                                tmp1 = np.einsum('m,n -> mn', np.conj(wave_coe2), wave_coe1)
                                # MRTA
                                if np.linalg.norm(vnk[ispin,ibnd])**2 > QTcts.TENPM4: # sanity check
                                    velocity_factor = 1.0 - np.dot(vnk[ispin,ibnd], vmkq[jbnd]) / np.linalg.norm(vnk[ispin,ibnd])**2  
                                else:
                                    velocity_factor = 1.0
                                # SERTA
                                #velocity_factor = 1.0
                                for ibranch in range(nbranches):
                                    if freq[ibranch] < self.phonon_frequency_cutoff:
                                        continue
                                    epc = self._epc_cal_element(k, q, freq[ibranch], phvec_wap[ibranch],
                                                            tmp1, self.grad_mat[ispin], phase_k, phase_kpq,
                                                            dipole_correction=False)
                                    g2_tmp = np.abs(epc) * np.abs(epc)
                                    delta_nk = w0gauss((eig_k[ispin,ibnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                                    delta_mkq = w0gauss((eig_kpq[ispin,jbnd] - self.efermi) * self.inv_smeark) * self.inv_smeark
                                    epc_strengths[ispin, iq, ibranch] += g2_tmp * delta_nk * delta_mkq
                                    epc_strengths_tr[ispin, iq, ibranch] += g2_tmp * delta_nk * delta_mkq * velocity_factor
            if self.rank==0:
                logger.step(send_rank+1)
        # epc_strengths / freq_qv is the lambda
        epc_strengths = epc_strengths * self.weight_k / np.array(dos_ef)[:, None, None]
        epc_strengths_tr = epc_strengths_tr * self.weight_k / np.array(dos_ef)[:, None, None]
        epc_strengths_tr[epc_strengths_tr < 0] = 0 #sanity check
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        # a2f, a2f_tr: shape (nomegas,2)
        a2f = np.einsum('sqv, wqv->ws', epc_strengths, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        a2f_tr = np.einsum('sqv, wqv->ws', epc_strengths_tr, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        # The factor 2 in PRB 92, 075405 (2015) formula (26) has been considered in weight_k.
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
            self.comm.Allreduce(MPI.IN_PLACE, a2f_tr, op=MPI.SUM)
        return omegas_list, a2f, a2f_tr, fermi_speed

    def transport_eliashberg_function_cal_epc_iso_collinear_sparse(self, k_grid, q_grid, dos_ef):
        """
        Calculate spin-resolved isotropic Eliashberg spectra with sparse gradients.

        This is the C-accelerated sparse-gradient counterpart of
        ``transport_eliashberg_function_cal_epc_iso_collinear_dense``. It computes
        ordinary and transport-weighted Eliashberg spectra independently for the two
        collinear spin channels.

        Args:
            k_grid (np.ndarray): Full k-point grid in fractional coordinates,
                shape (nks, 3).
            q_grid (np.ndarray): Full q-point grid in fractional coordinates,
                shape (nqs, 3).
            dos_ef (Sequence[float]): Spin-resolved density of states at the Fermi
                level, length 2.

        Returns:
            tuple: ``(omegas_list, a2f, a2f_tr, fermi_speed)``. ``a2f`` and
                ``a2f_tr`` have shape (nomegas, 2), and ``fermi_speed`` has shape
                (2, 3).
        """
        # ref: PHYSICAL REVIEW B 92, 075405 (2015) & Computer Physics Communications 209 (2016) 116–133

        nks = len(k_grid)
        nqs = len(q_grid)
        nmodes = int(3) * self.natoms
        nbands = len(self.bands_indices)

        # initial omegas
        if self.omega_range[0] < QTcts.TENPM10:
            omegas_list = np.arange(self.omega_range[0]+self.omega_step, self.omega_range[1]+self.omega_step/10, self.omega_step)
        else:
            omegas_list = np.arange(self.omega_range[0], self.omega_range[1]+self.omega_step/10, self.omega_step)
        nomegas = len(omegas_list)
        a2f = np.zeros((nomegas,2))
        a2f_tr = np.zeros((nomegas,2))
        # k_grid_all: shape LIST[(nks_local, 3)]
        # eigen: shape (2, nbands, nks_local)
        # eigen_vecs: shape (2, nks_local, nbands, norbs)
        # vnks: shape (2, nks_local, nbands, 3)
        # q_grid_local: shape (nqs_local, 3)
        # weights_q_local: shape (nqs_local, )
        # freq_grid: shape (nqs_local, nbranches)
        # boson_vecs: shape (nqs_local, nbranches, natoms_uc, 3)
        efocus_threshold, k_grid_all, eigen, eigen_vecs, vnks, q_grid_local, weights_q_local, freq_grid, phon_vecs, fermi_speed = \
            self._transport_prepare(k_grid, q_grid, self.bands_indices, self.efermi, is_metal=True)
        
        fermi_speed = np.sqrt(fermi_speed * self.weight_k / np.array(dos_ef)[:, None]) #(2,3)

        if self.rank == 0:
            for ispin in range(2):
                print(f"Fermi speed[spin-{ispin+1}](m/s)  x                y                z")
                print('                   {:12.4e}      {:12.4e}      {:12.4e}'.format(*(fermi_speed[ispin]*QTcts.BOHRtoM*QTcts.HARTREEtoINVS)))
                print('-'*100)
        nqs_local = len(q_grid_local)
        epc_strengths = np.zeros((2, nqs_local, nmodes))
        epc_strengths_tr = np.zeros((2, nqs_local, nmodes))
        atomic_mass = np.ascontiguousarray(self.atomic_mass)
        atomic_phases_all = np.exp(QTcts.JTWOPI * (self.graph_data.pos @ q_grid_local.T))
        phvec_waps_all = [
            np.ascontiguousarray(atomic_phases_all[:, iq][None, :, None] * phon_vecs[iq])
            for iq in range(nqs_local)
        ]
        phase_q_all = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ q_grid_local.T))
        if self.rank == 0:
            logger = time_logger(self.rank_size, 'transport_eliashberg_function_cal_epc_iso_collinear')

        # calculate electron-phonon coupling strength
        ik_all = -1
        for send_rank in range(self.rank_size):
            eigen_vecs_recv, eigen_recv, vnks_recv = self.comm.bcast((eigen_vecs, eigen, vnks), root=send_rank)
            for ik, k in enumerate(k_grid_all[send_rank]):
                ik_all += 1
                eig_k, wave_k = eigen_recv[:,:,ik], eigen_vecs_recv[:,ik]
                enk_match_table = (eig_k >= efocus_threshold[0]) & (eig_k <= efocus_threshold[1])
                if not enk_match_table.any():
                    continue
                vnk = vnks_recv[:,ik]
                phase_k = np.exp(QTcts.JTWOPI * (self.nbr_shift_of_cell_sc @ k)) # shape: (ncells,)
                for iq, q in enumerate(q_grid_local):
                    kpq = k+q
                    freq = freq_grid[iq]
                    phvec_wap = phvec_waps_all[iq] # shape (nbranches, natoms, 3)
                    eig_kpq, eigen_vecs_kpq = self._elec_cal_torch(kpq) # (2*norbs, 1), (1, 2*norbs, norbs)
                    eig_kpq = np.stack((eig_kpq[self.bands_indices, 0], eig_kpq[self.bands_indices+self.norbs, 0]), axis=0)
                    wave_kpq = np.stack((eigen_vecs_kpq[0, self.bands_indices], eigen_vecs_kpq[0, self.bands_indices+self.norbs]), axis=0)
                    emkq_match_table = (eig_kpq >= efocus_threshold[0]) & (eig_kpq <= efocus_threshold[1])
                    if not emkq_match_table.any():
                        continue
                    phase_kpq = phase_k * phase_q_all[:, iq] # shape: (ncells,)
                    for ispin in range(2):
                        vmkq = self._vel_nk_cal_from_HS(eig_kpq[ispin].reshape(nbands,1), wave_kpq[ispin].reshape(1,nbands,self.norbs), kpq, collinear_part_index = ispin) # (1, nbands, 3)
                        vmkq = vmkq[0]
                        epc_strengths_q = np.zeros((nmodes,), dtype=float, order='C')
                        epc_strengths_tr_q = np.zeros((nmodes,), dtype=float, order='C')
                        transport_eliashberg_function_cal_EPC_iso_helper_sparse(
                            np.ascontiguousarray(freq), atomic_mass, phvec_wap,
                            np.ascontiguousarray(phase_k), np.ascontiguousarray(phase_kpq), self.grad_mat[ispin], np.ascontiguousarray(wave_k[ispin]),
                            np.ascontiguousarray(wave_kpq[ispin]), np.ascontiguousarray(enk_match_table[ispin]), np.ascontiguousarray(emkq_match_table[ispin]), np.ascontiguousarray(eig_k[ispin]),
                            np.ascontiguousarray(eig_kpq[ispin]), np.ascontiguousarray(vnk[ispin]), np.ascontiguousarray(vmkq), epc_strengths_q, epc_strengths_tr_q,
                            self.phonon_frequency_cutoff, self.inv_smeark, self.efermi)
                        epc_strengths[ispin, iq] += epc_strengths_q
                        epc_strengths_tr[ispin, iq] += epc_strengths_tr_q
            if self.rank==0:
                logger.step(send_rank+1)
        # epc_strengths / freq_qv is the lambda
        epc_strengths = epc_strengths * self.weight_k / np.array(dos_ef)[:, None, None]
        epc_strengths_tr = epc_strengths_tr * self.weight_k / np.array(dos_ef)[:, None, None]
        epc_strengths_tr[epc_strengths_tr < 0] = 0
        delta_omegas = w0gauss(((freq_grid)[None,:,:] - omegas_list[:,None,None]) * self.inv_smearq) * self.inv_smearq # shape: (nomegas, nqs[local], nmodes)
        # a2f, a2f_tr: shape (nomegas,2)
        a2f = np.einsum('sqv, wqv->ws', epc_strengths, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        a2f_tr = np.einsum('sqv, wqv->ws', epc_strengths_tr, delta_omegas) /nqs / 2.0 # consistent with EPW. 
        # The factor 2 in PRB 92, 075405 (2015) formula (26) has been considered in weight_k.
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, a2f, op=MPI.SUM)
            self.comm.Allreduce(MPI.IN_PLACE, a2f_tr, op=MPI.SUM)
        return omegas_list, a2f, a2f_tr, fermi_speed
