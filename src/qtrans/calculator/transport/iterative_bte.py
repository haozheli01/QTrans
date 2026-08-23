"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
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
