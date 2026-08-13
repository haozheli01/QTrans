"""Mechanically extracted QTrans methods."""

from qtrans.data.basis import basis_def_dicts
from qtrans.numerics.sparse import build_reciprocal_from_sparseMat
from qtrans.numerics.sparse import build_reciprocal_from_sparseMat_collinear_split
from qtrans.numerics.sparse import build_reciprocal_from_sparseMat_soc
from scipy.linalg import block_diag
from scipy.linalg import eigh
import numpy as np
import torch


class _ElectronMixin:
    def _elec_cal_torch(self, k_grid):
        k_vec = k_grid.reshape(-1, 3)
        if self.spinor_mode == 'nc':
            eigen = []
            eigen_vecs = []
            for ik in range(len(k_vec)):
                # SK shape: (1, norbs/2, norbs/2) to (norbs/2, norbs/2)
                SK = build_reciprocal_from_sparseMat(self.graph_data.S_cell, k_vec[[ik]], self.graph_data.nbr_shift_of_cell)
                SK = np.squeeze(SK, axis=0)
                I = np.identity(2, dtype=SK.dtype)
                SK = np.kron(I,SK)
                # HK shape: (1, norbs, norbs) to (norbs, norbs)
                HK = build_reciprocal_from_sparseMat(self.graph_data.H_cell, k_vec[[ik]], self.graph_data.nbr_shift_of_cell)
                HK = np.squeeze(HK, axis=0)
                L = np.linalg.cholesky(SK)
                L_t = np.transpose(L.conj())
                L_inv = np.linalg.inv(L)
                L_t_inv = np.linalg.inv(L_t)
                Hs_np = np.matmul(np.matmul(L_inv, HK), L_t_inv)
                Hs = torch.from_numpy(Hs_np).to(torch.complex128).contiguous()
                w, v = torch.linalg.eigh(Hs)
                # The first eigen value is related to the v[:, 0]
                v = np.matmul(L_t_inv, v.numpy())
                eigen.append(w.numpy())
                eigen_vecs.append(v)
            eigen = np.swapaxes(np.array(eigen), 0, 1) # (norbs, nks)
            eigen_vecs = np.array(eigen_vecs) # (nks, norbs, norbs)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
        elif self.spinor_mode == 'collinear':
            eigen = []
            eigen_vecs = []
            for ik in range(len(k_vec)):
                # SK shape: (1, norbs, norbs) to (norbs, norbs)
                SK = build_reciprocal_from_sparseMat(self.graph_data.S_cell, k_vec[[ik]], self.graph_data.nbr_shift_of_cell)
                SK = np.squeeze(SK, axis=0)
                # HK shape: (2, 1, norbs, norbs) to (2, norbs, norbs)
                HK = build_reciprocal_from_sparseMat_collinear_split(self.graph_data.H_cell, k_vec[[ik]], self.graph_data.nbr_shift_of_cell)
                HK = np.squeeze(HK, axis=1)
                L = np.linalg.cholesky(SK)
                L_t = np.transpose(L.conj())
                L_inv = np.linalg.inv(L)
                L_t_inv = np.linalg.inv(L_t)
                Hs_list = []
                for ispin in range(2):
                    Hs = np.matmul(np.matmul(L_inv, HK[ispin]), L_t_inv)
                    Hs_list.append(Hs)
                Hs_np = np.stack(Hs_list, axis=0)
                Hs = torch.from_numpy(Hs_np).to(torch.complex128).contiguous()
                # w shape: (2, norbs)
                # v shape: (2, norbs, norbs)
                w, v = torch.linalg.eigh(Hs)
                # The first eigen value is related to the v[ispin, :, 0]
                v = np.matmul(L_t_inv[None, :, :], v.numpy())
                eigen.append(w.numpy())
                eigen_vecs.append(v)
            eigen = np.array(eigen)
            eigen = eigen.reshape(-1, 2*self.norbs)
            eigen = np.swapaxes(eigen, 0, 1) # (2*norbs, nks)
            eigen_vecs = np.array(eigen_vecs) # (nks, 2, norbs, norbs)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2) # (nks, 2, norbs, norbs)
            eigen_vecs = eigen_vecs.reshape(-1, 2*self.norbs, self.norbs) # (nks, 2*norbs, norbs)
        else:
            eigen = []
            eigen_vecs = []
            for ik in range(len(k_vec)):
                SK = build_reciprocal_from_sparseMat(self.graph_data.S_cell, k_vec[[ik]], self.graph_data.nbr_shift_of_cell)
                SK = np.squeeze(SK, axis=0)
                HK = build_reciprocal_from_sparseMat(self.graph_data.H_cell, k_vec[[ik]], self.graph_data.nbr_shift_of_cell)
                HK = np.squeeze(HK, axis=0)
                L = np.linalg.cholesky(SK)
                L_t = np.transpose(L.conj())
                L_inv = np.linalg.inv(L)
                L_t_inv = np.linalg.inv(L_t)
                Hs_np = np.matmul(np.matmul(L_inv, HK), L_t_inv)
                Hs = torch.from_numpy(Hs_np).to(torch.complex128).contiguous()
                w, v = torch.linalg.eigh(Hs)
                # The first eigen value is related to the v[:, 0]
                v = np.matmul(L_t_inv, v.numpy())
                eigen.append(w.numpy())
                eigen_vecs.append(v)
            eigen = np.swapaxes(np.array(eigen), 0, 1) # (norbs, nks)
            eigen_vecs = np.array(eigen_vecs) # (nks, norbs, norbs)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
        return eigen, eigen_vecs

    def _elec_cal(self, k_grid):
        k_vec = k_grid.reshape(-1, 3)
        SK = build_reciprocal_from_sparseMat(self.graph_data.S_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        # Calculate the electron wave function
        if self.spinor_mode == 'nc':
            HK = build_reciprocal_from_sparseMat_soc(self.graph_data.H_cell, k_vec, self.graph_data.nbr_shift_of_cell)
            I = np.identity(2, dtype=SK.dtype)
            SK = np.kron(I,SK)

            # diagonalization
            eigen = []
            eigen_vecs = []
            for ik in range(len(k_vec)):
                w, v = eigh(a=HK[0, ik], b=SK[ik])
                eigen.append(w)
                eigen_vecs.append(v)
                
            eigen = np.swapaxes(np.array(eigen), 0, 1) # (norbs, nk)
            eigen_vecs = np.array(eigen_vecs) # (nk, norbs, norbs)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
            
            lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK, eigen_vecs).real
            lamda = 1/np.sqrt(lamda) # shape: (nk, norbs)
            eigen_vecs = eigen_vecs*lamda[:,:,None]

        elif self.spinor_mode == 'collinear':
            # HK shape: (2, nks, norbs, norbs)
            HK = build_reciprocal_from_sparseMat_collinear_split(self.graph_data.H_cell, k_vec, self.graph_data.nbr_shift_of_cell)
            _, nks, norbs, _ = HK.shape
            # diagonalization
            eigen = []
            eigen_vecs = []
            for ik in range(nks):
                w_up, v_up = eigh(a=HK[0, ik], b=SK[ik])
                w_down, v_down = eigh(a=HK[1, ik], b=SK[ik])
                eigen.append([w_up, w_down])
                eigen_vecs.append([v_up, v_down])

            eigen = np.array(eigen)
            eigen = eigen.reshape(nks, 2*norbs)
            eigen = np.swapaxes(np.array(eigen), 0, 1) # (2*norbs, nks)
            eigen_vecs = np.array(eigen_vecs) # (nks, 2, norbs, norbs)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2) # (nks, 2, norbs, norbs)
            eigen_vecs = eigen_vecs.reshape(nks, 2*norbs, norbs) # (nks, 2*norbs, norbs)
            
            lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK, eigen_vecs).real
            lamda = 1.0 / np.sqrt(lamda) # shape: (nks, 2*norbs)
            eigen_vecs = eigen_vecs * lamda[:,:,None]

        else:
            HK = build_reciprocal_from_sparseMat(self.graph_data.H_cell, k_vec, self.graph_data.nbr_shift_of_cell)
        
            # diagonalization
            eigen = []
            eigen_vecs = []
            for ik in range(len(k_vec)):
                w, v = eigh(a=HK[ik], b=SK[ik])
                eigen.append(w)
                eigen_vecs.append(v)
                
            eigen = np.swapaxes(np.array(eigen), 0, 1) # (norbs, nk)
            eigen_vecs = np.array(eigen_vecs) # (nk, norbs, norbs)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
            
            lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK, eigen_vecs).real
            lamda = 1/np.sqrt(lamda) # shape: (nk, norbs)
            eigen_vecs = eigen_vecs*lamda[:,:,None]
        
        return eigen, eigen_vecs
