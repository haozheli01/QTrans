"""Mechanically extracted QTrans methods."""

from pymatgen.core.periodic_table import Element
from qtrans.core.constants import QTcts
from qtrans.core.errors import InputParameterError
from qtrans.data.basis import basis_def_dicts
from qtrans.io.abacus import read_abacus_input
from qtrans.io.abacus import write_abacus_input
from qtrans.io.openmx import write_openmx_input
from qtrans.numerics.sparse import build_sparse_matrix
from qtrans.numerics.sparse import build_sparse_matrix_collinear_split
from qtrans.numerics.sparse import build_sparse_matrix_soc
from qtrans.parallel.mpi import barrier_ranks
from scipy.sparse import csr_matrix
import json
import math
import numpy as np
import os
import re


class _GradientMixin:
    def _extract_mat_info_sc(self, mat_info_sc:dict):
        try:       
            cell_shift_array = mat_info_sc['cell_shift_array'] # shape: (ncells, 3)
            cell_index_map = mat_info_sc['cell_index_map'] # len: (ncells, )
            s2u_list = mat_info_sc['s2u_list'] # shape: (natoms_sc, )
            cell_shift_of_each_atom_in_sc = mat_info_sc['cell_shift_of_each_atom'] # len: (natoms_sc, 3)
            p2s_indices = mat_info_sc['p2s_indices'] # shape: (natoms_uc, )
        except:
            raise InputParameterError(1010, "grad_epc", "mat_info_sc_path")
        return cell_shift_array, cell_index_map, s2u_list, cell_shift_of_each_atom_in_sc, p2s_indices

    def _gen_pred_Hon_Hoff_sc(self):
        # Calculate the length of H for each structure
        nstructs_sc = self.natoms_uc * 3 + 1
        len_H = []
        for i in range(nstructs_sc):
            if self.spinor_mode == 'collinear':
                len_H.append(len(self.graph_dataset[i].Hon)*2)
                len_H.append(len(self.graph_dataset[i].Hoff)*2)
            elif self.spinor_mode == 'nc':
                len_H.append((len(self.graph_dataset[i].Hon)+len(self.graph_dataset[i].Hoff))*2)
            else:
                len_H.append(len(self.graph_dataset[i].Hon))
                len_H.append(len(self.graph_dataset[i].Hoff))
        try:
            pred_Hon_sc, pred_Hoff_sc = [], []
            idx = 0
            if self.spinor_mode == 'nc':
                pred_iHon_sc, pred_iHoff_sc = [], []
                for i in range(0, len(len_H)):
                    Hon_length = len(self.graph_dataset[i].Hon)
                    Hoff_length = len(self.graph_dataset[i].Hoff)
                    H_real = self.pred_H_sc[idx:idx+len_H[i]//2] # Hon, Hoff
                    H_imag = self.pred_H_sc[idx+len_H[i]//2:idx+len_H[i]] # iHon, iHoff

                    pred_Hon_sc.append(H_real[:Hon_length])
                    pred_Hoff_sc.append(H_real[Hon_length:])
                    pred_iHon_sc.append(H_imag[:Hon_length])
                    pred_iHoff_sc.append(H_imag[Hon_length:])
                    idx = idx + len_H[i]
            else:
                for i in range(0, len(len_H), 2):
                    pred_Hon_sc.append(self.pred_H_sc[idx:idx+len_H[i]])
                    idx = idx + len_H[i]
                    pred_Hoff_sc.append(self.pred_H_sc[idx:idx+len_H[i+1]])
                    idx = idx + len_H[i+1]
        except:
            raise InputParameterError(1012, "grad_epc", "pred_H_sc_path")
        if self.spinor_mode == 'nc':
            return  pred_Hon_sc, pred_Hoff_sc, pred_iHon_sc, pred_iHoff_sc
        else:
            return  pred_Hon_sc, pred_Hoff_sc

    def _gen_S_reduced_and_cell_index_map_000(self, data, cell_shift, edge_index, species,
                                              cell_shift_of_each_atom_in_sc, s2u_list_reduced, cell_index_map_reduced, orb_mask_mat):
        Son = data.Son.numpy().reshape(-1, self.nao_max, self.nao_max)
        Soff = data.Soff.numpy().reshape(-1, self.nao_max, self.nao_max)
        ncells = len(cell_index_map_reduced)
        # shape: (ncells, natoms_sc, nao_max, natoms_sc, nao_max)
        S_cell, _, _, cell_index_map, _ = build_sparse_matrix(
            species, cell_shift, self.nao_max, Son, Soff, edge_index, return_raw_mat=True)
        cell_index_map_000 = cell_index_map[(0,0,0)]
        S_000 = S_cell[cell_index_map_000] # shape: (natoms_sc, nao_max, natoms_sc, nao_max)
        del S_cell
        S_reduced = np.zeros((ncells, ncells, self.natoms_uc, self.nao_max, self.natoms_uc, self.nao_max))
        for i, Ci in enumerate(cell_shift_of_each_atom_in_sc): # ncells
            iatom = s2u_list_reduced[i]
            Ci = cell_index_map_reduced[tuple(Ci.tolist())]
            for j, Cj in enumerate(cell_shift_of_each_atom_in_sc): # ncells
                jatom = s2u_list_reduced[j]
                Cj = cell_index_map_reduced[tuple(Cj.tolist())]
                S_reduced[Ci,Cj,iatom,:,jatom,:] = S_000[i,:,j,:]
        S_reduced = S_reduced.reshape(ncells, ncells, self.natoms_uc*self.nao_max, self.natoms_uc*self.nao_max)
        S_reduced = S_reduced[:, :, orb_mask_mat > 0] # shape: (ncells, ncells, norbs, norbs)
        S_reduced = S_reduced.reshape(ncells, ncells, self.norbs, self.norbs)
        return S_reduced, cell_index_map_000

    def _gen_H_reduced(self, Hon, Hoff, cell_index_map_000, cell_shift, edge_index,
                       species, cell_shift_of_each_atom_in_sc, s2u_list_reduced, cell_index_map_reduced, orb_mask_mat, spinor_mode:str):
        ncells = len(cell_index_map_reduced)
        if spinor_mode == 'collinear':
            H_cell, _, _, _, _ = build_sparse_matrix_collinear_split(
                species, cell_shift, self.nao_max, Hon, Hoff, edge_index, return_raw_mat=True)
            H_000 = H_cell[:,cell_index_map_000]
            del H_cell
            H_reduced = np.zeros((2, ncells, ncells, self.natoms_uc, self.nao_max, self.natoms_uc, self.nao_max))
            for i, Ci in enumerate(cell_shift_of_each_atom_in_sc): # ncells
                iatom = s2u_list_reduced[i]
                Ci = cell_index_map_reduced[tuple(Ci.tolist())]
                for j, Cj in enumerate(cell_shift_of_each_atom_in_sc): # ncells
                    ja = s2u_list_reduced[j]
                    Cj = cell_index_map_reduced[tuple(Cj.tolist())]
                    H_reduced[:,Ci,Cj,iatom,:,ja,:] = H_000[:,i,:,j,:]
            H_reduced = H_reduced.reshape(2, ncells, ncells, self.natoms_uc*self.nao_max, self.natoms_uc*self.nao_max)
            H_reduced = H_reduced[:, :, :, orb_mask_mat > 0] # shape: (2, ncells, ncells, norbs, norbs)
            H_reduced = H_reduced.reshape(2, ncells, ncells, self.norbs, self.norbs)
        else:
            H_cell, _, _, _, _ = build_sparse_matrix(
                species, cell_shift, self.nao_max, Hon, Hoff, edge_index, return_raw_mat=True)
            H_000 = H_cell[cell_index_map_000]
            del H_cell
            H_reduced = np.zeros((ncells, ncells, self.natoms_uc, self.nao_max, self.natoms_uc, self.nao_max))
            for i, Ci in enumerate(cell_shift_of_each_atom_in_sc): # ncells
                ia = s2u_list_reduced[i]
                Ci = cell_index_map_reduced[tuple(Ci.tolist())]
                for j, Cj in enumerate(cell_shift_of_each_atom_in_sc): # ncells
                    ja = s2u_list_reduced[j]
                    Cj = cell_index_map_reduced[tuple(Cj.tolist())]
                    H_reduced[Ci,Cj,ia,:,ja,:] = H_000[i,:,j,:]
            H_reduced = H_reduced.reshape(ncells, ncells, self.natoms_uc*self.nao_max, self.natoms_uc*self.nao_max)
            H_reduced = H_reduced[:, :, orb_mask_mat > 0] # shape: (ncells, ncells, norbs, norbs)
            H_reduced = H_reduced.reshape(ncells, ncells, self.norbs, self.norbs)
        return H_reduced

    def _gen_H_reduced_soc(self, Hon, Hoff, iHon, iHoff, cell_index_map_000, cell_shift, edge_index,
                       species, cell_shift_of_each_atom_in_sc, s2u_list_reduced, cell_index_map_reduced, orb_mask_mat, spinor_mode:str):
        ncells = len(cell_index_map_reduced)
        H_cell, _, _, _, _ = build_sparse_matrix_soc(
            species, cell_shift, self.nao_max, Hon, Hoff, iHon, iHoff, edge_index, return_raw_mat=True)
        H_000 = H_cell[cell_index_map_000]
        del H_cell
        H_reduced = np.zeros((ncells, ncells, 2, self.natoms_uc, self.nao_max, 2, self.natoms_uc, self.nao_max), dtype=complex)
        for i, Ci in enumerate(cell_shift_of_each_atom_in_sc): # ncells
            ia = s2u_list_reduced[i]
            Ci = cell_index_map_reduced[tuple(Ci.tolist())]
            for j, Cj in enumerate(cell_shift_of_each_atom_in_sc): # ncells
                ja = s2u_list_reduced[j]
                Cj = cell_index_map_reduced[tuple(Cj.tolist())]
                H_reduced[Ci,Cj,:,ia,:,:,ja,:] = H_000[:,i,:,:,j,:]
        H_reduced = H_reduced.reshape(ncells, ncells, 2*self.natoms_uc*self.nao_max, 2*self.natoms_uc*self.nao_max)
        H_reduced = H_reduced[:, :, orb_mask_mat > 0] # shape: (ncells, ncells, norbs_soc*norbs_soc)
        norbs_soc = int(math.sqrt(H_reduced.size/(ncells*ncells)))
        H_reduced = H_reduced.reshape(ncells, ncells, norbs_soc, norbs_soc)
        return H_reduced

    def _gen_dS_dH(self, H_all:np.ndarray, S_all:np.ndarray):
        nabla_H = []
        nabla_S = []
        if self.fd_method == 'central':
            for i in range(1, 3 * self.natoms_uc + 1):
                dH = H_all[i] - H_all[i + 3 * self.natoms_uc]
                dH = dH / (2 * self.epc_sc_disp)
                dS = S_all[i] - S_all[i + 3 * self.natoms_uc]
                dS = dS / (2 * self.epc_sc_disp)
                nabla_H.append(dH)
                nabla_S.append(dS)
        else:
            for i in range(1, 3 * self.natoms_uc + 1):
                dH = H_all[i] - H_all[0]
                dH = dH / self.epc_sc_disp
                dS = S_all[i] - S_all[0]
                dS = dS / self.epc_sc_disp
                nabla_H.append(dH)
                nabla_S.append(dS)
        nabla_H = np.stack(nabla_H, -1).reshape(*dH.shape, self.natoms_uc, 3)
        nabla_S = np.stack(nabla_S, -1).reshape(*dS.shape, self.natoms_uc, 3)
        H_unperturb = H_all[0].copy()
        S_unperturb = S_all[0].copy()
        del H_all, S_all
        return nabla_H, nabla_S, H_unperturb, S_unperturb

    def _gen_dSon(self, dSon_dir:str):
        if not os.path.exists(dSon_dir):
            os.mkdir(dSon_dir)
        unique_element = np.unique(self.atoms_label)
        # dSon = {1: dSon_H, 8: dSon_O}
        dSon = dict()
        for each in unique_element:
            dSon_atoms_dict = dict()
            dSon_atoms_dict['chemical_symbols'] = [each, each, each, each]
            dSon_atom_positions = [np.dot(np.array([0.5, 0.5, 0.5]), np.array(self.unitcell_latt))]
            for i in range(3):
                tmp_atom_position = dSon_atom_positions[0].copy()
                tmp_atom_position[i] += self.dSon_delta_r
                dSon_atom_positions.append(tmp_atom_position)
            dSon_atoms_dict['positions'] = np.array(dSon_atom_positions)
            dSon_atoms_dict['cell'] = np.array(self.unitcell_latt)
            dSon_element_path = os.path.join(dSon_dir, each)
            if os.path.exists(dSon_element_path):
                print(f"Warning: Path '{dSon_element_path}' already exists.")
                print("Continuing execution: existing files in this directory will be overwritten.")
            else:
                os.mkdir(dSon_element_path)
            
            if self.DFT_software == 'openmx':
                write_openmx_input(os.path.join(dSon_element_path, 'openmx.dat'), dSon_atoms_dict, 
                                self.scf_uc_kpt, self.openmx_dftdata_dir,
                                spinor_mode='none', cwd='./', xc_func=self.xc_functional)
                os.chdir(dSon_element_path)
                os.system(f"{self.openmx_pp_path} openmx.dat > log.out 2>&1")
                os.system(f"{self.openmx_read_path} overlap.scfout")
            elif self.DFT_software == 'abacus':
                write_abacus_input(os.path.join(dSon_element_path, 'STRU'), dSon_atoms_dict, self.scf_uc_kpt, write_mode = 'STRU',
                                spinor_mode='none',dSon_cal=True)
                write_abacus_input(os.path.join(dSon_element_path, 'INPUT'), dSon_atoms_dict, self.scf_uc_kpt, write_mode = 'INPUT',
                                spinor_mode='none',pseudo_dir = self.abacus_dftdata_dir,orbital_dir=self.abacus_orb_dir,dSon_cal=True)
                write_abacus_input(os.path.join(dSon_element_path, 'KPT'), dSon_atoms_dict, self.scf_uc_kpt, write_mode = 'KPT',
                                spinor_mode='none',dSon_cal=True)
                os.chdir(dSon_element_path)
                os.system(f"{self.abacus_pp_path} > dSon.out 2>&1")
                self._gen_dSon_json_abacus(dSon_element_path, basis_def_dicts[self.basis_label])
            HS_path_tmp = os.path.join(dSon_element_path, 'HS.json')
            try:
                f_HS_dSon_tmp = open(HS_path_tmp)
                data_HS_dSon_tmp = json.load(f_HS_dSon_tmp)
            except:
                raise RuntimeError(f"Error(3005): Failed to generate dSon when preparing gradient matrix.")
            edge_index_HS_dSon_tmp = np.array(data_HS_dSon_tmp['edge_index'])
            cell_shift_HS_dSon_tmp = [tuple(cell_shift) for cell_shift in data_HS_dSon_tmp['cell_shift']]
            Soff_HS_dSon_tmp = np.array(data_HS_dSon_tmp['Soff'])
            Son_HS_dSon_tmp = np.array(data_HS_dSon_tmp['Son'])
            nao_max_element = int(math.sqrt(Son_HS_dSon_tmp.size / 4))
            Soff_HS_dSon_tmp = Soff_HS_dSon_tmp.reshape(-1, nao_max_element, nao_max_element)
            Son_HS_dSon_tmp = Son_HS_dSon_tmp.reshape(-1, nao_max_element, nao_max_element)
            index_list = np.where(edge_index_HS_dSon_tmp[0] == 0)[0]
            Son_cell = dict()
            for iedge in index_list:
                tar = edge_index_HS_dSon_tmp[1][iedge]
                icell = cell_shift_HS_dSon_tmp[iedge]
                if icell not in Son_cell:
                    Son_cell[icell] = dict()
                Son_cell[icell][tar] = Soff_HS_dSon_tmp[iedge]
            dSon_cell = dict() # dSon_cell[(x,x,x)][1,2,3]: shape: (nao_max, nao_max)
            for iedge in index_list:
                tar = edge_index_HS_dSon_tmp[1][iedge]
                icell = cell_shift_HS_dSon_tmp[iedge]
                if tar == 0:
                    continue
                if icell not in dSon_cell:        
                    dSon_cell[icell] = dict()
                if icell == (0,0,0):
                    dSon_cell[icell][tar] = (Son_cell[icell][tar]-Son_HS_dSon_tmp[0]) / self.dSon_delta_r
                else:
                    dSon_cell[icell][tar] = (Son_cell[icell][tar]-Son_cell[icell][0]) / self.dSon_delta_r
            dSon[Element(each).Z] = dSon_cell
            print(f"Finished on-site overlap gradient calculation for element: {each}.")
        return dSon

    def _gen_dSbar(self, nabla_S, dSon, cell_index_map_reduced):
        ncells = len(cell_index_map_reduced)
        atomic_numbers = [Element(each).Z for each in self.atoms_label]
        orb2atom_idx = np.repeat(np.arange(self.natoms_uc), repeats=self.atomic_orb_len_list, axis=0)
        dSbar = np.zeros_like(nabla_S) # shape: (ncells, ncells, norbs, norbs, natoms, 3)
        center_cell_idx = cell_index_map_reduced[(0,0,0)]
        over_orbs_tmp = np.arange(self.norbs)
        dSbar[center_cell_idx,:,over_orbs_tmp,:,orb2atom_idx[over_orbs_tmp],:] = -nabla_S[center_cell_idx,:,over_orbs_tmp,:,orb2atom_idx[over_orbs_tmp],:]
        start = 0
        for iatom in range(self.natoms_uc):
            dSbar[center_cell_idx,center_cell_idx,start:start+self.atomic_orb_len_list[iatom],start:start+self.atomic_orb_len_list[iatom],iatom,0] = dSon[atomic_numbers[iatom]][(0,0,0)][1] # x
            dSbar[center_cell_idx,center_cell_idx,start:start+self.atomic_orb_len_list[iatom],start:start+self.atomic_orb_len_list[iatom],iatom,1] = dSon[atomic_numbers[iatom]][(0,0,0)][2] # y
            dSbar[center_cell_idx,center_cell_idx,start:start+self.atomic_orb_len_list[iatom],start:start+self.atomic_orb_len_list[iatom],iatom,2] = dSon[atomic_numbers[iatom]][(0,0,0)][3] # z
            start += self.atomic_orb_len_list[iatom]
        dSbar = np.swapaxes(dSbar, axis1=1, axis2=2).reshape(ncells*self.norbs, ncells*self.norbs, self.natoms_uc, 3)
        return dSbar

    def _gen_grad_mat_dense(self, H_unperturb, S_unperturb, nabla_H, nabla_S, dSon, cell_index_map_reduced, spinor_mode:str):
        ncells = len(cell_index_map_reduced)
        S_den = np.swapaxes(S_unperturb, axis1=1, axis2=2).reshape(ncells*self.norbs, ncells*self.norbs)
        S_den_inv = np.linalg.inv(S_den)
        dSbar = self._gen_dSbar(nabla_S, dSon, cell_index_map_reduced)
        if spinor_mode == 'collinear':
            # build dense matrix
            H_den = np.swapaxes(H_unperturb, axis1=2, axis2=3).reshape(2, ncells*self.norbs, ncells*self.norbs)
            # tmp1 shape: (2, ncells*norbs, ncells*norbs)
            tmp1 = np.einsum('ij, mjk->mik', S_den_inv, H_den)
            sum1 = []
            for iatom in range(self.natoms_uc):
                for idir in range(3):
                    sum1.append(np.einsum('ij, mjk->mik', dSbar[:,:,iatom, idir], tmp1))
            sum1 = np.stack(sum1, axis=-1).reshape(2, ncells*self.norbs, ncells*self.norbs, self.natoms_uc, 3)
            sum2 = np.swapaxes(sum1, axis1=1, axis2=2)
            grad_mat = (sum1+sum2).reshape(2, ncells, self.norbs, ncells, self.norbs, self.natoms_uc, 3)
            grad_mat = np.swapaxes(grad_mat, axis1=2, axis2=3) # shape: (2, ncells, ncells, norbs, norbs, natoms_uc, 3)
        elif spinor_mode == 'nc':
            # build dense matrix
            H_den = np.swapaxes(H_unperturb, axis1=1, axis2=2).reshape(ncells*self.norbs*2, ncells*self.norbs*2)
            I = np.identity(2, dtype=complex)

            # # from (ncells, ncells, norbs, norbs) to (ncells, ncells, 2, norbs, 2, norbs)
            # S_tmp = np.einsum('ijkl,ab->ijakbl', S_unperturb, I)
            # # from (ncells, ncells, 2, norbs, 2, norbs) to (ncells, ncells, 2*norbs, 2*norbs)
            # S_tmp = S_tmp.reshape(ncells, ncells, self.norbs*2, self.norbs*2)
            # # from (ncells, ncells, 2*norbs, 2*norbs) to (ncells*2*norbs, ncells*2*norbs)
            # S_tmp = np.swapaxes(S_tmp, axis1=1, axis2=2).reshape(ncells*self.norbs*2, ncells*self.norbs*2)
            # S_den_inv_nc = np.linalg.inv(S_tmp)

            # lhz: more efficient way to get S_den_inv_nc without building the huge dense matrix S_tmp
            # shape: (ncells, norbs, ncells, norbs) --> (ncells, ncells, norbs, norbs)
            S_inv_tmp = S_den_inv.reshape(ncells, self.norbs, ncells, self.norbs).swapaxes(1, 2)
            # from (ncells, ncells, norbs, norbs) to (ncells, ncells, 2, norbs, 2, norbs)
            S_den_inv_nc = np.einsum('ijkl,ab->ijakbl', S_inv_tmp, I)
            # from (ncells, ncells, 2, norbs, 2, norbs) to (ncells, ncells, 2*norbs, 2*norbs)
            S_den_inv_nc = S_den_inv_nc.reshape(ncells, ncells, self.norbs*2, self.norbs*2)
            # from (ncells, ncells, 2*norbs, 2*norbs) to (ncells*2*norbs, ncells*2*norbs)
            S_den_inv_nc = np.swapaxes(S_den_inv_nc, axis1=1, axis2=2).reshape(ncells*self.norbs*2, ncells*self.norbs*2)

            tmp1 = np.dot(S_den_inv_nc, H_den) # shape: (ncells*norbs*2, ncells*norbs*2)
            sum1 = []
            dSbar_tmp = dSbar.reshape(ncells, self.norbs, ncells, self.norbs, self.natoms_uc, 3).swapaxes(1,2) # shape: (ncells, ncells, norbs, norbs, natoms_uc, 3)
            dSbar_nc = np.einsum('ijklmn,ab->ijakblmn', dSbar_tmp, I) # shape: (ncells, ncells, 2, norbs, 2, norbs, natoms_uc, 3)
            dSbar_nc = dSbar_nc.reshape(ncells, ncells, self.norbs*2, self.norbs*2, self.natoms_uc, 3) # shape: (ncells, ncells, 2*norbs, 2*norbs, natoms_uc, 3)
            dSbar_nc = dSbar_nc.swapaxes(1,2).reshape(ncells*self.norbs*2, ncells*self.norbs*2, self.natoms_uc, 3) # shape: (ncells*norbs*2, ncells*norbs*2, natoms_uc, 3)
            for iatom in range(self.natoms_uc):
                for idir in range(3):
                    sum1.append(np.dot(dSbar_nc[:,:,iatom, idir], tmp1))
            sum1 = np.stack(sum1, axis=-1).reshape(ncells*self.norbs*2, ncells*self.norbs*2, self.natoms_uc, 3)
            sum2 = np.swapaxes(sum1.conj(), axis1=0, axis2=1) # TEST
            grad_mat = (sum1+sum2).reshape(ncells, self.norbs*2, ncells, self.norbs*2, self.natoms_uc, 3)
            grad_mat = np.swapaxes(grad_mat, axis1=1, axis2=2) # shape: (ncells, ncells, 2*norbs, 2*norbs, natoms_uc, 3)
        else:
            # build dense matrix
            H_den = np.swapaxes(H_unperturb, axis1=1, axis2=2).reshape(ncells*self.norbs, ncells*self.norbs)
            tmp1 = np.dot(S_den_inv, H_den)
            sum1 = []
            for iatom in range(self.natoms_uc):
                for idir in range(3):
                    sum1.append(np.dot(dSbar[:,:,iatom, idir], tmp1))
            sum1 = np.stack(sum1, axis=-1).reshape(ncells*self.norbs, ncells*self.norbs, self.natoms_uc, 3)
            sum2 = np.swapaxes(sum1, axis1=0, axis2=1)
            grad_mat = (sum1+sum2).reshape(ncells, self.norbs, ncells, self.norbs, self.natoms_uc, 3)
            grad_mat = np.swapaxes(grad_mat, axis1=1, axis2=2) # shape: (ncells, ncells, norbs, norbs, natoms_uc, 3)
        grad_mat += nabla_H
        return grad_mat

    def _gen_grad_mat_sparse_from_dense(self, grad_mat:np.ndarray, spinor_mode:str):
        ncells = grad_mat.shape[1]
        if spinor_mode == 'collinear':
            grad_mat_dict = [dict(), dict()]
            for ispin in range(2):
                for icell in range(ncells):
                    for jcell in range(ncells):
                        for iatom in range(self.natoms_uc):
                            for idir in range(3):
                                grad_mat_small = grad_mat[ispin, icell, jcell, :, :, iatom, idir]
                                filtered_array = np.where(np.abs(grad_mat_small) > self.grad_sparse_thr, grad_mat_small, 0.0)
                                sparse_matrix = csr_matrix(filtered_array)
                                if sparse_matrix.nnz > 0:
                                    grad_mat_dict[ispin][(icell,jcell,iatom,idir)] = (sparse_matrix.data.copy(), sparse_matrix.indices.copy(), sparse_matrix.indptr.copy())
        else:
            grad_mat_dict = dict()
            for icell in range(ncells):
                for jcell in range(ncells):
                    for iatom in range(self.natoms_uc):
                        for idir in range(3):
                            grad_mat_small = grad_mat[icell, jcell, :, :, iatom, idir]
                            filtered_array = np.where(np.abs(grad_mat_small) > self.grad_sparse_thr, grad_mat_small, 0.0)
                            sparse_matrix = csr_matrix(filtered_array)
                            if sparse_matrix.nnz > 0:
                                grad_mat_dict[(icell,jcell,iatom,idir)] = (sparse_matrix.data.copy(), sparse_matrix.indices.copy(), sparse_matrix.indptr.copy())
        return grad_mat_dict

    def _gen_grad_mat_epc(self):
        # This function is serial
        # Note: xxx_path include filename, xxx_dir is a folder.
        work_dir = os.path.abspath(self.outdir)
        dSon_dir = os.path.join(work_dir, 'dSon')
        grad_mat_path = os.path.join(work_dir, 'grad_mat.npy')
        grad_mat_sparse_path = os.path.join(work_dir, 'grad_mat_sparse.npy')
        # get orbits mask matrix
        orb_mask_mat = self.orb_mask[:,None] * self.orb_mask[None,:]   # shape: [natoms_uc*nao_max, natoms_uc*nao_max]
        if self.spinor_mode == 'collinear':
            H_reshape_shape = (-1, 2, self.nao_max, self.nao_max)
        elif self.spinor_mode == 'nc':
            H_reshape_shape = (-1, 2 * self.nao_max, 2 * self.nao_max)
        else:
            H_reshape_shape = (-1, self.nao_max, self.nao_max)
        ######################################## Gather H and S ########################################
        os.chdir(work_dir)
        _, cell_index_map_reduced, s2u_list_reduced, cell_shift_of_each_atom_in_sc, p2s_indices = self._extract_mat_info_sc(self.mat_info_sc)
        if self.use_prediction:
            if self.spinor_mode == 'nc':
                pred_Hon_sc, pred_Hoff_sc, pred_iHon_sc, pred_iHoff_sc = self._gen_pred_Hon_Hoff_sc()
            else:
                pred_Hon_sc, pred_Hoff_sc = self._gen_pred_Hon_Hoff_sc()
        H_all = []
        S_all = []
        if self.gen_force_constants:
            atomic_forces = []
        for idx, data in enumerate(self.graph_dataset):
            cell_shift = data.cell_shift.numpy()
            edge_index = data.edge_index.numpy()
            species = data.z.numpy()
            if self.gen_force_constants and (idx > 0):
                try:
                    atomic_forces.append(data.atomic_forces.numpy())
                except:
                    raise RuntimeError("Error(3007): The graph data must contains atomic forces when you need to generate FORCE_SETS.")
            S_reduced, cell_index_map_000 = self._gen_S_reduced_and_cell_index_map_000(
                data, cell_shift, edge_index, species,
                cell_shift_of_each_atom_in_sc, s2u_list_reduced, cell_index_map_reduced, orb_mask_mat)
            S_all.append(S_reduced)
            if self.use_prediction:
                if self.spinor_mode == 'nc':
                    Hon = pred_Hon_sc[idx].reshape(*H_reshape_shape)
                    Hoff = pred_Hoff_sc[idx].reshape(*H_reshape_shape)
                    iHon = pred_iHon_sc[idx].reshape(*H_reshape_shape)
                    iHoff = pred_iHoff_sc[idx].reshape(*H_reshape_shape)
                else:
                    Hon = pred_Hon_sc[idx].reshape(*H_reshape_shape)
                    Hoff = pred_Hoff_sc[idx].reshape(*H_reshape_shape)
            else:
                Hon = data.Hon.numpy().reshape(*H_reshape_shape)
                Hoff = data.Hoff.numpy().reshape(*H_reshape_shape)
                if self.spinor_mode == 'nc':
                    iHon = data.iHon.numpy().reshape(*H_reshape_shape)
                    iHoff = data.iHoff.numpy().reshape(*H_reshape_shape)
            if self.spinor_mode == 'nc':
                orb_mask_soc = np.concatenate([self.orb_mask, self.orb_mask], axis=0) # shape: (2*natoms_uc*nao_max, )
                orb_mask_soc = orb_mask_soc[:,None] * orb_mask_soc[None,:]   # shape: (2*natoms_uc*nao_max, 2*natoms_uc*nao_max)
                H_reduced = self._gen_H_reduced_soc(Hon, Hoff, iHon, iHoff, 
                                                cell_index_map_000, cell_shift, edge_index, species,
                                                cell_shift_of_each_atom_in_sc, s2u_list_reduced, cell_index_map_reduced,
                                                orb_mask_soc, self.spinor_mode)
            else:
                H_reduced = self._gen_H_reduced(Hon, Hoff, 
                                                cell_index_map_000, cell_shift, edge_index, species,
                                                cell_shift_of_each_atom_in_sc, s2u_list_reduced, cell_index_map_reduced,
                                                orb_mask_mat, self.spinor_mode)
            H_all.append(H_reduced)
        del self.graph_dataset
        H_all = np.stack(H_all, 0)
        S_all = np.stack(S_all, 0)
        ####################################### FORCE SETS #######################################
        if self.gen_force_constants:
            atomic_forces = np.stack(atomic_forces, 0) # shape (natoms_uc*3, natoms_sc, 3)
            self._gen_force_constants_from_force(atomic_forces, p2s_indices)
            print("FORCE_CONSTANTS generated successfully.")
        print("Hamiltonian and overlap matrices gathered successfully.")
        ######################################## dS and dH ########################################
        nabla_H, nabla_S, H_unperturb, S_unperturb = self._gen_dS_dH(H_all, S_all)
        print("Finished computing gradients of Hamiltonian and overlap.")
        ######################################## dSon calulation ########################################
        dSon = self._gen_dSon(dSon_dir)
        ######################################## Gradient matrix calculation ########################################
        os.chdir(work_dir)
        grad_mat = self._gen_grad_mat_dense(H_unperturb, S_unperturb,
                                            nabla_H, nabla_S, dSon,
                                            cell_index_map_reduced, self.spinor_mode)
        np.save(file=grad_mat_path, arr=grad_mat)
        grad_mat_dict = self._gen_grad_mat_sparse_from_dense(grad_mat, self.spinor_mode)
        np.save(file=grad_mat_sparse_path, arr=grad_mat_dict)
        print("Finished computing the gradient matrix.")

    def prep_grad_mat_epc(self):
        if self.graph_data_sc_path:
            if self.rank == 0:
                self._gen_grad_mat_epc()
            barrier_ranks(self.comm)
        else:
            pass

    def _gen_dSon_json_abacus(self, folder:str, basis_def:dict):
        # Define paths for the required files
        output_folder_path = os.path.join(folder, 'OUT.QTrans')
        stru_file_path = os.path.join(folder, 'STRU')

        # Read crystal structure parameters
        try:
            species, lattice, positions, magmoms = read_abacus_input(stru_file_path)
            atomic_numbers = [Element(iatom).Z for iatom in species]
            lattice = lattice * QTcts.ANGtoBOHR # convert to atomic unit
            positions = positions * QTcts.ANGtoBOHR # convert to atomic unit
        except Exception as e:
            print(f"Error reading STRU file: {e}. Skipping...")
        
        # Load sparse overlap matrices
        SR = os.path.join(output_folder_path, 'data-SR-sparse_SPIN0.csr')
        # Generate graphs for overlap
        graphS = self.get_abacus_hamiltonian_graph(SR, atomic_numbers, lattice, positions, basis_def,
                                                    graph={}, skip=False, tojson=True)
        
        # Output file name
        filename = 'HS.json'
        graphS["Son"] = graphS.pop("Hon")
        graphS["Soff"] = graphS.pop("Hoff")

        # Write the graph data to the JSON file
        with open(filename, 'w') as f:
            json.dump(graphS, f, separators=[',', ':'])

        # Reopen the file for formatting (newlines after commas for readability)
        with open(filename, 'r') as f:
            file_content = f.read()
            formatted_content = re.sub(r', *"', ',\n"', file_content)

        # Write the formatted content back to the file
        with open(filename, 'w') as f:
            f.write(formatted_content)
