"""Mechanically extracted QTrans methods."""

from pymatgen.core.periodic_table import Element
from qtrans.core.constants import QTcts
from qtrans.io.openmx import read_openmx_input
from qtrans.io.openmx import read_openmx_out
from qtrans.numerics.arrays import average_close_values
from torch_geometric.data import Data
import json
import numpy as np
import os
import torch


class _OpenMXMixin:
    def _gather_HS_from_json(self, json_path:str, atomic_numbers, basis_def, with_S:bool=False):
        with open(json_path,'r') as f_HS_tmp:
            f_HS_tmp = open(json_path)
            data_HS_tmp = json.load(f_HS_tmp)
        pos = np.array(data_HS_tmp['pos'])
        edge_index = np.array(data_HS_tmp['edge_index'])
        inv_edge_idx = np.array(data_HS_tmp['inv_edge_idx'])
        nbr_shift = np.array(data_HS_tmp['nbr_shift'])
        cell_shift = np.array(data_HS_tmp['cell_shift'])
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print(f"inverse edge index do not match with edge index in {json_path}")
            return None
        Hon = data_HS_tmp['Hon'][0]
        Hoff = data_HS_tmp['Hoff'][0]
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        H = np.zeros((num_sub_matrix, self.nao_max**2))
        if with_S:
            Son = data_HS_tmp['Son']
            Soff = data_HS_tmp['Soff']
            S = np.zeros((num_sub_matrix, self.nao_max**2))
            for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hon, Son)):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src = atomic_numbers[i]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[i][mask] = np.array(sub_maxtrix_H)
                S[i][mask] = np.array(sub_maxtrix_S)
            num = 0
            for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hoff, Soff)):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src, tar = atomic_numbers[edge_index[0,num]], atomic_numbers[edge_index[1,num]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[num + len(atomic_numbers)][mask] = np.array(sub_maxtrix_H)
                if with_S:
                    S[num + len(atomic_numbers)][mask] = np.array(sub_maxtrix_S)
                num = num + 1
        else:
            for i, sub_maxtrix_H in enumerate(Hon):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src = atomic_numbers[i]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[i][mask] = np.array(sub_maxtrix_H)
            num = 0
            for i, sub_maxtrix_H in enumerate(Hoff):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src, tar = atomic_numbers[edge_index[0,num]], atomic_numbers[edge_index[1,num]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[num + len(atomic_numbers)][mask] = np.array(sub_maxtrix_H)
                num = num + 1
        if with_S:
            return pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H, S
        else:
            return pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H

    def _gather_HS_from_json_collinear(self, json_path:str, atomic_numbers, basis_def, with_S:bool=False):
        with open(json_path,'r') as f_HS_tmp:
            f_HS_tmp = open(json_path)
            data_HS_tmp = json.load(f_HS_tmp)
        pos = np.array(data_HS_tmp['pos'])
        edge_index = np.array(data_HS_tmp['edge_index'])
        inv_edge_idx = np.array(data_HS_tmp['inv_edge_idx'])
        nbr_shift = np.array(data_HS_tmp['nbr_shift'])
        cell_shift = np.array(data_HS_tmp['cell_shift'])
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print(f"inverse edge index do not match with edge index in {json_path}")
            return None
        Hon_u = data_HS_tmp['Hon'][0]
        Hoff_u = data_HS_tmp['Hoff'][0]
        Hon_d = data_HS_tmp['Hon'][1]
        Hoff_d = data_HS_tmp['Hoff'][1]
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        H = np.zeros((num_sub_matrix, 2, self.nao_max**2))
        if with_S:            
            Son = data_HS_tmp['Son']
            Soff = data_HS_tmp['Soff']
            S = np.zeros((num_sub_matrix, self.nao_max**2))
            for i, (sub_maxtrix_H_u, sub_maxtrix_H_d, sub_maxtrix_S) in enumerate(zip(Hon_u, Hon_d, Son)):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src = atomic_numbers[i]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[i,0][mask] = np.array(sub_maxtrix_H_u)
                H[i,1][mask] = np.array(sub_maxtrix_H_d)
                if with_S:
                    S[i][mask] = np.array(sub_maxtrix_S)
            num = 0
            for i, (sub_maxtrix_H_u, sub_maxtrix_H_d, sub_maxtrix_S) in enumerate(zip(Hoff_u, Hoff_d, Soff)):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src, tar = atomic_numbers[edge_index[0,num]], atomic_numbers[edge_index[1,num]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[num + len(atomic_numbers),0][mask] = np.array(sub_maxtrix_H_u)
                H[num + len(atomic_numbers),1][mask] = np.array(sub_maxtrix_H_d)
                if with_S:
                    S[num + len(atomic_numbers)][mask] = np.array(sub_maxtrix_S)
                num = num + 1
        else:
            for i, (sub_maxtrix_H_u, sub_maxtrix_H_d) in enumerate(zip(Hon_u, Hon_d)):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src = atomic_numbers[i]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[i,0][mask] = np.array(sub_maxtrix_H_u)
                H[i,1][mask] = np.array(sub_maxtrix_H_d)
            num = 0
            for i, (sub_maxtrix_H_u, sub_maxtrix_H_d) in enumerate(zip(Hoff_u, Hoff_d)):
                mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
                src, tar = atomic_numbers[edge_index[0,num]], atomic_numbers[edge_index[1,num]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[num + len(atomic_numbers),0][mask] = np.array(sub_maxtrix_H_u)
                H[num + len(atomic_numbers),1][mask] = np.array(sub_maxtrix_H_d)
                num = num + 1

        if with_S:
            return pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H, S
        else:
            return pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H

    def _gather_HS_from_json_nc(self, json_path:str, atomic_numbers, basis_def, with_S:bool=False, with_L:bool=False):
        with open(json_path,'r') as f_HS_tmp:
            f_HS_tmp = open(json_path)
            data_HS_tmp = json.load(f_HS_tmp)
        pos = np.array(data_HS_tmp['pos'])
        edge_index = np.array(data_HS_tmp['edge_index'])
        inv_edge_idx = np.array(data_HS_tmp['inv_edge_idx'])

        Hon = data_HS_tmp['Hon']
        Hoff = data_HS_tmp['Hoff']
        iHon = data_HS_tmp['iHon']
        iHoff = data_HS_tmp['iHoff']
        nbr_shift = np.array(data_HS_tmp['nbr_shift'])
        cell_shift = np.array(data_HS_tmp['cell_shift'])
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print(f"inverse edge index do not match with edge index in {json_path}")
            return None
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        Hks = np.zeros((num_sub_matrix, 4, self.nao_max**2))   
        iHks = np.zeros((num_sub_matrix, 3, self.nao_max**2))

        if with_S:
            Son = data_HS_tmp['Son']
            Soff = data_HS_tmp['Soff']
            S = np.zeros((num_sub_matrix, self.nao_max**2))
        if with_L:
            Lon = data_HS_tmp['Lon']
            Loff = data_HS_tmp['Loff']
            L = np.zeros((num_sub_matrix, self.nao_max**2, 3))
        # on-site
        for iatm in range(len(atomic_numbers)):
            mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
            src = atomic_numbers[iatm]
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            if with_S:
                S[iatm][mask] = np.array(Son[iatm])
            if with_L:
                L[iatm][mask] = np.array(Lon[iatm])
            for i in range(4):
                Hks[iatm, i][mask] = np.array(Hon[i][iatm]) 
            for i in range(3):
                iHks[iatm, i][mask] = np.array(iHon[i][iatm])    
        # off-site
        for iedge in range(len(edge_index[0])):
            mask = np.zeros((self.nao_max, self.nao_max), dtype=int)
            src, tar = atomic_numbers[edge_index[0, iedge]], atomic_numbers[edge_index[1, iedge]]
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            if with_S:
                S[len(atomic_numbers)+iedge][mask] = np.array(Soff[iedge])
            if with_L:
                L[len(atomic_numbers)+iedge][mask] = np.array(Loff[iedge])
            for i in range(4):
                Hks[iedge + len(atomic_numbers), i][mask] = np.array(Hoff[i][iedge]) 
            for i in range(3):
                iHks[iedge + len(atomic_numbers), i][mask] = np.array(iHoff[i][iedge])
        hamiltonian_real = np.zeros((num_sub_matrix, 2*self.nao_max, 2*self.nao_max)) 
        hamiltonian_real[:,:self.nao_max,:self.nao_max] = Hks[:,0,:].reshape(-1, self.nao_max, self.nao_max)
        hamiltonian_real[:,:self.nao_max, self.nao_max:] = Hks[:,2,:].reshape(-1,self.nao_max, self.nao_max)
        hamiltonian_real[:,self.nao_max:,:self.nao_max] = Hks[:,2,:].reshape(-1,self.nao_max, self.nao_max)
        hamiltonian_real[:,self.nao_max:,self.nao_max:] = Hks[:,1,:].reshape(-1,self.nao_max, self.nao_max)
        hamiltonian_real = hamiltonian_real.reshape(-1, (2*self.nao_max)**2)            
        hamiltonian_imag = np.zeros((num_sub_matrix, 2*self.nao_max, 2*self.nao_max)) 
        hamiltonian_imag[:,:self.nao_max,:self.nao_max] = iHks[:,0,:].reshape(-1, self.nao_max, self.nao_max)
        hamiltonian_imag[:,:self.nao_max, self.nao_max:] = (Hks[:,3,:] + iHks[:,2,:]).reshape(-1, self.nao_max, self.nao_max)
        hamiltonian_imag[:,self.nao_max:,:self.nao_max] = -(Hks[:,3,:] + iHks[:,2,:]).reshape(-1, self.nao_max, self.nao_max)
        hamiltonian_imag[:,self.nao_max:,self.nao_max:] = iHks[:,1,:].reshape(-1, self.nao_max, self.nao_max)
        hamiltonian_imag = hamiltonian_imag.reshape(-1, (2*self.nao_max)**2)

        H = {
            'real': hamiltonian_real,
            'imag': hamiltonian_imag
        }

        ret = [pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H]

        if with_S:
            ret.append(S)
        if with_L:
            ret.append(L)

        return tuple(ret)

    def _gen_graph_data_scf(self, folder:str, basis_def:dict):
        f_vbout = os.path.join(folder, self.vbout_filename)
        f_HSout = os.path.join(folder, self.HSout_filename)
        if self.with_nonscf:
            f_nonscf = os.path.join(folder, self.nonscf_filename)
        
        # read output file
        try:
            if self.pack_atomic_force:
                reader_out_dict = read_openmx_out(f_vbout, return_iter=True, return_magmom=True, return_total_energy=True, return_atomic_force=True)
                atomic_forces = reader_out_dict['atomic_forces']
            else:
                reader_out_dict = read_openmx_out(f_vbout, return_iter=True, return_magmom=True, return_total_energy=True)
            species = reader_out_dict['species']
            latt = reader_out_dict['latt']
            coordinates = reader_out_dict['coordinates']
            magmoms = reader_out_dict['magmoms']
            max_niter = reader_out_dict['max_niter']
            niter_scf = reader_out_dict['niter_scf']
            total_energy = reader_out_dict['total_energy']
            spinor_mode_tmp = reader_out_dict['spinor_mode']
            if self.spinor_mode != spinor_mode_tmp:
                raise RuntimeError(f"Error(3006): The spinor mode used in the OpenMX calculation is inconsistent with the setting specified during graph data packing.\n \
                                   Path: {f_vbout}")
            atomic_numbers = np.array([Element[s].Z for s in species])
            latt = latt * QTcts.ANGtoBOHR
            coordinates = coordinates * QTcts.ANGtoBOHR
            if self.spinor_mode == 'collinear':
                spin_length = np.abs(magmoms[:, 2])
                is_magnetic = spin_length > self.magnetic_tolerance
                spin_length[~is_magnetic] = 0.0
                spin_vec = np.zeros_like(magmoms, dtype=float)
                spin_vec[is_magnetic, 2] = np.sign(magmoms[is_magnetic, 2])
                spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
            elif self.spinor_mode == 'nc':
                spin_length = np.linalg.norm(magmoms, axis=1)
                spin_vec = np.zeros_like(magmoms, dtype=float)
                is_magnetic = spin_length > self.magnetic_tolerance
                spin_length[~is_magnetic] = 0.0
                spin_vec[is_magnetic, :] = magmoms[is_magnetic, :] / spin_length[is_magnetic][:, None]
                spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
        except Exception as e:
            print(f"Fail to read the OpenMX \'.out\' file in {folder} : {e}")
            return None
        max_niter = int(max_niter * self.converge_step_ratio)
        if niter_scf > max_niter:
            print(f"The number of self-consistent iterations is excessive in {folder}: {niter_scf} over {max_niter}")
            return None
        # read Hamiltonian and overlap matrix
        os.chdir(folder)
        os.system(f"{self.openmx_read_path} {f_HSout}")
        HS_json_path = os.path.join(folder, 'HS.json')
        try:
            if self.spinor_mode == 'none':
                pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H, S = self._gather_HS_from_json(HS_json_path, atomic_numbers, basis_def, with_S=True)
            elif self.spinor_mode == 'collinear':
                pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H, S = self._gather_HS_from_json_collinear(HS_json_path, atomic_numbers, basis_def, with_S=True)
            elif self.spinor_mode == 'nc':
                pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H, S = self._gather_HS_from_json_nc(HS_json_path, atomic_numbers, basis_def, with_S=True)
                hamiltonian_real = H['real']
                hamiltonian_imag = H['imag']
            os.remove(HS_json_path)
        except Exception as e:
            print(f"Failed to load Hamiltonian and overlap matrix from {f_HSout}")
            print(e)
            return None

        graph_tmp = Data(z=torch.LongTensor(atomic_numbers),
                        cell = torch.Tensor(latt[None,:,:]),
                        total_energy=torch.Tensor([total_energy]),
                        pos=torch.FloatTensor(pos),
                        node_counts=torch.LongTensor([len(atomic_numbers)]),
                        edge_index=torch.LongTensor(edge_index),
                        inv_edge_idx=torch.LongTensor(inv_edge_idx),
                        nbr_shift=torch.FloatTensor(nbr_shift),
                        cell_shift=torch.LongTensor(cell_shift),
                        Son = torch.FloatTensor(S[:pos.shape[0]]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:]))

        if self.spinor_mode == 'none':
            graph_tmp['Hon'] = torch.FloatTensor(H[:pos.shape[0]])
            graph_tmp['Hoff'] = torch.FloatTensor(H[pos.shape[0]:])
        elif self.spinor_mode == 'collinear':
            graph_tmp['Hon'] = torch.FloatTensor(H[:pos.shape[0]])
            graph_tmp['Hoff'] = torch.FloatTensor(H[pos.shape[0]:])
            graph_tmp['spin_length'] = torch.FloatTensor(spin_length)
            graph_tmp['spin_vec'] = torch.FloatTensor(spin_vec)
        elif self.spinor_mode == 'nc':
            graph_tmp['Hon'] = torch.FloatTensor(hamiltonian_real[:len(atomic_numbers)])
            graph_tmp['Hoff'] = torch.FloatTensor(hamiltonian_real[len(atomic_numbers):])
            graph_tmp['iHon'] = torch.FloatTensor(hamiltonian_imag[:len(atomic_numbers)])
            graph_tmp['iHoff'] = torch.FloatTensor(hamiltonian_imag[len(atomic_numbers):])
            graph_tmp['overlap'] = torch.FloatTensor(S)
            graph_tmp['spin_length'] = torch.FloatTensor(spin_length)
            graph_tmp['spin_vec'] = torch.FloatTensor(spin_vec)

        if self.pack_atomic_force:
            graph_tmp['atomic_forces'] = torch.FloatTensor(atomic_forces)

        if self.with_nonscf:
            # read non-scf Hamiltonian
            os.chdir(folder)
            os.system(f"{self.openmx_read_path} {f_nonscf}")
            HS_json_path = os.path.join(folder, 'HS.json')
            try:
                if self.spinor_mode == 'none':
                    _, _, _, _, _, H0 = self._gather_HS_from_json(HS_json_path, atomic_numbers, basis_def)
                elif self.spinor_mode == 'collinear':
                    _, _, _, _, _, H0 = self._gather_HS_from_json_collinear(HS_json_path, atomic_numbers, basis_def)
                elif self.spinor_mode == 'nc':
                    _, _, _, _, _, H0, L = self._gather_HS_from_json_nc(HS_json_path, atomic_numbers, basis_def, with_L=True)
                    hamiltonian_real0 = H0['real']
                    hamiltonian_imag0 = H0['imag']
                os.remove(HS_json_path)
            except Exception as e:
                print(f"Failed to load non-scf Hamiltonian matrix from {f_nonscf}")
                print(e)
                return None
            if self.spinor_mode == 'nc':
                graph_tmp['Hon0'] = torch.FloatTensor(hamiltonian_real0[:len(atomic_numbers)])
                graph_tmp['Hoff0'] = torch.FloatTensor(hamiltonian_real0[len(atomic_numbers):])
                graph_tmp['iHon0'] = torch.FloatTensor(hamiltonian_imag0[:len(atomic_numbers)])
                graph_tmp['iHoff0'] = torch.FloatTensor(hamiltonian_imag0[len(atomic_numbers):])
                graph_tmp['Lon'] = torch.FloatTensor(L[:pos.shape[0]])
                graph_tmp['Loff'] = torch.FloatTensor(L[pos.shape[0]:])
            else:
                graph_tmp['Hon0'] = torch.FloatTensor(H0[:pos.shape[0]])
                graph_tmp['Hoff0'] = torch.FloatTensor(H0[pos.shape[0]:])

        return graph_tmp

    def _gen_graph_data_nonscf(self, folder:str, basis_def:dict):
        f_stdin = os.path.join(folder, self.stdin_filename)
        f_nonscf = os.path.join(folder, self.nonscf_filename)
        
        # read standard input file
        try:
            species, latt, coordinates, magmoms, _, atomic_mag_constrained, atomic_u_enhances = read_openmx_input(f_stdin)
            if atomic_mag_constrained is not None:
                spinor_mode_tmp = 'nc'
            elif atomic_u_enhances is not None:
                spinor_mode_tmp = 'collinear'
            else:
                spinor_mode_tmp = 'none'
            if self.spinor_mode != spinor_mode_tmp:
                raise RuntimeError(f"Error(3006): The spinor mode used in the OpenMX calculation is inconsistent with the setting specified during graph data packing.\n \
                                   Path: {f_stdin}")
            atomic_numbers = np.array([Element[s].Z for s in species])
            latt = latt * QTcts.ANGtoBOHR
            coordinates = coordinates * QTcts.ANGtoBOHR
            if self.spinor_mode == 'collinear':
                spin_length = np.abs(magmoms[:, 2])
                is_magnetic = spin_length > self.magnetic_tolerance
                spin_length[~is_magnetic] = 0.0
                spin_vec = np.zeros_like(magmoms, dtype=float)
                spin_vec[is_magnetic, 2] = np.sign(magmoms[is_magnetic, 2])
                spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
            elif self.spinor_mode == 'nc':
                spin_length = np.linalg.norm(magmoms, axis=1)
                spin_vec = np.zeros_like(magmoms, dtype=float)
                is_magnetic = spin_length > self.magnetic_tolerance
                spin_length[~is_magnetic] = 0.0
                spin_vec[is_magnetic, :] = magmoms[is_magnetic, :] / spin_length[is_magnetic][:, None]
                spin_length = average_close_values(spin_length, self.moment_similar_tolerence)
        except Exception as e:
            print(f"Fail to read the OpenMX input file in {folder} : {e}")
            return None
        # read non-scf Hamiltonian
        os.chdir(folder)
        os.system(f"{self.openmx_read_path} {f_nonscf}")
        HS_json_path = os.path.join(folder, 'HS.json')
        try:
            if self.spinor_mode == 'none':
               pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H0, S = self._gather_HS_from_json(HS_json_path, atomic_numbers, basis_def, with_S=True)
            elif self.spinor_mode == 'collinear':
                pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H0, S = self._gather_HS_from_json_collinear(HS_json_path, atomic_numbers, basis_def, with_S=True)
            elif self.spinor_mode == 'nc':
                pos, edge_index, inv_edge_idx, nbr_shift, cell_shift, H0, S, L = self._gather_HS_from_json_nc(HS_json_path, atomic_numbers, basis_def, with_S=True, with_L=True)
                hamiltonian_real0 = H0['real']
                hamiltonian_imag0 = H0['imag']
            os.remove(HS_json_path)
        except:
            print(f"Failed to load Hamiltonian and overlap matrix from {f_nonscf}")
            return None
        
        graph_tmp = Data(z=torch.LongTensor(atomic_numbers),
                        cell = torch.Tensor(latt[None,:,:]),
                        total_energy=torch.Tensor([-9999.9]),
                        pos=torch.FloatTensor(pos),
                        node_counts=torch.LongTensor([len(atomic_numbers)]),
                        edge_index=torch.LongTensor(edge_index),
                        inv_edge_idx=torch.LongTensor(inv_edge_idx),
                        nbr_shift=torch.FloatTensor(nbr_shift),
                        cell_shift=torch.LongTensor(cell_shift),
                        Son = torch.FloatTensor(S[:pos.shape[0]]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:]))

        if self.spinor_mode == 'none':
            graph_tmp['Hon'] = torch.FloatTensor(H0[:pos.shape[0]])
            graph_tmp['Hoff'] = torch.FloatTensor(H0[pos.shape[0]:])
            graph_tmp['Hon0'] = torch.FloatTensor(H0[:pos.shape[0]])
            graph_tmp['Hoff0'] = torch.FloatTensor(H0[pos.shape[0]:])
        elif self.spinor_mode == 'collinear':
            graph_tmp['Hon'] = torch.FloatTensor(H0[:pos.shape[0]])
            graph_tmp['Hoff'] = torch.FloatTensor(H0[pos.shape[0]:])
            graph_tmp['Hon0'] = torch.FloatTensor(H0[:pos.shape[0]])
            graph_tmp['Hoff0'] = torch.FloatTensor(H0[pos.shape[0]:])
            graph_tmp['spin_length'] = torch.FloatTensor(spin_length)
            graph_tmp['spin_vec'] = torch.FloatTensor(spin_vec)
        elif self.spinor_mode == 'nc':
            graph_tmp['Hon'] = torch.FloatTensor(hamiltonian_real0[:len(atomic_numbers)])
            graph_tmp['Hoff'] = torch.FloatTensor(hamiltonian_real0[len(atomic_numbers):])
            graph_tmp['iHon'] = torch.FloatTensor(hamiltonian_imag0[:len(atomic_numbers)])
            graph_tmp['iHoff'] = torch.FloatTensor(hamiltonian_imag0[len(atomic_numbers):])
            graph_tmp['Hon0'] = torch.FloatTensor(hamiltonian_real0[:len(atomic_numbers)])
            graph_tmp['Hoff0'] = torch.FloatTensor(hamiltonian_real0[len(atomic_numbers):])
            graph_tmp['iHon0'] = torch.FloatTensor(hamiltonian_imag0[:len(atomic_numbers)])
            graph_tmp['iHoff0'] = torch.FloatTensor(hamiltonian_imag0[len(atomic_numbers):])
            graph_tmp['overlap'] = torch.FloatTensor(S)
            graph_tmp['spin_length'] = torch.FloatTensor(spin_length)
            graph_tmp['spin_vec'] = torch.FloatTensor(spin_vec)
            graph_tmp['Lon'] = torch.FloatTensor(L[:pos.shape[0]])
            graph_tmp['Loff'] = torch.FloatTensor(L[pos.shape[0]:])

        return graph_tmp
