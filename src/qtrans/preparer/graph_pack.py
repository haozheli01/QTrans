"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
from qtrans.data.basis import basis_def_dicts
from qtrans.io.filesystem import check_path_exist
from qtrans.io.phonon import write_force_constants_h5
from qtrans.parallel.mpi import gather_to_rank_root
from typing import List
import numpy as np
import os
import pickle


class _GraphPackMixin:
    def _check_files_for_graph(self, folders: List[str]) -> List[str]:
        """
        Check each folder to ensure required files exist.
        Return a list of folders that pass the check.
        """
        if self.DFT_software == 'openmx':
            # Conditionally add vbout
            if self.is_pred_graph:
                required_filenames = [self.stdin_filename, self.nonscf_filename]
            else:
                required_filenames = [self.HSout_filename, self.vbout_filename]
                if self.with_nonscf:
                    required_filenames.append(self.nonscf_filename)
            useful_folders = []

            for folder in folders:
                missing_files = []
                for filename in required_filenames:
                    filepath = os.path.join(folder, filename)
                    if not check_path_exist(filepath):
                        missing_files.append(filename)

                if not missing_files:
                    useful_folders.append(folder)
                else:
                    print(f"Neglect folder '{folder}': missing {', '.join(missing_files)}")
                    # raise RuntimeError(f"Error: Missing necessary files to generate graph data!")
        
        elif self.DFT_software == 'abacus':
            required_filenames = ['STRU']
            if self.magnetic_type == 'collinear':
                required_filenames_in_output_folder = ['data-HR-sparse_SPIN0.csr', 'data-HR-sparse_SPIN1.csr', 
                                                       'data-SR-sparse_SPIN0.csr', self.abacus_scf_log]
            else:
                required_filenames_in_output_folder = ['data-HR-sparse_SPIN0.csr',  
                                                       'data-SR-sparse_SPIN0.csr', self.abacus_scf_log]
            useful_folders = []
            for folder in folders:
                missing_files = []
                for filename in required_filenames:
                    filepath = os.path.join(folder, filename)
                    if not check_path_exist(filepath):
                        missing_files.append(filename)
                for filename in required_filenames_in_output_folder:
                    filepath = os.path.join(os.path.join(folder, self.abacus_out_folder), filename)
                    if not check_path_exist(filepath):
                        missing_files.append(filename)
                if not missing_files:
                    useful_folders.append(folder)
                else:
                    print(f"Neglect folder '{folder}': missing {', '.join(missing_files)}")
                    # raise RuntimeError(f"Error: Missing necessary files to generate graph data!")
        return useful_folders

    def pack_graph_data(self):
        # get basis defination
        basis_def = basis_def_dicts[self.basis_label]

        graphs_local = dict()
        
        for ifolder, each in enumerate(self.data_folders_local):
            if self.DFT_software == 'openmx':
                if self.is_pred_graph:
                    graph_tmp = self._gen_graph_data_nonscf(each, basis_def)
                else:
                    graph_tmp = self._gen_graph_data_scf(each, basis_def)
            elif self.DFT_software == 'abacus':
                graph_tmp = self._gen_graph_data_abacus(each, basis_def)
            if not (graph_tmp is None):
                graphs_local[self.data_folders_index_local[ifolder]] = graph_tmp

        graphs_all = gather_to_rank_root(graphs_local, self.comm, root=0)
        if self.rank == 0:
            graphs = {}
            for each in graphs_all:
                graphs.update(each)
            # sort the graphs in case of silently corrupt
            graphs = dict(sorted(graphs.items()))
            # in case of big graph data, save as pkl file
            graph_npz_path = os.path.join(self.outdir, 'graph_data_QTrans.npz')
            graph_pkl_path = os.path.join(self.outdir, 'graph_data_QTrans.pkl')
            try:
                np.savez(graph_npz_path, graph=graphs)
                print(f"QTrans graph data saved as npz: {graph_npz_path}")

            except OverflowError as e:
                print("QTrans np.savez failed (graph too large), switching to pickle...")
                print(f"Reason: {e}")

                with open(graph_pkl_path, "wb") as f:
                    pickle.dump(graphs, f, protocol=5)

                print(f"QTrans graph data saved as pickle: {graph_pkl_path}")

    def _gen_force_constants_from_force(self, forces:np.ndarray, p2s_indices:np.ndarray):
        _, natoms_sc, _ = forces.shape
        forces_evA = forces.reshape(self.natoms_uc, 3, natoms_sc, 3) * QTcts.HARTREEtoEV / QTcts.BOHRtoANG
        fcs_eVA2 =  -forces_evA / (self.epc_sc_disp * QTcts.BOHRtoANG) # shape (natoms_uc, 3, natoms_sc, 3)
        filename = os.path.join(self.outdir, 'FORCE_CONSTANTS_QTrans.hdf5')
        write_force_constants_h5(filename, fcs_eVA2, p2s_indices)

        # FORCE_SETS
        # with open(filename, 'w') as fout:
        #     fout.write(f"{natoms_sc}\n")
        #     fout.write(f"{nstructs}\n")
        #     for iatom_uc in range(self.natoms_uc):
        #         atom_id_sc = p2s_indices[iatom_uc]
        #         for idir in range(3):
        #             forces_this = forces_evA[iatom_uc, idir] # shape (natoms_sc, 3)
        #             disp = np.zeros(3)
        #             disp[idir] = self.epc_sc_disp * QTcts.BOHRtoANG
        #             fout.write(f"{atom_id_sc}\n")
        #             fout.write("{:20.16f} {:20.16f} {:20.16f}\n".format(*disp))
        #             # forces of supercell atoms
        #             for jatom_sc in range(natoms_sc):
        #                 fout.write("{:20.16f} {:20.16f} {:20.16f}\n".format(*forces_this[jatom_sc]))
        #             fout.write('\n')
