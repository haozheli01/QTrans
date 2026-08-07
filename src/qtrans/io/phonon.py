"""QTrans io.phonon helpers."""

import h5py
import numpy as np


def write_force_constants_h5(filepath:str, fcs:np.ndarray, p2s_indices:np.ndarray):
    # fcs: shape (natoms_uc, 3, natoms_sc, 3)
    fcs = np.transpose(fcs, (0, 2, 1, 3))
    with h5py.File(filepath, 'w') as f:
        f.create_dataset("force_constants", data=fcs)
        f.create_dataset("p2s_map", data=p2s_indices)
