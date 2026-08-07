"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
import numpy as np


class _PhononKernelMixin:
    def _phonon_cal(self, q_grid):
        q_grid = q_grid.reshape(-1, 3)
        dynmat_array = np.array([self.phonon.get_dynamical_matrix_at_q(q) for q in q_grid])
        eigvals, eigvecs = np.linalg.eigh(dynmat_array) # eigvals: (nq, nbranches), eigvecs: (nq, nbranches, nbranches)
        phon_vecs = np.transpose(eigvecs, (0, 2, 1)) # shape: (nq, nbranches, nbranches)
        freq_grid = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real) * QTcts.PHONOPYtoHARTREE # shape: (nq, nbranches)
        return freq_grid, phon_vecs
