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

    def _phonon_cal_q_batch(self, q_grid):
        """Build at most 4096 dynamical matrices using one Phonopy object.

        Used for metallic conductivity and semiconductor mobility under the
        nonmagnetic, no-SOC, sparse order-0 guard.  The configured
        DynamicalMatrix retains its NAC parameters
        and Gonze dataset between blocks.  With no nac_q_direction argument,
        the Gamma-point behavior matches the existing scalar call.
        """
        from phonopy.harmonic.dynamical_matrix import run_dynamical_matrix_solver_c

        q_grid = np.ascontiguousarray(np.asarray(q_grid, dtype=float).reshape(-1, 3))
        nbranches = 3 * self.natoms
        freq_grid = np.empty((len(q_grid), nbranches), dtype=float)
        phon_vecs = np.empty((len(q_grid), nbranches, nbranches), dtype=np.complex128)
        if not len(q_grid):
            return freq_grid, phon_vecs

        # force_constants/nac_params setters normally initialize this object.
        # Initialize once if needed, never once per q or per block.
        dynamical_matrix = self.phonon.dynamical_matrix
        if dynamical_matrix is None:
            self.phonon._set_dynamical_matrix()
            dynamical_matrix = self.phonon.dynamical_matrix
        if dynamical_matrix is None:
            raise RuntimeError('Phonopy dynamical matrix has not yet been built.')

        for start in range(0, len(q_grid), 4096):
            stop = min(start + 4096, len(q_grid))
            dynmat_block = run_dynamical_matrix_solver_c(
                dynamical_matrix, q_grid[start:stop])
            eigvals, eigvecs = np.linalg.eigh(dynmat_block)
            freq_grid[start:stop] = (
                np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
                * QTcts.PHONOPYtoHARTREE)
            phon_vecs[start:stop] = np.transpose(eigvecs, (0, 2, 1))
            del dynmat_block, eigvals, eigvecs
        return freq_grid, phon_vecs
