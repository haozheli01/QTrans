"""Numerical convergence helpers for imaginary-axis solvers."""

import warnings

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs


def mixed(old, new, fraction):
    return old + fraction * (new - old)


def scaled_residual(old_arrays, new_arrays, absolute_scale):
    residual = 0.0
    for old, new in zip(old_arrays, new_arrays):
        scale = max(float(np.max(np.abs(new))), float(absolute_scale))
        residual = max(residual, float(np.max(np.abs(new - old))) / scale)
    return residual


def dominant_real_eigenvalue(matvec, size, tolerance, max_iterations):
    """Return the eigenvalue with largest real part for a real linear map."""
    if size <= 0:
        raise ValueError("The linearized Eliashberg operator is empty.")
    if size == 1:
        return float(np.asarray(matvec(np.ones(1)))[0])

    if size <= 96:
        identity = np.eye(size)
        matrix = np.column_stack([matvec(identity[:, index]) for index in range(size)])
        values = np.linalg.eigvals(matrix)
        return float(np.max(values.real))

    operator = LinearOperator(
        shape=(size, size),
        matvec=lambda vector: np.asarray(matvec(vector), dtype=float),
        dtype=float,
    )
    initial = np.ones(size, dtype=float) / np.sqrt(float(size))
    try:
        values = eigs(
            operator,
            k=1,
            which="LR",
            v0=initial,
            tol=tolerance,
            maxiter=max_iterations,
            return_eigenvectors=False,
        )
    except ArpackNoConvergence as exc:
        if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
            raise RuntimeError(
                "The linearized Eliashberg eigenvalue solver did not converge."
            ) from exc
        warnings.warn(
            "ARPACK reached its iteration limit; using the best available "
            "Eliashberg eigenvalue.", RuntimeWarning
        )
        values = exc.eigenvalues
    value = values[np.argmax(values.real)]
    if abs(value.imag) > max(100.0 * tolerance, 1.0e-8):
        warnings.warn(
            "The leading Eliashberg eigenvalue has a non-negligible imaginary "
            "part ({:.3e}); check transition-table symmetry.".format(value.imag),
            RuntimeWarning,
        )
    return float(value.real)
