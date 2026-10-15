"""Memory-based k-point batching for the full-matrix electronic solver."""

from numbers import Integral


DEFAULT_K_BATCH_MEM_MIB = 256
DEFAULT_MAX_K_BATCH = 1024
_COMPLEX128_BYTES = 16
_MATRIX_ALLOWANCE = 8


def get_k_batch_size(calculator, max_batch=DEFAULT_MAX_K_BATCH):
    """Return a positive batch size bounded by max_batch.

    k_batch_mem is a per-process MiB budget for the electronic batch, not a
    total RSS limit. The solver returns full norbs-by-norbs eigenvectors even
    when only a few bands are selected. Allow eight complex128 matrices per k
    for accumulated output, its stacked copy, and solver/transformation work.
    This is a conservative batching estimate, not a backend workspace bound.
    At least one k is processed even when its estimate exceeds the budget.
    """
    norbs = getattr(calculator, 'norbs', None)
    memory_mib = getattr(calculator, 'k_batch_mem', DEFAULT_K_BATCH_MEM_MIB)
    for name, value in [('norbs', norbs), ('k_batch_mem', memory_mib),
                        ('max_batch', max_batch)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    bytes_per_k = _MATRIX_ALLOWANCE * _COMPLEX128_BYTES * int(norbs)**2
    capacity = int(memory_mib) * 1024**2 // bytes_per_k
    return max(1, min(int(max_batch), capacity))
