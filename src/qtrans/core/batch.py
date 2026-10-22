"""Shared wavefunction batch-size helpers."""

import numpy as np


DEFAULT_K_BATCH_MEM_MIB = 256


def get_k_batch_size(calculator, max_batch=1024):
    """Return a bounded k batch from the calculator's MiB budget."""
    mem_mib = getattr(calculator, 'k_batch_mem', DEFAULT_K_BATCH_MEM_MIB)
    if (isinstance(mem_mib, bool) or
            not isinstance(mem_mib, (int, np.integer)) or
            int(mem_mib) <= 0):
        raise ValueError('k_batch_mem must be a positive integer in MiB.')
    norbs = int(getattr(calculator, 'norbs'))
    wave_bytes = max(1, norbs ** 2 * np.dtype(np.complex128).itemsize)
    budget_bytes = int(mem_mib) * 1024 ** 2
    return min(int(max_batch), max(1, budget_bytes // wave_bytes))
