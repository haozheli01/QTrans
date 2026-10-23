"""Optional phonon-branch electron self-energy output and collective stages."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from qtrans.core.constants import QTcts
from qtrans.core.defaults import calculator_default_parameters
from qtrans.core.errors import InputParameterError
from qtrans.core.validators import validate_bool


def validate_mode_configuration(config):
    """Check the original input, including blocks the selected workflow ignores."""
    def value(block, key):
        return config.get(block, {}).get(
            key, calculator_default_parameters.get(block, {}).get(key))

    enabled = value('self-energy', 'SE_mode_resolved')
    try:
        validate_bool(enabled, 'SE_mode_resolved')
    except TypeError as error:
        raise InputParameterError(5305, 'self-energy', 'SE_mode_resolved', str(error))
    if not enabled:
        return False
    required = {
        ('basic', 'cal_mode'): 'band',
        ('dispersion', 'disp_renormalize'): True,
        ('dispersion', 'band_grid_mode'): False,
        ('advanced', 'read_sparse_grad_mat'): True,
        ('advanced', 'magnetic_type'): 'none',
        ('advanced', 'soc_switch'): False,
        ('phonon', 'apply_correction'): False,
        ('phonon', 'quadrupole_correction'): False,
        ('self-energy', 'SE_type'): 'electron',
    }
    invalid = []
    for (block, key), expected in required.items():
        actual = value(block, key)
        if isinstance(expected, bool):
            valid = isinstance(actual, bool) and actual is expected
        else:
            valid = isinstance(actual, str) and actual.lower() == expected
        if not valid:
            invalid.append(f'{block}.{key} must be {expected!r} (got {actual!r})')
    if invalid:
        raise InputParameterError(5306, 'self-energy', 'SE_mode_resolved',
                                  '; '.join(invalid))
    return True


@contextmanager
def mode_collective_stage(calculator, stage):
    """Synchronize local failures before the next collective, only when enabled.

    Every rank must enter each stage once. Do not place collectives inside its
    body, or call it once per locally owned q point.
    """
    if not getattr(calculator, 'SE_mode_resolved', False):
        yield
        return
    failure = None
    try:
        yield
    except Exception as error:
        failure = f'rank {calculator.rank}: {type(error).__name__}: {error}'
    comm = calculator.comm
    failures = [failure] if comm is None else comm.allgather(failure)
    failures = [item for item in failures if item is not None]
    if failures:
        raise RuntimeError(f'Electron self-energy modes ({stage}): ' + '; '.join(failures))


def mode_output_path(calculator):
    return Path(calculator.outdir) / 'self_energy' / 'electron_self_energy_modes.npz'


def check_mode_output(calculator):
    """Fail collectively before calculation if the new output already exists."""
    if not getattr(calculator, 'SE_mode_resolved', False):
        return
    with mode_collective_stage(calculator, 'output preflight'):
        if calculator.rank == 0:
            path = mode_output_path(calculator)
            if os.path.lexists(path):
                raise FileExistsError(f'{path} already exists; use a new output directory.')


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot encode {type(value).__name__} as JSON.')


def _json_text(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=_json_value)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(_json_text([array.dtype.str, array.shape]).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mode_metadata(calculator, kpoints_frac, qpoints_frac, rank_statistics):
    """Describe the actually loaded code, input convention, and numerical run."""
    import qtrans
    from qtrans.extensions import native
    from mpi4py import MPI

    package_dir = Path(qtrans.__file__).resolve().parent
    source_hashes = {str(path.relative_to(package_dir)): _file_hash(path)
                     for path in sorted(package_dir.rglob('*.py'))}
    native_path = Path(native._native.__file__).resolve()
    source_root = package_dir.parent.parent
    c_hashes = {str(path.relative_to(source_root)): _file_hash(path)
                for path in (source_root / 'c').glob('*.c')}
    weights = np.asarray(calculator.weight_q, dtype=float)
    if weights.ndim == 0:
        weights = np.full(len(qpoints_frac), weights.item())
    configuration = getattr(calculator, '_self_energy_modes_input', None)
    metadata = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'program_version': qtrans.__version__,
        'package_path': str(package_dir),
        'python_source_sha256': source_hashes,
        'python_source_tree_sha256': hashlib.sha256(_json_text(source_hashes).encode()).hexdigest(),
        'c_source_sha256': c_hashes,
        'native_path': str(native_path),
        'native_sha256': _file_hash(native_path),
        'configuration_hash_kind': 'canonical_json_of_original_configuration',
        'configuration_sha256': (hashlib.sha256(_json_text(configuration).encode()).hexdigest()
                                 if configuration is not None else None),
        'temperature_K': float(calculator.temperature / QTcts.KELVINtoHARTREE),
        'efermi_eV': float(calculator.efermi * QTcts.HARTREEtoEV),
        'k_size': getattr(calculator, 'k_size', None),
        'q_size': getattr(calculator, 'q_size', None),
        'kpoints_frac_sha256': _array_hash(kpoints_frac),
        'qpoints_frac_sha256': _array_hash(qpoints_frac),
        'q_weights_sha256': _array_hash(weights),
        'q_weight_sum': float(weights.sum()),
        'q_point_count': len(qpoints_frac),
        'q_split_mode': getattr(calculator, 'q_split_mode', 'contiguous'),
        'initial_bands_1based': np.asarray(calculator.bands_indices) + 1,
        'final_bands_1based': np.asarray(calculator.bands_indices) + 1,
        'branch_index_convention': 'frequency_sorted_at_each_q',
        'summed_axes': ['q', 'final_electron_band'],
        'sigma_mode_axes': ['k', 'initial_electron_band', 'phonon_branch', 'energy'],
        'sigma_total_axes': ['k', 'initial_electron_band', 'energy'],
        'self_energy_unit': 'eV',
        'energy_reference': 'efermi',
        'zero_reference': 'subtract_Re_Sigma_at_E_zero_per_branch',
        'electron_degeneracy_tolerance_Hartree': float(QTcts.TENPM6),
        'initial_state_mask_convention': 'before_electron_degeneracy_average; do_not_reapply',
        'spinor_mode': calculator.spinor_mode,
        'long_range_correction': False,
        'mpi_size': calculator.rank_size,
        'mpi_library': MPI.Get_library_version(),
        'thread_environment': {key: os.environ.get(key) for key in
                               ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                'OPENBLAS_NUM_THREADS', 'MKL_THREADING_LAYER')},
        'rank_statistics': rank_statistics,
    }
    for name in ('smeark', 'e_thr', 'phonon_frequency_cutoff', 'over_band_edge'):
        metadata[name + '_eV'] = float(getattr(calculator, name) * QTcts.HARTREEtoEV)
    return metadata


def write_mode_output(calculator):
    """Called by all ranks; root writes once, then all ranks release the payload."""
    if not getattr(calculator, 'SE_mode_resolved', False):
        return
    try:
        with mode_collective_stage(calculator, 'NPZ output'):
            if calculator.rank == 0:
                payload = calculator._electron_self_energy_modes
                path = mode_output_path(calculator)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also protects against a race after preflight.
                with path.open('xb') as stream:
                    np.savez(stream, **payload)
    finally:
        calculator._electron_self_energy_modes = None


def make_mode_payload(calculator, modes, total, energies, bare, kpoints_frac,
                      initial_mask, metadata):
    return {
        'sigma_mode_eV': modes * QTcts.HARTREEtoEV,
        'sigma_total_eV': total * QTcts.HARTREEtoEV,
        'energies_eV': energies * QTcts.HARTREEtoEV,
        'bare_energies_eV': bare * QTcts.HARTREEtoEV,
        'kpoints_frac': np.array(kpoints_frac, dtype=np.float64, copy=True),
        'bands_1based': np.asarray(calculator.bands_indices, dtype=np.int64) + 1,
        'branches_1based': np.arange(1, modes.shape[2] + 1, dtype=np.int64),
        'initial_state_mask': np.array(initial_mask, dtype=bool, copy=True),
        'metadata_json': np.array(_json_text(metadata)),
    }
