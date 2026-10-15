"""Configuration and lossless output for phonon-resolved electron self-energy."""

from datetime import datetime
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

from qtrans.core.constants import QTcts
from qtrans.core.errors import InputParameterError
from qtrans.core.validators import validate_bool


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Cannot serialize {}'.format(type(value).__name__))


def validate_mode_configuration(config):
    """Validate the original input, including blocks unused by other workflows."""
    se = config.get('self-energy', {})
    enabled = se.get('SE_mode_resolved', False)
    try:
        validate_bool(enabled, 'SE_mode_resolved')
        if not enabled:
            return None
        basic = config.get('basic', {})
        dispersion = config.get('dispersion', {})
        advanced = config.get('advanced', {})
        phonon = config.get('phonon', {})
        if (str(basic.get('cal_mode', '')).lower() != 'band' or
                dispersion.get('disp_renormalize', False) is not True or
                dispersion.get('band_grid_mode', False) or
                str(se.get('SE_type', 'electron')).lower() != 'electron' or
                advanced.get('read_sparse_grad_mat', False) is not True or
                advanced.get('soc_switch', False) or
                str(advanced.get('magnetic_type', 'none')).lower() != 'none' or
                phonon.get('apply_correction', False) or
                phonon.get('quadrupole_correction', False)):
            raise ValueError(
                'Requires band path renormalization, SE_type=electron, sparse gradients, '
                'nonmagnetic/no SOC, and both long-range correction options disabled.')
    except (TypeError, ValueError) as error:
        raise InputParameterError(5305, 'self-energy', 'SE_mode_resolved', error) from error
    payload = json.dumps(config, sort_keys=True, default=_json_value).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def mode_output_path(calculator):
    return Path(calculator.outdir) / 'self_energy' / 'electron_self_energy_modes.npz'


@contextmanager
def collective_mode_errors(calculator, stage):
    """Synchronize local-only work failures before entering the next collective."""
    if not getattr(calculator, 'SE_mode_resolved', False):
        yield
        return
    error = None
    try:
        yield
    except Exception as exc:
        error = 'rank {}: {}: {}'.format(calculator.rank, type(exc).__name__, exc)
    errors = [error] if calculator.comm is None else calculator.comm.allgather(error)
    failures = [message for message in errors if message is not None]
    if failures:
        raise RuntimeError('Mode self-energy {} failed: {}'.format(stage, '; '.join(failures)))


def preflight_mode_output(calculator):
    """Reject an existing result on every rank before starting the calculation."""
    error = None
    if calculator.rank == 0:
        try:
            path = mode_output_path(calculator)
            if os.path.lexists(path):
                raise FileExistsError('Refusing to overwrite {}; use a new outdir.'.format(path))
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = str(exc)
    if calculator.comm is not None:
        error = calculator.comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _mode_metadata(calculator, q_grid, result):
    import qtrans
    from qtrans import _native
    from mpi4py import MPI

    package_dir = Path(qtrans.__file__).resolve().parent
    # Hash the actual imported Python tree; a version string alone is not provenance.
    hashes = {str(path.relative_to(package_dir)): _file_hash(path)
              for path in sorted(package_dir.rglob('*.py'))}
    weights = np.asarray(calculator.weight_q)
    weight_sum = float(weights * len(q_grid)) if weights.ndim == 0 else float(weights.sum())
    metadata = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'qtrans_version': qtrans.__version__,
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'package_path': str(package_dir),
        'extension_path': str(Path(_native.__file__).resolve()),
        'extension_sha256': _file_hash(_native.__file__),
        'python_source_sha256': hashes,
        'input_configuration_sha256': getattr(calculator, '_se_mode_input_sha256', None),
        'temperature_K': calculator.temperature / QTcts.KELVINtoHARTREE,
        'fermi_energy_eV': calculator.efermi * QTcts.HARTREEtoEV,
        'smeark_eV': calculator.smeark * QTcts.HARTREEtoEV,
        'e_thr_eV': calculator.e_thr * QTcts.HARTREEtoEV,
        'phonon_frequency_cutoff_eV': calculator.phonon_frequency_cutoff * QTcts.HARTREEtoEV,
        'initial_state_window_eV': np.array([-1, 1]) * calculator.over_band_edge * QTcts.HARTREEtoEV,
        'initial_and_final_bands_1based': np.asarray(calculator.bands_indices) + 1,
        'k_size': calculator.k_size,
        'q_size': calculator.q_size,
        'q_count': len(q_grid),
        'q_weight_sum': weight_sum,
        'q_points_sha256': hashlib.sha256(np.ascontiguousarray(q_grid, dtype=np.float64).tobytes()).hexdigest(),
        'q_weights_sha256': hashlib.sha256(np.ascontiguousarray(weights, dtype=np.float64).tobytes()).hexdigest(),
        'q_split_mode': getattr(calculator, 'q_split_mode', 'contiguous'),
        'mpi_size': calculator.rank_size,
        'mpi_library': (MPI.Get_library_version().strip() if MPI.Is_initialized()
                        else 'MPI runtime not initialized in serial execution'),
        'omp_num_threads_env': os.environ.get('OMP_NUM_THREADS'),
        'native_profiles_by_rank': result['profiles'],
        'mode_array_bytes_per_rank': result['sigma_mode'].nbytes,
        'axes': ['k', 'initial_band', 'phonon_branch', 'energy'],
        'units': 'eV',
        'energy_reference': 'Fermi level',
        'self_energy_convention': 'retarded; Sigma(E) - Re Sigma(0)',
        'branch_index_convention': 'frequency_sorted_at_each_q',
        'q_summed': True,
        'initial_state_mask_convention': 'before_electronic_degeneracy_average',
        'electronic_degeneracy_tolerance_Hartree': QTcts.TENPM6,
        'electronic_degeneracy_convention': 'same existing average applied independently to every branch',
        'spinor_mode': calculator.spinor_mode,
        'long_range_correction': False,
    }
    return metadata


def write_mode_output(calculator, k_grid, q_grid, eigen, omega, total, result):
    """Write on root, broadcasting failures so other ranks do not report success."""
    error = None
    if calculator.rank == 0:
        try:
            metadata = _mode_metadata(calculator, q_grid, result)
            path = mode_output_path(calculator)
            # Exclusive creation also closes the race after the early preflight.
            with open(path, 'xb') as stream:
                np.savez(
                    stream,
                    sigma_mode_eV=result['sigma_mode'] * QTcts.HARTREEtoEV,
                    sigma_total_eV=total * QTcts.HARTREEtoEV,
                    energies_eV=omega * QTcts.HARTREEtoEV,
                    bare_energies_eV=eigen * QTcts.HARTREEtoEV,
                    kpoints_frac=np.asarray(k_grid),
                    bands_1based=np.asarray(calculator.bands_indices, dtype=np.int64) + 1,
                    branches_1based=np.arange(1, result['sigma_mode'].shape[2] + 1),
                    initial_state_mask=result['initial_state_mask'],
                    metadata_json=np.array(json.dumps(metadata, sort_keys=True, default=_json_value)),
                )
        except Exception as exc:
            error = 'Failed to write mode self-energy: {}'.format(exc)
    if calculator.comm is not None:
        error = calculator.comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)
