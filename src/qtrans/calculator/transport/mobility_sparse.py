"""Bounded non-spin sparse EPC transport loop.

This module is called only by the conductivity/mobility sparse guard in
``scattering.py``.  Other transport physics keeps its established path.
"""

from mpi4py import MPI
import os
from qtrans.core.constants import QTcts
from qtrans.core.logging import time_logger
from qtrans.extensions import prepare_sparse_grad_cache
from qtrans.extensions import scattering_rate_cal_EPC_helper_sparse_batch
from qtrans.numerics.distributions import bose_weight, fermi_weight
from qtrans.parallel.mpi import allreduce_array_in_place_chunks
from qtrans.parallel.mpi import iter_bcast_array_by_rank_chunks
from qtrans.core.batch import get_k_batch_size
import numpy as np


_MPI_BUDGET = 256 * 1024 ** 2
_CHANNEL_DTYPE = [('ik', int), ('ikpq_ibz', int), ('ikpq_fbz', int),
                  ('iband', int), ('jband', int), ('trans_prob', float)]
_Q_PHASE_BLOCK_SIZE = 4096
_TRACE_ENV = 'QTRANS_TRACE_MOBILITY_STARTUP'


def validate_commensurate_lookup(calculator, k_ibz, k_fbz, grid2ir, q_frac):
    """Return a reason to use direct solves, or ``None`` for a safe lookup.

    ``k_ibz`` and ``k_fbz`` are Cartesian; ``q_frac`` is fractional.  This
    checks both mesh origins and the two independently built IBZ row maps.
    """
    mesh = np.asarray(calculator.k_size, dtype=int)
    qmesh = np.asarray(calculator.q_size, dtype=int)
    if (mesh.shape != (3,) or qmesh.shape != (3,) or
            np.any(mesh < 1) or np.any(qmesh < 1)):
        return 'invalid k or q mesh dimensions'
    nfull = int(np.prod(mesh))
    k_ibz = np.asarray(k_ibz)
    k_fbz = np.asarray(k_fbz)
    q_frac = np.asarray(q_frac)
    mapped = np.asarray(grid2ir) if grid2ir is not None else None
    ibte_map = np.asarray(calculator.ir_indices)
    if (k_ibz.ndim != 2 or k_ibz.shape[1] != 3 or
            k_fbz.shape != (nfull, 3) or q_frac.ndim != 2 or
            q_frac.shape[1] != 3 or mapped is None or
            mapped.shape != (nfull,) or ibte_map.shape != (nfull,)):
        return 'incomplete full k mesh or IBZ map'
    if (np.any(mapped < 0) or np.any(mapped >= len(k_ibz)) or
            np.any(ibte_map < 0) or np.any(ibte_map >= len(k_ibz)) or
            not np.array_equal(mapped, ibte_map)):
        return 'IBZ mapping row order differs'
    if getattr(calculator, 'MC_sampling', 'none') != 'none':
        return 'q points use Monte Carlo sampling'
    if np.any(mesh % qmesh != 0):
        return 'q mesh does not divide k mesh'
    for name, kpoints in (('FBZ', k_fbz), ('IBZ', k_ibz)):
        # The mesh builder uses ``k_frac @ rec_latt``.  Invert that same
        # matrix here: ``_car2frac`` uses stored ``latt.T`` whose roundtrip
        # residual reaches 7.87e-6 mesh units on the MoS2 270x270 grid.
        frac = np.linalg.solve(calculator.graph_data.rec_latt.T,
                               kpoints.T).T
        scaled = np.mod(frac, 1.0) * mesh[None, :]
        if not np.allclose(scaled, np.rint(scaled), atol=1e-8, rtol=0):
            return f'{name} k coordinates have a shifted or off-grid origin'
    full_indices = calculator._get_kpq_BZ_index(k_fbz)
    if not np.array_equal(full_indices, np.arange(nfull)):
        return 'FBZ integer indexing differs'
    ibz_indices = calculator._get_kpq_BZ_index(k_ibz)
    if (np.any(ibz_indices < 0) or np.any(ibz_indices >= nfull) or
            not np.array_equal(mapped[ibz_indices], np.arange(len(k_ibz)))):
        return 'IBZ representatives do not match energy table rows'
    qscaled = np.mod(q_frac, 1.0) * qmesh[None, :]
    if not np.allclose(qscaled, np.rint(qscaled), atol=1e-8, rtol=0):
        return 'q coordinates have a shifted or off-grid origin'
    return None


def _inside_window(eigs, threshold, is_metal, is_hole, margin=0.0):
    if is_metal:
        return ((eigs >= threshold[0] - margin) &
                (eigs <= threshold[1] + margin))
    if is_hole:
        return eigs >= threshold - margin
    return eigs <= threshold + margin


def _candidate_indices(calculator, eig_k, eig_kpq, freq, initial_mask,
                       threshold, is_metal):
    """Conservative table prefilter; exact match is rebuilt after the solve."""
    # The MoS2 270x270 geometry spot check found up to 2.107e-6 Ha between
    # a direct solve and its symmetry-mapped IBZ energy.  The wider margin
    # only admits extra candidates; the direct solve rebuilds exact masks
    # and match tables before any physical contribution is accumulated.
    margin = max(1e-4, abs(calculator.e_thr) * 1e-4)
    final_mask = _inside_window(eig_kpq, threshold, is_metal,
                                calculator.is_hole, margin)
    active = initial_mask.any(axis=1) & final_mask.any(axis=1)
    if not np.any(active):
        return np.empty(0, dtype=int)
    valid_freq = freq >= calculator.phonon_frequency_cutoff
    if not np.any(valid_freq):
        return np.empty(0, dtype=int)
    diff = np.abs(np.abs(eig_k[:, :, None] - eig_kpq[:, None, :])[:, :, :, None]
                  - freq[None, None, None, :])
    matched = np.any((diff < calculator.e_thr + margin) &
                     valid_freq[None, None, None, :], axis=(1, 2, 3))
    return np.flatnonzero(active & matched)


def _collect_useful_k_wave_table(k_grid_all, eigen_local, wave_local,
                                 comm, chunk_bytes=_MPI_BUDGET):
    """Collect the selected initial-k table once before q-block processing."""
    local = np.ascontiguousarray(wave_local)
    if local.ndim != 3:
        raise ValueError('Sparse mobility wave table must be rank-3.')
    counts = [len(chunk) for chunk in k_grid_all]
    if comm is None:
        if len(k_grid_all) != 1 or eigen_local.shape[1] != counts[0]:
            raise ValueError('Initial-k data do not match the serial k split.')
        return (np.ascontiguousarray(k_grid_all[0]),
                np.ascontiguousarray(eigen_local.T), local.copy())
    rank = comm.Get_rank()
    if eigen_local.shape[1] != counts[rank]:
        raise ValueError('Initial-k eigenvalues do not match the local k split.')
    nbands = int(eigen_local.shape[0])
    nrows = int(sum(counts))
    starts = np.insert(np.cumsum(counts[:-1], dtype=np.int64), 0, 0)
    k_global = np.concatenate(k_grid_all, axis=0) if nrows else np.empty((0, 3))
    eig_global = np.empty((nrows, nbands), dtype=eigen_local.dtype)
    wave_global = np.empty((nrows,) + local.shape[1:], dtype=local.dtype)
    for owner, offset, wave_chunk in iter_bcast_array_by_rank_chunks(
            local, comm, chunk_bytes=chunk_bytes):
        count = len(wave_chunk)
        if rank == owner:
            eig_chunk = np.ascontiguousarray(
                eigen_local[:, offset:offset + count].T)
        else:
            eig_chunk = np.empty((count, nbands), dtype=eigen_local.dtype)
        comm.Bcast(eig_chunk, root=owner)
        start = int(starts[owner] + offset)
        wave_global[start:start + count] = wave_chunk
        eig_global[start:start + count] = eig_chunk
    return np.ascontiguousarray(k_global), eig_global, wave_global


def _trace_startup_record(calculator, label, local_seconds, extra=None):
    """Optional rank timing collector; enabled only by the trace env flag."""
    if calculator.comm is None:
        return
    payload = (int(calculator.rank), float(local_seconds), extra or {})
    records = calculator.comm.gather(payload, root=0)
    if calculator.rank == 0:
        slow = max(records, key=lambda item: item[1])
        fast = min(records, key=lambda item: item[1])
        avg = sum(item[1] for item in records) / len(records)
        extra_text = ''
        if extra:
            extra_parts = []
            for key in sorted(extra):
                values = [float(item[2].get(key, 0.0)) for item in records]
                unit = '' if key.endswith('_size') else 's'
                extra_parts.append(
                    f'{key}_min={min(values):.6f}{unit} '
                    f'{key}_avg={sum(values) / len(values):.6f}{unit} '
                    f'{key}_max={max(values):.6f}{unit}')
            extra_text = ' ' + ' '.join(extra_parts)
        print(f'TRACE {label}: min={fast[1]:.6f}s avg={avg:.6f}s '
              f'max={slow[1]:.6f}s slow_rank={slow[0]}{extra_text}',
              flush=True)


def scattering_rate_sparse_none_batched(calculator, k_grid, q_grid,
                                        band_indice, band_edge_energy,
                                        mob_level='erta', is_metal=False,
                                        is_rmp=False):
    """Calculate sparse mobility rates with bounded q/k tiles.

    The q phase table is built at most ``_Q_PHASE_BLOCK_SIZE`` q points at a
    time.  ``hash-cyclic`` ownership has already been resolved by
    ``get_q_rank_indices`` without a global argsort.
    """
    nbands = len(band_indice)
    nks = len(k_grid)
    nbranches = calculator._get_nbranches()
    mode_resolved = bool(calculator.mode_resolved)
    is_ibte = mob_level == 'ibte'
    use_lr_batch = (
        calculator.cal_mode == 'conductivity'
        and not is_metal
        and calculator.spinor_mode == 'none'
        and getattr(calculator, 'magnetic_type', None) == 'none'
        and not calculator.soc_switch
        and calculator.read_sparse_grad_mat
        and getattr(calculator, 'LRC_taylor_order', None) == 0)
    rate_all = np.full((nbands, nks), np.inf)
    rate_resolved = (np.full((nbands, nks, nbranches), np.inf)
                     if mode_resolved else None)
    trace = os.environ.get(_TRACE_ENV) == '1'
    trace_t0 = getattr(calculator, '_mobility_trace_t0', None)
    (threshold, k_grid_all, eigen, wave_local, vnks, q_local, weight_local,
     freq_grid, phon_vecs, apply_correction) = calculator._transport_prepare(
         k_grid, q_grid, band_indice, band_edge_energy, is_metal)
    q_indices = getattr(calculator, '_mobility_q_global_indices',
                        np.arange(len(q_local)))
    table = getattr(calculator, '_mobility_kpq_energy_ibz', None)
    grid2ir = getattr(calculator, '_mobility_grid2ir_idx', None)
    if table is not None and (table.ndim != 2 or table.shape[1] != nbands or
                              grid2ir is None):
        raise RuntimeError('Sparse mobility k+q lookup has inconsistent shape.')

    if calculator.rank == 0:
        logger = time_logger(
            total_cycles=max(1, (len(q_local) + _Q_PHASE_BLOCK_SIZE - 1) //
                             _Q_PHASE_BLOCK_SIZE),
            routine_name='scattering_rate_cal_epc_sparse')
    k_global, eig_global, wave_global = _collect_useful_k_wave_table(
        k_grid_all, eigen, wave_local, calculator.comm,
        chunk_bytes=_MPI_BUDGET)
    if len(k_global) != nks or len(eig_global) != nks:
        raise RuntimeError('Collected useful-k table has an inconsistent size.')
    batch_size = get_k_batch_size(calculator)
    initial_mask = np.asarray(
        _inside_window(eig_global, threshold, is_metal,
                       calculator.is_hole), dtype=bool)
    for global_k in range(nks):
        rate_all[initial_mask[global_k], global_k] = 0.0
        if mode_resolved:
            rate_resolved[initial_mask[global_k], global_k, :] = 0.0
    phase_k_all = np.ascontiguousarray(np.exp(
        QTcts.JTWOPI *
        (k_global @ calculator.nbr_shift_of_cell_sc.T)))
    mass = np.ascontiguousarray(calculator.atomic_mass, dtype=np.float64)
    channels = []
    cache_t0 = MPI.Wtime() if trace else None
    cache = prepare_sparse_grad_cache(calculator.grad_mat)
    cache_seconds = MPI.Wtime() - cache_t0 if trace else 0.0
    if trace and trace_t0 is not None:
        _trace_startup_record(
            calculator, 'startup_ready_before_q_block',
            MPI.Wtime() - trace_t0,
            {'cache_s': cache_seconds,
             'qsplit_s': getattr(calculator, '_mobility_trace_qsplit_s', -1.0),
             'phonon_s': getattr(calculator, '_mobility_trace_phonon_s', -1.0)})
    nq = len(q_local)
    qblocks = max(1, (nq + _Q_PHASE_BLOCK_SIZE - 1) // _Q_PHASE_BLOCK_SIZE)
    phase_seconds_total = 0.0
    first_progress_reported = False
    try:
        for qblock in range(qblocks):
            qstart = qblock * _Q_PHASE_BLOCK_SIZE
            qstop = min(qstart + _Q_PHASE_BLOCK_SIZE, nq)
            qblock_points = np.ascontiguousarray(q_local[qstart:qstop])
            phase_t0 = MPI.Wtime() if trace else None
            q_phases = np.ascontiguousarray(np.exp(
                QTcts.JTWOPI *
                (calculator.nbr_shift_of_cell_sc @ qblock_points.T)).T)
            atomic_phases = np.ascontiguousarray(np.exp(
                QTcts.JTWOPI *
                (calculator.graph_data.pos @ qblock_points.T)).T)
            phase_seconds_total += MPI.Wtime() - phase_t0 if trace else 0.0
            for start in range(0, nks, batch_size):
                stop = min(start + batch_size, nks)
                k_block = np.ascontiguousarray(k_global[start:stop])
                eig_k = np.ascontiguousarray(eig_global[start:stop])
                wave_k = np.ascontiguousarray(wave_global[start:stop])
                global_k = np.arange(start, stop, dtype=int)
                mask_k = _inside_window(eig_k, threshold, is_metal,
                                         calculator.is_hole)
                active = np.flatnonzero(mask_k.any(axis=1))
                if not active.size:
                    continue
                k_active = np.ascontiguousarray(k_block[active])
                eig_active = np.ascontiguousarray(eig_k[active])
                wave_active = np.ascontiguousarray(wave_k[active])
                mask_active = np.ascontiguousarray(mask_k[active])
                ik_active = global_k[active]
                phase_active = np.ascontiguousarray(phase_k_all[ik_active])
                for iq_block, q in enumerate(qblock_points):
                    iq = qstart + iq_block
                    freq = np.ascontiguousarray(freq_grid[iq], dtype=np.float64)
                    if not np.any(freq >= calculator.phonon_frequency_cutoff):
                        continue
                    kpq = np.ascontiguousarray(k_active + q)
                    bz_indices = calculator._get_kpq_BZ_index(kpq) if (
                        table is not None or is_ibte) else None
                    if table is not None:
                        eig_prefilter = np.ascontiguousarray(
                            table[grid2ir[bz_indices]])
                        selected = _candidate_indices(
                            calculator, eig_active, eig_prefilter, freq,
                            mask_active, threshold, is_metal)
                    else:
                        selected = np.arange(len(kpq), dtype=int)
                    if not selected.size:
                        continue
                    eig_raw, wave_raw = calculator._elec_cal_torch(kpq[selected])
                    eig_kpq = np.ascontiguousarray(eig_raw[band_indice, :].T)
                    wave_kpq = np.ascontiguousarray(wave_raw[:, band_indice, :])
                    del eig_raw, wave_raw
                    mask_kpq = _inside_window(eig_kpq, threshold, is_metal,
                                              calculator.is_hole)
                    matches = np.ascontiguousarray(np.stack([
                        calculator._get_match_table(
                            eig_active[index], eig_kpq[row], freq)
                        for row, index in enumerate(selected)]))
                    exact = mask_kpq.any(axis=1) & matches.any(axis=(1, 2, 3))
                    if not np.any(exact):
                        continue
                    selected = selected[exact]
                    eig_kpq = np.ascontiguousarray(eig_kpq[exact])
                    wave_kpq = np.ascontiguousarray(wave_kpq[exact])
                    mask_kpq = np.ascontiguousarray(mask_kpq[exact])
                    matches = np.ascontiguousarray(matches[exact])
                    eig_in = np.ascontiguousarray(eig_active[selected])
                    wave_in = np.ascontiguousarray(wave_active[selected])
                    mask_in = np.ascontiguousarray(mask_active[selected])
                    phase_in = np.ascontiguousarray(phase_active[selected])
                    phase_out = np.ascontiguousarray(
                        phase_in * q_phases[iq_block][None, :])
                    phvec = np.ascontiguousarray(
                        atomic_phases[iq_block][None, :, None] * phon_vecs[iq])
                    corr = np.zeros(
                        (len(selected), nbands, nbands, nbranches),
                        dtype=np.complex128)
                    if apply_correction[iq]:
                        if use_lr_batch:
                            corr = calculator._get_LR_correction_kq_batch(
                                q, freq, phvec, matches, mask_in,
                                k_active[selected], wave_in, wave_kpq)
                        else:
                            for row, index in enumerate(selected):
                                corr[row] = calculator._get_LR_correction_kq(
                                    q, freq, phvec, matches[row], mask_in[row],
                                    k_active[index], wave_in[row], wave_kpq[row],
                                    return_tmp1=False)
                    corr = np.ascontiguousarray(corr, dtype=np.complex128)
                    rate_batch = np.zeros((len(selected), nbands), dtype=np.float64)
                    resolved_batch = (
                        np.zeros((len(selected), nbands, nbranches), dtype=np.float64)
                        if mode_resolved else None)
                    trans_batch = (
                        np.zeros((len(selected), nbands, nbands), dtype=np.float64)
                        if is_ibte else None)
                    scattering_rate_cal_EPC_helper_sparse_batch(
                        freq, mass, phvec, phase_in, phase_out, cache,
                        wave_in, wave_kpq, mask_in, mask_kpq, matches,
                        eig_in, eig_kpq,
                        np.ascontiguousarray(bose_weight(
                            freq, calculator.temperature)),
                        np.ascontiguousarray(fermi_weight(
                            eig_kpq - calculator.efermi,
                            calculator.temperature)),
                        corr, rate_batch, resolved_batch, trans_batch,
                        np.ascontiguousarray(fermi_weight(
                            eig_in - calculator.efermi,
                            calculator.temperature)),
                        calculator.inv_smearq, float(weight_local[iq]),
                        int(is_rmp), int(is_ibte))
                    for row, index in enumerate(selected):
                        ik = int(ik_active[index])
                        rate_all[:, ik] += rate_batch[row]
                        if mode_resolved:
                            rate_resolved[:, ik, :] += resolved_batch[row]
                        if is_ibte:
                            fbz = int(bz_indices[selected[row]])
                            ibz = int(calculator.ir_indices[fbz])
                            for ib in np.flatnonzero(mask_in[row]):
                                for jb in np.flatnonzero(mask_kpq[row]):
                                    channels.append((
                                        ik, int(q_indices[iq]), ibz, fbz,
                                        int(ib), int(jb),
                                        float(trans_batch[row, ib, jb])))
            if trace and not first_progress_reported:
                _trace_startup_record(
                    calculator, 'first_progress_global',
                    MPI.Wtime() - trace_t0,
                    {'q_block_size': _Q_PHASE_BLOCK_SIZE,
                     'phase_s': phase_seconds_total})
                first_progress_reported = True
            if calculator.rank == 0:
                logger.step(qblock + 1)
    finally:
        del cache

    rate_all *= QTcts.TWOPI
    allreduce_array_in_place_chunks(rate_all, calculator.comm)
    if mode_resolved:
        allreduce_array_in_place_chunks(rate_resolved, calculator.comm)
    if calculator.comm is not None:
        calculator.comm.Barrier()
    if calculator.rank == 0 and mode_resolved:
        np.save('rate_mode_resolved.npy', rate_resolved)
    rate_all[np.abs(rate_all) < QTcts.TENPM160] = np.inf
    if is_ibte:
        channels.sort(key=lambda item: (item[0], item[1], item[4], item[5]))
        records = [(ik, ibz, fbz, ib, jb, probability)
                   for ik, _, ibz, fbz, ib, jb, probability in channels]
        return rate_all, eigen, vnks, np.array(records, dtype=_CHANNEL_DTYPE)
    return rate_all, eigen, vnks
