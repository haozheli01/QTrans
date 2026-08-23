"""Text and NPZ output for Eliashberg solver results."""

import os

import numpy as np

from qtrans.core.constants import QTcts


def _validate_index(index):
    if (
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
    ):
        raise ValueError("result index must be a non-negative integer.")
    return int(index)


def _gap_zero_frequency(result):
    gap = np.asarray(result.gap)
    if gap.ndim == 1:
        return float(gap[0])
    state_weights = result.metadata.get("weights")
    if state_weights is None:
        return float(np.mean(gap[:, 0]))
    return float(np.sum(np.asarray(state_weights) * gap[:, 0]))


def write_summary(output_directory, results):
    path = os.path.join(output_directory, "eliashberg_summary.dat")
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# approximation mu_star Tc_K tc_status solution_T_K "
            "gap_iw0_meV converged iterations residual\n"
        )
        for result in results:
            tc_kelvin = (
                result.tc * QTcts.HARTREEtoKELVIN
                if np.isfinite(result.tc) else np.nan
            )
            stream.write(
                "{} {:.8f} {:.10e} {} {:.10e} {:.10e} {} {} {:.10e}\n"
                .format(
                    result.approximation.value,
                    result.mu_star,
                    tc_kelvin,
                    result.tc_status,
                    result.solution_temperature * QTcts.HARTREEtoKELVIN,
                    _gap_zero_frequency(result) * QTcts.HARTREEtoMEV,
                    int(result.converged),
                    result.iterations,
                    result.residual,
                )
            )
    return path


def write_result(output_directory, result, save_matsubara=True, index=0):
    index = _validate_index(index)
    filename = "eliashberg_result_{}_mu_{}.npz".format(
        result.approximation.value, "{:04d}".format(index)
    )
    path = os.path.join(output_directory, filename)
    payload = {
        "approximation": np.asarray(result.approximation.value),
        "mu_star": np.asarray(result.mu_star, dtype=float),
        "tc": np.asarray(result.tc, dtype=float),
        "tc_status": np.asarray(result.tc_status),
        "solution_temperature": np.asarray(result.solution_temperature, dtype=float),
        "converged": np.asarray(result.converged, dtype=bool),
        "iterations": np.asarray(result.iterations, dtype=np.int64),
        "residual": np.asarray(result.residual, dtype=float),
        "tc_temperatures": np.asarray(result.tc_temperatures, dtype=float),
        "tc_eigenvalues": np.asarray(result.tc_eigenvalues, dtype=float),
    }
    if save_matsubara:
        payload.update({
            "matsubara_frequencies": np.asarray(
                result.matsubara_frequencies, dtype=float
            ),
            "z": np.asarray(result.z, dtype=float),
            "phi": np.asarray(result.phi, dtype=float),
            "gap": np.asarray(result.gap, dtype=float),
        })
        if result.chi is not None:
            payload["chi"] = np.asarray(result.chi, dtype=float)
    for name, value in result.metadata.items():
        if isinstance(value, np.ndarray):
            output_name = name if name.startswith("state_") else "state_{}".format(name)
            payload[output_name] = value
    np.savez_compressed(path, **payload)
    return path


def write_results(output_directory, results, save_matsubara=True):
    paths = [write_summary(output_directory, results)]
    paths.extend(
        write_result(output_directory, result, save_matsubara, index=index)
        for index, result in enumerate(results)
    )
    return paths
