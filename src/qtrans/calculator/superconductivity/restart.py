"""Strict nonlinear-state restart files for Eliashberg calculations."""

import os

import numpy as np

from qtrans._version import __version__

from .state import NonlinearState


def _validate_index(index):
    if (
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
    ):
        raise ValueError("restart index must be a non-negative integer.")
    return int(index)


def _scalar_value(data, key):
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.ndim != 0:
        return None
    return value.item()


def restart_path(output_directory, approximation, index):
    index = _validate_index(index)
    filename = "eliashberg_restart_{}_mu_{}.npz".format(
        approximation.value, "{:04d}".format(index)
    )
    return os.path.join(output_directory, filename)


def load_restart(path, approximation, mu_star, temperature, fingerprint):
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        checks = {
            "approximation": approximation.value,
            "qtrans_version": __version__,
            "problem_fingerprint": fingerprint,
        }
        for key, expected in checks.items():
            if str(_scalar_value(data, key)) != str(expected):
                return None
        converged = _scalar_value(data, "converged")
        if not isinstance(converged, (bool, np.bool_)) or not converged:
            return None
        stored_mu_star = _scalar_value(data, "mu_star")
        stored_temperature = _scalar_value(data, "temperature")
        if stored_mu_star is None or stored_temperature is None:
            return None
        try:
            stored_mu_star = float(stored_mu_star)
            stored_temperature = float(stored_temperature)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(stored_mu_star) or not np.isfinite(stored_temperature):
            return None
        if not np.isclose(stored_mu_star, mu_star, rtol=0.0, atol=1.0e-14):
            return None
        if not np.isclose(
            stored_temperature, temperature, rtol=0.0, atol=1.0e-14
        ):
            return None
        if "z" not in data or "phi" not in data:
            return None
        z = np.asarray(data["z"], dtype=float)
        phi = np.asarray(data["phi"], dtype=float)
        chi = np.asarray(data["chi"], dtype=float) if "chi" in data else None
        if z.shape != phi.shape or not (
            np.all(np.isfinite(z)) and np.all(np.isfinite(phi))
        ):
            return None
        if chi is not None and (chi.shape != z.shape or not np.all(np.isfinite(chi))):
            return None
    return NonlinearState(z=z, phi=phi, chi=chi)


def save_restart(
    path, approximation, mu_star, temperature, fingerprint, nonlinear_state,
    converged=False,
):
    if not isinstance(converged, (bool, np.bool_)):
        raise ValueError("converged must be a boolean.")
    payload = {
        "approximation": np.asarray(approximation.value),
        "qtrans_version": np.asarray(__version__),
        "problem_fingerprint": np.asarray(fingerprint),
        "mu_star": np.asarray(mu_star, dtype=float),
        "temperature": np.asarray(temperature, dtype=float),
        "converged": np.asarray(converged, dtype=bool),
        "z": np.asarray(nonlinear_state.z, dtype=float),
        "phi": np.asarray(nonlinear_state.phi, dtype=float),
    }
    if nonlinear_state.chi is not None:
        payload["chi"] = np.asarray(nonlinear_state.chi, dtype=float)
    np.savez_compressed(path, **payload)
