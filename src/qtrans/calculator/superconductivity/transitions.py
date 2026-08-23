"""EPC data-provider boundary for Eliashberg solvers."""

import hashlib
import os

import numpy as np

from .enums import EliashbergApproximation
from .state import EPCTransitionTable, IsotropicFSRProblem, StateResolvedProblem


_TRANSITION_KEYS = (
    "source_state",
    "target_state",
    "phonon_frequency",
    "coupling_weight",
)


def _positive_alpha2f_samples(phonon_frequencies, alpha2f):
    frequencies = np.asarray(phonon_frequencies, dtype=float)
    spectrum = np.asarray(alpha2f, dtype=float)
    mask = np.isfinite(frequencies) & np.isfinite(spectrum) & (frequencies > 0.0)
    frequencies = frequencies[mask]
    spectrum = spectrum[mask]
    if frequencies.size < 2:
        raise ValueError("The QTrans alpha^2 F grid has fewer than two positive samples.")
    order = np.argsort(frequencies)
    return frequencies[order], spectrum[order]


class EPCStateProvider:
    """Build solver problems without exposing calculator internals to solvers."""

    def isotropic_fsr(self, phonon_frequencies, alpha2f):
        frequencies, spectrum = _positive_alpha2f_samples(
            phonon_frequencies, alpha2f
        )
        return IsotropicFSRProblem(frequencies, spectrum)

    def state_resolved(self, path, approximation):
        return load_state_resolved_problem(path, approximation)


def load_state_resolved_problem(path, approximation):
    """Load a normalized state-resolved problem from a non-pickled NPZ file.

    Required arrays are ``state_weights``, ``source_state``, ``target_state``,
    ``phonon_frequency``, and ``coupling_weight``. FBW files additionally need
    ``energies`` in Hartree relative to the fixed normal-state Fermi level.
    """
    approximation = EliashbergApproximation.parse(approximation)
    if approximation == EliashbergApproximation.ISOTROPIC_FSR:
        raise ValueError("iso_fsr uses the QTrans alpha^2 F provider, not an NPZ table.")
    path = os.path.abspath(os.fspath(path))
    with np.load(path, allow_pickle=False) as data:
        required = ("state_weights",) + _TRANSITION_KEYS
        missing = [name for name in required if name not in data]
        if approximation.is_fbw and "energies" not in data:
            missing.append("energies")
        if missing:
            raise ValueError(
                "Eliashberg data file {!r} is missing arrays: {}."
                .format(path, ", ".join(missing))
            )
        state_weights = np.asarray(data["state_weights"], dtype=float)
        transitions = EPCTransitionTable(
            number_of_states=state_weights.size,
            source_state=data["source_state"],
            target_state=data["target_state"],
            phonon_frequency=data["phonon_frequency"],
            coupling_weight=data["coupling_weight"],
        )
        energies = np.asarray(data["energies"], dtype=float) \
            if "energies" in data else None
        standard = set(required) | {"energies", "approximation"}
        metadata = {
            name: np.asarray(data[name])
            for name in data.files
            if name not in standard and name.startswith("state_")
        }
        if "approximation" in data:
            stored = str(np.asarray(data["approximation"]).item())
            if EliashbergApproximation.parse(stored) != approximation:
                raise ValueError(
                    "Eliashberg data file approximation {!r} does not match {!r}."
                    .format(stored, approximation.value)
                )
    return StateResolvedProblem(
        approximation=approximation,
        state_weights=state_weights,
        transitions=transitions,
        energies=energies,
        state_metadata=metadata,
    )


def problem_fingerprint(problem):
    """Return a deterministic checksum used to validate nonlinear restarts."""
    digest = hashlib.sha256()

    def update(array):
        array = np.ascontiguousarray(np.asarray(array))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())

    if isinstance(problem, IsotropicFSRProblem):
        update(problem.phonon_frequencies)
        update(problem.alpha2f)
    else:
        update(problem.state_weights)
        if problem.energies is not None:
            update(problem.energies)
        update(problem.transitions.source_state)
        update(problem.transitions.target_state)
        update(problem.transitions.phonon_frequency)
        update(problem.transitions.coupling_weight)
    return digest.hexdigest()
