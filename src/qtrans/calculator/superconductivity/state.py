"""Validated data contracts shared by the Eliashberg solvers."""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

from .enums import EliashbergApproximation


def _float_array(value, name, ndim=None):
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError("{} must have {} dimensions, got shape {}.".format(
            name, ndim, array.shape
        ))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or infinite values.".format(name))
    return np.ascontiguousarray(array)


def _integer_array(value, name, ndim=None):
    """Convert an integer-valued array without silently truncating floats."""
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError("{} must have {} dimensions, got shape {}.".format(
            name, ndim, array.shape
        ))
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("{} must contain integer values, not booleans.".format(name))
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("{} must contain integer values.".format(name))
    if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)):
        raise ValueError("{} must contain finite integer values.".format(name))
    return np.ascontiguousarray(array, dtype=np.int64)


@dataclass(frozen=True)
class EliashbergSettings:
    """Internal solver settings; all temperatures and energies use Hartree."""

    approximation: EliashbergApproximation
    temperature_min: float
    temperature_max: float
    temperature_step: float
    matsubara_cutoff: float
    coulomb_cutoff: float
    max_iterations: int = 500
    eigen_max_iterations: int = 300
    max_matsubara_points: int = 20000
    convergence_tol: float = 1.0e-8
    eigen_tolerance: float = 1.0e-7
    tc_tolerance: float = 0.1
    mixing: float = 0.3
    initial_gap: float = 1.0e-4
    save_matsubara: bool = True
    restart: bool = True

    def __post_init__(self):
        approximation = EliashbergApproximation.parse(self.approximation)
        object.__setattr__(self, "approximation", approximation)
        positive = {
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "temperature_step": self.temperature_step,
            "matsubara_cutoff": self.matsubara_cutoff,
            "coulomb_cutoff": self.coulomb_cutoff,
            "convergence_tol": self.convergence_tol,
            "eigen_tolerance": self.eigen_tolerance,
            "tc_tolerance": self.tc_tolerance,
            "initial_gap": self.initial_gap,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be positive, got {!r}.".format(name, value))
        if self.temperature_max <= self.temperature_min:
            raise ValueError("temperature_max must be larger than temperature_min.")
        if not 0.0 < self.mixing <= 1.0:
            raise ValueError("mixing must be in the interval (0, 1].")
        for name in ("max_iterations", "eigen_max_iterations", "max_matsubara_points"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError("{} must be a positive integer.".format(name))
        for name in ("save_matsubara", "restart"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError("{} must be a boolean.".format(name))


@dataclass(frozen=True)
class IsotropicFSRProblem:
    """Sampled isotropic alpha^2 F(omega) on a positive phonon grid."""

    phonon_frequencies: np.ndarray
    alpha2f: np.ndarray

    def __post_init__(self):
        frequencies = _float_array(self.phonon_frequencies, "phonon_frequencies", 1)
        alpha2f = _float_array(self.alpha2f, "alpha2f", 1)
        if frequencies.shape != alpha2f.shape:
            raise ValueError("phonon_frequencies and alpha2f must have the same shape.")
        if frequencies.size < 2:
            raise ValueError("At least two alpha^2 F samples are required.")
        if np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("phonon_frequencies must be strictly increasing.")
        if np.any(frequencies <= 0.0):
            raise ValueError("phonon_frequencies must be positive.")
        alpha_scale = max(float(np.max(np.abs(alpha2f))), 1.0)
        if np.any(alpha2f < -1.0e-12 * alpha_scale):
            raise ValueError("alpha2f must be non-negative within numerical tolerance.")
        object.__setattr__(self, "phonon_frequencies", frequencies)
        object.__setattr__(self, "alpha2f", np.maximum(alpha2f, 0.0))


@dataclass(frozen=True)
class EPCTransitionTable:
    """Sparse, directed state-to-state phonon spectral weights.

    For FSR, ``coupling_weight[e]`` is the integrated alpha^2 F weight
    (Hartree), including Fermi-surface quadrature weights. For FBW it is
    the weighted spectral coefficient C_e (Hartree**2): w_j * |g_e|**2
    for discrete states, or u_j times the integrated spectral weight for
    energy bins, where u_j = w_j / N_F. Electronic/q-point weights are
    already included; solvers must not multiply them in again. Neither
    convention includes a pi prefactor.
    """

    number_of_states: int
    source_state: np.ndarray
    target_state: np.ndarray
    phonon_frequency: np.ndarray
    coupling_weight: np.ndarray

    def __post_init__(self):
        if (
            isinstance(self.number_of_states, bool)
            or not isinstance(self.number_of_states, (int, np.integer))
            or self.number_of_states <= 0
        ):
            raise ValueError("number_of_states must be a positive integer.")
        source = _integer_array(self.source_state, "source_state", 1)
        target = _integer_array(self.target_state, "target_state", 1)
        frequency = _float_array(self.phonon_frequency, "phonon_frequency", 1)
        coupling = _float_array(self.coupling_weight, "coupling_weight", 1)
        size = source.size
        if size == 0 or any(array.size != size for array in (target, frequency, coupling)):
            raise ValueError("All transition arrays must have the same non-zero length.")
        if np.any(source < 0) or np.any(source >= self.number_of_states):
            raise ValueError("source_state contains an out-of-range state index.")
        if np.any(target < 0) or np.any(target >= self.number_of_states):
            raise ValueError("target_state contains an out-of-range state index.")
        if np.any(frequency <= 0.0):
            raise ValueError("Transition phonon frequencies must be positive.")
        if np.any(coupling < 0.0):
            raise ValueError("Transition coupling weights must be non-negative.")
        object.__setattr__(self, "source_state", source)
        object.__setattr__(self, "target_state", target)
        object.__setattr__(self, "phonon_frequency", frequency)
        object.__setattr__(self, "coupling_weight", coupling)


@dataclass(frozen=True)
class StateResolvedProblem:
    """State-resolved FSR or fixed-chemical-potential FBW problem.

    FBW requires unnormalized ``fbw_energy_weights`` u_j = w_j / N_F
    in Hartree for the Coulomb energy sum. ``state_weights`` are normalized
    averaging weights and are used only for reporting in FBW.
    """

    approximation: EliashbergApproximation
    state_weights: np.ndarray
    transitions: EPCTransitionTable
    energies: Optional[np.ndarray] = None
    state_metadata: Mapping[str, np.ndarray] = field(default_factory=dict)
    fbw_energy_weights: Optional[np.ndarray] = None

    def __post_init__(self):
        approximation = EliashbergApproximation.parse(self.approximation)
        if approximation == EliashbergApproximation.ISOTROPIC_FSR:
            raise ValueError("Use IsotropicFSRProblem for the iso_fsr approximation.")
        weights = _float_array(self.state_weights, "state_weights", 1)
        if weights.size != self.transitions.number_of_states:
            raise ValueError("state_weights length does not match number_of_states.")
        if np.any(weights < 0.0) or np.sum(weights) <= 0.0:
            raise ValueError("state_weights must be non-negative and have a positive sum.")
        weights = np.ascontiguousarray(weights / np.sum(weights))
        energies = self.energies
        fbw_energy_weights = None
        if approximation.is_fbw:
            if energies is None:
                raise ValueError("FBW problems require state energies relative to E_F.")
            energies = _float_array(energies, "energies", 1)
            if energies.size != weights.size:
                raise ValueError("energies length does not match number_of_states.")
            if self.fbw_energy_weights is None:
                raise ValueError(
                    "FBW problems require fbw_energy_weights = w_j / N_F in "
                    "Hartree; these cannot be inferred from normalized "
                    "state_weights. FBW coupling_weight must use Hartree**2."
                )
            fbw_energy_weights = _float_array(
                self.fbw_energy_weights, "fbw_energy_weights", 1
            )
            if fbw_energy_weights.size != weights.size:
                raise ValueError(
                    "fbw_energy_weights length does not match number_of_states."
                )
            if np.any(fbw_energy_weights < 0.0) or not np.any(fbw_energy_weights > 0.0):
                raise ValueError(
                    "fbw_energy_weights must be non-negative and have a positive sum."
                )
        elif energies is not None:
            energies = _float_array(energies, "energies", 1)
        if not isinstance(self.state_metadata, Mapping):
            raise TypeError("state_metadata must be a mapping of names to arrays.")
        metadata = {}
        for name, value in self.state_metadata.items():
            if not isinstance(name, str):
                raise TypeError("state_metadata keys must be strings.")
            array = np.asarray(value)
            if array.dtype.kind == "O":
                raise TypeError("state_metadata values must be typed NumPy-compatible arrays.")
            metadata[name] = np.ascontiguousarray(array)
        object.__setattr__(self, "approximation", approximation)
        object.__setattr__(self, "state_weights", weights)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "fbw_energy_weights", fbw_energy_weights)
        object.__setattr__(self, "state_metadata", metadata)


@dataclass
class NonlinearState:
    z: np.ndarray
    phi: np.ndarray
    chi: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TcSearchResult:
    tc: float
    status: str
    temperatures: np.ndarray
    eigenvalues: np.ndarray


@dataclass
class EliashbergResult:
    approximation: EliashbergApproximation
    mu_star: float
    tc: float
    tc_status: str
    solution_temperature: float
    matsubara_frequencies: np.ndarray
    z: np.ndarray
    phi: np.ndarray
    gap: np.ndarray
    chi: Optional[np.ndarray]
    converged: bool
    iterations: int
    residual: float
    tc_temperatures: np.ndarray
    tc_eigenvalues: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)
