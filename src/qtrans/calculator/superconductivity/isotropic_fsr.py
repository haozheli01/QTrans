"""Isotropic Fermi-surface-restricted Eliashberg solver."""

import numpy as np

from .convergence import dominant_real_eigenvalue, mixed, scaled_residual
from .matsubara import positive_matsubara_frequencies, search_transition_temperature
from .phonon_kernel import IsotropicPhononKernel
from .state import NonlinearState


class IsotropicFSRSolver:
    """Solve scalar singlet Eliashberg equations on the imaginary axis."""

    def __init__(self, problem, settings):
        self.problem = problem
        self.settings = settings
        self.kernel = IsotropicPhononKernel(
            problem.phonon_frequencies, problem.alpha2f
        )

    def _frequencies(self, temperature):
        return positive_matsubara_frequencies(
            temperature,
            self.settings.matsubara_cutoff,
            self.settings.max_matsubara_points,
        )

    def _normal_z(self, temperature, frequencies):
        values = np.ones_like(frequencies)
        correction = self.kernel.apply(values, temperature, "difference")
        z = 1.0 + np.pi * temperature * correction / frequencies
        if np.any(z <= 0.0) or not np.all(np.isfinite(z)):
            raise RuntimeError("The isotropic FSR normal-state Z is not positive and finite.")
        return z

    def leading_eigenvalue(self, temperature, mu_star):
        frequencies = self._frequencies(temperature)
        z = self._normal_z(temperature, frequencies)
        denominator = frequencies * z
        coulomb_mask = frequencies <= self.settings.coulomb_cutoff

        def matvec(vector):
            pair_amplitude = np.asarray(vector, dtype=float) / denominator
            phonon = self.kernel.apply(pair_amplitude, temperature, "sum")
            coulomb = 2.0 * mu_star * np.sum(pair_amplitude[coulomb_mask])
            return np.pi * temperature * (phonon - coulomb)

        return dominant_real_eigenvalue(
            matvec,
            frequencies.size,
            self.settings.eigen_tolerance,
            self.settings.eigen_max_iterations,
        )

    def solve_linearized_tc(self, mu_star):
        return search_transition_temperature(
            lambda temperature: self.leading_eigenvalue(temperature, mu_star),
            self.settings.temperature_min,
            self.settings.temperature_max,
            self.settings.temperature_step,
            self.settings.tc_tolerance,
        )

    def solve_nonlinear_gap(self, temperature, mu_star, initial_state=None):
        frequencies = self._frequencies(temperature)
        shape = frequencies.shape
        if initial_state is not None and initial_state.z.shape == shape:
            z = np.array(initial_state.z, dtype=float, copy=True)
            phi = np.array(initial_state.phi, dtype=float, copy=True)
        else:
            z = self._normal_z(temperature, frequencies)
            phi = z * self.settings.initial_gap

        coulomb_mask = frequencies <= self.settings.coulomb_cutoff
        residual = np.inf
        converged = False
        iterations = 0
        for iterations in range(1, self.settings.max_iterations + 1):
            denominator = np.sqrt((frequencies * z) ** 2 + phi ** 2)
            normal_amplitude = frequencies * z / denominator
            pair_amplitude = phi / denominator

            z_new = 1.0 + (
                np.pi * temperature / frequencies
                * self.kernel.apply(normal_amplitude, temperature, "difference")
            )
            coulomb = 2.0 * mu_star * np.sum(pair_amplitude[coulomb_mask])
            phi_new = np.pi * temperature * (
                self.kernel.apply(pair_amplitude, temperature, "sum") - coulomb
            )
            if np.any(z_new <= 0.0) or not (
                np.all(np.isfinite(z_new)) and np.all(np.isfinite(phi_new))
            ):
                raise RuntimeError("The isotropic FSR iteration produced invalid values.")

            z_mixed = mixed(z, z_new, self.settings.mixing)
            phi_mixed = mixed(phi, phi_new, self.settings.mixing)
            residual = scaled_residual(
                (z, phi),
                (z_mixed, phi_mixed),
                self.settings.initial_gap,
            )
            z, phi = z_mixed, phi_mixed
            if residual <= self.settings.convergence_tol:
                converged = True
                break

        return (
            frequencies,
            NonlinearState(z=z, phi=phi, chi=np.zeros_like(z)),
            converged,
            iterations,
            residual,
        )
