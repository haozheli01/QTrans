"""Fixed-chemical-potential full-bandwidth Eliashberg solver."""

import numpy as np

from .convergence import dominant_real_eigenvalue, mixed, scaled_residual
from .matsubara import positive_matsubara_frequencies, search_transition_temperature
from .phonon_kernel import TransitionPhononKernel
from .state import NonlinearState


class _FullBandwidthSolverBase:
    """Shared imaginary-axis FBW equations with chi fixed to zero."""

    solver_name = "full-bandwidth"

    def __init__(self, problem, settings):
        self.problem = problem
        self.settings = settings
        self.kernel = TransitionPhononKernel(problem.transitions)

    def _frequencies(self, temperature):
        return positive_matsubara_frequencies(
            temperature,
            self.settings.matsubara_cutoff,
            self.settings.max_matsubara_points,
        )

    def _normal_z(self, temperature, frequencies):
        number_of_states = self.problem.transitions.number_of_states
        shape = (number_of_states, frequencies.size)
        z = np.ones(shape, dtype=float)
        frequency_grid = frequencies[None, :]
        energy_grid = self.problem.energies[:, None]
        for _ in range(self.settings.max_iterations):
            denominator = np.sqrt((frequency_grid * z) ** 2 + energy_grid ** 2)
            amplitude = frequency_grid * z / denominator
            z_new = 1.0 + (
                np.pi * temperature / frequency_grid
                * self.kernel.apply(amplitude, temperature, "difference")
            )
            if np.any(z_new <= 0.0) or not np.all(np.isfinite(z_new)):
                raise RuntimeError(
                    "The {} normal-state Z is not positive and finite."
                    .format(self.solver_name)
                )
            z_mixed = mixed(z, z_new, self.settings.mixing)
            residual = scaled_residual((z,), (z_mixed,), 1.0)
            z = z_mixed
            if residual <= self.settings.convergence_tol:
                return z
        raise RuntimeError(
            "The {} normal-state Z did not converge in {} iterations."
            .format(self.solver_name, self.settings.max_iterations)
        )

    def _coulomb_term(self, amplitude, frequencies, mu_star):
        mask = frequencies <= self.settings.coulomb_cutoff
        weighted = self.problem.state_weights[:, None] * amplitude[:, mask]
        return 2.0 * mu_star * np.sum(weighted)

    def leading_eigenvalue(self, temperature, mu_star):
        frequencies = self._frequencies(temperature)
        z = self._normal_z(temperature, frequencies)
        shape = z.shape
        denominator = np.sqrt(
            (frequencies[None, :] * z) ** 2
            + self.problem.energies[:, None] ** 2
        )

        def matvec(vector):
            amplitude = np.asarray(vector, dtype=float).reshape(shape) / denominator
            phonon = self.kernel.apply(amplitude, temperature, "sum")
            coulomb = self._coulomb_term(amplitude, frequencies, mu_star)
            return (np.pi * temperature * (phonon - coulomb)).ravel()

        return dominant_real_eigenvalue(
            matvec,
            z.size,
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
        number_of_states = self.problem.transitions.number_of_states
        shape = (number_of_states, frequencies.size)
        if initial_state is not None and initial_state.z.shape == shape:
            z = np.array(initial_state.z, dtype=float, copy=True)
            phi = np.array(initial_state.phi, dtype=float, copy=True)
        else:
            z = self._normal_z(temperature, frequencies)
            phi = z * self.settings.initial_gap

        frequency_grid = frequencies[None, :]
        energy_grid = self.problem.energies[:, None]
        residual = np.inf
        converged = False
        iterations = 0
        for iterations in range(1, self.settings.max_iterations + 1):
            denominator = np.sqrt(
                (frequency_grid * z) ** 2 + energy_grid ** 2 + phi ** 2
            )
            normal_amplitude = frequency_grid * z / denominator
            pair_amplitude = phi / denominator
            z_new = 1.0 + (
                np.pi * temperature / frequency_grid
                * self.kernel.apply(normal_amplitude, temperature, "difference")
            )
            coulomb = self._coulomb_term(
                pair_amplitude, frequencies, mu_star
            )
            phi_new = np.pi * temperature * (
                self.kernel.apply(pair_amplitude, temperature, "sum") - coulomb
            )
            if np.any(z_new <= 0.0) or not (
                np.all(np.isfinite(z_new)) and np.all(np.isfinite(phi_new))
            ):
                raise RuntimeError(
                    "The {} nonlinear iteration produced invalid values."
                    .format(self.solver_name)
                )
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


class IsotropicFBWSolver(_FullBandwidthSolverBase):
    """Energy-bin averaged full-bandwidth solver with fixed mu and chi=0."""

    solver_name = "isotropic FBW"
