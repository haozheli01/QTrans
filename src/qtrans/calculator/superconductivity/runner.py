"""QTrans-facing orchestration for the four Eliashberg approximations."""

import traceback

import numpy as np

from qtrans.core.constants import QTcts

from .anisotropic_fbw import AnisotropicFBWSolver
from .anisotropic_fsr import AnisotropicFSRSolver
from .enums import EliashbergApproximation
from .isotropic_fbw import IsotropicFBWSolver
from .isotropic_fsr import IsotropicFSRSolver
from .outputs import write_results
from .restart import load_restart, restart_path, save_restart
from .state import EliashbergResult, EliashbergSettings
from .transitions import EPCStateProvider, problem_fingerprint


_SOLVERS = {
    EliashbergApproximation.ISOTROPIC_FSR: IsotropicFSRSolver,
    EliashbergApproximation.ANISOTROPIC_FSR: AnisotropicFSRSolver,
    EliashbergApproximation.ISOTROPIC_FBW: IsotropicFBWSolver,
    EliashbergApproximation.ANISOTROPIC_FBW: AnisotropicFBWSolver,
}


class EliashbergRunner:
    """Translate QTrans calculator state into standalone solver problems."""

    def __init__(self, calculator):
        self.calculator = calculator
        self.settings = EliashbergSettings(
            approximation=calculator.eliashberg_approximation,
            temperature_min=calculator.temperature_range[0],
            temperature_max=calculator.temperature_range[1],
            temperature_step=calculator.temperature_range[2],
            matsubara_cutoff=calculator.matsubara_cutoff,
            coulomb_cutoff=calculator.coulomb_cutoff,
            max_iterations=calculator.max_iterations,
            eigen_max_iterations=calculator.eigen_max_iterations,
            max_matsubara_points=calculator.max_matsubara_points,
            convergence_tol=calculator.convergence_tol,
            eigen_tolerance=calculator.eigen_tolerance,
            tc_tolerance=calculator.tc_tolerance,
            mixing=calculator.mixing,
            initial_gap=calculator.initial_gap,
            save_matsubara=calculator.save_matsubara,
            restart=calculator.restart,
        )
        self.provider = EPCStateProvider()

    def _check_spin_support(self):
        if (
            self.calculator.soc_switch
            or self.calculator.spinor_mode in ("nc", "collinear")
        ):
            if self.calculator.rank == 0:
                print(
                    "Eliashberg scalar-singlet solving is unavailable for SOC, "
                    "non-collinear, or collinear magnetic calculations."
                )
            raise NotImplementedError(
                "Eliashberg equation solving currently supports only non-magnetic, "
                "non-SOC calculations."
            )

    def _build_problem(self, phonon_frequencies=None, alpha2f=None):
        approximation = self.settings.approximation
        if approximation == EliashbergApproximation.ISOTROPIC_FSR:
            return self.provider.isotropic_fsr(phonon_frequencies, alpha2f)
        if self.calculator.eliashberg_data_path is None:
            raise NotImplementedError(
                "{} requires a normalized state-resolved NPZ file through "
                "eliashberg_data_path; automatic QTrans EPC state extraction "
                "is not implemented yet."
                .format(approximation.value)
            )
        return self.provider.state_resolved(
            self.calculator.eliashberg_data_path, approximation
        )

    def _solve_root(self, problem):
        solver = _SOLVERS[self.settings.approximation](problem, self.settings)
        fingerprint = problem_fingerprint(problem)
        results = []
        solution_temperature = self.settings.temperature_min
        for mu_index, mu_star in enumerate(
            np.asarray(self.calculator.mus, dtype=float)
        ):
            tc_search = solver.solve_linearized_tc(float(mu_star))
            path = restart_path(
                self.calculator.outdir, self.settings.approximation, mu_index
            )
            initial_state = None
            if self.settings.restart:
                initial_state = load_restart(
                    path,
                    self.settings.approximation,
                    float(mu_star),
                    solution_temperature,
                    fingerprint,
                )
            frequencies, state, converged, iterations, residual = (
                solver.solve_nonlinear_gap(
                    solution_temperature, float(mu_star), initial_state
                )
            )
            if self.settings.restart:
                save_restart(
                    path,
                    self.settings.approximation,
                    float(mu_star),
                    solution_temperature,
                    fingerprint,
                    state,
                    converged,
                )
            metadata = {}
            if hasattr(problem, "state_weights"):
                metadata["weights"] = np.asarray(problem.state_weights)
                metadata.update(problem.state_metadata)
                if problem.energies is not None:
                    metadata["energies"] = np.asarray(problem.energies)
                if self.settings.approximation.is_fbw:
                    metadata["state_fbw_energy_weights"] = np.asarray(
                        problem.fbw_energy_weights
                    )
            gap = state.phi / state.z
            results.append(EliashbergResult(
                approximation=self.settings.approximation,
                mu_star=float(mu_star),
                tc=tc_search.tc,
                tc_status=tc_search.status,
                solution_temperature=solution_temperature,
                matsubara_frequencies=frequencies,
                z=state.z,
                phi=state.phi,
                gap=gap,
                chi=state.chi,
                converged=converged,
                iterations=iterations,
                residual=residual,
                tc_temperatures=tc_search.temperatures,
                tc_eigenvalues=tc_search.eigenvalues,
                metadata=metadata,
            ))
        write_results(
            self.calculator.outdir, results, self.settings.save_matsubara
        )
        return results

    def run(self):
        self._check_spin_support()
        approximation = self.settings.approximation
        frequencies = None
        alpha2f = None
        if approximation == EliashbergApproximation.ISOTROPIC_FSR:
            frequencies, alpha2f = self.calculator.eliashberg_spectrum_cal()
            if self.calculator.rank == 0:
                self.calculator._write_a2f_spectrum(frequencies, alpha2f)
            if np.asarray(alpha2f).ndim == 2:
                alpha2f = np.sum(alpha2f, axis=0)

        rank = self.calculator.rank
        comm = self.calculator.comm
        if comm is None:
            problem = self._build_problem(frequencies, alpha2f)
            results = self._solve_root(problem)
            self._print_summary(results)
            return results

        results = None
        failure = None
        if rank == 0:
            try:
                problem = self._build_problem(frequencies, alpha2f)
                results = self._solve_root(problem)
            except Exception as exc:
                failure = (
                    type(exc).__name__, str(exc), traceback.format_exc()
                )
        failure = comm.bcast(failure, root=0)
        if failure is not None:
            error_name, message, details = failure
            if error_name == "NotImplementedError":
                raise NotImplementedError(message)
            raise RuntimeError("Eliashberg root-rank failure:\n{}".format(details))
        if rank == 0:
            self._print_summary(results)
        return results

    def _print_summary(self, results):
        if self.calculator.rank != 0:
            return
        print("Eliashberg approximation = {}".format(
            self.settings.approximation.value
        ))
        print("{:>10} {:>14} {:>18} {:>12}".format(
            "mu_star", "Tc (K)", "gap_iw0 (meV)", "status"
        ))
        for result in results:
            tc = (
                result.tc * QTcts.HARTREEtoKELVIN
                if np.isfinite(result.tc) else np.nan
            )
            gap = result.gap
            if gap.ndim == 1:
                gap0 = gap[0]
            else:
                gap0 = np.sum(result.metadata["weights"] * gap[:, 0])
            print("{:10.5f} {:14.6e} {:18.6e} {:>12}".format(
                result.mu_star,
                tc,
                gap0 * QTcts.HARTREEtoMEV,
                result.tc_status,
            ))
