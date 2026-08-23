"""Allen-Dynes critical-temperature calculation."""

import numpy as np


class _AllenDynesMixin:
    def _Allen_Dynes_Tc_cal(
            self, epc_strength: float, mu: np.ndarray,
            omega_logave: float):
        """Calculate the Allen-Dynes critical temperature in Hartree."""
        exponent = -1.04 * (1.0 + epc_strength) / (
            epc_strength - mu * (1.0 + 0.62 * epc_strength)
        )
        return omega_logave / 1.2 * np.exp(exponent)
