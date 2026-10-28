"""Allen-Dynes critical-temperature calculation."""

import numpy as np


class _AllenDynesMixin:
    def _Allen_Dynes_Tc_cal(
            self, epc_strength: float, mu: np.ndarray,
            omega_logave: float):
        """Return Tc in Hartree, or zero outside the attractive regime."""
        denominator = np.asarray(
            epc_strength - mu * (1.0 + 0.62 * epc_strength)
        )
        tc = np.zeros_like(denominator, dtype=float)
        valid = denominator > 0.0
        tc[valid] = omega_logave / 1.2 * np.exp(
            -1.04 * (1.0 + epc_strength) / denominator[valid]
        )
        return tc[()] if tc.ndim == 0 else tc
