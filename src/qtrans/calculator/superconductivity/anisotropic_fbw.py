"""Band/k-resolved full-bandwidth Eliashberg solver."""

from .isotropic_fbw import _FullBandwidthSolverBase


class AnisotropicFBWSolver(_FullBandwidthSolverBase):
    """Reference anisotropic FBW solver with fixed chemical potential and chi=0."""

    solver_name = "anisotropic FBW"
