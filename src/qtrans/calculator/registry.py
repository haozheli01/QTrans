"""Calculation mode registry used by the public calculator facade."""

from types import MappingProxyType


CALCULATION_MODES = MappingProxyType(
    {
        "band": "plot_band",
        "boltzmann": "boltzmann_cal",
        "breakdown": "breakdown_cal",
        "conductivity": "conductivity_cal",
        "dos": "plot_dos",
        "epc": "plot_epc",
        "fermi_surface": "plot_fermi_surface",
        "phonon": "plot_phonon",
        "pomega": "pomega_cal",
        "resistivity": "resistivity_cal",
        "superconductivity": "superconductivity_cal",
    }
)


def resolve_calculation(instance, mode):
    """Return the bound action for a mode, or ``None`` when unsupported."""
    method_name = CALCULATION_MODES.get(mode)
    if method_name is None:
        return None
    return getattr(instance, method_name)

