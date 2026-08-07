"""Preparation mode registry used by the public preparer facade."""

from types import MappingProxyType


PREPARATION_MODES = MappingProxyType(
    {
        "grad-epc": "prep_grad_mat_epc",
        "graph-pack": "pack_graph_data",
        "vasp-ph": "prep_force_sets_from_vasp",
    }
)


def resolve_preparation(instance, mode):
    """Return the bound action for a mode, or ``None`` when unsupported."""
    method_name = PREPARATION_MODES.get(mode)
    if method_name is None:
        return None
    return getattr(instance, method_name)

