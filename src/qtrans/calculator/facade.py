from .lifecycle import _LifecycleMixin
from .graph import _GraphMixin
from .config import _ConfigMixin
from .mesh import _MeshMixin
from .phonons import _PhononKernelMixin
from .electrons import _ElectronMixin
from .epc import _EPCMixin
from .long_range import _LongRangeMixin
from .fermi import _FermiMixin
from .workflows.pomega import _PomegaMixin
from .superconductivity.pairing_spectrum import _PairingSpectrumMixin
from .superconductivity.allen_dynes import _AllenDynesMixin
from .workflows.superconductivity import _SuperconductivityMixin
from .workflows.resistivity import _ResistivityMixin
from .workflows.conductivity import _ConductivityMixin
from .workflows.breakdown import _BreakdownMixin
from .workflows.boltzmann import _BoltzmannMixin
from .transport.transport_spectrum import _TransportSpectrumMixin
from .transport.common import _TransportCommonMixin
from .transport.scattering import _ScatteringMixin
from .transport.transport_solver import _TransportSolverMixin
from .transport.iterative_bte import _IterativeBTEMixin
from .transport.realtime_bte import _RealtimeBTEMixin
from .spectra.self_energy import _SelfEnergyMixin
from .workflows.dos import _DOSMixin
from .outputs import _OutputMixin
from .workflows.dispersion import _DispersionMixin
from .workflows.epc import _EPCPlotMixin


class QTransCalculator(
    _LifecycleMixin,
    _GraphMixin,
    _ConfigMixin,
    _MeshMixin,
    _PhononKernelMixin,
    _ElectronMixin,
    _EPCMixin,
    _LongRangeMixin,
    _FermiMixin,
    _PomegaMixin,
    _PairingSpectrumMixin,
    _AllenDynesMixin,
    _SuperconductivityMixin,
    _ResistivityMixin,
    _ConductivityMixin,
    _BreakdownMixin,
    _BoltzmannMixin,
    _TransportSpectrumMixin,
    _TransportCommonMixin,
    _ScatteringMixin,
    _TransportSolverMixin,
    _IterativeBTEMixin,
    _RealtimeBTEMixin,
    _SelfEnergyMixin,
    _DOSMixin,
    _OutputMixin,
    _DispersionMixin,
    _EPCPlotMixin,
):
    """Public QTrans calculator facade."""

    pass
