"""QTrans core.constants helpers."""

from typing import Any
import math


class Constants:
    # used constant
    EWALD_LN_CUTOFF = 14.0          # exp(-14) ~ 1E-6
    EWALD_SCALE = 1.0               # unit in (2PI/alat)^{2}
    EXP_MAXARG = 200.0
    MAX_BUFFER_BYTES = 1.5 * (1024 ** 3)  # 1.5 GB
    AXIS_MAP = {'h': 0, 'k': 1, 'l': 2}
    LOG_LENGTH = 120

    # numbers
    PI = math.pi
    TWOPI = 2.0 * math.pi
    FOURPI = 4.0 * math.pi
    TWOPI_SQUARE = TWOPI * TWOPI
    SQRT_0P5 = math.sqrt(0.5)
    SQRT_2 = math.sqrt(2.0)
    INV_SQRT_2 = 1.0 / math.sqrt(2.0)
    SQRT_INVPI = math.sqrt(1.0 / math.pi)
    TENPM160 = 1.0E-160
    TENPM80 = 1.0E-80
    TENPM40 = 1.0E-40
    TENPM10 = 1.0E-10
    TENPM8 = 1.0E-8
    TENPM6 = 1.0E-6
    TENPM5 = 1.0E-5
    TENPM4 = 1.0E-4
    TENPM3 = 1.0E-3
    TENPP3 = 1.0E3
    TENPP6 = 1.0E6
    TENPP8 = 1.0E8
    TENPP40 = 1.0E40
    TENPP60 = 1.0E60
    TENPP80 = 1.0E80
    TENPP200 = 1.0E200
    JTWOPI = 2j * math.pi
    JFOURPI = 2.0 * JTWOPI

    # physical constants
    AMU = 1.6605402e-27                     # [kg]
    EV = 1.60217733e-19                     # [J]
    MASS_E = 9.10938215e-31                 # [kg]
    SPEED_LIGHT = 299792458                 # [m/s]
    PERMEABILITY_VACUUM = 4.0e-7 * math.pi # [Hartree/m]
    PERMITTIVITY_VACUUM = 1.0 / PERMEABILITY_VACUUM / (SPEED_LIGHT ** 2)  # [(C^2)/N*(m^2)]
    PLANCK_CONSTANT_EV = 4.13566733e-15     # [eV s]
    HBAR_EV = PLANCK_CONSTANT_EV / (2.0 * math.pi)  # [eV s]
    HARTREE = MASS_E * EV / 16.0 / (math.pi ** 2) / (PERMITTIVITY_VACUUM ** 2) / (HBAR_EV ** 2)  # [eV]
    BOHR = 4.0e+10 * math.pi * PERMITTIVITY_VACUUM * (HBAR_EV ** 2) / MASS_E  # [angstrom]
    BOLTZMANN_CONSTANT_SI = 1.3806504e-23   # [J/K]
    BOLTZMANN_CONSTANT_EV = BOLTZMANN_CONSTANT_SI / EV  # [eV/K]

    # transformation constants
    BOHRMAGtoHARTREE = 0.5
    EVtoPS = HBAR_EV * 1.0E12
    EVtoS = HBAR_EV
    EVtoINVPS = 1.0 / EVtoPS
    EVtoINVS = 1.0 / EVtoS
    KELVINtoHARTREE = BOLTZMANN_CONSTANT_EV / HARTREE
    HARTREEtoKELVIN = 1.0 / KELVINtoHARTREE
    EVtoHARTREE = 1.0 / HARTREE
    MEVtoHARTREE = EVtoHARTREE / 1000.0     # MEV means millielectron volts here
    HARTREEtoEV = HARTREE
    HARTREEtoMEV = 1.0 / MEVtoHARTREE       # MEV means millielectron volts here
    HARTREEtoJ = HARTREEtoEV * EV
    BOHRtoANG = BOHR
    ANGtoBOHR = 1.0 / BOHR
    HARTREEtoINVS = HARTREEtoEV * EVtoINVS
    HARTREEtoINVPS = HARTREEtoEV * EVtoINVPS
    MtoBOHR = 1.0E10 * ANGtoBOHR
    BOHRtoM = 1.0 / MtoBOHR
    CMtoBOHR = 1.0E8 * ANGtoBOHR
    BOHRtoCM = 1.0 / CMtoBOHR
    PHONOPYtoEV = HBAR_EV * math.sqrt(1.0E20 * EV / AMU)  # SQRT(eV / Ang^2 / AMU) to eV
    PHONOPYtoHARTREE = PHONOPYtoEV * EVtoHARTREE
    HARTREEtoAMPERE = HARTREE * EV / HBAR_EV
    HARTREEtoVOLT = HARTREE
    HARTREEtoOHM = HBAR_EV / EV

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(f"Cannot modify constant '{name}'!")

    def __delattr__(self, name: str) -> None:
        raise TypeError(f"Cannot delete constant '{name}'!")

QTcts = Constants()
