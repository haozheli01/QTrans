"""Names used by the superconductivity configuration and solver registry."""

from enum import Enum


class SuperconductivityMethod(str, Enum):
    ALLEN_DYNES = "allen_dynes"
    ELIASHBERG = "eliashberg"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                "Unsupported superconductivity method {!r}; choose one of: {}."
                .format(value, choices)
            ) from exc


class EliashbergApproximation(str, Enum):
    ISOTROPIC_FSR = "iso_fsr"
    ANISOTROPIC_FSR = "aniso_fsr"
    ISOTROPIC_FBW = "iso_fbw"
    ANISOTROPIC_FBW = "aniso_fbw"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                "Unsupported Eliashberg approximation {!r}; choose one of: {}."
                .format(value, choices)
            ) from exc

    @property
    def is_fbw(self):
        return self in (self.ISOTROPIC_FBW, self.ANISOTROPIC_FBW)

    @property
    def is_anisotropic(self):
        return self in (self.ANISOTROPIC_FSR, self.ANISOTROPIC_FBW)
