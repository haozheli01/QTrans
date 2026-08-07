"""QTrans numerics.distributions helpers."""

import numpy as np


def fermi_weight(E, kbT):
    # Here E means E - \mu 
    x = E / kbT
    result = np.empty_like(x)
    result[x > 100] = 0.0
    result[x < -100] = 1.0
    mask = (x >= -100) & (x <= 100)
    result[mask] = 1.0 / (np.exp(x[mask]) + 1.0)
    return result

def minus_dfermi(E, kbT):
    return fermi_weight(E, kbT) * fermi_weight(-E, kbT) / kbT

def bose_weight(E, kbT):
    E = np.asarray(E, dtype=np.float64)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = E / kbT
        result = 1.0 / np.expm1(x)  
    result = np.where(x > 700, 0.0, result)
    result = np.where(x <= 0, 0.0, result)     
    if np.isscalar(E) or np.ndim(E) == 0:
        return result.item()  
    return result
