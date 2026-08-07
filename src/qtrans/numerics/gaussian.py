"""QTrans numerics.gaussian helpers."""

from qtrans.core.constants import QTcts
from scipy.special import erfc
import numpy as np


def gauss_freq(x):
    return 0.5 * erfc(-x * QTcts.SQRT_0P5)

# ref from QE Modules/wgauss.f90
def wgauss(x, ngauss:int=0):
    if ngauss == 0:
        result = gauss_freq(x*QTcts.SQRT_2)
    elif ngauss == -99:
        x = np.asarray(x)
        result = np.zeros_like(x)
        result[x > 200.0] = 1.0
        mask = (x >= -200.0) & (x <= 200.0)
        result[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
    return result

# ref from QE Modules/w0gauss.f90
def w0gauss(x, ngauss:int=0):
    if ngauss == 0:
        # Simple Gaussian (most common case) — early return
        arg = np.fmin(200.0, x * x)
        return np.exp(-arg) * QTcts.SQRT_INVPI

    if ngauss == -99:
        # Fermi-Dirac smearing — single cosh instead of two exp calls
        arg = np.fmin(36.0, np.abs(x))
        return 1.0 / (2.0 + 2.0 * np.cosh(arg))

    if ngauss == -1:
        # cold smearing (Marzari-Vanderbilt-DeVita-Payne)
        arg = np.fmin(200.0, np.square(x - QTcts.INV_SQRT_2))
        return QTcts.SQRT_INVPI * np.exp(-arg) * (2.0 - QTcts.SQRT_2 * x)

    if ngauss > 10 or ngauss < 0:
        print('Higher order smearing of {} is untested and unstable'.format(np.abs(ngauss)))
        return 0.0

    # Methfessel-Paxton (ngauss >= 1)
    arg = np.fmin(200.0, x * x)
    exp_neg_arg = np.exp(-arg)
    result = exp_neg_arg * QTcts.SQRT_INVPI
    hd = 0.0
    hp = exp_neg_arg
    a = QTcts.SQRT_INVPI
    for i in range(ngauss):
        ni2 = 2 * i
        hd = 2.0 * x * hp - 2.0 * ni2 * hd
        a = -a / ((i + 1) * 4.0)
        hp = 2.0 * x * hd - 2.0 * (ni2 + 1) * hp
        result = result + a * hp

    return result
