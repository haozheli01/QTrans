"""QTrans numerics.random helpers."""

import numpy as np


def random_uniform(nvec:int, random_seed:int=0):
    """
    Get a set of vectors which meet the uniform distribution.

    args:
        nvec (int): The number of vectors.
        random_seed (int): The random seed.

    returns:
        vecs (np.ndarray): The random vectors. # shape: (nvec, 3)
        weights (np.ndarray): The weights corresponding to the vecs. # shape: (nvec, )
    """
    np.random.seed(random_seed)
    vecs = np.random.rand(nvec, 3)
    weights = np.ones(nvec) / float(nvec)
    return vecs, weights

# ref from Perturbo pert_utils.f90
def random_cauchy(nvec:int, cauchy_scale:float=0.05, random_seed:int=0):
    """
    Get a set of vectors which meet the cauchy distribution.

    args:
        nvec (int): The number of vectors.
        cauchy_scale (float): The scale parameter sigma in cauchy distribution.
        random_seed (int): The random seed.

    returns:
        vecs (np.ndarray): The random vectors. # shape: (nvec, 3)
        weights (np.ndarray): The weights corresponding to the vecs. # shape: (nvec, )
    """
    
    vecs = np.zeros((nvec, 3))
    weights = np.zeros(nvec)

    # set seed
    np.random.seed(random_seed)
    pa = 0.5 + np.arctan(-0.5 / cauchy_scale) / np.pi
    pb = 0.5 + np.arctan(0.5 / cauchy_scale) / np.pi
    wtmp = 1.0 / nvec
    for i in range(nvec):
        temp = np.random.rand(3)
        temp = pa + temp * (pb - pa)
        vecs[i, :] = cauchy_scale * np.tan((temp - 0.5) * np.pi)
        temp = ((vecs[i, :] / cauchy_scale) ** 2 + 1.0) * cauchy_scale * np.pi * (pb - pa)
        weights[i] = wtmp
        for j in range(3):
            weights[i] = weights[i] * temp[j]

    return vecs, weights
