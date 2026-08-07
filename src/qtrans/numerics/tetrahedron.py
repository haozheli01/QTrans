"""QTrans numerics.tetrahedron helpers."""

import numpy as np


def get_tetrahedron_weight(e_corner, e_surface):
    """
    Get the weights of the four corners in one tetrahedron.
    References: Phys. Rev. B 49, 16223 (1994).
                weight_dos.f90 in Perturbo Package.
    Args:
        e_corner (np.ndarray): The energies of the four corners. # shape (4, )
        e_surface (float): The energy of the integral surface.

    Returns:
        weights (np.ndarray): Weights of the four corners. # shape (4, )
    """
    
    e_index = np.argsort(e_corner)
    e = e_corner[e_index]
    w = np.zeros(4, dtype=np.float64)
    if e_surface < e[0]:
        w[:] = 0.0
    elif e_surface < e[1] or (e_surface == e[1] and e[1] == e[2] == e[3] and e[1] > e[0]):
        e01 = e_surface - e[0]
        e21 = e[1] - e[0]
        e31 = e[2] - e[0]
        e41 = e[3] - e[0]
        factor = e01 * e01 / (e21 * e31 * e41)
        w[1] = factor * e01 / e21
        w[2] = factor * e01 / e31
        w[3] = factor * e01 / e41
        w[0] = 3.0 * factor - w[1] - w[2] - w[3]
    elif e_surface < e[2]:
        e01 = e_surface - e[0]
        e02 = e_surface - e[1]
        e30 = e[2] - e_surface
        e40 = e[3] - e_surface
        e31 = e[2] - e[0]
        e32 = e[2] - e[1]
        e41 = e[3] - e[0]
        e42 = e[3] - e[1]
        dc1 = 0.5 * e01 / (e41 * e31)
        c1 = 0.5 * e01 * dc1
        c2 = 0.25 / (e41 * e32 * e31)
        dc2 = c2 * (e02 * e30 + e01 * e30 - e01 * e02)
        c2 = c2 * e01 * e02 * e30
        c3 = 0.25 / (e42 * e32 * e41)
        dc3 = c3 * (2.0 * e02 * e40 - e02 * e02)
        c3 = c3 * (e02 * e02 * e40)
        w[0] = dc1 + (dc1 + dc2) * e30 / e31 + (dc1 + dc2 + dc3) * e40 / e41
        w[0] -= (c1 + c2) / e31 + (c1 + c2 + c3) / e41
        w[1] = dc1 + dc2 + dc3 + (dc2 + dc3) * e30 / e32 + dc3 * e40 / e42
        w[1] -= (c2 + c3) / e32 + c3 / e42
        w[2] = (dc1 + dc2) * e01 / e31 + (dc2 + dc3) * e02 / e32
        w[2] += (c1 + c2) / e31 + (c2 + c3) / e32
        w[3] = (dc1 + dc2 + dc3) * e01 / e41 + dc3 * e02 / e42
        w[3] += (c1 + c2 + c3) / e41 + c3 / e42
    elif e_surface < e[3]:
        e41 = e[3] - e[0]
        e42 = e[3] - e[1]
        e43 = e[3] - e[2]
        e40 = e[3] - e_surface
        factor = e40 * e40 / (e41 * e42 * e43)
        w[0] = factor * e40 / e41
        w[1] = factor * e40 / e42
        w[2] = factor * e40 / e43
        w[3] = 3.0 * factor - w[0] - w[1] - w[2]
    else:
        w[:] = 0.0

    # Reorder weights to match the original order of e_corner
    weights = np.zeros(4, dtype=np.float64)
    weights[e_index] = w[:]

    return weights

def get_tetrahedron_sum_weight_batch(e_corners, e_surface):
    """
    Vectorized version: compute sum_weight for many tetrahedra at once.
    References: Phys. Rev. B 49, 16223 (1994).
                weight_dos.f90 in Perturbo Package.

    Args:
        e_corners (np.ndarray): Sorted corner energies, shape (N, 4).
        e_surface (float): The energy of the integral surface.

    Returns:
        sum_weights (np.ndarray): Shape (N,).
    """
    e = np.sort(e_corners, axis=1)  # (N, 4)
    e0, e1, e2, e3 = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
    es = e_surface

    sum_weights = np.zeros(e.shape[0])

    # Region 1: e0 <= es < e1 (or degenerate case)
    mask1 = (es >= e0) & (
        (es < e1) | ((es == e1) & (e1 == e2) & (e2 == e3) & (e1 > e0))
    )
    if mask1.any():
        e01 = es - e0[mask1]
        e21 = e1[mask1] - e0[mask1]
        e31 = e2[mask1] - e0[mask1]
        e41 = e3[mask1] - e0[mask1]
        sum_weights[mask1] = 3.0 * e01 * e01 / (e21 * e31 * e41)

    # Region 2: e1 <= es < e2
    mask2 = (es >= e1) & (es < e2) & ~mask1
    if mask2.any():
        e02 = es - e1[mask2]
        e21 = e1[mask2] - e0[mask2]
        e31 = e2[mask2] - e0[mask2]
        e32 = e2[mask2] - e1[mask2]
        e41 = e3[mask2] - e0[mask2]
        e42 = e3[mask2] - e1[mask2]
        sum_weights[mask2] = 3.0 * (e21 + 2.0*e02 - (e31+e42)*e02*e02/(e32*e42)) / (e31*e41)

    # Region 3: e2 <= es < e3
    mask3 = (es >= e2) & (es < e3) & ~mask1 & ~mask2
    if mask3.any():
        e41 = e3[mask3] - e0[mask3]
        e42 = e3[mask3] - e1[mask3]
        e43 = e3[mask3] - e2[mask3]
        e40 = e3[mask3] - es
        sum_weights[mask3] = 3.0 * e40 * e40 / (e41 * e42 * e43)

    # Region: es >= e3 or es < e0 → 0 (already initialized)

    return sum_weights

def tetrahedron_integral(tetras, enks, e_surface):
    """
    Calculate the density of state at e_surface, using tetrahedron method.

    Args:
        tetras (np.ndarray): Indices of k-points forming each tetrahedron. # shape (ntetras, 4)
        enks (np.ndarray): The eigen energies for electrons in corresponding k-points. # shape (nbands, nks)
        e_surface (float): The energy of the integral surface.

    Returns:
        dos (float): The density of state at e_surface.
    """
    # Gather corner energies for all bands and tetrahedra at once
    # enks[:, tetras] has shape (nbands, ntetras, 4)
    e_corners = enks[:, tetras].reshape(-1, 4)  # (nbands * ntetras, 4)
    return get_tetrahedron_sum_weight_batch(e_corners, e_surface).sum()
