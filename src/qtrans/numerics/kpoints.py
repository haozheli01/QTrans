"""QTrans numerics.kpoints helpers."""

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np


class kpoints_generator:
    """Generate k-point paths along high-symmetry directions in the Brillouin zone.

    Parameters
    ----------
    dim_k : int
        Dimensionality of k-space.
    lat : array-like
        Lattice vectors as rows of a matrix.
    per : list or tuple, optional
        Indices of periodic directions. Defaults to first `dim_k` dimensions.
    """

    def __init__(
        self,
        dim_k: int = 3,
        lat: Optional[Union[np.ndarray, List]] = None,
        per: Optional[Union[List[int], Tuple[int, ...]]] = None,
    ):
        self._dim_k = dim_k
        self._lat = np.asarray(lat) if lat is not None else None

        if per is None:
            self._per = list(range(self._dim_k))
        else:
            if len(per) != self._dim_k:
                raise ValueError(
                    f"Length of 'per' ({len(per)}) must equal dim_k ({self._dim_k})."
                )
            self._per = list(per)

    def __repr__(self) -> str:
        return f"KPointsGenerator(dim_k={self._dim_k}, per={self._per})"

    @staticmethod
    def _nice_float(x: float, just: int, rnd: int) -> str:
        return str(round(x, rnd)).rjust(just)

    def k_path(
        self,
        kpts: Union[str, List, np.ndarray],
        nk: int,
        report: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate an interpolated k-point path.

        Parameters
        ----------
        kpts : str or array-like
            High-symmetry k-points defining the path nodes.
            Special strings: 'full', 'fullc', 'half' (1D only).
        nk : int
            Total number of k-points along the path.
        report : bool
            Whether to print a path report.

        Returns
        -------
        k_vec : ndarray, shape (nk, dim_k)
            Interpolated k-point coordinates.
        k_dist : ndarray, shape (nk,)
            Accumulated distance along the path for each k-point.
        k_node : ndarray, shape (n_nodes,)
            Accumulated distance at each node.
        """
        k_list = np.asarray(kpts)

        # In 1D, convert a flat vector to column
        if k_list.ndim == 1 and self._dim_k == 1:
            k_list = k_list[:, np.newaxis]

        # Validate dimensions
        if k_list.ndim != 2 or k_list.shape[1] != self._dim_k:
            raise ValueError(
                f"k-point dimension mismatch: input has {k_list.shape[-1]}, "
                f"expected {self._dim_k}."
            )

        n_nodes = k_list.shape[0]
        if nk < n_nodes:
            raise ValueError(
                f"nk ({nk}) must be >= number of nodes ({n_nodes})."
            )

        # Compute k-space metric tensor
        lat_per = self._lat[self._per]
        k_metric = np.linalg.inv(lat_per @ lat_per.T)

        # Compute accumulated distances at each node
        k_node = np.zeros(n_nodes)
        for n in range(1, n_nodes):
            dk = k_list[n] - k_list[n - 1]
            k_node[n] = k_node[n - 1] + np.sqrt(dk @ k_metric @ dk)

        # Determine node indices in the interpolated array
        node_index = [0]
        for n in range(1, n_nodes - 1):
            node_index.append(int(round(k_node[n] / k_node[-1] * (nk - 1))))
        node_index.append(nk - 1)

        # Vectorized interpolation between consecutive nodes
        k_dist = np.zeros(nk)
        k_vec = np.zeros((nk, self._dim_k))
        k_vec[0] = k_list[0]

        for n in range(1, n_nodes):
            n_i, n_f = node_index[n - 1], node_index[n]
            if n_f == n_i:
                continue
            frac = np.linspace(0.0, 1.0, n_f - n_i + 1)
            k_dist[n_i : n_f + 1] = k_node[n - 1] + frac * (k_node[n] - k_node[n - 1])
            k_vec[n_i : n_f + 1] = (
                k_list[n - 1][np.newaxis, :] + frac[:, np.newaxis] * (k_list[n] - k_list[n - 1])[np.newaxis, :]
            )

        if report:
            self._print_report(k_list, k_node, node_index, lat_per, k_metric)

        return k_vec, k_dist, k_node

    def _print_report(
        self,
        k_list: np.ndarray,
        k_node: np.ndarray,
        node_index: List[int],
        lat_per: np.ndarray,
        k_metric: np.ndarray,
    ) -> None:
        """Print a summary of the k-path."""
        n_nodes = k_list.shape[0]

        if self._dim_k == 1:
            print(f" Path in 1D BZ defined by nodes at {k_list.flatten()}")
        else:
            print("----- k_path report begin ----------")
            with np.printoptions(precision=5):
                print("real-space lattice vectors\n", lat_per)
                print("k-space metric tensor\n", k_metric)
                print("internal coordinates of nodes\n", k_list)
                if lat_per.shape[0] == lat_per.shape[1]:
                    lat_per_inv = np.linalg.inv(lat_per).T
                    print("reciprocal-space lattice vectors\n", lat_per_inv)
                    kpts_cart = k_list @ lat_per_inv
                    print("cartesian coordinates of nodes\n", kpts_cart)
                print("list of segments:")
                for n in range(1, n_nodes):
                    dk_str = self._nice_float(k_node[n] - k_node[n - 1], 7, 5)
                    print('  length = ' + dk_str + '  from ', k_list[n-1], ' to ', k_list[n])
                print("node distance list:", k_node)
                print("node index list:   ", np.array(node_index))
            print("----- k_path report end ------------")
        print()
