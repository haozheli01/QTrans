import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class DistributionTests(unittest.TestCase):
    def test_gaussian_at_origin(self):
        from qtrans.numerics.gaussian import w0gauss

        self.assertAlmostEqual(float(w0gauss(0.0)), 1.0 / math.sqrt(math.pi))

    def test_fermi_distribution_limits_and_midpoint(self):
        from qtrans.numerics.distributions import fermi_weight

        energies = np.array([-1000.0, 0.0, 1000.0])
        actual = fermi_weight(energies, 1.0)
        np.testing.assert_allclose(actual, [1.0, 0.5, 0.0])

    def test_bose_distribution_handles_nonpositive_energy(self):
        from qtrans.numerics.distributions import bose_weight

        actual = bose_weight(np.array([-1.0, 0.0, 1.0]), 1.0)
        self.assertEqual(actual[0], 0.0)
        self.assertEqual(actual[1], 0.0)
        self.assertAlmostEqual(actual[2], 1.0 / np.expm1(1.0))


class KPointTests(unittest.TestCase):
    def test_linear_k_path(self):
        from qtrans.numerics.kpoints import kpoints_generator

        generator = kpoints_generator(dim_k=3, lat=np.eye(3))
        points, distances, nodes = generator.k_path(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            nk=3,
            report=False,
        )
        np.testing.assert_allclose(
            points,
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
        )
        np.testing.assert_allclose(distances, [0.0, 0.25, 0.5])
        np.testing.assert_allclose(nodes, [0.0, 0.5])


class ParallelHelperTests(unittest.TestCase):
    def test_split_sections_are_balanced(self):
        from qtrans.parallel.mpi import get_mpi_split_sections, split_into_rank

        indices = get_mpi_split_sections(10, 3)
        np.testing.assert_array_equal(indices, [4, 7])
        chunks = [split_into_rank(np.arange(10), 0, rank, 3) for rank in range(3)]
        self.assertEqual([len(chunk) for chunk in chunks], [4, 3, 3])
        np.testing.assert_array_equal(np.concatenate(chunks), np.arange(10))


class SparseMatrixTests(unittest.TestCase):
    def test_reciprocal_transform_for_two_cells(self):
        from qtrans.numerics.sparse import build_reciprocal_from_sparseMat

        matrices = np.array([[[1.0]], [[2.0]]], dtype=complex)
        shifts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        k_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        actual = build_reciprocal_from_sparseMat(matrices, k_points, shifts)
        np.testing.assert_allclose(actual[:, 0, 0], [3.0, -1.0])


if __name__ == "__main__":
    unittest.main()

