import unittest

import torch

from phase_transition.subspace.core import projection_fraction, subspace_overlap


class SubspaceTests(unittest.TestCase):
    def test_identical_subspaces_have_unit_overlap(self):
        basis = torch.eye(6)[:, :3]
        overlap, largest, cosines = subspace_overlap(basis, basis, 3)
        self.assertAlmostEqual(overlap, 1.0)
        self.assertAlmostEqual(largest, 1.0)
        self.assertEqual(len(cosines), 3)

    def test_projection_fraction_is_squared_norm_fraction(self):
        gradient = torch.tensor([3.0, 4.0])
        basis = torch.tensor([[1.0], [0.0]])
        self.assertAlmostEqual(projection_fraction(gradient, basis, 1, 1e-12), 9.0 / 25.0)


if __name__ == "__main__":
    unittest.main()
