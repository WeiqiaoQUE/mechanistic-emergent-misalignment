import unittest

import torch

from phase_transition.causal import interventions


class InterventionTests(unittest.TestCase):
    def setUp(self):
        interventions.RANK = 1

    def test_decomposition_reconstructs_update(self):
        update = torch.tensor([2.0, 3.0])
        basis = torch.tensor([[1.0], [0.0]])
        parallel, perpendicular, norm = interventions.decompose(update, basis, 1)
        self.assertTrue(torch.allclose(parallel + perpendicular, update))
        self.assertAlmostEqual(norm, 2.0)

    def test_random_control_has_requested_removed_norm(self):
        update = torch.tensor([3.0, 4.0])
        basis = torch.tensor([[1.0], [0.0]])
        result = interventions.norm_matched_removal(update, basis, 1, 2.0)
        self.assertAlmostEqual(float((update - result).norm()), 2.0)


if __name__ == "__main__":
    unittest.main()
