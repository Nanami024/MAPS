import tempfile
import unittest
from pathlib import Path

import numpy as np

from pacer_marl.model import RoleMixtureActorCritic


class ModelTest(unittest.TestCase):
    def test_distribution_and_roundtrip(self):
        model = RoleMixtureActorCritic(12, 4, context_dim=6, num_roles=3, seed=5)
        obs = np.linspace(-1, 1, 12)
        context = np.zeros(6)
        p, gate, experts, *_ = model.distribution(obs, context)
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertAlmostEqual(float(gate.sum()), 1.0)
        self.assertEqual(experts.shape, (3, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            model.save(path)
            restored = RoleMixtureActorCritic.load(path)
            p2, *_ = restored.distribution(obs, context)
            np.testing.assert_allclose(p, p2)


if __name__ == "__main__":
    unittest.main()

