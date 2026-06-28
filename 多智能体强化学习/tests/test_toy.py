import unittest

import numpy as np

from pacer_marl.envs.toy_coordination import ProceduralCoordinationEnv


class ToyEnvironmentTest(unittest.TestCase):
    def test_shapes_and_episode_end(self):
        env = ProceduralCoordinationEnv(seed=3, horizon=6)
        obs = env.reset()
        self.assertEqual(obs.shape, (1, env.observation_dim))
        for index in range(6):
            result = env.step(np.array([index % env.num_actions]))
        self.assertTrue(result.done)
        self.assertEqual(result.rewards.shape, (1,))
        self.assertIn("partner_type", result.info)

    def test_invalid_action(self):
        env = ProceduralCoordinationEnv(seed=0)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.array([99]))


if __name__ == "__main__":
    unittest.main()

