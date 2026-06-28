import tempfile
import unittest
from pathlib import Path

from pacer_marl.cli import smoke_config
from pacer_marl.trainer import evaluate, train


class SmokeTest(unittest.TestCase):
    def test_end_to_end(self):
        config = smoke_config(episodes=5)
        config["environment"]["horizon"] = 8
        with tempfile.TemporaryDirectory() as tmp:
            model = train(config, tmp)
            result = evaluate(config, model, episodes=3)
            self.assertTrue((Path(tmp) / "checkpoint_final.npz").exists())
            self.assertTrue((Path(tmp) / "metrics.csv").exists())
            self.assertEqual(result["episodes"], 3)


if __name__ == "__main__":
    unittest.main()

