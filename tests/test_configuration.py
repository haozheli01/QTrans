import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ConfigurationOwnershipTests(unittest.TestCase):
    def test_snapshot_is_deep_and_does_not_mutate_caller(self):
        from qtrans.config import snapshot_configuration

        original = {"basic": {"mesh": [2, 2, 2]}}
        snapshot = snapshot_configuration(original)
        snapshot["basic"]["mesh"][0] = 8
        snapshot.pop("basic")
        self.assertEqual(original, {"basic": {"mesh": [2, 2, 2]}})

    def test_calculator_defaults_are_not_shared_between_instances(self):
        from qtrans.calculator import QTransCalculator
        from qtrans.core.defaults import calculator_default_parameters

        class CalculatorWithoutParsing(QTransCalculator):
            def _parse_input(self, config):
                self.captured_config = config

        config = {"basic": {"marker": [1]}}
        with redirect_stdout(StringIO()):
            first = CalculatorWithoutParsing(config)
            second = CalculatorWithoutParsing(config)

        first.k_size[0] = 99
        first.captured_config["basic"]["marker"].append(2)

        self.assertEqual(second.k_size, [32, 32, 32])
        self.assertEqual(calculator_default_parameters["transport"]["k_size"], [32, 32, 32])
        self.assertEqual(config, {"basic": {"marker": [1]}})


if __name__ == "__main__":
    unittest.main()

