import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class CalculationRegistryTests(unittest.TestCase):
    def test_registry_resolves_bound_action(self):
        from qtrans.calculator.registry import resolve_calculation

        calls = []

        class Calculator:
            def plot_dos(self):
                calls.append("dos")

        action = resolve_calculation(Calculator(), "dos")
        action()
        self.assertEqual(calls, ["dos"])

    def test_registry_rejects_mutation(self):
        from qtrans.calculator.registry import CALCULATION_MODES

        with self.assertRaises(TypeError):
            CALCULATION_MODES["new-mode"] = "run_new_mode"

    def test_unknown_calculation_returns_none(self):
        from qtrans.calculator.registry import resolve_calculation

        self.assertIsNone(resolve_calculation(object(), "unknown"))

    def test_calculator_run_uses_registry(self):
        from qtrans.calculator import QTransCalculator

        calls = []

        class Calculator(QTransCalculator):
            def plot_dos(self):
                calls.append("dos")

            def _show_job_success(self):
                calls.append("success")

        calculator = Calculator.__new__(Calculator)
        calculator.cal_mode = "dos"
        calculator.run()
        self.assertEqual(calls, ["dos", "success"])


class PreparationRegistryTests(unittest.TestCase):
    def test_registry_resolves_bound_action(self):
        from qtrans.preparer.registry import resolve_preparation

        calls = []

        class Preparer:
            def pack_graph_data(self):
                calls.append("graph-pack")

        action = resolve_preparation(Preparer(), "graph-pack")
        action()
        self.assertEqual(calls, ["graph-pack"])

    def test_preparer_run_uses_registry(self):
        from qtrans.preparer import QTransPreparer

        calls = []

        class Preparer(QTransPreparer):
            def pack_graph_data(self):
                calls.append("graph-pack")

            def _show_job_success(self):
                calls.append("success")

        preparer = Preparer()
        preparer.rank = 0
        preparer.prep_mode = "graph-pack"
        preparer.run()
        self.assertEqual(calls, ["graph-pack", "success"])


if __name__ == "__main__":
    unittest.main()
