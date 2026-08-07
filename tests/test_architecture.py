import ast
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def class_method_names(class_type):
    names = set()
    for base in class_type.__mro__:
        if base is object:
            continue
        for name, value in base.__dict__.items():
            if callable(value):
                names.add(name)
    return names


class FacadeTests(unittest.TestCase):
    def test_calculator_preserves_complete_method_surface(self):
        from qtrans.calculator import QTransCalculator

        self.assertEqual(len(class_method_names(QTransCalculator)), 162)
        with redirect_stdout(StringIO()):
            instance = QTransCalculator()
        self.assertIsInstance(instance, QTransCalculator)

    def test_preparer_preserves_complete_method_surface(self):
        from qtrans.preparer import QTransPreparer

        self.assertEqual(len(class_method_names(QTransPreparer)), 43)
        with redirect_stdout(StringIO()):
            instance = QTransPreparer()
        self.assertIsInstance(instance, QTransPreparer)


class DependencyTests(unittest.TestCase):
    def test_legacy_root_modules_are_removed(self):
        legacy_names = [
            "QTrans_calculator.py",
            "QTrans_preparer.py",
            "QTrans_spectrum.py",
            "QTrans_error.py",
            "QTrans_validator.py",
            "utils.py",
            "run_QTrans.py",
        ]
        remaining = [name for name in legacy_names if (PROJECT_ROOT / name).exists()]
        self.assertEqual(remaining, [])

    def test_package_has_no_star_imports(self):
        violations = []
        for path in (SRC_ROOT / "qtrans").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if any(alias.name == "*" for alias in node.names):
                        violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [])

    def test_low_level_packages_do_not_import_high_level_packages(self):
        violations = []
        for package in ["core", "data", "numerics", "io", "parallel"]:
            for path in (SRC_ROOT / "qtrans" / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    if node.module.startswith("qtrans.calculator") or node.module.startswith("qtrans.preparer"):
                        violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [])

    def test_generated_modules_stay_below_transition_size_limit(self):
        oversized = []
        for path in (SRC_ROOT / "qtrans").rglob("*.py"):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > 1000:
                oversized.append((str(path.relative_to(PROJECT_ROOT)), line_count))
        self.assertEqual(oversized, [])


if __name__ == "__main__":
    unittest.main()
