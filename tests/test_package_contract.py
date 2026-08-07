import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class PackageContractTests(unittest.TestCase):
    def test_package_has_current_version(self):
        import qtrans

        self.assertEqual(qtrans.__version__, "1.0.0")

    def test_native_boundary_exports_expected_functions(self):
        import qtrans.extensions as extensions

        expected = {
            "breakdown_rate_cal_EPC_helper_sparse",
            "dynamics_setup_helper_sparse",
            "electron_self_energy_cal_EPC_helper_sparse",
            "eliashberg_spectrum_cal_EPC_helper_sparse",
            "eliashberg_spectrum_cal_EPC_helper_sparse_soc",
            "epc_elements_cal_helper_sparse",
            "scattering_rate_cal_EPC_helper_sparse",
            "scattering_rate_cal_EPC_helper_sparse_soc",
            "transport_eliashberg_function_cal_EPC_iso_helper_sparse",
        }
        self.assertTrue(expected.issubset(set(extensions.__all__)))

    def test_missing_native_extension_has_actionable_error(self):
        from qtrans.extensions import (
            NativeExtensionUnavailable,
            epc_elements_cal_helper_sparse,
        )

        try:
            epc_elements_cal_helper_sparse()
        except NativeExtensionUnavailable as error:
            self.assertIn("C compiler", str(error))
        except TypeError:
            self.skipTest("The native extension is already built in this environment.")


if __name__ == "__main__":
    unittest.main()
