"""Superconductivity calculation workflow."""

import os

from qtrans.core.constants import QTcts
from qtrans.calculator.superconductivity import EliashbergRunner


class _SuperconductivityMixin:
    def superconductivity_cal(self):
        """Dispatch to Allen-Dynes or an imaginary-axis Eliashberg solver."""
        if self.sc_method == 'eliashberg':
            return EliashbergRunner(self).run()
        else:
            return self._allen_dynes_superconductivity_cal()

    def _write_a2f_spectrum(self, omegas, a2f):
        if self.rank != 0:
            return
        path = os.path.join(self.outdir, "a2f.dat")
        with open(path, 'w', encoding='utf-8', newline='\n') as fout:
            if self.spinor_mode == 'collinear':
                a2f_tot = a2f[0] + a2f[1]
                fout.write(
                    f"{'#omega (meV)':>6}    {'a2f_tot':>2}    "
                    f"{'a2f_up':>2}    {'a2f_dn':>2}\n"
                )
                for iomega, omega in enumerate(omegas):
                    fout.write(
                        f"{omega * QTcts.HARTREEtoMEV:6.4f}        "
                        f"{a2f_tot[iomega]:6.4f}        "
                        f"{a2f[0, iomega]:6.4f}        "
                        f"{a2f[1, iomega]:6.4f}\n"
                    )
            else:
                fout.write(f"{'#omega (meV)':>6}    {'a2f':>2}\n")
                for iomega, omega in enumerate(omegas):
                    fout.write(
                        f"{omega * QTcts.HARTREEtoMEV:6.4f}        "
                        f"{a2f[iomega]:6.4f}\n"
                    )

    def _allen_dynes_superconductivity_cal(self):
        omegas, a2f = self.eliashberg_spectrum_cal()
        self._write_a2f_spectrum(omegas, a2f)
        if self.spinor_mode == 'collinear':
            a2f_tot = a2f[0] + a2f[1]
        epc_strength = self._epc_strength_cal(a2f, omegas)
        if self.spinor_mode == 'collinear':
            epc_strength_tot = epc_strength[0] + epc_strength[1]
            logave_freq = self._logave_freq_cal(
                epc_strength_tot, a2f_tot, omegas
            )
            tc = self._Allen_Dynes_Tc_cal(
                epc_strength_tot, self.mus, logave_freq
            )
        else:
            logave_freq = self._logave_freq_cal(
                epc_strength, a2f, omegas
            )
            tc = self._Allen_Dynes_Tc_cal(
                epc_strength, self.mus, logave_freq
            )
        if self.rank == 0:
            if self.spinor_mode == 'collinear':
                print(
                    f"Lambda_tot = {epc_strength_tot}, "
                    f"Lambda_up = {epc_strength[0]}, "
                    f"Lambda_dn = {epc_strength[1]}"
                )
            else:
                print(f"Lambda = {epc_strength}")
            print(f"Omega log = {logave_freq * QTcts.HARTREEtoMEV} meV")
            print(f"{'mu':>4}    {'Tc (K)':>10}")
            print('\n'.join(
                f"{mu:6.4f}    {value * QTcts.HARTREEtoKELVIN:.6e}"
                for mu, value in zip(self.mus, tc)
            ))
        return tc
