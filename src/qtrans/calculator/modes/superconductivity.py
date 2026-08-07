"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
import os


class _SuperconductivityMixin:
    def superconductivity_cal(self):
        """
        Calculate for superconductivity.

        Args:

        Returns:
        """
        omegas, a2f = self.eliashberg_spectrum_cal()
        if self.spinor_mode == 'collinear':
            a2f_tot = a2f[0] + a2f[1] # tot a2f
        if self.rank == 0:
            fout = open(os.path.join(self.outdir, "a2f.dat"), 'w')
            if self.spinor_mode == 'collinear':
                fout.write(f"{'#omega (meV)':>6}    {'a2f_tot':>2}    {'a2f_up':>2}    {'a2f_dn':>2}\n")
                for iomega, omega in enumerate(omegas):
                    fout.write(f"{omega * QTcts.HARTREEtoMEV:6.4f}        {a2f_tot[iomega]:6.4f}        {a2f[0, iomega]:6.4f}        {a2f[1, iomega]:6.4f}\n")
            else:
                fout.write(f"{'#omega (meV)':>6}    {'a2f':>2}\n")
                for iomega, omega in enumerate(omegas):
                    fout.write(f"{omega * QTcts.HARTREEtoMEV:6.4f}        {a2f[iomega]:6.4f}\n")
            fout.close()
        epc_strength = self._epc_strength_cal(a2f, omegas)
        if self.spinor_mode == 'collinear':
            epc_strength_tot = epc_strength[0] + epc_strength[1]
            logave_freq = self._logave_freq_cal(epc_strength_tot, a2f_tot, omegas)
            Tc = self._Allen_Dynes_Tc_cal(epc_strength_tot, self.mus, logave_freq)
        else:
            logave_freq = self._logave_freq_cal(epc_strength, a2f, omegas)
            Tc = self._Allen_Dynes_Tc_cal(epc_strength, self.mus, logave_freq)
        if self.rank == 0:
            if self.spinor_mode == 'collinear':
                print(f"Lambda_tot = {epc_strength_tot}, Lambda_up = {epc_strength[0]}, Lambda_dn = {epc_strength[1]}")
            else:
                print(f"Lambda = {epc_strength}")
            print(f"Omega log = {logave_freq * QTcts.HARTREEtoMEV} meV")
            print(f"{'mu':>4}    {'Tc (K)':>10}")
            mu_Tc_str = '\n'.join([f"{self.mus[imu]:6.4f}    {Tc[imu] * QTcts.HARTREEtoKELVIN:.6e}" for imu in range(len(self.mus))])
            print(mu_Tc_str)

    def pomega_cal(self):
        """
        Calculate for Pomega.

        Args:

        Returns:
        """
        omegas, a2f = self._Pomega_cal_epc()
        if self.spinor_mode == 'collinear':
            a2f_tot = a2f[0] + a2f[1] # tot a2f
        if self.rank == 0:
            fout = open(os.path.join(self.outdir, "pomega.dat"), 'w')
            if self.spinor_mode == 'collinear':
                fout.write(f"{'#omega (meV)':>6}    {'a2f_tot':>2}    {'a2f_up':>2}    {'a2f_dn':>2}\n")
                for iomega, omega in enumerate(omegas):
                    fout.write(f"{omega * QTcts.HARTREEtoMEV:6.4f}        {a2f_tot[iomega]:6.4f}        {a2f[0, iomega]:6.4f}        {a2f[1, iomega]:6.4f}\n")
            else:
                fout.write(f"{'#omega (meV)':>6}    {'a2f':>2}\n")
                for iomega, omega in enumerate(omegas):
                    fout.write(f"{omega * QTcts.HARTREEtoMEV:6.4f}        {a2f[iomega]:6.4f}\n")
            fout.close()
