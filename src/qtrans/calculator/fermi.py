"""Mechanically extracted QTrans methods."""

from qtrans.core.constants import QTcts
from qtrans.core.logging import log
from qtrans.numerics.distributions import fermi_weight
from qtrans.numerics.gaussian import w0gauss
from qtrans.numerics.gaussian import wgauss
import numpy as np
import sys


class _FermiMixin:
    def _get_ecbm(self, enks, icbm):
        """
        Get the energy of CBM.

        Args:
            enks (np.ndarray): (nbnd, nks) electron energys 
            icbm (int): the band index of CBM
        
        Returns:
            ecbm: The energy of CBM in Hartree.
        """
        return np.min(enks[icbm,:])

    def _get_evbm(self, enks, ivbm):
        """
        Get the energy of VBM.

        Args:
            enks (np.ndarray): (nbnd, nks) electron energys 
            icbm (int): the band index of VBM
        
        Returns:
            evbm: The energy of VBM in Hartree.
        """
        return np.max(enks[ivbm,:])

    def _sumkg(self, enks, degauss, ngauss, ene):
        # ref from PW sumkg.f90
        nbnd, nks = enks.shape
        result = 0.0
        for ik in range(nks):
            tmp = np.sum(wgauss((ene - enks[:,ik]) / degauss, ngauss))
            result = result + self.weight_k[ik] * tmp
        return result

    def _get_fermi_level_insulator(self, enks, iband_edge):
        """
        Calculate the fermi energy for a given carrier density, and then recalculate the carrier density through fermi level.
        Ref from EPW utilities.f90.

        Args:
            enks (np.ndarray): (nbnd, nks) electron energys 
            iband_edge (int): the first index of band edge in enks
        
        Returns:
            efermi: The fermi energy in Hartree
            carrier_density: The carrier density in a.u.
            band_edge: The CBM or CBM energy.
        """
        efermi = 0.0
        nbnd = enks.shape[0]
        nks = enks.shape[1]
        carrier_density = 0.0
        carrier_small_judge = QTcts.TENPM80 / (QTcts.CMtoBOHR ** 3)
        if self.is_hole:
            band_edge = self._get_evbm(enks, iband_edge)
            log("VBM = {:10.6f} eV".format(band_edge * QTcts.HARTREEtoEV), rank=self.rank)
            ks_exp = np.zeros((nbnd, nks))
            for ibnd in range(nbnd):
                for ik in range(nks):
                    arg = (enks[ibnd,ik] - band_edge) / self.temperature
                    if arg < -QTcts.EXP_MAXARG:
                        ks_exp[ibnd,ik] = 0.0
                    else:
                        ks_exp[ibnd,ik] = np.exp(arg)
            eup = QTcts.TENPM160
            elw = QTcts.TENPP40 # 1.0
            converged = False
            for _ in range(self.fermi_maxiter):
                ef = np.sqrt(eup) * np.sqrt(elw)
                hole_density = 0.0
                for ibnd in range(iband_edge+1):
                    for ik in range(nks):
                        if ks_exp[ibnd,ik] * ef > QTcts.TENPP60:
                            fnk = 0.0
                        else:
                            fnk = 1.0 / (ks_exp[ibnd,ik] * ef + 1.0)
                        hole_density += (1.0 - fnk) * self.weight_k[ik]
                hole_density *= self.inv_cell
                if np.abs(hole_density) < carrier_small_judge:
                    rel_err = -QTcts.TENPP3
                else:
                    rel_err = (hole_density - np.abs(self.ncarrier)) / hole_density
                if np.abs(rel_err) < QTcts.TENPM5:
                    efermi = band_edge - (np.log(ef) * self.temperature)
                    converged = True
                    break
                elif rel_err > QTcts.TENPM5:
                    elw = ef
                else:
                    eup = ef
            for ibnd in range(iband_edge+1):
                for ik in range(nks):
                    fnk = fermi_weight(enks[ibnd,ik] - efermi, self.temperature)
                    carrier_density += (1.0 - fnk) * self.weight_k[ik]
        else:
            band_edge = self._get_ecbm(enks, iband_edge)
            log("CBM = {:10.6f} eV".format(band_edge * QTcts.HARTREEtoEV), rank=self.rank)
            ks_expcb = np.zeros((nbnd, nks))
            for ibnd in range(nbnd):
                for ik in range(nks):
                    arg = (enks[ibnd,ik] - band_edge) / self.temperature
                    if arg > QTcts.EXP_MAXARG:
                        ks_expcb[ibnd,ik] = QTcts.TENPP200
                    else:
                        ks_expcb[ibnd,ik] = np.exp(arg)
            eup = QTcts.TENPM40 # 1.0
            elw = QTcts.TENPP80
            converged = False
            for _ in range(self.fermi_maxiter):
                ef = np.sqrt(eup) * np.sqrt(elw)
                electron_density = 0.0
                for ibnd in range(iband_edge, nbnd):
                    for ik in range(nks):
                        if ks_expcb[ibnd,ik] * ef > QTcts.TENPP60:
                            fnk = 0.0
                        else:
                            fnk = 1.0 / (ks_expcb[ibnd,ik] * ef + 1.0)
                        electron_density += fnk * self.weight_k[ik]
                electron_density *= self.inv_cell
                if np.abs(electron_density) < carrier_small_judge:
                    rel_err = QTcts.TENPP3
                else:
                    rel_err = (electron_density - np.abs(self.ncarrier)) / electron_density
                if np.abs(rel_err) < QTcts.TENPM5:
                    efermi = band_edge - (np.log(ef) * self.temperature)
                    converged = True
                    break
                elif rel_err > QTcts.TENPM5:
                    eup = ef
                else:
                    elw = ef
            for ibnd in range(iband_edge, nbnd):
                for ik in range(nks):
                    fnk = fermi_weight(enks[ibnd,ik] - efermi, self.temperature)
                    carrier_density += fnk * self.weight_k[ik]
        if not converged:
            raise RuntimeError("The insulator fermi level cannot converge.", '6002')
        return efermi, carrier_density, band_edge

    def _get_fermi_level_metal(self, enks, degauss, ngauss, nelec):
        # ref from PW efermig.f90
        elw = min(1.0E8, np.min(enks))
        eup = max(-1.0E8, np.max(enks))
        eup = eup + 2 * degauss
        elw = elw - 2 * degauss
        sumkup = self._sumkg(enks, degauss, ngauss, eup)
        sumklw = self._sumkg(enks, degauss, ngauss, elw)
        if (sumkup - nelec) < -QTcts.TENPM10 or (sumklw - nelec) > QTcts.TENPM10:
            raise RuntimeError("Cannot bracket Ef.", '6002')
        ef = 0.0
        converged = False
        for _ in range(self.fermi_maxiter):
            ef = (eup + elw) / 2.0
            sumkmid = self._sumkg(enks, degauss, ngauss, ef)
            if np.abs(sumkmid - nelec) < QTcts.TENPM10:
                converged = True
                break
            elif (sumkmid - nelec) < -QTcts.TENPM10:
                elw = ef
            else:
                eup = ef
        if not converged:
            raise RuntimeError("The metal fermi level cannot converge.", '6002')
        return ef

    def _get_ef_dos(self, enks, efermi, inv_degauss, ngauss):
        dos = 0.0
        for _, ekks in enumerate(enks):
            for ik, enk in enumerate(ekks):
                delta_f3 = w0gauss((enk - efermi) * inv_degauss, ngauss=ngauss) * inv_degauss
                dos = dos + delta_f3 * self.weight_k[ik]
        return dos

    def _print_mesh_mpi_check(self, nps_fbz:int, nps_ibz:int, nps_useful:int, nfps:int, reduced_type:str='k'):
        if self.rank == 0:
            print()
            print("-"*100)
            print("|                                  BZ POINT DISTRIBUTION CHECK (PARALLEL)                          |")
            print("-"*100)
            if reduced_type == 'k':
                print(f"   Number of FBZ k-points              : {nps_fbz}")
                print(f"   Number of IBZ k-points              : {nps_ibz}")
                print(f"   Number of contributing IBZ k-points : {nps_useful}")
                print(f"   Number of FBZ q-points              : {nfps}")
            elif reduced_type == 'q':
                print(f"   Number of FBZ q-points              : {nps_fbz}")
                print(f"   Number of IBZ q-points              : {nps_ibz}")
                print(f"   Number of contributing IBZ q-points : {nps_useful}")
                print(f"   Number of FBZ k-points              : {nfps}")   
            else:
                raise RuntimeError(f"Error(7003): Unsupported reducible mesh type: {reduced_type}.")
            print(f"   Number of MPI ranks being used      : {self.rank_size}\n")
        sys.stdout.flush()
