"""QTrans data.valence helpers."""

from pymatgen.core.periodic_table import Element
import numpy as np


num_valence_openmx = {
    Element['H'].Z:  1,   Element['He'].Z: 2,   Element['Li'].Z: 3,   Element['Be'].Z: 2,   Element['B'].Z:  3,
    Element['C'].Z:  4,   Element['N'].Z:  5,   Element['O'].Z:  6,   Element['F'].Z:  7,   Element['Ne'].Z: 8,
    Element['Na'].Z: 9,   Element['Mg'].Z: 8,   Element['Al'].Z: 3,   Element['Si'].Z: 4,   Element['P'].Z:  5,
    Element['S'].Z:  6,   Element['Cl'].Z: 7,   Element['Ar'].Z: 8,   Element['K'].Z:  9,   Element['Ca'].Z: 10,
    Element['Sc'].Z: 11,  Element['Ti'].Z: 12,  Element['V'].Z:  13,  Element['Cr'].Z: 14,  Element['Mn'].Z: 15,
    Element['Fe'].Z: 16,  Element['Co'].Z: 17,  Element['Ni'].Z: 18,  Element['Cu'].Z: 19,  Element['Zn'].Z: 20,
    Element['Ga'].Z: 13,  Element['Ge'].Z: 4,   Element['As'].Z: 15,  Element['Se'].Z: 6,   Element['Br'].Z: 7,
    Element['Kr'].Z: 8,   Element['Rb'].Z: 9,   Element['Sr'].Z: 10,  Element['Y'].Z:  11,  Element['Zr'].Z: 12,
    Element['Nb'].Z: 13,  Element['Mo'].Z: 14,  Element['Tc'].Z: 15,  Element['Ru'].Z: 14,  Element['Rh'].Z: 15,
    Element['Pd'].Z: 16,  Element['Ag'].Z: 17,  Element['Cd'].Z: 12,  Element['In'].Z: 13,  Element['Sn'].Z: 14,
    Element['Sb'].Z: 15,  Element['Te'].Z: 16,  Element['I'].Z:  7,   Element['Xe'].Z: 8,   Element['Cs'].Z: 9,
    Element['Ba'].Z: 10,  Element['La'].Z: 11,  Element['Ce'].Z: 12,  Element['Pr'].Z: 13,  Element['Nd'].Z: 14,
    Element['Pm'].Z: 15,  Element['Sm'].Z: 16,  Element['Dy'].Z: 20,  Element['Ho'].Z: 21,  Element['Lu'].Z: 11,
    Element['Hf'].Z: 12,  Element['Ta'].Z: 13,  Element['W'].Z:  12,  Element['Re'].Z: 15,  Element['Os'].Z: 14,
    Element['Ir'].Z: 15,  Element['Pt'].Z: 16,  Element['Au'].Z: 17,  Element['Hg'].Z: 18,  Element['Tl'].Z: 19,
    Element['Pb'].Z: 14,  Element['Bi'].Z: 15
}
num_val_openmx = np.zeros((99,), dtype=int)
for k in num_valence_openmx.keys():
    num_val_openmx[k] = num_valence_openmx[k]

num_valence_siesta = {
    1:1,2:2,
    3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
    11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,
    19:1,20:2,22:12,31:3,33:5,Element['Mo'].Z:6,
}
num_val_siesta = np.zeros((99,), dtype=int)
for k in num_valence_siesta.keys():
    num_val_siesta[k] = num_valence_siesta[k]

num_valence_abacus = {
    Element['H'].Z:  1,   Element['He'].Z: 2,   Element['Li'].Z: 3,   Element['Be'].Z: 4,   Element['B'].Z:  3,
    Element['C'].Z:  4,   Element['N'].Z:  5,   Element['O'].Z:  6,   Element['F'].Z:  7,   Element['Ne'].Z: 8,
    Element['Na'].Z: 9,   Element['Mg'].Z: 10,  Element['Al'].Z: 11,  Element['Si'].Z: 4,   Element['P'].Z:  5,
    Element['S'].Z:  6,   Element['Cl'].Z: 7,   Element['Ar'].Z: 8,   Element['K'].Z:  9,   Element['Ca'].Z: 10,
    Element['Sc'].Z: 11,  Element['Ti'].Z: 12,  Element['V'].Z:  13,  Element['Cr'].Z: 14,  Element['Mn'].Z: 15,
    Element['Fe'].Z: 16,  Element['Co'].Z: 17,  Element['Ni'].Z: 18,  Element['Cu'].Z: 19,  Element['Zn'].Z: 20,
    Element['Ga'].Z: 13,  Element['Ge'].Z: 14,  Element['As'].Z: 5,   Element['Se'].Z: 6,   Element['Br'].Z: 7,
    Element['Kr'].Z: 8,   Element['Rb'].Z: 9,   Element['Sr'].Z: 10,  Element['Y'].Z:  11,  Element['Zr'].Z: 12,
    Element['Nb'].Z: 13,  Element['Mo'].Z: 14,  Element['Tc'].Z: 15,  Element['Ru'].Z: 16,  Element['Rh'].Z: 17,
    Element['Pd'].Z: 18,  Element['Ag'].Z: 19,  Element['Cd'].Z: 20,  Element['In'].Z: 13,  Element['Sn'].Z: 14,
    Element['Sb'].Z: 15,  Element['Te'].Z: 16,  Element['I'].Z:  17,  Element['Xe'].Z: 18,  Element['Cs'].Z: 9,
    Element['Ba'].Z: 10,  Element['La'].Z: 11,  Element['Hf'].Z: 26,  Element['Ta'].Z: 27,  Element['W'].Z:  28,  Element['Re'].Z: 15,
    Element['Os'].Z: 16,  Element['Ir'].Z: 17,  Element['Pt'].Z: 18,  Element['Au'].Z: 19,  Element['Hg'].Z: 20,
    Element['Tl'].Z: 13,  Element['Pb'].Z: 14,  Element['Bi'].Z: 15,
}
num_val_abacus = np.zeros((99,), dtype=int)
for k in num_valence_abacus.keys():
    num_val_abacus[k] = num_valence_abacus[k]
