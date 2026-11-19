.. _valence_electrons_title:

=========================
Valence Electron Counts
=========================

This page lists the element-specific valence electron counts stored in
**QTrans** for each electronic-structure interface. The values follow the
current definitions in ``src/qtrans/data/valence.py``.

* :ref:`OpenMX <valence_electrons_openmx>`
* :ref:`SIESTA <valence_electrons_siesta>`
* :ref:`ABACUS <valence_electrons_abacus>`

**QTrans** selects the table using ``basic.Ham_type`` and sums the counts
for all atoms to determine the reference number of electrons in the unit
cell. These counts are distinct from the number of atomic orbitals and from
an element's chemical valency. See :ref:`ao_define_title` for the orbital
basis definitions. ``Ham_type: honpas`` uses the SIESTA table.

.. important::

   Valence electron counts depend on the pseudopotential, including whether
   semicore states are treated as valence states. These tables document the
   defaults implemented in **QTrans**, rather than universal values for all
   pseudopotentials used by a given software package. Check consistency
   with the pseudopotentials used to generate your electronic data.
   Elements absent from a table have no explicit valence-count definition
   for that interface; their omission does not imply zero valence electrons.

.. _valence_electrons_openmx:

OpenMX
------

.. list-table::
   :header-rows: 1
   :stub-columns: 0
   :class: custom-table center-align-text
   :name: valence-openmx-table

   * - Element
     - Atomic Number
     - Valence Electrons per Atom
   * - H
     - 1
     - 1
   * - He
     - 2
     - 2
   * - Li
     - 3
     - 3
   * - Be
     - 4
     - 2
   * - B
     - 5
     - 3
   * - C
     - 6
     - 4
   * - N
     - 7
     - 5
   * - O
     - 8
     - 6
   * - F
     - 9
     - 7
   * - Ne
     - 10
     - 8
   * - Na
     - 11
     - 9
   * - Mg
     - 12
     - 8
   * - Al
     - 13
     - 3
   * - Si
     - 14
     - 4
   * - P
     - 15
     - 5
   * - S
     - 16
     - 6
   * - Cl
     - 17
     - 7
   * - Ar
     - 18
     - 8
   * - K
     - 19
     - 9
   * - Ca
     - 20
     - 10
   * - Sc
     - 21
     - 11
   * - Ti
     - 22
     - 12
   * - V
     - 23
     - 13
   * - Cr
     - 24
     - 14
   * - Mn
     - 25
     - 15
   * - Fe
     - 26
     - 16
   * - Co
     - 27
     - 17
   * - Ni
     - 28
     - 18
   * - Cu
     - 29
     - 19
   * - Zn
     - 30
     - 20
   * - Ga
     - 31
     - 13
   * - Ge
     - 32
     - 4
   * - As
     - 33
     - 15
   * - Se
     - 34
     - 6
   * - Br
     - 35
     - 7
   * - Kr
     - 36
     - 8
   * - Rb
     - 37
     - 9
   * - Sr
     - 38
     - 10
   * - Y
     - 39
     - 11
   * - Zr
     - 40
     - 12
   * - Nb
     - 41
     - 13
   * - Mo
     - 42
     - 14
   * - Tc
     - 43
     - 15
   * - Ru
     - 44
     - 14
   * - Rh
     - 45
     - 15
   * - Pd
     - 46
     - 16
   * - Ag
     - 47
     - 17
   * - Cd
     - 48
     - 12
   * - In
     - 49
     - 13
   * - Sn
     - 50
     - 14
   * - Sb
     - 51
     - 15
   * - Te
     - 52
     - 16
   * - I
     - 53
     - 7
   * - Xe
     - 54
     - 8
   * - Cs
     - 55
     - 9
   * - Ba
     - 56
     - 10
   * - La
     - 57
     - 11
   * - Ce
     - 58
     - 12
   * - Pr
     - 59
     - 13
   * - Nd
     - 60
     - 14
   * - Pm
     - 61
     - 15
   * - Sm
     - 62
     - 16
   * - Dy
     - 66
     - 20
   * - Ho
     - 67
     - 21
   * - Lu
     - 71
     - 11
   * - Hf
     - 72
     - 12
   * - Ta
     - 73
     - 13
   * - W
     - 74
     - 12
   * - Re
     - 75
     - 15
   * - Os
     - 76
     - 14
   * - Ir
     - 77
     - 15
   * - Pt
     - 78
     - 16
   * - Au
     - 79
     - 17
   * - Hg
     - 80
     - 18
   * - Tl
     - 81
     - 19
   * - Pb
     - 82
     - 14
   * - Bi
     - 83
     - 15

:ref:`Back to top <valence_electrons_title>`

.. _valence_electrons_siesta:

SIESTA
------

.. list-table::
   :header-rows: 1
   :stub-columns: 0
   :class: custom-table center-align-text
   :name: valence-siesta-table

   * - Element
     - Atomic Number
     - Valence Electrons per Atom
   * - H
     - 1
     - 1
   * - He
     - 2
     - 2
   * - Li
     - 3
     - 1
   * - Be
     - 4
     - 2
   * - B
     - 5
     - 3
   * - C
     - 6
     - 4
   * - N
     - 7
     - 5
   * - O
     - 8
     - 6
   * - F
     - 9
     - 7
   * - Ne
     - 10
     - 8
   * - Na
     - 11
     - 1
   * - Mg
     - 12
     - 2
   * - Al
     - 13
     - 3
   * - Si
     - 14
     - 4
   * - P
     - 15
     - 5
   * - S
     - 16
     - 6
   * - Cl
     - 17
     - 7
   * - Ar
     - 18
     - 8
   * - K
     - 19
     - 1
   * - Ca
     - 20
     - 2
   * - Ti
     - 22
     - 12
   * - Ga
     - 31
     - 3
   * - As
     - 33
     - 5
   * - Mo
     - 42
     - 6

:ref:`Back to top <valence_electrons_title>`

.. _valence_electrons_abacus:

ABACUS
------

.. list-table::
   :header-rows: 1
   :stub-columns: 0
   :class: custom-table center-align-text
   :name: valence-abacus-table

   * - Element
     - Atomic Number
     - Valence Electrons per Atom
   * - H
     - 1
     - 1
   * - He
     - 2
     - 2
   * - Li
     - 3
     - 3
   * - Be
     - 4
     - 4
   * - B
     - 5
     - 3
   * - C
     - 6
     - 4
   * - N
     - 7
     - 5
   * - O
     - 8
     - 6
   * - F
     - 9
     - 7
   * - Ne
     - 10
     - 8
   * - Na
     - 11
     - 9
   * - Mg
     - 12
     - 10
   * - Al
     - 13
     - 11
   * - Si
     - 14
     - 4
   * - P
     - 15
     - 5
   * - S
     - 16
     - 6
   * - Cl
     - 17
     - 7
   * - Ar
     - 18
     - 8
   * - K
     - 19
     - 9
   * - Ca
     - 20
     - 10
   * - Sc
     - 21
     - 11
   * - Ti
     - 22
     - 12
   * - V
     - 23
     - 13
   * - Cr
     - 24
     - 14
   * - Mn
     - 25
     - 15
   * - Fe
     - 26
     - 16
   * - Co
     - 27
     - 17
   * - Ni
     - 28
     - 18
   * - Cu
     - 29
     - 19
   * - Zn
     - 30
     - 20
   * - Ga
     - 31
     - 13
   * - Ge
     - 32
     - 14
   * - As
     - 33
     - 5
   * - Se
     - 34
     - 6
   * - Br
     - 35
     - 7
   * - Kr
     - 36
     - 8
   * - Rb
     - 37
     - 9
   * - Sr
     - 38
     - 10
   * - Y
     - 39
     - 11
   * - Zr
     - 40
     - 12
   * - Nb
     - 41
     - 13
   * - Mo
     - 42
     - 14
   * - Tc
     - 43
     - 15
   * - Ru
     - 44
     - 16
   * - Rh
     - 45
     - 17
   * - Pd
     - 46
     - 18
   * - Ag
     - 47
     - 19
   * - Cd
     - 48
     - 20
   * - In
     - 49
     - 13
   * - Sn
     - 50
     - 14
   * - Sb
     - 51
     - 15
   * - Te
     - 52
     - 16
   * - I
     - 53
     - 17
   * - Xe
     - 54
     - 18
   * - Cs
     - 55
     - 9
   * - Ba
     - 56
     - 10
   * - La
     - 57
     - 11
   * - Hf
     - 72
     - 26
   * - Ta
     - 73
     - 27
   * - W
     - 74
     - 28
   * - Re
     - 75
     - 15
   * - Os
     - 76
     - 16
   * - Ir
     - 77
     - 17
   * - Pt
     - 78
     - 18
   * - Au
     - 79
     - 19
   * - Hg
     - 80
     - 20
   * - Tl
     - 81
     - 13
   * - Pb
     - 82
     - 14
   * - Bi
     - 83
     - 15

:ref:`Back to top <valence_electrons_title>`

