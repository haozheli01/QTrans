.. _tutorial-phonon-prep:

==============================
Phonon Force Data Preparation
==============================

This tutorial prepares ``FORCE_SETS`` for **QTrans** using the
finite-displacement workflow in
`Phonopy <https://phonopy.github.io/phonopy/>`_: generate displaced
supercells, calculate their atomic forces with a DFT code, and collect the
forces into a Phonopy dataset. Bulk Al examples are provided for VASP and
ABACUS.

The VASP force calculations are stored in
``tutorials/Al/openmx/phonopy/`` because they supply phonons for the OpenMX
transport examples. The electronic Hamiltonian and phonon forces do not
have to come from the same DFT code, but their structures, atom ordering,
and physical settings must be consistent.

.. important::

   **QTrans** reads ``FORCE_SETS`` using Phonopy's default VASP convention:
   **displacements in angstrom and forces in eV/angstrom**. A ``FORCE_SETS``
   file does not identify its units. Generating it successfully with Phonopy
   does not guarantee that its units are suitable for **QTrans**.

   In particular, Phonopy's ABACUS interface writes displacements in bohr
   and forces in eV/angstrom. Convert the displacement vectors before
   supplying that dataset to **QTrans**; leave the force values unchanged.

VASP force calculations
------------------------

For DFT parameter definitions and execution details, consult the
`VASP documentation <https://www.vasp.at/wiki/>`_ and the
`Phonopy VASP interface <https://phonopy.github.io/phonopy/vasp.html>`_.
The supplied ``supercells/INCAR``, ``KPOINTS``, and potential inputs provide
an example setup; converge the force accuracy for your own calculation.

Step 1. Generate displaced supercells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the tutorial repository root, enter the example directory and generate
a ``3 x 3 x 3`` supercell from the primitive-cell ``POSCAR``:

.. code-block:: bash

    cd tutorials/Al/openmx/phonopy/
    phonopy -d --dim="3 3 3"

VASP is Phonopy's default interface, so no calculator flag is needed.
The reference files include ``SPOSCAR`` (the undisplaced supercell),
``POSCAR-001`` (a displaced supercell), and ``phonopy_disp.yaml`` (the
structure and displacement information). This Al example has 27 atoms in
the supercell and one symmetry-independent displacement. Other materials
can require several ``POSCAR-xxx`` files.

Keep ``phonopy_disp.yaml`` in the working directory. For this example,
copy the displaced structure into the prepared force-calculation directory:

.. code-block:: bash

    cp POSCAR-001 supercells/POSCAR

Step 2. Calculate forces
^^^^^^^^^^^^^^^^^^^^^^^^^

Run a static, self-consistent VASP calculation in ``supercells/`` with the
displaced atomic positions fixed. Do not relax the displaced structure.
For example, using your VASP executable and MPI configuration:

.. code-block:: bash

    cd supercells/
    mpirun -np 8 vasp_std > vasp.out
    cd ..

The supplied reference ``INCAR`` uses ``ENCUT = 500`` eV,
``EDIFF = 1E-08``, ``IBRION = -1``, and ``NSW = 0``.
Adjust parallel settings and potential paths for your installation.
Confirm electronic convergence and the presence of atomic forces in
``vasprun.xml``. If several displacements are generated, calculate each in
its own directory with consistent DFT settings.

Step 3. Collect forces
^^^^^^^^^^^^^^^^^^^^^^^

From the directory containing ``phonopy_disp.yaml``, run:

.. code-block:: bash

    phonopy -f ./supercells/vasprun.xml

For several displacement directories, pass all output files in the same
order as the displaced structures. With appropriately named directories,
the usual command is:

.. code-block:: bash

    phonopy -f ./*/vasprun.xml

This generates ``FORCE_SETS``. VASP-interface displacements are already in
angstrom and forces in eV/angstrom, so no unit conversion is needed.
Reference inputs and outputs are in ``ref/``. After checking the result,
copy it to the data directory used by the Al examples:

.. code-block:: bash

    cp FORCE_SETS ../data/FORCE_SETS

ABACUS force calculations
--------------------------

For DFT parameters and execution details, consult the
`ABACUS documentation <https://abacus.deepmodeling.com/en/latest/>`_ and the
`Phonopy ABACUS interface <https://phonopy.github.io/phonopy/abacus.html>`_.
The example is in ``tutorials/Al/abacus/phonopy/``; reference structures,
force outputs, and the conversion script are in ``ref/``.

Step 1. Generate displaced supercells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the tutorial repository root, run:

.. code-block:: bash

    cd tutorials/Al/abacus/phonopy/
    phonopy --abacus -d --dim="3 3 3"

Here ``--abacus`` selects the ABACUS interface and reads the primitive-cell
``STRU``. Phonopy generates ``STRU-001`` and ``phonopy_disp.yaml`` for this
27-atom Al supercell. Copy the displaced structure into the prepared
calculation directory:

.. code-block:: bash

    cp STRU-001 supercells/STRU

Step 2. Calculate forces
^^^^^^^^^^^^^^^^^^^^^^^^^

Use a static SCF calculation (``calculation scf``) and enable force output
with ``cal_force 1`` in ``supercells/INPUT``. The supplied example uses
``basis_type lcao``, ``ecutwfc 100`` Ry, and ``scf_thr 1e-9``. Update
``pseudo_dir``, ``orbital_dir``, and the referenced potential and orbital
files for your installation. Keep the displaced atomic positions fixed.

.. code-block:: bash

    cd supercells/
    mpirun -np 8 abacus > abacus.out
    cd ..

For ``suffix Al``, the force output needed by Phonopy is
``supercells/OUT.Al/running_scf.log``. Check SCF convergence and force
output before collecting the dataset. Repeat for every displaced
supercell if more than one is generated.

Step 3. Collect forces and convert units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the directory containing the ABACUS ``phonopy_disp.yaml``, run:

.. code-block:: bash

    phonopy --abacus -f ./supercells/OUT.Al/running_scf.log

For multiple displacements, supply all corresponding ``running_scf.log``
files in displacement order. The resulting ``FORCE_SETS`` uses the ABACUS
interface convention:

.. list-table:: 
   :header-rows: 1
   :widths: 30 35 35

   * - Quantity
     - Original ABACUS-interface dataset
     - Dataset for **QTrans**
   * - Displacement
     - bohr
     - angstrom (multiply by approximately 0.529177)
   * - Force
     - eV/angstrom
     - eV/angstrom (unchanged)

The **QTrans** script ``ref/phonon_convert.py`` converts this type-1
finite-displacement dataset using the unit factor from the installed
Phonopy version:

.. code-block:: bash

    python ref/phonon_convert.py --calculator abacus --input FORCE_SETS --output FORCE_SETS_converted
    cp FORCE_SETS_converted ../data/FORCE_SETS

The script preserves the original file and refuses to overwrite an
existing output. If ``FORCE_SETS_converted`` already exists, choose a new
output filename. Convert the original ABACUS-interface file only once;
its units cannot be inferred automatically from its contents.

In the supplied reference, the first displacement changes from
``[-0.0141421356237309, 0, 0.0141421356237309]`` bohr to approximately
``[-0.007483696, 0, 0.007483696]`` angstrom. The force components remain
unchanged. Last digits can differ slightly with the physical constants
used by different Phonopy versions.

Keep the original ``FORCE_SETS`` together with the ABACUS
``phonopy_disp.yaml`` for calculations through Phonopy's ABACUS interface.
The converted copy is for **QTrans**'s VASP-unit reader; do not combine it
with ABACUS-unit displacements in a subsequent Phonopy calculation.

Use the force data in **QTrans**
---------------------------------

The Al calculation inputs read ``../data/FORCE_SETS`` together with
``../data/Al-pri.vasp`` and ``phonon.supercell_matrix`` set to a diagonal
``3 x 3 x 3`` supercell. Use the matching primitive structure in angstrom,
with the same cell orientation and atom ordering as the force calculation.
The phonon supercell size is independent of the supercell used to prepare
EPC gradients.

See :ref:`tutorial-dispersion` and :ref:`tutorial-dos` for complete
**QTrans** inputs that calculate phonon dispersions and DOS. The
:doc:`../workflow` summarizes the input conventions for other interfaces.
Check the units of the actual Phonopy interface before reusing force data
from another DFT code; do not apply the ABACUS conversion indiscriminately.
