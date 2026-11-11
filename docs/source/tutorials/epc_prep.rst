.. _tutorial-epc-grad-prep:

================================
EPC Gradient Matrix Preparation
================================

This tutorial prepares the real-space electron-phonon coupling (EPC)
gradients using ``prep_mode: grad-epc`` in **QTrans**. OpenMX aluminum (Al)
is the main example; the following sections explain the changes for
collinear magnetic iron (Fe) and for Al using ABACUS.

The workflow has three stages: generate unperturbed and displaced supercells,
obtain their electronic graph data, and construct the gradient matrices.
For DFT settings and matrix extraction, see :ref:`tutorial_obtain_hs` and
the official `OpenMX <https://www.openmx-square.org/>`_ and
`ABACUS <https://abacus.deepmodeling.com/en/latest/>`_ documentation.

Paths are relative to the tutorial repository root. Update installation and
pseudopotential paths in all YAML files before running. Each case includes
previously generated outputs in ``ref/`` for comparison. Complete **QTrans** YAML
configurations are shown below; replace ``/path/to/...`` with your installation
paths. Parameters not specified in these files use the program defaults.

1. OpenMX: Al as the main example
-------------------------------------

The example is in ``tutorials/Al/openmx/epc-prep/``. It contains
``Al-pri.vasp``, ``input_epc_1.yaml``, and ``input_epc_2.yaml``.

Step 1. Generate the supercell structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first configuration generates structures without constructing gradients.
The complete ``input_epc_1.yaml`` is:

.. code-block:: yaml

   basic:
     prep_mode: 'grad-epc'
     outdir: './'

   software_path:
     openmx_read_path: /path/to/read_openmx
     openmx_dftdata_dir: /path/to/DFT_DATA19
   material:
     struct_path: ./Al-pri.vasp
     nao_max: 14
     DFT_software: openmx

   grad_epc:
     epc_sc_matrix: [3, 3, 3]

Leave ``grad_epc.graph_data_sc_path`` unset in this first stage.
:ref:`struct_path <prep_struct_path>` supplies the primitive-cell structure;
this example uses POSCAR format. :ref:`nao_max <prep_nao_max>` must match the
orbital basis used for the electronic calculations.

:ref:`epc_sc_matrix <prep_epc_sc_matrix>` specifies positive odd supercell
multipliers. Without an explicit size, **QTrans** determines it from
:ref:`epc_sc_auto_length <prep_epc_sc_auto_length>`. Here we use
``[3, 3, 3]`` as a quick tutorial example.
Converge the supercell size for your system, taking account
of the orbital cutoff radii listed in :ref:`ao_define_title`.

From the tutorial repository root, run:

.. code-block:: bash

   cd tutorials/Al/openmx/epc-prep/
   QTrans_prep --config input_epc_1.yaml > grad_epc_1.out

The calculation writes ``QT.UNITCELL`` in POSCAR format. If
:ref:`std_symm_cell <prep_std_symm_cell>` is enabled, this is the standardized
primitive cell; use a consistent reference cell throughout subsequent
phonon, electronic, and EPC calculations.

The ``perturb/`` directory contains ``mat_info_sc.npy`` and four OpenMX
inputs, ``QTepc-0.dat`` through ``QTepc-3.dat``. In general, the workflow
creates :math:`3N+1` structures for :math:`N` atoms in the unit cell:
structure 0 is unperturbed, and the remaining structures displace each atom
in a representative cell along x, y, and z. The displacement amplitude is
set by :ref:`epc_sc_disp <prep_epc_sc_disp>` (default: 0.01 angstrom).

.. important::

   Review the generated DFT settings before running the supercells,
   especially the basis, pseudopotentials, k-point sampling, and SCF
   tolerance. Keep all settings identical across the displaced and
   unperturbed structures. Preserve the generated structure numbering and
   its matching ``mat_info_sc.npy``.

Step 2. Run DFT and pack the supercell graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a separate calculation folder for each structure. For this one-atom Al
cell, the following commands illustrate the sequence from ``epc-prep/``:

.. code-block:: bash

   cd perturb/
   for i in 0 1 2 3; do
       mkdir -p "$i"
       cp "QTepc-${i}.dat" "$i/openmx.dat"
       (cd "$i" && mpirun -np 8 openmx openmx.dat > openmx.out)
   done

The generated OpenMX inputs use ``System.Name QTrans``. Therefore graph
packing reads ``QTrans.scfout`` and ``QTrans.out`` in each calculation
folder, rather than the ``crystal.*`` filenames in the unit-cell tutorial.
Check that all four SCF calculations have converged.

Next, pack the supercell results into a single graph dataset.
Create ``input_graph.yaml`` in ``perturb/`` using the reference configuration
at ``ref/perturb/input_graph.yaml`` as a guide:

.. code-block:: yaml

   basic:
     prep_mode: 'graph-pack'
     outdir: './'

   software_path:
     openmx_read_path: /path/to/read_openmx

   material:
     nao_max: 14
     DFT_software: openmx

   graph_pack:
     HSout_filename: 'QTrans.scfout'
     vbout_filename: 'QTrans.out'
     data_folder: ./*/
     converge_step_ratio: 0.9

From ``perturb/``, run:

.. code-block:: bash

   mpirun -np 4 QTrans_prep --config input_graph.yaml > graph.out
   cd ..

Verify that all four folders were included. The output
``perturb/graph_data_QTrans.npz`` must contain the unperturbed structure
followed by the three displaced structures in the original order.
Missing or unconverged structures must be corrected before constructing
gradients. See :ref:`tutorial_obtain_hs` for graph-packing details.

Step 3. Construct the gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second configuration reads the supercell graphs and mapping. In
``input_epc_2.yaml``, use the following complete configuration:

.. code-block:: yaml

   basic:
     prep_mode: 'grad-epc'
     outdir: './'

   software_path:
     openmx_pp_path: /path/to/openmx_postprocess
     openmx_read_path: /path/to/read_openmx
     openmx_dftdata_dir: /path/to/DFT_DATA19

   material:
     struct_path: ./Al-pri.vasp
     nao_max: 14
     DFT_software: openmx

   grad_epc:
     graph_data_sc_path: ./perturb/graph_data_QTrans.npz
     mat_info_sc_path: ./perturb/mat_info_sc.npy

Unlike structure generation, this stage requires ``openmx_postprocess`` as
well as ``read_openmx``. Obtain these tools from the
`HamGNN project <https://github.com/QuantumLab-ZY/HamGNN/>`_.
**QTrans** invokes these tools to compute the onsite
overlap derivatives in ``dSon/``; finite differences of the electronic
Hamiltonian alone are not sufficient in a nonorthogonal atomic-orbital basis.

From ``tutorials/Al/openmx/epc-prep/``, run with a single process:

.. code-block:: bash

   QTrans_prep --config input_epc_2.yaml > grad_epc_2.out

Check for ``QTRANS JOB SUCCESS`` and compare the outputs with ``ref/``.
The resulting ``grad_mat.npy`` is the dense gradient matrix;
``grad_mat_sparse.npy`` is its sparse representation, controlled by
:ref:`grad_sparse_thr <prep_grad_sparse_thr>`.

2. OpenMX: magnetic Fe
--------------------------

The Fe example is in ``tutorials/Fe/openmx/epc-prep/``.
Follow the Al sequence with ``Fe-pri.dat`` and the supplied
``input_epc_1.yaml`` and ``input_epc_2.yaml``. The complete first configuration is:

.. code-block:: yaml

   basic:
     prep_mode: 'grad-epc'
     outdir: './'

   software_path:
     openmx_dftdata_dir: /path/to/DFT_DATA19

   material:
     struct_path: ./Fe-pri.dat
     nao_max: 14
     DFT_software: openmx
     magnetic_type: collinear

   grad_epc:
     epc_sc_matrix: [3, 3, 3]

The OpenMX structure input carries magnetic information in addition to
atomic positions. Use spin populations consistent with the converged
reference magnetic state, rather than arbitrary initial guesses. The
prepared supercells, SCF calculations, graph packing, and gradient
construction must all describe the same collinear, no-SOC system.
See the Fe discussion in :ref:`tutorial_obtain_hs`.

From the tutorial repository root, start with:

.. code-block:: bash

   cd tutorials/Fe/openmx/epc-prep/
   QTrans_prep --config input_epc_1.yaml > grad_epc_1.out

Run the four generated OpenMX supercells as for Al. When packing them,
include ``material.magnetic_type: collinear`` in ``input_graph.yaml``;
the reference file is ``ref/perturb/input_graph.yaml`` and uses
``converge_step_ratio: 0.95``. It reads ``QTrans.scfout`` and ``QTrans.out``
from ``./*/``. Check that all four magnetic calculations are included.

The complete Fe ``perturb/input_graph.yaml`` is:

.. code-block:: yaml

   basic:
     prep_mode: 'graph-pack'
     outdir: './'

   software_path:
     openmx_read_path: /path/to/read_openmx

   material:
     nao_max: 14
     DFT_software: openmx
     magnetic_type: collinear

   graph_pack:
     HSout_filename: 'QTrans.scfout'
     vbout_filename: 'QTrans.out'
     data_folder: ./*/
     converge_step_ratio: 0.95


The complete Fe ``input_epc_2.yaml`` is:

.. code-block:: yaml

   basic:
     prep_mode: 'grad-epc'
     outdir: './'

   software_path:
     openmx_pp_path: /path/to/openmx_postprocess
     openmx_read_path: /path/to/read_openmx
     openmx_dftdata_dir: /path/to/DFT_DATA19

   material:
     struct_path: ./Fe-pri.dat
     nao_max: 14
     DFT_software: openmx
     magnetic_type: collinear

   grad_epc:
     epc_sc_matrix: [3, 3, 3]
     graph_data_sc_path: ./perturb/graph_data_QTrans.npz
     mat_info_sc_path: ./perturb/mat_info_sc.npy

Then, from ``epc-prep/``, run:

.. code-block:: bash

   QTrans_prep --config input_epc_2.yaml > grad_epc_2.out

The second configuration supplies ``./perturb/graph_data_QTrans.npz`` and
``./perturb/mat_info_sc.npy``, just as for Al. Compare the dense and sparse
gradient outputs and logs with ``ref/``. Keep the collinear magnetic
setting when using these datasets in subsequent calculations.

3. ABACUS: Al
-----------------

The ABACUS example is in ``tutorials/Al/abacus/epc-prep/``. The workflow
remains structure generation, DFT and graph packing, then gradient
construction. Use the supplied ``STRU`` instead of ``Al-pri.vasp``, because
it contains the structure, species, and orbital-file information for ABACUS.

Step 1. Generate ABACUS supercells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update the executable and data-directory paths in both preparation files.
The complete ``input_epc_1.yaml`` is:

.. code-block:: yaml

   basic:
     prep_mode: 'grad-epc'
     outdir: './'

   software_path:
     abacus_pp_path: /path/to/abacus
     abacus_dftdata_dir: /path/to/SG15_ONCV
     abacus_orb_dir: /path/to/SG15-std

   material:
     nao_max: 40
     DFT_software: abacus
     struct_path: ./STRU

   grad_epc:
     epc_sc_matrix: [3, 3, 3]

From the tutorial repository root, run:

.. code-block:: bash

   cd tutorials/Al/abacus/epc-prep/
   QTrans_prep --config input_epc_1.yaml > grad_epc_1.out

The generated ``perturb/QTepc-0.dat`` through ``QTepc-3.dat`` are ABACUS
``STRU`` files despite their ``.dat`` extension. Keep the accompanying
``mat_info_sc.npy`` for the final stage.

Step 2. Run ABACUS and pack the graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Place each generated structure in its own numbered directory as ``STRU``.
Provide matching ``INPUT`` and ``KPT`` files; reference templates are in
``ref/perturb/``. Update their pseudopotential and orbital paths and retain
``basis_type lcao``, ``calculation scf``, ``suffix Al``, and
``out_mat_hs2 1``. With these files prepared in ``perturb/``, run:

.. code-block:: bash

   cd perturb/
   for i in 0 1 2 3; do
       mkdir -p "$i"
       cp "QTepc-${i}.dat" "$i/STRU"
       cp INPUT KPT "$i/"
       (cd "$i" && mpirun -np 16 abacus > abacus.out)
   done

Check convergence and the matrix outputs in each ``OUT.Al/`` directory.
The reference graph-packing configuration is ``ref/perturb/input.yaml``.
The complete ``perturb/input.yaml`` for graph packing is shown below.
Update its software paths before running:

.. code-block:: yaml

   basic:
     # prep_mode: 'grad-epc'
     prep_mode: 'graph-pack'
     outdir: './'

   software_path:
     abacus_pp_path: /path/to/abacus
     abacus_dftdata_dir: /path/to/SG15_ONCV
     abacus_orb_dir: /path/to/SG15-std

   material:
     nao_max: 40
     DFT_software: abacus
     # magnetic_type: collinear
     # struct_path: /path/to/STRU

   graph_pack:
     data_folder: ./*/
     abacus_out_folder: OUT.Al
     abacus_scf_log: running_scf.log

   grad_epc:
     epc_sc_matrix: [3, 3, 3]
     # fd_method: central

Run from ``perturb/``:

.. code-block:: bash

   mpirun -np 4 QTrans_prep --config input.yaml > pack.out
   cd ..

Verify that all four supercells were packed in the original order. Compare
with ``ref/perturb/pack.out`` and ``ref/perturb/graph_data_QTrans.npz``.

Step 3. Construct the ABACUS gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The supplied ``input_epc_2.yaml`` reads the same graph and mapping paths
as the OpenMX examples. Its complete contents are:

.. code-block:: yaml

   basic:
     prep_mode: 'grad-epc'
     outdir: './'

   software_path:
     abacus_pp_path: /path/to/abacus
     abacus_dftdata_dir: /path/to/SG15_ONCV
     abacus_orb_dir: /path/to/SG15-std

   material:
     nao_max: 40
     DFT_software: abacus
     struct_path: ./STRU

   grad_epc:
     graph_data_sc_path: ./perturb/graph_data_QTrans.npz
     mat_info_sc_path: ./perturb/mat_info_sc.npy

Keep ``DFT_software: abacus``, ``nao_max: 40``, and ``struct_path: ./STRU``.
The ``abacus_pp_path`` executable is invoked during the onsite-overlap
calculation in ``dSon/``; use an executable compatible with this step and
ensure the configured pseudopotential and orbital files are available.
This example supplies the ABACUS executable through that parameter.

Run from ``tutorials/Al/abacus/epc-prep/``:

.. code-block:: bash

   QTrans_prep --config input_epc_2.yaml > grad_epc_2.out

The outputs are ``grad_mat.npy`` and ``grad_mat_sparse.npy``. Compare the
completion log and matrices with ``ref/grad_epc_2.out`` and the reference
matrix files.

Using the prepared data
---------------------------

In the calculator configuration, set ``epc.grad_mat_path`` to the selected
gradient file and ``epc.mat_info_sc_path`` to its matching mapping. Use
``advanced.read_sparse_grad_mat: True`` with ``grad_mat_sparse.npy`` and
``False`` with ``grad_mat.npy``. The electronic unit-cell data, phonon
reference cell, basis, magnetic settings, and gradients must be consistent.

For OpenMX Al, the generated files are handed to the calculation tutorials
as follows (paths are relative to the tutorial repository root):

.. list-table:: 
   :header-rows: 1
   :widths: 40 30 30

   * - Generated file under ``tutorials/Al/openmx/``
     - Destination under the same directory
     - Calculator input key
   * - ``graph-pack/graph_data_QTrans.npz``
     - ``data/graph_data.npz``
     - ``basic.graph_data_path_uc``
   * - ``epc-prep/grad_mat.npy`` or ``epc-prep/grad_mat_sparse.npy``
     - ``data/grad_mat.npy`` or ``data/grad_mat_sparse.npy``
     - ``epc.grad_mat_path``
   * - ``epc-prep/perturb/mat_info_sc.npy``
     - ``data/mat_info_sc.npy``
     - ``epc.mat_info_sc_path``

After successfully generating the gradients, run from the tutorial
repository root:

.. code-block:: bash

   cp -i tutorials/Al/openmx/epc-prep/grad_mat.npy tutorials/Al/openmx/data/grad_mat.npy
   cp -i tutorials/Al/openmx/epc-prep/grad_mat_sparse.npy tutorials/Al/openmx/data/grad_mat_sparse.npy
   cp -i tutorials/Al/openmx/epc-prep/perturb/mat_info_sc.npy tutorials/Al/openmx/data/mat_info_sc.npy

The ``-i`` option asks before replacing supplied example data. Alternatively,
keep all generated files in a separate directory and update the paths in
the calculator YAML. Copy the matching mapping whenever the gradients are
replaced; a mapping from another supercell preparation is not interchangeable.
For OpenMX Fe or ABACUS Al, use the corresponding material/software paths
and copy the outputs of that case rather than the Al/OpenMX files.

These steps do not prepare the phonon inputs. Follow
:ref:`tutorial-phonon-prep` for ``FORCE_SETS`` and its reference structure,
and keep the phonon and EPC structures consistent. The complete calculator
YAML files in the following tutorials show the final input paths.

If using predicted Hamiltonians instead of SCF Hamiltonians, set
``grad_epc.use_prediction: True`` and provide ``grad_epc.pred_H_sc_path``
together with matching supercell graph data. A unit-cell prediction alone
cannot replace the full set of supercell predictions. See
:ref:`tutorial_obtain_hs` for the prediction workflow.

.. note::

   Run ``grad-epc`` with a single process: its main preparation and gradient
   work is performed on rank 0. The separate ``graph-pack`` step supports
   MPI distribution across supercell folders, as illustrated above. MPI
   parallelism in the DFT commands is controlled by the external DFT code.
