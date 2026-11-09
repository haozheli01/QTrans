.. _tutorial_obtain_hs:

==================================================
Generation of Hamiltonian and Overlap Matrices
==================================================

This tutorial explains how to prepare electronic data for **QTrans** using the
aluminum (Al) examples. **QTrans** stores the structure, orbital basis, Hamiltonian,
and overlap matrices in graph data, normally ``graph_data.npz``.

Paths below are relative to the tutorial repository root. Run graph packing
from the relevant ``graph-pack/`` directory so that relative paths in the
YAML configuration resolve correctly. Each example includes a ``ref/``
directory with previously generated inputs and outputs for comparison.


Based on DFT Calculations
-------------------------

1. From ABACUS
^^^^^^^^^^^^^^

For detailed DFT parameter settings and instructions on using ABACUS, refer
to the `official ABACUS documentation <https://abacus.deepmodeling.com/en/latest/>`_.
The steps below focus on generating and converting the outputs needed by
**QTrans**.

The example is located in ``tutorials/Al/abacus/graph-pack/``. Reference
outputs are provided in ``tutorials/Al/abacus/graph-pack/ref/``.

Step 1. Prepare the ABACUS input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DFT/`` directory contains ``INPUT``, ``STRU``, and ``KPT`` for an
LCAO self-consistent calculation of aluminum. Before running:

* Update ``pseudo_dir`` and ``orbital_dir`` in ``INPUT`` to your local
  pseudopotential and numerical-orbital directories.
* Ensure that these directories contain ``Al_ONCV_PBE-1.0.upf`` and
  ``Al_gga_7au_100Ry_4s4p1d.orb``, respectively, as specified in ``STRU``.
* Make the ``abacus`` executable available in your shell. If using the
  supplied ``sub_abacus`` submission script, adapt its scheduler settings,
  modules, and executable path to your cluster.

The key settings in ``INPUT`` are:

.. code-block:: text

   suffix          Al
   basis_type      lcao
   calculation     scf
   out_mat_hs2     1

``out_mat_hs2`` enables the real-space Hamiltonian and overlap matrix output
required for graph packing. With ``suffix`` set to ``Al``, ABACUS writes
these files to ``OUT.Al/``. The supplied input uses an energy cutoff of
150 Ry, ``scf_thr: 1e-9``, 10 bands, Gaussian smearing with
``smearing_sigma: 0.01``, and a Gamma-centered ``10 x 10 x 10`` k-point mesh.
These settings describe this example; check convergence for your own system.

Step 2. Run the self-consistent calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, run:

.. code-block:: bash

   cd tutorials/Al/abacus/graph-pack/DFT/
   mpirun -np 16 abacus > abacus.out
   cd ..

Check SCF convergence in ``DFT/OUT.Al/running_scf.log``. Graph packing
requires the original ``DFT/STRU`` and the following output files:

.. code-block:: text

   DFT/OUT.Al/
     running_scf.log
     data-HR-sparse_SPIN0.csr
     data-SR-sparse_SPIN0.csr

Step 3. Configure graph packing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``tutorials/Al/abacus/graph-pack/``, configure ``input_graph.yaml`` as
follows, replacing the data-directory paths with your local paths:

.. code-block:: yaml

   basic:
     prep_mode: graph-pack
     outdir: ./

   software_path:
     abacus_dftdata_dir: /path/to/SG15_ONCV
     abacus_orb_dir: /path/to/SG15-std

   material:
     nao_max: 40
     DFT_software: abacus

   graph_pack:
     data_folder: ./DFT/
     abacus_out_folder: OUT.Al
     abacus_scf_log: running_scf.log

The supplied ``data_folder: ./DFT`` is equivalent to ``./DFT/`` above.
``abacus_out_folder`` is relative to each selected calculation folder and
must match the ABACUS output directory. ``abacus_scf_log`` identifies the log inside that directory.

``nao_max: 40`` selects **QTrans**'s ABACUS orbital definition for this example;
it is not the number of orbitals on each atom. Keep this definition
consistent with the numerical orbitals used in the DFT calculation.
**QTrans** reads the ABACUS CSR matrices directly.

Step 4. Pack and inspect the result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run from ``tutorials/Al/abacus/graph-pack/``:

.. code-block:: bash

   QTrans_prep --config input_graph.yaml > graph.out

The resulting ``graph_data_QTrans.npz`` is written to ``basic.outdir`` and
can be supplied through ``basic.graph_data_path_uc`` for unit-cell calculations.
Compare with ``ref/graph.out`` and ``ref/graph_data_QTrans.npz`` in this
``graph-pack/`` directory. The reference directory
also includes the DFT inputs and the required ``OUT.Al`` files.

2. From OpenMX
^^^^^^^^^^^^^^

For detailed DFT parameter settings and instructions on using OpenMX, refer
to the manuals on the `official OpenMX website <https://www.openmx-square.org/>`_.
The steps below focus on generating and converting the outputs needed by
**QTrans**.

The example is located in ``tutorials/Al/openmx/graph-pack/``. Its ``ref/``
directory contains outputs for reference.

Step 1. Prepare the executables and input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install OpenMX and the ``read_openmx`` matrix-export executable available
from the `HamGNN repository <https://github.com/QuantumLab-ZY/HamGNN/>`_.
**QTrans** invokes ``read_openmx`` to extract Hamiltonian and overlap matrices
from OpenMX binary outputs into a temporary ``HS.json`` file.

Before running the example:

* Update ``DATA.PATH`` in ``DFT/openmx_Al.dat`` to your OpenMX pseudopotential
  and atomic-orbital data directory.
* Update ``software_path.openmx_read_path`` in ``input_graph.yaml`` to an
  executable ``read_openmx``. The supplied value, ``./read_openmx``, assumes
  the executable is in the ``graph-pack`` directory; it is not included there.
* Make sure ``openmx`` is available in your shell, or use its full path in
  the command below.

The supplied input uses the GGA-PBE functional, the ``Al7.0-s2p2d1`` orbital
basis, and the ``Al_PBE19`` pseudopotential without spin polarization.
It sets ``scf.energycutoff`` to 200 Ry, ``scf.Kgrid`` to ``18 18 18``,
``scf.ElectronicTemperature`` to 300 K, and ``scf.criterion`` to
``1.0e-12`` Hartree. Keep ``HS.fileout on`` enabled to write the binary
Hamiltonian and overlap data required by **QTrans**. Check convergence
before applying these settings to another system.

Step 2. Run the self-consistent calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, run:

.. code-block:: bash

   cd tutorials/Al/openmx/graph-pack/DFT/
   mpirun -np 8 openmx openmx_Al.dat > openmx_scf.out
   cd ..

The input sets ``System.Name`` to ``crystal``, so the files needed for graph
packing are ``DFT/crystal.scfout`` and ``DFT/crystal.out``.
``openmx_scf.out`` is the redirected terminal output; it is distinct from
``crystal.out``, which **QTrans** reads for structure and convergence information.
Check that the SCF calculation converged before continuing.

Step 3. Configure graph packing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplied ``input_graph.yaml`` uses:

.. code-block:: yaml

   basic:
     prep_mode: graph-pack
     outdir: ./

   software_path:
     openmx_read_path: /path/to/read_openmx

   material:
     nao_max: 14
     DFT_software: openmx

   graph_pack:
     HSout_filename: crystal.scfout
     vbout_filename: crystal.out
     data_folder: ./DFT/
     converge_step_ratio: 0.9

Replace the executable path with your installation path.
:ref:`nao_max <prep_nao_max>` selects the orbital definition used to pack the
data and must match the electronic-structure setup. This DFT example uses
``14`` for the aluminum basis used here.

:ref:`HSout_filename <prep_HSout_filename>` identifies the binary matrix file,
and :ref:`vbout_filename <prep_vbout_filename>` identifies the OpenMX text
output. :ref:`data_folder <prep_data_folder>` selects the calculation folders.
For multiple calculations, it can be a glob pattern such as ``./DFT/*/``;
folders missing required files are reported and skipped.

The parameter :ref:`converge_step_ratio <prep_converge_step_ratio>` filters
calculations using their SCF iteration count relative to ``scf.maxIter``.
The example uses ``0.9``. This filter supplements checking the SCF convergence
and does not set the convergence tolerance of OpenMX.

Step 4. Pack and inspect the result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``tutorials/Al/openmx/graph-pack/``, run:

.. code-block:: bash

   QTrans_prep --config input_graph.yaml > graph_pack.out

The output is ``graph_data_QTrans.npz`` in ``basic.outdir``. Check that
``graph_pack.out`` reports one useful folder and successful completion.
Reference files are available at:

.. code-block:: text

   ref/
     DFT/crystal.scfout
     DFT/crystal.out
     DFT/openmx_scf.out
     graph_data_QTrans.npz
     graph_pack.out

Reference logs may contain paths from the original calculation environment;
use the paths shown above for the current directory layout.

For unit-cell calculations, provide the generated dataset through
``basic.graph_data_path_uc`` in the calculator configuration. To prepare EPC
from displaced-supercell data, continue with :ref:`tutorial-epc-grad-prep`.

.. warning::

   **QTrans** creates and removes ``HS.json`` in each selected calculation
   folder during matrix extraction. Use folders without an existing
   ``HS.json`` and avoid concurrent graph-packing jobs in the same folders.

Based on HamGNN Prediction with OpenMX
--------------------------------------------

We can also predict the Hamiltonian using **HamGNN**.
The prediction example is located in
``tutorials/Al/openmx/predict_HamGNN/``. It uses the universal Hamiltonian model
:cite:`zhongZhongUniversalMachineLearning2024`, available from
`Zenodo <https://doi.org/10.5281/zenodo.10827117>`_.

Prepare graph data for prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prediction route first constructs graph data containing the structure,
overlap matrices, and non-self-consistent Hamiltonian contribution. **HamGNN**
then predicts the Hamiltonian using this graph data and the universal model.

From the repository root, enter the DFT directory:

.. code-block:: bash

   cd tutorials/Al/openmx/predict_HamGNN/DFT/

Update ``DATA.PATH`` in ``openmx_Al.dat`` and ensure that
``openmx_postprocess`` and ``read_openmx`` are compatible with your OpenMX
installation. Obtain these tools from the
`HamGNN project <https://github.com/QuantumLab-ZY/HamGNN/>`_ and place the
executables in this DFT directory, or update the command and configuration
to use their installation paths.

Run the preprocessing command:

.. code-block:: bash

   mpirun -np 1 ./openmx_postprocess openmx_Al.dat > openmx_nonscf.out

This step produces ``overlap.scfout``. The graph-packing configuration uses
``DFT_software: openmx`` and ``nao_max: 26`` to match the universal model's
orbital definition. Configure the relevant block as follows:

.. code-block:: yaml

   graph_pack:
     HSout_filename: null
     vbout_filename: null
     nonscf_filename: overlap.scfout
     stdin_filename: openmx_Al.dat
     data_folder: ./

Here, ``stdin_filename`` supplies the structure and basis information from
the OpenMX input. Use YAML ``null`` for disabled filenames, and leave
``vbout_filename`` unset or null to select the non-SCF graph-packing route.
Set ``software_path.openmx_read_path`` to the local ``./read_openmx`` or
another compatible executable.

From this DFT directory, pack the data and copy it to the filename expected
by HamGNN:

.. code-block:: bash

   QTrans_prep --config input_graph.yaml > graph_pack.out
   cp graph_data_QTrans.npz graph_data.npz

**HamGNN** expects ``graph_data.npz`` in the directory specified by
``dataset_params.graph_data_path``. The supplied ``predict/HamGNN_config.yaml``
sets this path to ``../DFT/``. Previously generated preprocessing outputs
are available in ``predict_HamGNN/ref/DFT/``.

Prediction configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Before running inference, update ``setup.checkpoint_path`` in
``predict/HamGNN_config.yaml`` to the downloaded model and use a compatible
**HamGNN** environment. The example selects ``setup.stage: test``,
``output_nets.HamGNN_out.nao_max: 26``, and ``add_H0: True``. Keep the model,
orbital definition, graph data, and non-self-consistent contribution consistent.

Run the prediction from the ``predict`` directory:

.. code-block:: bash

   cd ../predict/
   HamGNN2.0 --config HamGNN_config.yaml > HamGNN_test.out

The predicted Hamiltonian is written to
``version_0/prediction_hamiltonian.npy`` for the reference run. Reference
outputs are provided in ``predict_HamGNN/ref/predict/``, including
``HamGNN_test.out`` and the prediction file. The output version directory
may change for subsequent runs.

Use the predicted Hamiltonian together with the corresponding graph data;
the structures, ordering, and orbital basis must match. For EPC gradient
preparation, set ``grad_epc.use_prediction: True`` and supply the prediction
through ``grad_epc.pred_H_sc_path`` together with matching supercell graph
data and mapping, as described in :ref:`tutorial-epc-grad-prep`.

For HamGNN installation and inference details, consult the
`HamGNN documentation <https://github.com/QuantumLab-ZY/HamGNN/>`_.


.. note::

   The ``graph-pack`` mode of ``QTrans_prep`` supports MPI parallel execution
   across multiple calculation folders, including displaced supercells.
   Each rank builds the graphs for its assigned folders; rank 0 gathers,
   sorts, and saves all graphs in a single output file. Each rank may process
   one or more folders, depending on the number of folders and MPI ranks.

   For multiple folders selected by ``graph_pack.data_folder``, you can run:

   .. code-block:: bash

      mpirun -np 4 QTrans_prep --config input_graph.yaml > graph_pack.out

   A single folder is processed by one rank, so the single-folder examples
   above use a single process. Gradient matrix generation with ``grad-epc``
   is performed on rank 0 and does not benefit from MPI parallel execution.
