.. _tutorial-epc-grad-prep:

================================
EPC Gradient Matrix Preparation
================================

In this section we show how to prepare the EPC gradient matrix in QTrans via the ``grad-epc`` mode of **QTrans_prep**, 
using aluminum and iron as examples.

.. warning::
    The ``grad-epc`` mode of ``QTrans_prep`` runs in serial.


EPC Gradient Matrix of Al
------------------------------

This tutorial is located in ``tutorials/Al/epc_prep/``. 
Reference outputs are provided in ``tutorials/Al/epc_prep/ref/`` , 
which can be used to compare with your own results to verify that **QTrans** is running correctly.

1. Generation of Perturbed Supercell Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**QTrans** computes the electron-phonon coupling strength based on the finite-displacement method.
As a first step, we need to obtain the structures of the unperturbed and perturbed supercells.

.. code-block:: none

    [user]$ cd tutorials/Al/epc_prep/ && ls
    Al-pri.vasp  input_epc_1.yaml  input_epc_2.yaml

Firstly, we need to prepare a structure file for aluminum (Al) in a format supported by ``QTrans_prep``.
The ``Al-pri.vasp`` is the structure of aluminum (Al) primitive cell that in POSCAR format.
However, **QTrans** support severl other formats, which can be found in :ref:`struct_path <prep_struct_path>`.
Next, we configure the ``input_epc_1.yaml`` file accordingly.
The paths specified in ``software_path`` block depend on the locations of your DFT softwares.
And you can set :ref:`nao_max <prep_nao_max>` and :ref:`DFT_software <prep_DFT_software>` according to your own DFT calculations.
The size of the supercells can be specified using :ref:`epc_sc_matrix <prep_epc_sc_matrix>` 
in the ``grad_epc`` block and must be positive odd integers.
If not set, **QTrans** would automatically determine the size based on the parameter
to the parameter :ref:`epc_sc_auto_length <prep_epc_sc_auto_length>`, with a default length of 10 angstroms for the supercell.
Here, we use ``[3, 3, 3]`` for aluminum as a quick tutorial example.

.. note::
    In practice, the size of the supercells should be larger than the cutoff radius of each element in your material.
    The cutoff radii can be found in :ref:`Atomic Orbital Basis Definition <ao_define_title>`.

With the ``input_epc_1.yaml`` configured, you can now execute the following command:

.. code-block:: none
    
    [user]$ QTrans_prep --config ./input_epc_1.yaml > grad_epc_1.out

After the task is completed, the ``QT.UNITCELL`` file will be available in the directory specified by :ref:`outdir <prep_outdir>`.
It contains the unit cell structure in POSCAR format.
If :ref:`std_symm_cell <prep_std_symm_cell>` is set to **True**, the file will contain the standard primitive cell.  
Otherwise, it will reproduce the structure provided in :ref:`struct_path <prep_struct_path>`.
We recommend using the standard primitive cell for all your calculations.

And also a folder named ``perturb`` will be created.
Inside this folder, the program generates a file named ``mat_info_sc.npy``, which contains the mapping between the supercell and the unit cell.
It also includes :math:`3N + 1` input files formatted for the :ref:`DFT_software <prep_DFT_software>` specified in the input,
where :math:`N` is the number of atoms in the unit cell. 
The :math:`3N + 1` structures include one unperturbed supercell and :math:`3N` perturbed supercells,
with the perturbations applied along the Cartesian directions (x, y, z) for each atom in a representative unit cell of the supercell.
The perturbation displacement is specified by :ref:`epc_sc_disp <prep_epc_sc_disp>` with a default value of 0.01 angstrom.

.. warning::
    The generated input files use a set of preliminary DFT parameters intended for initial setup.
    You are suggested to review and adjust these parameters before running the actual calculations.

2. Hamiltonian and Overlap Matrices for Perturbed Supercell Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we need to obtain the Hamiltonian and overlap matrices for the supercells mentioned above.
Please refer to :ref:`Generation of Hamiltonian and Overlap Matrices <tutorial_obtain_hs>` for details on how to generate them.
Here, we use **OpenMX** to perform the self-consistent DFT calculations and extract the Hamiltonian and overlap matrices,
which are saved in ``graph_data_QTrans.npz``. 
You can also predict and generate the Hamiltonian matrices of those supercells using **HamGNN**.

3. Generation of Gradient Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    
Now, we need to configure the ``input_epc_2.yaml`` file according to the above generated files.
Set :ref:`graph_data_sc_path <prep_graph_data_sc_path>` to ``./perturb/graph_data_QTrans.npz``,
and :ref:`mat_info_sc_path <prep_mat_info_sc_path>` to ``./perturb/mat_info_sc.npy``.
Then, we could issue the following command:

.. code-block:: none

    [user]$ QTrans_prep --config ./input_epc_2.yaml > grad_epc_2.out

Once the task is completed, the files ``grad_mat.npy`` and ``grad_mat_sparse.npy`` will be generated.
Additionally, the program will create a temporary folder named ``dSon``,
which contains the necessary information for computing the gradients of the onsite elements of the overlap matrices.

.. note::
    If you choose to predict the Hamiltonian of those supercells,
    set :ref:`use_prediction <prep_use_prediction>` to ``True``,
    and :ref:`pred_H_sc_path <prep_pred_H_sc_path>` to the location of your predicted Hamiltonian file.


EPC Gradient Matrix of BCC-Fe
------------------------------

This tutorial is located in ``tutorials/Fe/epc_prep/``. 
Reference outputs are provided in ``tutorials/Fe/epc_prep/ref/`` , 
which can be used to compare with your own results to verify that **QTrans** is running correctly.

The process is almost the same as that for aluminum.
However, since BCC-phase iron (BCC-Fe) is a ferromagnetic system,
you need to specify the magnetic-related parameters in the input files for **QTrans**.
We highly recommend using the **OpenMX** input file as the value for the ``struct_path`` parameter,
as it contains both structural and magnetic information.

.. note::
    The magnetic moments of magnetic atoms in the current **OpenMX** input file should correspond to the 
    values obtained after self-consistent calculations, rather than the initial guesses.

