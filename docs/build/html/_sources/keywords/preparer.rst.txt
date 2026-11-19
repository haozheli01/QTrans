============================
Keywords of QTrans_prep
============================

Structure of the input data
---------------------------

We use a ``yaml`` file as the input configuration.
For different ``prep_mode`` values, the program requires different input blocks.
However, the parameters in **basic** block must always be defined in the input file.
The format is as follows:

.. code-block:: none

    baisc:
        prep_mode: <prep_mode>
        outdir: <outdir>

    block_name1:
        key1: value1
        key2: value2
    
    block_name2:
        key3: value3
        key4: value4

Keywords
---------------------------

.. _prep_title_keywords:

Blocks
^^^^^^^^^^^^

    .. container:: left-align-text

        **Block <basic>:**
        
            :ref:`prep_mode <prep_prep_mode>`,
            :ref:`outdir <prep_outdir>`

        **Block <software_path>:**

            :ref:`vasp_path <prep_vasp_path>`,
            :ref:`vaspkit_path <prep_vaspkit_path>`,
            :ref:`vasp_pot_dir <prep_vasp_pot_dir>`,
            :ref:`openmx_path <prep_openmx_path>`,
            :ref:`openmx_pp_path <prep_openmx_pp_path>`,
            :ref:`openmx_read_path <prep_openmx_read_path>`,
            :ref:`openmx_dftdata_dir <prep_openmx_dftdata_dir>`,
            :ref:`abacus_path <prep_abacus_path>`,
            :ref:`abacus_pp_path <prep_abacus_pp_path>`,
            :ref:`jx_path <prep_jx_path>`

        **Block <material>:**
        
            :ref:`struct_path <prep_struct_path>`,
            :ref:`std_symm_cell <prep_std_symm_cell>`,
            :ref:`symm_tol <prep_symm_tol>`,
            :ref:`magnetic_tolerance <prep_magnetic_tolerance>`,
            :ref:`magnetic_type <prep_magnetic_type>`,
            :ref:`soc_switch <prep_soc_switch>`,
            :ref:`DFT_software <prep_DFT_software>`,
            :ref:`nao_max <prep_nao_max>`

        **Block <scf_params>:**

            :ref:`scf_uc_kpt <prep_scf_uc_kpt>`,
            :ref:`scf_rec_density <prep_scf_rec_density>`

        **Block <grad_epc>:**

            :ref:`dSon_delta_r <prep_dSon_delta_r>`,
            :ref:`grad_sparse_thr <prep_grad_sparse_thr>`,
            :ref:`epc_sc_matrix <prep_epc_sc_matrix>`,
            :ref:`epc_sc_auto_length <prep_epc_sc_auto_length>`,
            :ref:`epc_sc_disp <prep_epc_sc_disp>`,
            :ref:`graph_data_sc_path <prep_graph_data_sc_path>`,
            :ref:`pred_H_sc_path <prep_pred_H_sc_path>`,
            :ref:`mat_info_sc_path <prep_mat_info_sc_path>`,
            :ref:`use_prediction <prep_use_prediction>`

        **Block <graph_pack>:**

            :ref:`HSout_filename <prep_HSout_filename>`,
            :ref:`nonscf_filename <prep_nonscf_filename>`,
            :ref:`stdin_filename <prep_stdin_filename>`,
            :ref:`vbout_filename <prep_vbout_filename>`,
            :ref:`data_folder <prep_data_folder>`,
            :ref:`converge_step_ratio <prep_converge_step_ratio>`,

basic block
^^^^^^^^^^^^^^

.. _prep_prep_mode:
.. param-card:: prep_mode
  :options: grad-epc
  :default: <b><i>MUST SET BY USER</i></b>
  :block: basic

   Preparation mode.

:ref:`Back to top <prep_title_keywords>`

.. _prep_outdir:
.. param-card:: outdir
   :options: <b><i>*STRING*</i></b>
   :default: <b><i>MUST SET BY USER</i></b>
   :block: basic

    The output directory.

:ref:`Back to top <prep_title_keywords>`

software_path block
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _prep_vasp_path:
.. param-card:: vasp_path
   :options: <b><i>*STRING*</i></b>
   :default: ./vasp_std
   :block: software_path

    The path to the VASP executable.

:ref:`Back to top <prep_title_keywords>`

.. _prep_vaspkit_path:
.. param-card:: vaspkit_path
   :options: <b><i>*STRING*</i></b>
   :default: ./vaspkit
   :block: software_path

    The path to the VASPkit executable.

:ref:`Back to top <prep_title_keywords>`

.. _prep_vasp_pot_dir:
.. param-card:: vasp_pot_dir
   :options: <b><i>*STRING*</i></b>
   :default: ./
   :block: software_path

    Directory containing the VASP pseudopotential files.

:ref:`Back to top <prep_title_keywords>`

.. _prep_openmx_path:
.. param-card:: openmx_path
   :options: <b><i>*STRING*</i></b>
   :default: ./openmx
   :block: software_path

    The path to the OpenMX executable.

:ref:`Back to top <prep_title_keywords>`

.. _prep_openmx_pp_path:
.. param-card:: openmx_pp_path
   :options: <b><i>*STRING*</i></b>
   :default: ./openmx_postprocess
   :block: software_path

    The path to the openmx_postprocess executable.

:ref:`Back to top <prep_title_keywords>`

.. _prep_openmx_read_path:
.. param-card:: openmx_read_path
   :options: <b><i>*STRING*</i></b>
   :default: ./read_openmx
   :block: software_path

    The path to the read_openmx executable.

:ref:`Back to top <prep_title_keywords>`

.. _prep_openmx_dftdata_dir:
.. param-card:: openmx_dftdata_dir
   :options: <b><i>*STRING*</i></b>
   :default: ./
   :block: software_path

    Directory containing OpenMX pseudopotentials and orbital basis files.

:ref:`Back to top <prep_title_keywords>`

.. _prep_abacus_path:
.. param-card:: abacus_path
   :options: <b><i>*STRING*</i></b>
   :default: ./abacus
   :block: software_path

    The path to the ABACUS executable.

:ref:`Back to top <prep_title_keywords>`

.. _prep_abacus_pp_path:
.. param-card:: abacus_pp_path
   :options: <b><i>*STRING*</i></b>
   :default: ./abacus_postprocess
   :block: software_path

    The path to the abacus_postprocess executable.

:ref:`Back to top <prep_title_keywords>`

.. _prep_jx_path:
.. param-card:: jx_path
   :options: <b><i>*STRING*</i></b>
   :default: ./jx
   :block: software_path

    The path to the JX executable.

:ref:`Back to top <prep_title_keywords>`

material block
^^^^^^^^^^^^^^^^

.. _prep_struct_path:
.. param-card:: struct_path
   :options: POSCAR | *.vasp | *.xsf | *.dat | *.npz
   :default: ./POSCAR
   :block: material

    The path to the structure file.

    .. list-table:: Supported Structure File Formats
        :header-rows: 0
        :stub-columns: 0
        :class: custom-table center-align-text 
        :name: struct-path-table
        
        * - POSCAR
          - Structure file in VASP POSCAR format
        * - .vasp
          - Structure file in VASP POSCAR format
        * - .xsf
          - XCrySDen structure file
        * - .dat
          - OpenMX input file
        * - .npz
          - HamGNN graph data file


:ref:`Back to top <prep_title_keywords>`

.. _prep_std_symm_cell:
.. param-card:: std_symm_cell
   :options: True | False
   :default: False
   :block: material

    If **True**, the input structure will be converted to
    a standard lattice prior to use.

:ref:`Back to top <prep_title_keywords>`

.. _prep_symm_tol:
.. param-card:: symm_tol
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.00001
   :block: material

    Tolerance (in angstroms) for symmetry finding.

:ref:`Back to top <prep_title_keywords>`

.. _prep_magnetic_tolerance:
.. param-card:: magnetic_tolerance
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.01
   :block: material

    Threshold (in Bohr magneton) for identifying magnetic atoms.

:ref:`Back to top <prep_title_keywords>`

.. _prep_moment_similar_tolerence:
.. param-card:: moment_similar_tolerence
   :options: <b><i>*FLOAT*</i></b>
   :default: -1.0
   :block: material

    Threshold (in Bohr magneton) to judge if two atomic magnetic moments
    are similar enough to be considered identical.

:ref:`Back to top <prep_title_keywords>`

.. _prep_magnetic_type:
.. param-card:: magnetic_type
   :options: 'none' | 'collinear' | 'nc'
   :default: 'none'
   :block: material

    Specifies the magnetic configuration of the system.

    .. list-table:: Magnetic Configuration
        :header-rows: 0
        :stub-columns: 0
        :class: custom-table center-align-text 
        :name: prep-magnetic-conf-table
        
        * - 'none'
          - Non-magnetic
        * - 'collinear'
          - Collinear spin configuration
        * - 'nc'
          - Non-collinear spin configuration

:ref:`Back to top <prep_title_keywords>`

.. _prep_soc_switch:
.. param-card:: soc_switch
   :options: True | False
   :default: False
   :block: material

    If **True**, the system will be treated with spin-orbit coupling (SOC).

:ref:`Back to top <prep_title_keywords>`

.. _prep_DFT_software:
.. param-card:: DFT_software
   :options: 'openmx' | 'abacus' | 'siesta'
   :default: None
   :block: material

    Selects the DFT software associated with the input data and generated files.

    - ``openmx``: OpenMX interface.
    - ``abacus``: ABACUS interface.
    - ``siesta``: SIESTA basis definitions are listed in
      :ref:`Atomic Orbital Basis Definition <ao_define_title>`, but the current
      preparer input validation does not yet accept this option.

    Use ``openmx`` or ``abacus`` for the preparation workflows documented here.

:ref:`Back to top <prep_title_keywords>`

.. _prep_nao_max:
.. param-card:: nao_max
   :options: 13 | 14 | 15 | 19 | 26 | 27 | 40
   :default: 26
   :block: material

    Determines which predefined atomic orbital basis set to use,
    based on the specified integer value and :ref:`DFT_software <prep_DFT_software>`.
    The basis set is specified as {**DFT_software**}_{**nao_max**}.
    More details can be found in :ref:`Input Keywords/Atomic Orbital Basis Definition <ao_define_title>`.

    .. list-table:: Atomic Orbital Basis
        :header-rows: 0
        :stub-columns: 0
        :class: custom-table center-align-text 
        :name: prep-orbit-basis-table
        
        * - openmx_14
          - **OpenMX** basis set specifying a maximum of **14** orbitals per element
        * - openmx_19
          - **OpenMX** basis set specifying a maximum of **19** orbitals per element
        * - openmx_26
          - **OpenMX** basis set specifying a maximum of **26** orbitals per element
        * - siesta_13
          - **SIESTA** basis set specifying a maximum of **13** orbitals per element
        * - siesta_19
          - **SIESTA** basis set specifying a maximum of **19** orbitals per element
        * - abacus_13
          - **ABACUS** basis set specifying a maximum of **13** orbitals per element
        * - abacus_15
          - **ABACUS** basis set specifying a maximum of **15** orbitals per element
        * - abacus_27
          - **ABACUS** basis set specifying a maximum of **27** orbitals per element
        * - abacus_40
          - **ABACUS** basis set specifying a maximum of **40** orbitals per element

:ref:`Back to top <prep_title_keywords>`

scf_params block
^^^^^^^^^^^^^^^^^

.. _prep_scf_uc_kpt:
.. param-card:: scf_uc_kpt
   :options: 'auto' | <b><i>*LIST[INT, INT, INT]*</i></b>
   :default: 'auto'
   :block: scf_params

    The size of the reciprocal-space **k**-point
    sampling grid. If set to ``'auto'``,
    the grid will be automatically generated based on
    :ref:`scf_rec_density <prep_scf_rec_density>`.

:ref:`Back to top <prep_title_keywords>`

.. _prep_scf_rec_density:
.. param-card:: scf_rec_density
   :options: <b><i>*FLOAT*</i></b>
   :default: 50.0
   :block: scf_params

    The reciprocal-space **k**-point sampling density for
    SCF calculations,  
    specified in units of [angstrom].  
    If :ref:`scf_uc_kpt <prep_scf_uc_kpt>` is ``'auto'``, the k-point grid will be generated
    based on this value and the lattice constants.

:ref:`Back to top <prep_title_keywords>`

grad_epc block
^^^^^^^^^^^^^^^^

.. _prep_dSon_delta_r:
.. param-card:: dSon_delta_r
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.001
   :block: grad_epc

    Small displacement (in angstroms)
    applied to compute the gradient of onsite overlap matrix elements.

:ref:`Back to top <prep_title_keywords>`

.. _prep_grad_sparse_thr:
.. param-card:: grad_sparse_thr
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.0001
   :block: grad_epc

    The threshold (in Hartree/bohr) below which
    elements in the sparse gradient matrix are neglected.

:ref:`Back to top <prep_title_keywords>`

.. _prep_epc_sc_matrix:
.. param-card:: epc_sc_matrix
   :options: 'auto' | <b><i>*LIST[INT, INT, INT]*</i></b>
   :default: 'auto'
   :block: grad_epc

    The supercell size used for EPC gradient matrix calculation.
    Note that all three integers must be odd.
    If set to ``'auto'``,
    the supercell size will be automatically determined
    based on :ref:`epc_sc_auto_length <prep_epc_sc_auto_length>`.

:ref:`Back to top <prep_title_keywords>`

.. _prep_epc_sc_auto_length:
.. param-card:: epc_sc_auto_length
   :options: <b><i>*FLOAT*</i></b>
   :default: 10.0
   :block: grad_epc

    If :ref:`epc_sc_matrix <prep_epc_sc_matrix>` is set to ``'auto'``,  
    the supercell size required for EPC gradient matrix calculation  
    will be automatically determined using this value, specified in angstroms.

:ref:`Back to top <prep_title_keywords>`

.. _prep_epc_sc_disp:
.. param-card:: epc_sc_disp
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.01
   :block: grad_epc

    Small displacement (in angstroms) used to compute
    the gradient of the Hamiltonian for EPC calculation,
    using the finite displacement method.

:ref:`Back to top <prep_title_keywords>`

.. _prep_graph_data_sc_path:
.. param-card:: graph_data_sc_path
   :options: <b><i>*STRING*</i></b>
   :default: None
   :block: grad_epc

    Path to the graph data of supercells used to
    compute the Hamiltonian gradient for EPC calculation,
    using the finite displacement method.

:ref:`Back to top <prep_title_keywords>`

.. _prep_pred_H_sc_path:
.. param-card:: pred_H_sc_path
   :options: <b><i>*STRING*</i></b>
   :default: None
   :block: grad_epc

    Path to the prediction Hamiltonian of supercells used to
    compute the Hamiltonian gradient for EPC calculation,
    using the finite displacement method.
    Used only if :ref:`use_prediction <prep_use_prediction>` is ``True``.

:ref:`Back to top <prep_title_keywords>`

.. _prep_mat_info_sc_path:
.. param-card:: mat_info_sc_path
   :options: <b><i>*STRING*</i></b>
   :default: None
   :block: grad_epc

    Path to the ``mat_info_sc.npy`` file generated by **QTrans**.
    It should contain the mapping between unit cell and supercells used to
    compute the Hamiltonian gradient for EPC calculation,
    using the finite displacement method.

:ref:`Back to top <prep_title_keywords>`

.. _prep_use_prediction:
.. param-card:: use_prediction
   :options: True | False
   :default: False
   :block: grad_epc

    If ``True``, the program reads the Hamiltonians from
    :ref:`pred_H_sc_path <prep_pred_H_sc_path>`
    instead of from :ref:`graph_data_sc_path <prep_graph_data_sc_path>`.
    However, it still requires other information from :ref:`graph_data_sc_path <prep_graph_data_sc_path>`.

:ref:`Back to top <prep_title_keywords>`

graph_pack block
^^^^^^^^^^^^^^^^

.. _prep_HSout_filename:
.. param-card:: HSout_filename
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: graph_pack

    Specifies the name of the file containing the
    self-consistent Hamiltonian and overlap matrix.
    If set to ``None``, the graph data will be constructed using
    non-self-consistent part of Hamiltonian, which requires :ref:`stdin_filename <prep_stdin_filename>`
    and :ref:`nonscf_filename <prep_nonscf_filename>`.
    Otherwise, the graph data will be built using self-consistent Hamiltonian inside this file
    and also other information from :ref:`vbout_filename <prep_vbout_filename>`.
    In this case, non-self-consistent part of Hamiltonian data can
    also be included by specifying :ref:`nonscf_filename <prep_nonscf_filename>`.

:ref:`Back to top <prep_title_keywords>`

.. _prep_nonscf_filename:
.. param-card:: nonscf_filename
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: graph_pack

    The name of the file that contains the non-self-consistent
    Hamiltonian and overlap matrix.

:ref:`Back to top <prep_title_keywords>`

.. _prep_stdin_filename:
.. param-card:: stdin_filename
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: graph_pack

    The name of the DFT software input file.
    Has no effect unless :ref:`HSout_filename <prep_HSout_filename>` is ``None``.

:ref:`Back to top <prep_title_keywords>`

.. _prep_vbout_filename:
.. param-card:: vbout_filename
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: graph_pack

    The name of the DFT software output file.
    Has no effect unless :ref:`HSout_filename <prep_HSout_filename>` is not ``None``.

:ref:`Back to top <prep_title_keywords>`

.. _prep_data_folder:
.. param-card:: data_folder
   :options: <b><i>*STRING*</i></b>
   :default: None
   :block: graph_pack

    Specifies the directories of the folders with the raw data
    to be packed into graph data.
    This path will be used by the **glob.glob()** function to match all
    directories.
    Note that this value must point to the directory of folders, not files.

:ref:`Back to top <prep_title_keywords>`

.. _prep_converge_step_ratio:
.. param-card:: converge_step_ratio
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.9
   :block: graph_pack

    Defines the maximum acceptable ratio between
    the number of self-consistent field (SCF) steps
    actually taken to reach convergence
    and the maximum number of SCF steps specified in the DFT input settings.
    Data entries with a convergence ratio
    exceeding this threshold will be excluded from the graph data generation process.

:ref:`Back to top <prep_title_keywords>`
