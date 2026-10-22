============================
Keywords of QTrans_run
============================

Structure of the input data
---------------------------

We use a ``yaml`` file as the input configuration.
For different ``cal_mode`` values, the program requires different input blocks.
However, the parameters in **basic** block must always be defined in the input file.
The format is as follows:

.. code-block:: none

    baisc:
        cal_mode: <cal_mode>
        graph_data_path_uc: <graph_data_path_uc>
        nao_max: <nao_max>
        Ham_type: <Ham_type>
        outdir: <outdir>

    block_name1:
        key1: value1
        key2: value2
    
    block_name2:
        key3: value3
        key4: value4

Keywords
---------------------------

.. _cal_title_keywords:

Blocks
^^^^^^^^^^^^

    .. container:: left-align-text

        **Block <basic>:**
        
            :ref:`cal_mode <cal_cal_mode>`,
            :ref:`graph_data_path_uc <cal_graph_data_path_uc>`,
            :ref:`nao_max <cal_nao_max>`,
            :ref:`Ham_type <cal_Ham_type>`,
            :ref:`outdir <cal_outdir>`

        **Block <advanced>:**

            :ref:`read_sparse_grad_mat <cal_read_sparse_grad_mat>`,
            :ref:`soc_switch <cal_soc_switch>`,
            :ref:`magnetic_type <cal_magnetic_type>`,
            :ref:`magnetic_tolerance <cal_magnetic_tolerance>`,
            :ref:`moment_similar_tolerence <cal_moment_similar_tolerence>`,
            :ref:`doping_charge <cal_doping_charge>`,
            :ref:`system_2D <cal_system_2D>`,
            :ref:`tetra_integral <cal_tetra_integral>`
        
        **Block <dos>:**

            :ref:`dos_qp <cal_dos_qp>`,
            :ref:`dos_npoints <cal_dos_npoints>`,
            :ref:`dos_erange <cal_dos_erange>`,
            :ref:`dos_mesh_size <cal_dos_mesh_size>`,
            :ref:`dos_smear <cal_dos_smear>`,
            :ref:`dos_gauss_type <cal_dos_gauss_type>`
        
        **Block <dispersion>:**

            :ref:`disp_renormalize <cal_disp_renormalize>`,
            :ref:`disp_lambda <cal_disp_lambda>`,
            :ref:`high_symmetry_points <cal_high_symmetry_points>`,
            :ref:`high_symmetry_labels <cal_high_symmetry_labels>`,
            :ref:`nks_path <cal_nks_path>`,
            :ref:`dispersion_select_index <cal_dispersion_select_index>`,
            :ref:`epc_path_fix_k <cal_epc_path_fix_k>`,
            :ref:`save_eigen_vectors <cal_save_eigen_vectors>`,
        
        **Block <phonon>:**

            :ref:`supercell_matrix <cal_supercell_matrix>`,
            :ref:`unitcell_filename <cal_unitcell_filename>`,
            :ref:`force_infos_filename <cal_force_infos_filename>`,
            :ref:`apply_correction <cal_apply_correction>`,
            :ref:`quadrupole_correction <cal_quadrupole_correction>`,
            :ref:`Q_file <cal_Q_file>`,
            :ref:`BECs <cal_BECs>`,
            :ref:`DL <cal_DL>`,
            :ref:`thickness <cal_thickness>`,
            :ref:`chiral_projection <cal_chiral_projection>`

        **Block <epc>:**

            :ref:`grad_mat_path <cal_grad_mat_path>`,
            :ref:`mat_info_sc_path <cal_mat_info_sc_path>`,
            :ref:`cell_range_cut <cal_cell_range_cut>`,
            :ref:`phonon_frequency_cutoff <cal_phonon_frequency_cutoff>`,
            :ref:`LRC_taylor_order <cal_LRC_taylor_order>`,
            :ref:`q_cut <cal_q_cut>`,
            :ref:`range_separation_length_L <cal_range_separation_length_L>`,
            :ref:`vaccum_length <cal_vaccum_length>`

        **Block <transport>:**

            :ref:`k_size <cal_k_size>`,
            :ref:`q_size <cal_q_size>`,
            :ref:`bands_indices <cal_bands_indices>`,
            :ref:`fermi_maxiter <cal_fermi_maxiter>`,
            :ref:`fermi_shift <cal_fermi_shift>`,
            :ref:`temperature <cal_temperature>`,
            :ref:`smeark <cal_smeark>`,
            :ref:`smearq <cal_smearq>`,
            :ref:`gauss_type <cal_gauss_type>`,
            :ref:`e_thr <cal_e_thr>`,
            :ref:`is_hole <cal_is_hole>`,
            :ref:`mode_resolved <cal_mode_resolved>`

        **Block <conductivity>:**

            :ref:`over_band_edge <cal_over_band_edge_conductivity>`,
            :ref:`MC_sampling <cal_MC_sampling>`,
            :ref:`polar_split <cal_polar_split>`,
            :ref:`cauchy_scale <cal_cauchy_scale>`,
            :ref:`sampling_seed <cal_sampling_seed>`,
            :ref:`nsamples <cal_nsamples>`,
            :ref:`ncarrier <cal_ncarrier_conductivity>`,
            :ref:`mob_level <cal_mob_level>`,
            :ref:`polar_rate_path <cal_polar_rate_path>`,
            :ref:`rmp_rate_path <cal_rmp_rate_path>`,
            :ref:`epc_rate_path <cal_epc_rate_path>`,
            :ref:`scattering_channel_dir <cal_scattering_channel_dir>`,
            :ref:`ibte_error <cal_ibte_error>`,
            :ref:`ibte_cycle <cal_ibte_cycle>`

        **Block <superconductivity>:**

            :ref:`sc_method <cal_sc_method>`,
            :ref:`mus <cal_mus>`,
            :ref:`omega_range <cal_omega_range_sc>`,
            :ref:`omega_step <cal_omega_step_sc>`

        **Block <pomega>:**

            :ref:`omega_range <cal_omega_range_sc>`,
            :ref:`omega_step <cal_omega_step_sc>`,
            :ref:`pomega_restart <cal_pomega_restart>`

        **Block <breakdown>**

            :ref:`fbd_erange <cal_fbd_erange>`,
            :ref:`fbd_rate_file <cal_fbd_rate_file>`,
            :ref:`fbd_effective_mass <cal_fbd_effective_mass>`,
            :ref:`fbd_exp_bandgap <cal_fbd_exp_bandgap>`

        **Block <resistivity>**

            :ref:`ziman_nc <cal_ziman_nc>`,
            :ref:`omega_range <cal_omega_range_resistivity>`,
            :ref:`omega_step <cal_omega_step_resistivity>`

        **Block <boltzmann>**

            :ref:`over_band_edge <cal_over_band_edge_boltzmann>`,
            :ref:`ncarrier <cal_ncarrier_boltzmann>`,
            :ref:`temperature_lat <cal_temperature_lat>`,
            :ref:`time_step <cal_time_step>`,
            :ref:`energy_step <cal_energy_step>`,
            :ref:`num_step <cal_num_step>`,
            :ref:`norm_dist <cal_norm_dist>`,
            :ref:`boltz_solver <cal_boltz_solver>`,
            :ref:`output_step <cal_output_step>`,
            :ref:`efield <cal_efield>`

        **Block <self-energy>**

            :ref:`SE_type <cal_SE_type>`,
            :ref:`SE_erange <cal_SE_erange>`,
            :ref:`over_band_edge <cal_over_band_edge_self_energy>`,
            :ref:`ncarrier <cal_ncarrier_self_energy>`


basic block
^^^^^^^^^^^^^^

.. _cal_cal_mode:
.. param-card:: cal_mode
   :options: 'band' | 'phonon' | 'dos' | 'epc' | 'superconductivity' | 'pomega' | 'conductivity' | </br>
    'breakdown' | 'resistivity' | 'boltzmann'
   :default: <b><i>MUST SET BY USER</i></b>
   :block: basic

    Calculation mode.

:ref:`Back to top <cal_title_keywords>`

.. _cal_graph_data_path_uc:
.. param-card:: graph_data_path_uc
   :options: <b><i>*STRING*</i></b>
   :default: <b><i>MUST SET BY USER</i></b>
   :block: basic

    Path to the graph data of the unit cell.

:ref:`Back to top <cal_title_keywords>`

.. _cal_nao_max:
.. param-card:: nao_max
   :options: 13 | 14 | 15 | 19 | 26 | 27 | 40
   :default: <b><i>MUST SET BY USER</i></b>
   :block: basic

    Determines which predefined atomic orbital basis set to use,
    based on the specified integer value and :ref:`Ham_type <cal_Ham_type>`.
    The basis set is specified as {**Ham_type**}_{**nao_max**}.
    More details can be found in :ref:`Input Keywords/Atomic Orbital Basis Definition <ao_define_title>`.

    .. list-table:: Atomic Orbital Basis
        :header-rows: 0
        :stub-columns: 0
        :class: custom-table center-align-text 
        :name: cal-orbit-basis-table
        
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

:ref:`Back to top <cal_title_keywords>`


.. _cal_Ham_type:
.. param-card:: Ham_type
   :options: 'openmx' | 'siesta' | 'abacus'
   :default: <b><i>MUST SET BY USER</i></b>
   :block: basic

    The DFT software used to prepare input data for **QTrans**.

:ref:`Back to top <cal_title_keywords>`

.. _cal_outdir:
.. param-card:: outdir
   :options: <b><i>*STRING*</i></b>
   :default: <b><i>MUST SET BY USER</i></b>
   :block: basic

    The output directory.

:ref:`Back to top <cal_title_keywords>`


advanced block
^^^^^^^^^^^^^^

.. _cal_read_sparse_grad_mat:
.. param-card:: read_sparse_grad_mat
   :options: True | False
   :default: False
   :block: advanced

    If **True**, the program will read the sparse gradient matrix.

:ref:`Back to top <cal_title_keywords>`

.. _cal_soc_switch:
.. param-card:: soc_switch
   :options: True | False
   :default: False
   :block: advanced

    If **True**, the system will be treated with spin-orbit coupling (SOC).

:ref:`Back to top <cal_title_keywords>`

.. _cal_magnetic_type:
.. param-card:: magnetic_type
   :options: 'none' | 'collinear' | 'nc'
   :default: 'none'
   :block: advanced

    Specifies the magnetic configuration of the system.

    .. list-table:: Magnetic Configuration
        :header-rows: 0
        :stub-columns: 0
        :class: custom-table center-align-text 
        :name: cal-magnetic-conf-table
        
        * - 'none'
            - Non-magnetic
        * - 'collinear'
            - Collinear spin configuration
        * - 'nc'
            - Non-collinear spin configuration

:ref:`Back to top <cal_title_keywords>`

.. _cal_magnetic_tolerance:
.. param-card:: magnetic_tolerance
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.01
   :block: advanced

    Threshold (in Bohr magneton) for identifying magnetic atoms.

:ref:`Back to top <cal_title_keywords>`

.. _cal_moment_similar_tolerence:
.. param-card:: moment_similar_tolerence
   :options: <b><i>*FLOAT*</i></b>
   :default: -1.0
   :block: advanced

    Threshold (in Bohr magneton) to judge if two atomic magnetic moments
    are similar enough to be considered identical.

:ref:`Back to top <cal_title_keywords>`

.. _cal_doping_charge:
.. param-card:: doping_charge
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.0
   :block: advanced

    The extra charge (in elementary charge) in the system.
    Negative values indicate electron doping.
    The Fermi level is adjusted accordingly using the rigid-band approximation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_system_2D:
.. param-card:: system_2D
   :options: True | False
   :default: False
   :block: advanced

    If **True**, the system will be treated as 2D system.

:ref:`Back to top <cal_title_keywords>`

.. _cal_tetra_integral:
.. param-card:: tetra_integral
   :options: True | False
   :default: False
   :block: advanced

    If **True**, the program uses the tetrahedron method for
    Brillouin zone integration. Currently, it is only supported
    when :ref:`cal_mode <cal_cal_mode>` is set to **'dos'**.

:ref:`Back to top <cal_title_keywords>`

dos block
^^^^^^^^^^^^^^

.. _cal_dos_qp:
.. param-card:: dos_qp
   :options: 'electron' | 'phonon'
   :default: 'electron'
   :block: dos

    Specifies the particle whose density of states (DOS) is to be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_dos_npoints:
.. param-card:: dos_npoints
   :options: <b><i>*INT*</i></b>
   :default: 1000
   :block: dos

    Specifies the number of energy sampling points
    used in the density of states (DOS) calculation.  
    The energy range defined in :ref:`dos_erange <cal_dos_erange>`
    is uniformly partitioned into this number of intervals.

:ref:`Back to top <cal_title_keywords>`

.. _cal_dos_erange:
.. param-card:: dos_erange
   :options: <b><i>*LIST[FLOAT, FLOAT]*</i></b>
   :default: None
   :block: dos

    Specifies the energy range (in eV), given as **[min, max]**,
    over which the density of states (DOS) is calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_dos_mesh_size:
.. param-card:: dos_mesh_size
   :options: <b><i>*LIST[INT, INT, INT]*</i></b>
   :default: [32, 32, 32]
   :block: dos

    Specifies the size of the reciprocal-space **k**-point sampling grid
    used for density of states (DOS) calculations.

:ref:`Back to top <cal_title_keywords>`

.. _cal_dos_smear:
.. param-card:: dos_smear
   :options: <b><i>*FLOAT*</i></b>
   :default: 10.0
   :block: dos

    Specifies the smearing parameter for density of states (DOS) calculations.
    Read :ref:`tutorials/smearing method <tutorial-smearing>` for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_dos_gauss_type:
.. param-card:: dos_gauss_type
   :options: -99 | -1 | <b><i>*UNSIGNED INT*</i></b>
   :default: 0
   :block: dos

    Specifies the smearing method for density of states (DOS) calculations.
    Read :ref:`tutorials/smearing method <tutorial-smearing>` for more details.

:ref:`Back to top <cal_title_keywords>`


dispersion block
^^^^^^^^^^^^^^^^^^

.. _cal_disp_renormalize:
.. param-card:: disp_renormalize
   :options: True | False
   :default: False
   :block: dispersion

    If **True**, the electron spectral functional along the electron dispersion will be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_disp_lambda:
.. param-card:: disp_lambda
   :options: True | False
   :default: False
   :block: dispersion

    If **True**, the electron-phonon coupling strength along the phonon dispersion will be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_high_symmetry_points:
.. param-card:: high_symmetry_points
   :options: <b><i>*LIST[LISTS[INT, INT, INT]]*</i></b>
   :default: None
   :block: dispersion

    The high-symmetry points used to calculate the dispersion relation.  
    Formatted as: **[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]**.

:ref:`Back to top <cal_title_keywords>`

.. _cal_high_symmetry_labels:
.. param-card:: high_symmetry_labels
   :options: <b><i>*LIST[STRINGS]*</i></b>
   :default: None
   :block: dispersion

    The  labels of the high-symmetry points
    used to calculate the dispersion relation.  
    Formatted as: ['**G**', '**L**'].

:ref:`Back to top <cal_title_keywords>`

.. _cal_nks_path:
.. param-card:: nks_path
   :options: <b><i>*INT*</i></b>
   :default: 200
   :block: dispersion

    Specifies the number of **k**-points sampled
    along the path for the dispersion relation calculation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_dispersion_select_index:
.. param-card:: dispersion_select_index
   :options: <b><i>*STRING*</i></b> | <b><i>LIST[INTS]</i></b>
   :default: None
   :block: dispersion

    Specifies the band or branch indices 
    for the dispersion relation calculation.
    Read :ref:`tutorials/Dispersion of Quasiparticle <tutorial-dispersion>`
    and :ref:`tutorials/EPC Path <tutorial-epc-path>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_epc_path_fix_k:
.. param-card:: epc_path_fix_k
   :options: <b><i>*LIST[FLOAT, FLOAT, FLOAT]*</i></b>
   :default: None
   :block: dispersion

    Specifies the fixed **k**-point for the EPC path calculation.
    Read :ref:`tutorials/EPC Path <tutorial-epc-path>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_save_eigen_vectors:
.. param-card:: save_eigen_vectors
   :options: 'none' | 'ascii'
   :default: 'none'
   :block: dispersion

    Specifies the format used to save eigenvectors
    after solving the quasiparticle eigenproblem.

   .. list-table:: Eigenvectors Saving Format
        :header-rows: 0
        :stub-columns: 0
        :class: custom-table center-align-text
        :name: cal-eigenvec-save-table

        * - 'none'
          - Do not save the eigenvectors.
        * - 'ascii'
          - Save eigenvectors in ASCII format.

:ref:`Back to top <cal_title_keywords>`

phonon block
^^^^^^^^^^^^^^

.. _cal_supercell_matrix:
.. param-card:: supercell_matrix
   :options: <b><i>*LIST[LISTS[FLOAT, FLOAT, FLOAT]]*</i></b>
   :default: None
   :block: phonon

    Specifies the :math:`3 \times 3` matrix used to construct the supercell
    from the unit cell for phonon calculations with the Phonopy package.

:ref:`Back to top <cal_title_keywords>`

.. _cal_unitcell_filename:
.. param-card:: unitcell_filename
   :options: <b><i>*STRING*</i></b>
   :default: './POSCAR'
   :block: phonon

    The unit cell POSCAR file used for phonon calculations with the Phonopy package.

:ref:`Back to top <cal_title_keywords>`

.. _cal_force_infos_filename:
.. param-card:: force_infos_filename
   :options: <b><i>*STRING*</i></b>
   :default: './FORCE_SETS'
   :block: phonon

    The FORCE_SETS file used for phonon calculations with the Phonopy package.

:ref:`Back to top <cal_title_keywords>`

.. _cal_apply_correction:
.. param-card:: apply_correction
   :options: True | False
   :default: False
   :block: phonon

    If **True**, the system is treated as polar,
    and the non-analytic correction to the phonon frequencies is included.
    In addition, the dipole contribution tothe long-range electron-phonon
    coupling matrix elements is included.

:ref:`Back to top <cal_title_keywords>`

.. _cal_quadrupole_correction:
.. param-card:: quadrupole_correction
   :options: True | False
   :default: False
   :block: phonon

    *bjr*

:ref:`Back to top <cal_title_keywords>`

.. _cal_Q_file:
.. param-card:: Q_file
   :options: <b><i>*STRING*</i></b>
   :default: './quadrupole.fmt'
   :block: phonon

    *bjr*

:ref:`Back to top <cal_title_keywords>`

.. _cal_BECs:
.. param-card:: BECs
   :options: <b><i>*LIST[LISTS[LISTS[FLOAT, FLOAT, FLOAT]]]*</i></b>
   :default: None
   :block: phonon

    The Born effective charge used for long-range corrections in polar system.
    Should be a :math:`\text{number_of_atoms} \times 3 \times 3` tensor.

:ref:`Back to top <cal_title_keywords>`

.. _cal_DL:
.. param-card:: DL
   :options: <b><i>*LIST[LISTS[FLOAT, FLOAT, FLOAT]]*</i></b>
   :default: None
   :block: phonon

    The dielectric constant matrix used for long-range corrections in polar system.  
    Should be a :math:`3 \times 3` matrix.

:ref:`Back to top <cal_title_keywords>`

.. _cal_thickness:
.. param-card:: thickness
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.0
   :block: phonon

    The effective thickness used for long-range corrections in 2D polar system.

:ref:`Back to top <cal_title_keywords>`

.. _cal_chiral_projection:
.. param-card:: chiral_projection
   :options: True | False
   :default: False
   :block: phonon

    If **True**, the phonon eigenvectors will be projected onto
    XY-chiral phonon modes.

:ref:`Back to top <cal_title_keywords>`

epc block
^^^^^^^^^^^^^^

.. _cal_grad_mat_path:
.. param-card:: grad_mat_path
   :options: <b><i>*STRING*</i></b>
   :default: './grad_mat.npy'
   :block: epc

    Path to the Hamiltonian gradient matrix generated by **QTrans**.
    Read :ref:`tutorials/EPC Gradient Matrix Preparation <tutorial-epc-grad-prep>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_mat_info_sc_path:
.. param-card:: mat_info_sc_path 
   :options: <b><i>*STRING*</i></b>
   :default: './mat_info_sc.npy'
   :block: epc

    Path to the file containing the mappings
    between supercells and the unit cell, generated by **QTrans**.
    Read :ref:`tutorials/EPC Gradient Matrix Preparation <tutorial-epc-grad-prep>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_cell_range_cut:
.. param-card:: cell_range_cut
   :options: <b><i>*LIST[INT, INT, INT]*</i></b>
   :default: [2, 2, 2]
   :block: epc

    Specifies the range of supercell lattice vectors to be
    included in the summation when calculating the electron phonon coupling matrix.
    The supercell with lattice vector which is beyond this value will be neglected.

:ref:`Back to top <cal_title_keywords>`

.. _cal_phonon_frequency_cutoff:
.. param-card:: phonon_frequency_cutoff
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.5
   :block: epc

    Phonon energy cutoff (in meV).
    Phonon modes with energies below this threshold will be ignored in the related calculation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_LRC_taylor_order:
.. param-card:: LRC_taylor_order
   :options: 0 | 1
   :default: 0
   :block: epc

    The taylor expansion order used in
    long-range correction of electron phonon coupling matrix
    in polar system.
    Read :ref:`Theory/Electron Phonon Coupling <theory_epc>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_q_cut:
.. param-card:: q_cut
   :options: <b><i>*FLOAT*</i></b>
   :default: 100.0
   :block: epc

    Specifies the cutoff for applying long-range corrections:
    only q-points with magnitudes smaller than this value will be included. 
    The unit is the length of the first reciprocal lattice vector.

:ref:`Back to top <cal_title_keywords>`

.. _cal_range_separation_length_L:
.. param-card:: range_separation_length_L
   :options: <b><i>*FLOAT*</i></b>
   :default: 100.0
   :block: epc

    *bjr*

:ref:`Back to top <cal_title_keywords>`

.. _cal_vaccum_length:
.. param-card:: vaccum_length
   :options: <b><i>*FLOAT*</i></b>
   :default: 30.0
   :block: epc

    *bjr*

:ref:`Back to top <cal_title_keywords>`


transport block
^^^^^^^^^^^^^^^^^

.. _cal_k_size:
.. param-card:: k_size
   :options: <b><i>*LIST[INT, INT, INT]*</i></b>
   :default: [32, 32, 32]
   :block: transport

    The size of **k** grid for integral over Brillouin Zone (BZ).

:ref:`Back to top <cal_title_keywords>`

.. _cal_q_size:
.. param-card:: q_size
   :options: <b><i>*LIST[INT, INT, INT]*</i></b>
   :default: [32, 32, 32]
   :block: transport

    The size of **q** grid for integral over Brillouin Zone (BZ).

:ref:`Back to top <cal_title_keywords>`

.. _cal_bands_indices:
.. param-card:: bands_indices
   :options: <b><i>*STRING* or LIST[INT]*</i></b>
   :default: None
   :block: transport

    Specifies the band indices relevant to transport property calculations.  
    Band indices start from 1.

    This parameter accepts either:
   
    - A string defining ranges, e.g. ``'1-6'`` or ``'1-3,5,7-8'``  
    - A list of integers, e.g. ``[1, 2, 3]`` or ``[5]``

:ref:`Back to top <cal_title_keywords>`

.. _cal_fermi_maxiter:
.. param-card:: fermi_maxiter
   :options: <b><i>*INT*</i></b>
   :default: 100
   :block: transport

    Maximum number of iterations for the bisection method used in Fermi level calculation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_fermi_shift:
.. param-card:: fermi_shift
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.0
   :block: transport

    Manual shift of the Fermi level (in eV).

:ref:`Back to top <cal_title_keywords>`

.. _cal_temperature:
.. param-card:: temperature
   :options: <b><i>*FLOAT*</i></b>
   :default: 300.0
   :block: transport

    Temperature of the system, in Klevin.

:ref:`Back to top <cal_title_keywords>`

.. _cal_smeark:
.. param-card:: smeark
   :options: <b><i>*FLOAT*</i></b>
   :default: 25
   :block: transport

    Smearing width (in meV) applied to the delta function in the summation over electronic states.

:ref:`Back to top <cal_title_keywords>`

.. _cal_smearq:
.. param-card:: smearq
   :options: <b><i>*FLOAT*</i></b>
   :default: 25
   :block: transport

    Smearing width (in meV) applied to the delta function in the summation over phonon states.

:ref:`Back to top <cal_title_keywords>`

.. _cal_gauss_type:
.. param-card:: gauss_type
   :options: <b><i>*INT*</i></b>
   :default: 0
   :block: transport

    The type of gaussian function. Read :ref:`tutorials/Smearing method <tutorial-smearing>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_e_thr:
.. param-card:: e_thr
   :options: <b><i>*FLOAT*</i></b>
   :default: 75
   :block: transport

    Energy threshold for the match table (in meV). Typically three times of the smearing width applied to the delta function.

:ref:`Back to top <cal_title_keywords>`

.. _cal_is_hole:
.. param-card:: is_hole
   :options: True | False
   :default: False
   :block: transport

    If True, the carriers are assumed to be holes in transport calculations.

:ref:`Back to top <cal_title_keywords>`

.. _cal_mode_resolved:
.. param-card:: mode_resolved
   :options: True | False
   :default: False
   :block: transport

    If **True**, the mode-resolved scattering rate will be saved.

:ref:`Back to top <cal_title_keywords>`


conductivity block
^^^^^^^^^^^^^^^^^^^

.. _cal_over_band_edge_conductivity:
.. param-card:: over_band_edge
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.2
   :block: conductivity

    Specifies the width (in eV) of the energy window centered around 
    the Fermi level or band edge. This window determines the range of 
    electronic states included in transport-related calculations.

:ref:`Back to top <cal_title_keywords>`

.. _cal_MC_sampling:
.. param-card:: MC_sampling
   :options: 'none' | 'uniform' | 'Cauchy'
   :default: 'none'
   :block: conductivity

    Specifies the type of importance sampling used in Monte Carlo integration.  

    - ``none``: No importance sampling.  
    - ``uniform``: Uniform random sampling.  
    - ``Cauchy``: Cauchy-distributed sampling, which emphasizes sampling near singularities.

:ref:`Back to top <cal_title_keywords>`

.. _cal_polar_split:
.. param-card:: polar_split
   :options: 'none' | 'polar' | 'rmp'
   :default: 'none'
   :block: conductivity

    Specifies whether to separate the polar and non-polar contributions to the scattering rate. 

    - ``none``: No splitting is applied.  
    - ``polar``: Computes only the polar (long-range) part of the scattering rate.  
    - ``rmp``: Computes only the remainder (non-polar) part.

    See :ref:`theory/Electron Phonon Coupling <theory_epc>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_cauchy_scale:
.. param-card:: cauchy_scale
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.035
   :block: conductivity

    The scale parameter :math:`\gamma` of the Cauchy distribution used for importance sampling.  
    This is only effective when ``MC_sampling = 'cauchy'``.

    The probability density function is given by:

    .. math::

        f(x) = \frac{1}{\pi} \cdot \frac{\gamma}{x^2 + \gamma^2}

    where :math:`\gamma` is the scale parameter.

:ref:`Back to top <cal_title_keywords>`

.. _cal_sampling_seed:
.. param-card:: sampling_seed
   :options: <b><i>*INT*</i></b>
   :default: 1
   :block: conductivity

    Random seed used for Monte Carlo sampling to ensure reproducibility.

:ref:`Back to top <cal_title_keywords>`

.. _cal_nsamples:
.. param-card:: nsamples
   :options: <b><i>*INT*</i></b>
   :default: 1000000
   :block: conductivity

    The number of sampling **q** points in the Monte Carlo integration.
    This is only effective when ``MC_sampling = 'cauchy'``.

:ref:`Back to top <cal_title_keywords>`

.. _cal_ncarrier_conductivity:
.. param-card:: ncarrier
   :options: <b><i>*FLOAT*</i></b>
   :default: 10000000000000000.0
   :block: conductivity

    The carrier concentration in :math:`\text{cm}^{-3}`.

:ref:`Back to top <cal_title_keywords>`

.. _cal_mob_level:
.. param-card:: mob_level
   :options: 'erta' | 'ibte'
   :default: 'erta'
   :block: conductivity

    Specifies the method used for electrical conductivity calculation.

    - ``erta``: Self-energy relaxation time approximation.  
    - ``ibte``: Iterative Boltzmann Transport Equation approach.

    See :ref:`theory/Conductivity Calculation <tutorial-conductivity>`
    for more details.

:ref:`Back to top <cal_title_keywords>`

.. _cal_polar_rate_path:
.. param-card:: polar_rate_path
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: conductivity

    Path to the precomputed scattering rate file for the polar part of the interaction.

:ref:`Back to top <cal_title_keywords>`

.. _cal_rmp_rate_path:
.. param-card:: rmp_rate_path
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: conductivity

    Path to the precomputed scattering rate file for the remainder (non-polar) part of the interaction.

:ref:`Back to top <cal_title_keywords>`

.. _cal_epc_rate_path:
.. param-card:: epc_rate_path
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: conductivity

    Path to the precomputed total scattering rate file.

.. _cal_scattering_channel_dir:
.. param-card:: scattering_channel_dir
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: conductivity

    Path to the folder containing the precomputed scattering channels required for iterative Boltzmann transport equation calculations.

:ref:`Back to top <cal_title_keywords>`

.. _cal_ibte_error:
.. param-card:: ibte_error
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.00001
   :block: conductivity

    Convergence threshold for the iterative solution of the Boltzmann Transport Equation (IBTE),  
    defined as the maximum allowed error (in Siemens/m) between consecutive iterations.  
    This parameter is only effective when ``mob_level = 'ibte'``.

:ref:`Back to top <cal_title_keywords>`

.. _cal_ibte_cycle:
.. param-card:: ibte_cycle
   :options: <b><i>*INT*</i></b>
   :default: 200
   :block: conductivity

    Maximum number of iterations allowed in the IBTE calculation loop.

:ref:`Back to top <cal_title_keywords>`

superconductivity block
^^^^^^^^^^^^^^^^^^^^^^^^

.. _cal_sc_method:
.. param-card:: sc_method
   :options: 'allen_dynes'
   :default: 'allen_dynes'
   :block: superconductivity

    Specifies the method used for superconductivity calculation.

    - ``allen_dynes``: Allen-Dynes modified McMillan equation

:ref:`Back to top <cal_title_keywords>`

.. _cal_mus:
.. param-card:: mus
   :options: <b><i>*LIST[FLOAT, FLOAT, FLOAT]*</i></b>
   :default: [0.05, 0.25, 0.01]
   :block: superconductivity

    The effective Coulomb potentials in Allen-Dynes formula.
    Should be given as **[min, max, step]**.

:ref:`Back to top <cal_title_keywords>`

.. _cal_omega_range_sc:
.. param-card:: omega_range
   :options: <b><i>*LIST[FLOAT, FLOAT]*</i></b>
   :default: [0, 100]
   :block: superconductivity

    Specifies the energy range (in meV),
    given as **[min, max]**,
    over which the Eliashberg spectrum function will be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_omega_step_sc:
.. param-card:: omega_step
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.01
   :block: superconductivity

    Specifies the energy step (in meV)
    used for the Eliashberg spectral function.

:ref:`Back to top <cal_title_keywords>`

pomega block
^^^^^^^^^^^^^

.. _cal_omega_range_po:
.. param-card:: omega_range
   :options: <b><i>*LIST[FLOAT, FLOAT]*</i></b>
   :default: [0.0, 100.0]
   :block: pomega

    Specifies the energy range (in meV),
    given as **[min, max]**,
    over which the frequency-dependent nesting function will be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_omega_step_po:
.. param-card:: omega_step
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.01
   :block: pomega

    Specifies the energy step (in meV)
    used for the frequency-dependent nesting function.

:ref:`Back to top <cal_title_keywords>`

.. _cal_pomega_restart:
.. param-card:: pomega_restart
   :options: True | False
   :default: False
   :block: pomega

    If True, the Pomega calculation will be restarted from '{self.outdir}/Pomega_restart.npz'.

:ref:`Back to top <cal_title_keywords>`


breakdown block
^^^^^^^^^^^^^^^^

.. _cal_fbd_erange:
.. param-card:: fbd_erange
   :options: <b><i>*LIST[FLOAT, FLOAT, FLOAT]*</i></b>
   :default: [0.0, 1.0, 0.01]
   :block: breakdown

    Specifies the energy range and step (in eV),
    given as **[min, max, step]**,
    over which the breakdown properties will be calculated.
    The energy is referenced to the conduction band minimum.

:ref:`Back to top <cal_title_keywords>`

.. _cal_fbd_rate_file:
.. param-card:: fbd_rate_file
   :options: None | <b><i>*STRING*</i></b>
   :default: None
   :block: breakdown

    **This feature is not yet implemented.**
    Path to the data file containing
    the computed scattering rates and energy loss rates,
    generated by **QTrans**.
    If not **None**, the program will
    read the scattering rates and energy loss rates from the file
    instead of computing them.

:ref:`Back to top <cal_title_keywords>`

.. _cal_fbd_effective_mass:
.. param-card:: fbd_effective_mass
   :options: <b><i>*FLOAT*</i></b>
   :default: 1.0
   :block: breakdown

    The electron effective mass (in :math:`m_\text{e}`)
    in intrinsic breakdown field strength formula.


:ref:`Back to top <cal_title_keywords>`

.. _cal_fbd_exp_bandgap:
.. param-card:: fbd_exp_bandgap
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.0
   :block: breakdown

    The band gap value (in eV)
    in intrinsic breakdown field strength formula.

:ref:`Back to top <cal_title_keywords>`

resistivity block
^^^^^^^^^^^^^^^^^^

.. _cal_ziman_nc:
.. param-card:: ziman_nc
   :options: <b><i>*FLOAT*</i></b>
   :default: -9999.9
   :block: resistivity

    Number of charge carriers per unit cell used in Ziman's resistivity formula.  
    If set to a non-positive value, **QTrans** will automatically estimate it based 
    on the number of band crossings at the Fermi level.

:ref:`Back to top <cal_title_keywords>`

.. _cal_omega_range_resistivity:
.. param-card:: omega_range
   :options: <b><i>*LIST[FLOAT,FLOAT]*</i></b>
   :default: [0, 100]
   :block: resistivity

    Specifies the energy range (in meV),
    given as **[min, max]**,
    over which the Eliashberg spectral function :math:`\alpha^2 F(\omega)` and 
    the Eliashberg transport spectral function :math:`\alpha_{tr}^2 F(\omega)`
    will be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_omega_step_resistivity:
.. param-card:: omega_step
   :options: <b><i>*INT*</i></b>
   :default: 200
   :block: resistivity

    Specifies the energy step (in meV)
    used for calculating the Eliashberg spectral function :math:`\alpha^2 F(\omega)`
    or the Eliashberg transport spectral function :math:`\alpha_{tr}^2 F(\omega)`.

:ref:`Back to top <cal_title_keywords>`

boltzmann block
^^^^^^^^^^^^^^^^

.. _cal_over_band_edge_boltzmann:
.. param-card:: over_band_edge
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.2
   :block: boltzmann

    Specifies the width (in eV) of the energy window centered around 
    the Fermi level or band edge. This window determines the range of 
    electronic states included in Boltzmann calculations.

:ref:`Back to top <cal_title_keywords>`

.. _cal_ncarrier_boltzmann:
.. param-card:: ncarrier
   :options: <b><i>*FLOAT*</i></b>
   :default: 10000000000000000.0
   :block: boltzmann

    The carrier concentration in :math:`\text{cm}^{-3}`.

:ref:`Back to top <cal_title_keywords>`

.. _cal_temperature_lat:
.. param-card:: temperature_lat
   :options: <b><i>*FLOAT*</i></b>
   :default: 300.0
   :block: boltzmann

    Temperature of the lattice system, in Klevin.

:ref:`Back to top <cal_title_keywords>`

.. _cal_time_step:
.. param-card:: time_step
   :options: <b><i>*FLOAT*</i></b>
   :default: 1.0
   :block: boltzmann

    Time step for solving the real-time Boltzmann equation, in femtosecond.

:ref:`Back to top <cal_title_keywords>`

.. _cal_energy_step:
.. param-card:: energy_step
   :options: <b><i>*FLOAT*</i></b>
   :default: 1.0
   :block: boltzmann

    Energy step for solving the real-time Boltzmann equation, in meV.

:ref:`Back to top <cal_title_keywords>`

.. _cal_num_step:
.. param-card:: num_step
   :options: <b><i>*INT*</i></b>
   :default: 100
   :block: boltzmann

    Number of steps for solving the real-time Boltzmann equation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_norm_dist:
.. param-card:: norm_dist
   :options: True | False
   :default: False
   :block: boltzmann

    If True, the distribution will be normalized after each iteration when solving the real-time Boltzmann transport equation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_boltz_solver:
.. param-card:: boltz_solver
    :options: 'euler' | 'rk4'
    :default: 'euler'
    :block: boltzmann

    Solution method for the real-time Boltzmann transport equation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_output_step:
.. param-card:: output_step
   :options: <b><i>*INT*</i></b>
   :default: 1
   :block: boltzmann

    Number of steps between outputs when solving the real-time Boltzmann equation.

:ref:`Back to top <cal_title_keywords>`

.. _cal_efield:
.. param-card:: efield
    :options: <b><i>*LIST[FLOAT,FLOAT,FLOAT]*</i></b>
    :default: [0.0,0.0,0.0]
    :block: boltzmann

    The external electric field strength used when solving the real-time Boltzmann equation, in V/cm.

:ref:`Back to top <cal_title_keywords>`

self-energy block
^^^^^^^^^^^^^^^^^^

.. _cal_SE_type:
.. param-card:: SE_type
   :options: 'electron'
   :default: 'electron'
   :block: self-energy

    Specifies the quasiparticle type of which the self-energy will be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_SE_erange:
.. param-card:: SE_erange
    :options: <b><i>*LIST[FLOAT,FLOAT,FLOAT]*</i></b>
    :default: [0.0,0.0,0.0]
    :block: self-energy

    Specifies the energy range (in eV), given as **[min, max, step]**, over which the spectral function will be calculated.

:ref:`Back to top <cal_title_keywords>`

.. _cal_over_band_edge_self_energy:
.. param-card:: over_band_edge
   :options: <b><i>*FLOAT*</i></b>
   :default: 0.2
   :block: self-energy

    Specifies the width (in eV) of the energy window centered around 
    the Fermi level or band edge. This window determines the range of 
    electronic states included in self-energy calculations.

:ref:`Back to top <cal_title_keywords>`

.. _cal_ncarrier_self_energy:
.. param-card:: ncarrier
   :options: <b><i>*FLOAT*</i></b>
   :default: 10000000000000000.0
   :block: self-energy

    The carrier concentration in :math:`\text{cm}^{-3}`.

:ref:`Back to top <cal_title_keywords>`