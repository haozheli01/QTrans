"""QTrans exception types."""


class InputParameterError(Exception):
    """
    Exception raised for errors in the input parameters.

    Attributes:
        block: The name of the block that contains the parameter.
        parameter: The name of the parameter that caused the error
        index: The index of the error.
    """

    """
    Error Table:
    1001: Cannot find <basic> block in input file.
    1002: The prep_mode in <basic> block is not supported.
    1003: The struct_path in material block is not existed or the format is not supported.
    1004: The format of ph_sc_matrix in phonon_prep block is not supported.
    1005: The format of vasp_uc_kpt in vasp_params block is not supported.
    1006: The format of epc_sc_matrix in grad_epc block is not supported.
    1007: The magnetic_type in material block is not supported.
    1008: The DFT_software in material block is not supported.
    1009: The nao_max in material block is not supported.
    1010: The mat_info_sc_path in grad_epc block is not existed or the format is not supported.
    1011: The graph_data_sc_path in grad_epc block is not existed or the format is not supported.
    1012: The pred_H_sc_path in grad_epc block is not existed or the format is not supported.
    1013: The elements of epc_sc_matrix in grad_epc block need to be positive odd integers.
    1014: The HSout_filename in graph_pack block is not existed or the format is not supported.
    1015: The nonscf_filename in graph_pack block is not existed or the format is not supported.
    1016: The stdin_filename in graph_pack block is not existed or the format is not supported.
    1017: The pack_atomic_force in graph_pack block is supported only while packing DFT graph data.
    1018: The abacus_scf_log in graph_pack block is not existed or the format is not supported.
    1019: The exchange-correlation function is not supported.

    5001: Cannot find the required <basic> block in the input file, or a required parameter is missing.
    5002: The 'cal_mode' in <basic> block is not supported.
    5003: The file specified by 'graph_data_path_uc' in the <basic> block does not exist or is not readable.
    5004: The 'Ham_type' in <basic> block is not supported.
    5005: The atomic basis set as '{Ham_type}_{nao_max}' is not supported.
    5006: The directory specified by 'outdir' in the <basic> block is not accessible or cannot be created.
    5007: ...
    5021: The 'read_sparse_grad_mat' in <advanced> block must be a boolean.
    5022: The 'soc_switch' in <advanced> block must be a boolean.
    5023: The 'magnetic_type' in <advanced> block is not supported.
    5024: The 'magnetic_tolerance' in <advanced> block must be a number.
    5025: The 'moment_similar_tolerence' in <advanced> block must be a number.
    5026: The 'doping_charge' in <advanced> block must be a number.
    5027: The 'system_2D' in <advanced> block must be a boolean.
    5028: The 'tetra_integral' in <advanced> block must be a boolean.
    5029: ...
    5031: The 'smooth_gauge' in <advanced> block must be a boolean.
    5032: ...
    5041: The 'supercell_matrix' in <phonon> block must be a 3*3 float matrix.
    5042: The file specified by 'unitcell_filename' in the <phonon> block does not exist or is not readable.
    5043: The file specified by 'force_infos_filename' in the <phonon> block does not exist or is not readable.
    5044: The 'apply_correction' in <phonon> block must be a boolean.
    5045: The 'quadrupole_correction' in <phonon> block must be a boolean.
    5046: The file specified by 'Q_file' in the <phonon> block does not exist or is not readable.
    5047: The 'BECs' in <phonon> block must be a {natoms}*3*3 float matrix.
    5048: The 'DL' in <phonon> block must be a 3*3 float matrix.
    5049: The 'thickness' in <phonon> block must be a non-negative float.
    5050: The 'chiral_projection' in <phonon> block must be a boolean.
    5051: ...
    5081: The 'disp_renormalize' in <dispersion> block must be a boolean.
    5082: The 'disp_lambda' in <dispersion> block must be a boolean.
    5083: The 'disp_nesting' in <dispersion> block must be a boolean.
    5084: The 'high_symmetry_points' in <dispersion> block is not supported.
    5085: The 'high_symmetry_labels' in <dispersion> block is not supported.
    5086: The 'nks_path' in <dispersion> block must be a positive integer.
    5087: The 'plot_2D_axes' in <dispersion> block is not supported.
    5088: The 'plot_2D_zcut' in <dispersion> block must be a number.
    5089: The 'plot_2D_ncuts' in <dispersion> block must be a positive integer.
    5090: The 'plot_2D_smear' in <dispersion> block must be a positive number.
    5091: The 'dispersion_select_index' in <dispersion> block is not supported.
    5092: The 'epc_path_fix_k' in <dispersion> block must be a list of 3 numbers.
    5093: The 'save_eigen_vectors' in <dispersion> block is not supported.
    5094: ...
    5101: The 'dos_qp' in <dos> block is not supported.
    5102: The 'dos_npoints' in <dos> block must be a positive integer.
    5103: The 'dos_erange' in <dos> block must be set as '[emin, emax]'.
    5104: The 'dos_mesh_size' in <dos> block must be set as '[npx, npy, npz]'.
    5105: The 'dos_smear' in <dos> block must be a positive number.
    5106: The 'dos_gauss_type' in <dos> block is not supported.
    5107: ...
    5141: The file specified by 'grad_mat_path' in the <epc> block does not exist or is not readable.
    5142: The file specified by 'mat_info_sc_path' in the <epc> block does not exist or is not readable.
    5143: The 'cell_range_cut' in <epc> block must be a list of 3 integers.
    5144: The 'LRC_taylor_order' in <epc> block must be 0 or 1.
    5145: The 'q_cut' in <epc> block must be a number.
    5146: ...
    5148: The 'phonon_frequency_cutoff' in <epc> block must be a number.
    5182: The 'k_size' in <transport> block must be set as '[nkx, nky, nkz]'.
    5183: The 'q_size' in <transport> block must be set as '[nqx, nqy, nqz]'.
    5184: The 'bands_indices' in <transport> block is not supported.
    5185: The 'fermi_maxiter' in <transport> block must be a positive integer.
    5186: The 'fermi_shift' in <transport> block must be a number.
    5187: The 'temperature' in <transport> block must be a positive number.
    5189: The 'smeark' in <transport> block must be a positive number.
    5190: The 'smearq' in <transport> block must be a positive number.
    5191: The 'gauss_type' in <transport> block must be an integer.
    5192: The 'e_thr' in <transport> block must be a positive number.
    5193: The 'is_hole' in <transport> block must be a boolean.
    5194: The 'mode_resolved' in <transport> block must be a boolean.
    5195: ...
    5201: The 'over_band_edge' in <conductivity> block must be a number.
    5202: The 'MC_sampling' in <conductivity> block is not supported.
    5203: The 'polar_split' in <conductivity> block is not supported.
    5204: The 'cauchy_scale' in <conductivity> block must be a positive number.
    5205: The 'sampling_seed' in <conductivity> block must be a number.
    5206: The 'nsamples' in <conductivity> block must be a positive integer.
    5207: The 'ncarrier' in <conductivity> block must be a number.
    5208: The 'mob_level' in <conductivity> block is not supported.
    5209: The file specified by 'polar_rate_path' in the <conductivity> block does not exist or is not readable.
    5210: The file specified by 'rmp_rate_path' in the <conductivity> block does not exist or is not readable.
    5211: The file specified by 'epc_rate_path' in the <conductivity> block does not exist or is not readable.
    5212: ...
    5213: The folder specified by 'scattering_channel_dir' in the <conductivity> block does not exist.
    5214: The 'ibte_error' in <conductivity> block must be a positive number.
    5215: The 'ibte_cycle' in <conductivity> block must be a positive integer.
    5216: ...
    5221: The 'ziman_nc' in <resistivity> block must be a number.
    5222: The 'omega_range' in <resistivity> block must be a list of 2 numbers.
    5223: The 'omega_step' in <resistivity> block must be a positive number.
    5224: ...
    5241: The 'mus' in <superconductivity> block must be a list of 3 numbers.
    5242: The 'omega_range' in <superconductivity> block must be a list of 2 numbers.
    5243: The 'omega_step' in <superconductivity> block must be a positive number.
    5244: The 'sc_method' in <superconductivity> block is not supported.
    5245: The Eliashberg approximation is not supported.
    5246: The 'epc_restart' flag must be a boolean.
    5247: The Eliashberg temperature range is invalid.
    5248: The Matsubara cutoff must be positive.
    5249: The Coulomb cutoff must be positive.
    5250: The nonlinear iteration limit must be a positive integer.
    5251: The eigenvalue iteration limit must be a positive integer.
    5252: The Matsubara point limit must be a positive integer.
    5253: The nonlinear convergence tolerance must be positive.
    5254: The eigenvalue tolerance must be positive.
    5255: The Tc tolerance must be positive.
    5256: The mixing fraction must be in (0, 1].
    5257: The initial gap must be positive.
    5258: The 'save_matsubara' flag must be a boolean.
    5259: The Eliashberg 'restart' flag must be a boolean.
    5260: A readable state-resolved Eliashberg data file is required.
    5261: The 'fbd_erange' in <breakdown> block must be a list of 3 numbers.
    5262: The file specified by 'fbd_rate_file' in the <breakdown> block does not exist or is not readable.
    5263: The 'fbd_effective_mass' in <breakdown> block must be a positive number.
    5264: The 'fbd_exp_bandgap' in <breakdown> block must be a positive number.
    5265: ...
    5281: The 'over_band_edge' in <boltzmann> block must be a number.
    5282: The 'ncarrier' in <boltzmann> block must be a number.
    5283: The 'temperature_lat' in <boltzmann> block must be a positive number.
    5284: The 'time_step' in <boltzmann> block must be a positive number.
    5285: The 'energy_step' in <boltzmann> block must be a positive number.
    5286: The 'num_step' in <boltzmann> block must be a positive integer.
    5287: The 'norm_dist' in <boltzmann> block must be a boolean.
    5288: The 'boltz_solver' in <boltzmann> block is not supported.
    5289: The 'output_step' in <boltzmann> block must be a positive integer.
    5290: The 'efield' in <boltzmann> block must be a list of 3 numbers.
    5291: ...
    5301: The 'SE_type' in <self-energy> block is not supported.
    5302: The 'SE_erange' in <self-energy> block must be a list of 3 numbers.
    5303: The 'over_band_edge' in <self-energy> block must be a number.
    5304: The 'ncarrier' in <self-energy> block must be a number.
    5305: The file specified by 'epc_se_path' in the <self-energy> block does not exist or is not readable.
    5306: ...
    5307: The file specified by 'disp_path' in the <self-energy> block does not exist or is not readable.
    5308: ...
    5321: The file specified by 'mesh_file' in the <fermi_surface> block does not exist or is not readable.

    5441: The 'omega_range' in <pomega> block must be a list of 2 numbers.
    5442: The 'omega_step' in <pomega> block must be a positive number.
    5443: ...


    """


    def __init__(self, index:int, block:str, parameter:str=None, infos:str=None):
        self.index = index
        self.block = block
        self.parameter = parameter
        self.infos = infos
        if self.parameter == None:
            super().__init__(f"Error({index}): Please check the {block} input block.\n{infos}")
        else:
            super().__init__(f"Error({index}): Please check the input parameter {parameter} in block {block}.\n{infos}")


"""
IOError:
2001: Failed to write POSCAR file.
2002: Failed to write KPOINTS file.
2003: Failed to write POTCAR file.
2004: Failed to write INCAR file.

6001: 
6011: Failed to write spectrum function data.
6012: Failed to write self energy data.
6013: Failed to read band dispersion data.
6014: Failed to read self energy data.

RuntimeError:
3001: Failed to generate POTCAR file using vaspkit.
3002: Failed to generate INCAR file using pymatgen.
3003: Failed to create folder in phonon calculation.
3004: Failed to create <perturb> folder in electron-phonon-coupling gradient matrix calculation.
3005: Failed to generate dSon when preparing gradient matrix.
3006: The spinor mode used in the OpenMX calculation is inconsistent with the setting specified during graph data packing.
3007: The graph data does not contain atomic forces while generating FORCE_CONSTANTS.

7001: Superconductivity can only be calculated for metallic systems.
7002: Failed to create <self_energy> folder while outputing self energies.
7003: Unsupported reducible mesh type.
7004: When derive coupling strength from electron self energy, the energy range of it must contain zero.
7005: The number of k-points in the provided band dispersion does not match nks_path in the dispersion block when calculating the spectral function.
7006: The nesting function can only be calculated for metallic systems.

NotImplementedError:
9001: Currently, LRC_taylor_order = 1 is supported only for non-magnetic systems without SOC.
9002: Renormalization is not implemented for the current calculation mode.
9003: ...
9004: The k grid should be either 1D, 2D or 3D.
9005: Lambda calculation is not implemented for the current calculation mode.


Special RuntimeError:
9999: Serious problem. Please contact the developer team.
"""
