"""QTrans core.defaults helpers."""

from typing import Any
from typing import Dict


calculator_default_parameters:Dict[str, Dict[str, Any]] = {
    'advanced': {
        'read_sparse_grad_mat': False,
        'soc_switch': False,
        'magnetic_type': 'none',
        'magnetic_tolerance': 0.01,
        'moment_similar_tolerence': -1.0,
        'doping_charge': 0.0,
        'system_2D': False,
        'tetra_integral': False,
        'spin_length_scale': 1.0,
    },
    'dos': {
        'dos_qp': 'electron',  # electron, phonon
        'dos_npoints': 1000,
        'dos_erange': [0.0, 1.0],  # [emin, emax] in eV
        'dos_mesh_size': [32,32,32],
        'dos_smear': 10.0, # meV
        'dos_gauss_type': 0,
    },
    'dispersion': {
        'disp_renormalize': False,
        'disp_lambda': False,
        'disp_nesting': False,
        'high_symmetry_points': None,  # use in 1D-plot and 2D-plot
        'high_symmetry_labels': None,  # use in 1D-plot
        'nks_path': 200,  # use in 1D-plot
        'plot_2D_axes': None,  # use in 2D-plot
        'plot_2D_zcut': 0.0,  # use in 2D-plot
        'plot_2D_ncuts': 200,  # use in 2D-plot
        'plot_2D_smear': 10.0,  # meV, use in 2D-plot
        'dispersion_select_index': None,
        'epc_path_fix_k': None,
        'save_eigen_vectors': None, # 'ascii'
        'epc_grid_mode': False,
        'epc_k_mesh': [8, 8, 8],
        'epc_q_mesh': [8, 8, 8],
        'band_grid_mode': False,
        'band_k_mesh': [8, 8, 8],
    },
    'phonon': {
        'supercell_matrix': None,
        'unitcell_filename': "./POSCAR",
        'force_infos_filename': "./FORCE_SETS",
        'apply_correction': False,
        'quadrupole_correction': False,
        'Q_file': "./quadrupole.fmt",
        'BECs': [],
        'DL': [],
        'thickness': 0.0,
        'chiral_projection': False,
    },
    'epc': {
        'grad_mat_path': "./grad_mat.npy",
        'mat_info_sc_path': "./mat_info_sc.npy",
        'cell_range_cut': [10, 10, 10],
        'LRC_taylor_order': 0,
        'q_cut': 100.0,    # The q cut for dipole correction. We can have a test to determine the best q_cut.
        'range_separation_length_L': 100.0,    # Ang, range separation length for 2D long-range correction
        'vaccum_length': 30.0,                 # Ang, vacuum length for 2D material
    },
    'transport': {
        'k_size': [32,32,32],
        'q_size': [32,32,32],
        'bands_indices': None,
        'fermi_maxiter': 100,
        'fermi_shift': 0.0, # eV
        'temperature': 300.0,    # K
        'frequency_cutoff': 0.5,    # meV
        'smeark': 25.0,    # meV
        'smearq': 25.0,    # meV
        'gauss_type': 0,    # the type of gaussian function
        'e_thr': 75.0,  # meV
        'is_hole': False,
        'mode_resolved': True,
    },
    'conductivity': {
        'over_band_edge': 0.2,    # eV
        'MC_sampling': "none",
        'polar_split': "none",
        'cauchy_scale': 0.035,
        'sampling_seed': 1,
        'nsamples': 1000000,
        'ncarrier': 10000000000000000,  # 10^16 cm^-3
        'mob_level': "ERTA", # 'ERTA'
        'polar_rate_path': None,
        'rmp_rate_path': None,
        'epc_rate_path': None,
        'scattering_channel_dir': None,
        'ibte_error': 0.00001, # Siemens/m
        'ibte_cycle': 200,
    },
    'superconductivity': {
        'mus': [0.05, 0.25, 0.01],
        'omega_range': [0.0, 100.0], # meV
        'omega_step': 0.01,  # meV
        'epc_restart': False,
    },
    'pomega': {
        'omega_range': [0.0, 100.0], # meV
        'omega_step': 0.01,  # meV
        'pomega_restart': False,
    },
    'breakdown': {
        'fbd_erange': [0.0, 1.0, 0.01], # eV [emin, emax, estep]
        'fbd_rate_file': None,
        'fbd_effective_mass': 1.0,
        'fbd_exp_bandgap': 0.0,
    },
    'resistivity': {
        'ziman_nc': -9999.9,
        'omega_range': [0, 100], # meV
        'omega_step': 0.01,  # meV
    },
    'boltzmann': {
        'over_band_edge': 0.2,    # eV
        'ncarrier': 10000000000000000,  # 10^16 cm^-3
        'temperature_lat': 300.0,    # K
        'time_step': 1.0,    # fs
        'energy_step': 1.0,    # meV
        'num_step': 100, 
        'norm_dist': False,
        'boltz_solver': "euler", # 'euler' or 'rk4'
        'output_step': 1,    # fs
        'efield': [0.0,0.0,0.0], # external E field, V/cm
    },
    'self-energy': {
        'SE_type': None, # electron, phonon
        'SE_erange': [-1.0, 1.0, 0.01],  # [emin, emax, estep] in eV
        'over_band_edge': 0.2, # eV
        'ncarrier': 10000000000000000,  # 10^16 cm^-3
        'epc_se_path': None,
        'disp_path': None,
    },
    'fermi_surface': {
        'mesh_file': './kpt.dat',
    },
}

preparer_default_parameters:Dict[str, Dict[str, Any]] = {
    'software_path':{
        'vasp_path': './vasp_std',
        'vaspkit_path': './vaspkit',
        'vasp_pot_dir': './',
        'openmx_path': './openmx',
        'openmx_pp_path': './openmx_postprocess',
        'openmx_read_path': './read_openmx',
        'openmx_dftdata_dir': './',
        'abacus_path': './abacus',
        'abacus_pp_path': './abacus_postprocess',
        'abacus_dftdata_dir': './',
        'abacus_orb_dir': './',
        'jx_path': './jx'
    },
    'material':{
        'struct_path': None,
        'std_symm_cell': False,
        'symm_tol': 0.00001,
        'magnetic_tolerance': 0.01,
        'moment_similar_tolerence': -1.0,
        'magnetic_type': 'none',
        'soc_switch': False,
        'DFT_software': None,
        'hse_switch': False,
        'nao_max': 26,
        'xc_functional': 'GGA-PBE',
    },
    'phonon_prep':{
        'ph_sc_matrix': 'auto',
        'ph_sc_auto_length': 10.0, #  ang
        'ph_sc_disp': 0.01, # ang
    },
    'scf_params':{
        'scf_uc_kpt': 'auto',
        'scf_rec_density': 50.0, # ang
    },
    'grad_epc':{
        'dSon_delta_r': 0.001, # ang
        'grad_sparse_thr': 0.00001, # Hartree
        'epc_sc_matrix': 'auto',
        'epc_sc_auto_length': 10.0, #  ang
        'epc_sc_disp': 0.01, # ang
        'graph_data_sc_path': None,
        'pred_H_sc_path': None,
        'mat_info_sc_path': None,
        'use_prediction': False,
        'gen_force_constants': False,
        'fd_method': 'forward',
    },
    'graph_pack':{
        'HSout_filename': None,
        'nonscf_filename': None,
        'stdin_filename': None,
        'vbout_filename': None,
        'data_folder': None,  # using glob.glob to match all files
        'converge_step_ratio': 0.9,
        'pack_atomic_force': False,
        'abacus_scf_log': None,
        'abacus_out_folder': None,
    },
}
