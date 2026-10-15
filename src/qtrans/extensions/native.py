"""Validated access to the optional compiled QTrans implementation."""

try:
    from qtrans import _native
except ImportError as error:
    _native = None
    _native_import_error = error
else:
    _native_import_error = None


class NativeExtensionUnavailable(RuntimeError):
    """Raised when a calculation needs the QTrans native extension."""


def _call(name, *args):
    if _native is None:
        raise NativeExtensionUnavailable(
            "The QTrans native extension is unavailable. Reinstall QTrans with "
            "a supported C compiler and OpenMP runtime. Original error: {}".format(
                _native_import_error
            )
        )
    try:
        function = getattr(_native, name)
    except AttributeError:
        raise NativeExtensionUnavailable(
            "The installed QTrans native extension does not provide {!r}.".format(
                name
            )
        )
    return function(*args)


def eliashberg_spectrum_cal_EPC_helper_sparse(*args):
    return _call("eliashberg_spectrum_cal_EPC_helper_sparse", *args)


def eliashberg_spectrum_cal_EPC_helper_sparse_batch(*args):
    return _call("eliashberg_spectrum_cal_EPC_helper_sparse_batch", *args)


def eliashberg_spectrum_cal_EPC_helper_sparse_soc(*args):
    return _call("eliashberg_spectrum_cal_EPC_helper_sparse_soc", *args)


def epc_elements_cal_helper_sparse(*args):
    return _call("epc_elements_cal_helper_sparse", *args)


def scattering_rate_cal_EPC_helper_sparse_soc(*args):
    return _call("scattering_rate_cal_EPC_helper_sparse_soc", *args)


def scattering_rate_cal_EPC_helper_sparse(*args):
    return _call("scattering_rate_cal_EPC_helper_sparse", *args)


def breakdown_rate_cal_EPC_helper_sparse(*args):
    return _call("breakdown_rate_cal_EPC_helper_sparse", *args)


def transport_eliashberg_function_cal_EPC_iso_helper_sparse(*args):
    return _call("transport_eliashberg_function_cal_EPC_iso_helper_sparse", *args)


def dynamics_setup_helper_sparse(*args):
    return _call("dynamics_setup_helper_sparse", *args)


def electron_self_energy_cal_EPC_helper_sparse(*args):
    return _call("electron_self_energy_cal_EPC_helper_sparse", *args)


def prepare_sparse_grad_cache(*args):
    return _call("prepare_sparse_grad_cache", *args)


def electron_self_energy_cal_EPC_helper_sparse_batch(*args):
    """Accumulate total self-energy; optional argument 18 receives branch terms.

    The optional buffer is writable C-contiguous complex128 with axes
    (k, initial band, phonon branch, energy), in Hartree. Both outputs are
    accumulated in place and must be initialized by the caller.
    """
    return _call("electron_self_energy_cal_EPC_helper_sparse_batch", *args)


def prepare_grad_mat(*args):
    return _call("prepare_grad_mat", *args)


def free_grad_mat_context(*args):
    return _call("free_grad_mat_context", *args)


def epc_profile_reset():
    return _call("epc_profile_reset")


def epc_profile_snapshot():
    return _call("epc_profile_snapshot")
