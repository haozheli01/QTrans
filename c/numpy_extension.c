#include <Python.h>
#include <numpy/arrayobject.h>


// PyObject *einsum_mnij_mnij(PyObject *self, PyObject *args);
// PyObject *einsum_ij_mn2mnij(PyObject *self, PyObject *args);
// PyObject *einsum_m_n2mn(PyObject *self, PyObject *args);

// PyObject *phonon_modes_sum(PyObject *self, PyObject *args);

// PyObject *eliashberg_spectrum_cal_helper(PyObject *self, PyObject *args);
PyObject *eliashberg_spectrum_cal_EPC_helper_sparse(PyObject *self, PyObject *args);
PyObject *eliashberg_spectrum_cal_EPC_helper_sparse_soc(PyObject *self, PyObject *args);

PyObject *epc_elements_cal_helper_sparse(PyObject *self, PyObject *args);

PyObject *scattering_rate_cal_EPC_helper_sparse_soc(PyObject *self, PyObject *args);
PyObject *scattering_rate_cal_EPC_helper_sparse(PyObject *self, PyObject *args);

PyObject *breakdown_rate_cal_EPC_helper_sparse(PyObject *self, PyObject *args);

PyObject *transport_eliashberg_function_cal_EPC_iso_helper_sparse(PyObject *self, PyObject *args);

PyObject *dynamics_setup_helper_sparse(PyObject *self, PyObject *args);

PyObject *electron_self_energy_cal_EPC_helper_sparse(PyObject *self, PyObject *args);

static PyMethodDef functions[] = {
    {"eliashberg_spectrum_cal_EPC_helper_sparse", (PyCFunction)eliashberg_spectrum_cal_EPC_helper_sparse, METH_VARARGS, 0},
    {"eliashberg_spectrum_cal_EPC_helper_sparse_soc", (PyCFunction)eliashberg_spectrum_cal_EPC_helper_sparse_soc, METH_VARARGS, 0},
    {"epc_elements_cal_helper_sparse", (PyCFunction)epc_elements_cal_helper_sparse, METH_VARARGS, 0},
    {"scattering_rate_cal_EPC_helper_sparse_soc", (PyCFunction)scattering_rate_cal_EPC_helper_sparse_soc, METH_VARARGS, 0},
    {"scattering_rate_cal_EPC_helper_sparse", (PyCFunction)scattering_rate_cal_EPC_helper_sparse, METH_VARARGS, 0},
    {"breakdown_rate_cal_EPC_helper_sparse", (PyCFunction)breakdown_rate_cal_EPC_helper_sparse, METH_VARARGS, 0},
    {"transport_eliashberg_function_cal_EPC_iso_helper_sparse", (PyCFunction)transport_eliashberg_function_cal_EPC_iso_helper_sparse, METH_VARARGS, 0},
    {"dynamics_setup_helper_sparse", (PyCFunction)dynamics_setup_helper_sparse, METH_VARARGS, 0},
    {"electron_self_energy_cal_EPC_helper_sparse", (PyCFunction)electron_self_energy_cal_EPC_helper_sparse, METH_VARARGS, 0},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "C-extension for QTrans",
    -1,
    functions};

PyMODINIT_FUNC PyInit__native(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&moduledef);
}


