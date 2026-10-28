#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#define INTP(a) ((int *)PyArray_DATA(a))
#define DOUBLEP(a) ((double *)PyArray_DATA(a))
#define FLOATP(a) ((float *)PyArray_DATA(a))
#define DCOMPLEXP(a) ((double complex *)PyArray_DATA(a))
#define FCOMPLEXP(a) ((float complex *)PyArray_DATA(a))
#define BOOLP(a) ((bool *)PyArray_DATA(a))

// 定义宏来计算一维索引
#define INDEX2D(dim2, i, j) ((i) * (dim2) + (j))
#define INDEX3D(dim2, dim3, i, j, k) ((i) * ((dim2) * (dim3)) + (j) * (dim3) + (k))
#define INDEX4D(dim2, dim3, dim4, i, j, k, l) ((i) * ((dim2) * (dim3) * (dim4)) + (j) * ((dim3) * (dim4)) + (k) * (dim4) + (l))
#define INDEX5D(dim2, dim3, dim4, dim5, i, j, k, l, m) ((i) * ((dim2) * (dim3) * (dim4) * (dim5)) + (j) * ((dim3) * (dim4) * (dim5)) + (k) * ((dim4) * (dim5)) + (l) * (dim5) + (m))
#define INDEX6D(dim2, dim3, dim4, dim5, dim6, i, j, k, l, m, n) ((i) * ((dim2) * (dim3) * (dim4) * (dim5) * (dim6)) + (j) * ((dim3) * (dim4) * (dim5) * (dim6)) + (k) * ((dim4) * (dim5) * (dim6)) + (l) * ((dim5) * (dim6)) + (m) * (dim6) + (n))

#define UTILS_SQRT_INVPI 0.5641895835477563
#define UTILS_INV_SQRT_2 0.7071067811865475
#define UTILS_SQRT_2 1.4142135623730951

#define TRACE printf("Executing line %d in file %s\n", __LINE__, __FILE__);

double w0gauss(double *x, int *ngauss, double *result, double *arg)
{
    *result = 0.0;
    // arg;

    if (*ngauss == -99)
    {
        // Fermi-Dirac smearing
        if (fabs(*x) < 36.0)
        {
            *result = 1.0 / (2.0 + exp(-*x) + exp(*x));
        }
    }
    else if (*ngauss == -1)
    {
        // cold smearing  (Marzari-Vanderbilt-DeVita-Payne)
        *arg = fmin(200.0, (*x - UTILS_INV_SQRT_2) * (*x - UTILS_INV_SQRT_2));
        *result = UTILS_SQRT_INVPI * exp(-*arg) * (2.0 - UTILS_SQRT_2 * (*x));
    }
    else if (*ngauss > 10 || *ngauss < 0)
    {
        printf("Higher order smearing of %d is untested and unstable\n", abs(*ngauss));
    }
    else
    {
        // Methfessel-Paxton
        *arg = fmin(200.0, *x * (*x));
        *result = exp(-*arg) * UTILS_SQRT_INVPI;
        if (*ngauss == 0)
        {
            return *result;
        }
        else
        {
            double hd = 0.0;
            double hp = exp(-*arg);
            int ni = 0;
            double a = UTILS_SQRT_INVPI;

            for (int i = 0; i < *ngauss; i++)
            {
                hd = 2.0 * (*x) * hp - 2.0 * ni * hd;
                ni = ni + 1;
                a = -a / ((i + 1) * 4.0);
                hp = 2.0 * (*x) * hd - 2.0 * ni * hp;
                ni = ni + 1;
                *result = *result + a * hp;
            }

            return *result;
        }
    }

    return *result;
}


/* Match Python w0gauss for superconductivity without changing other kernels. */
static inline double superconductivity_delta(double x, int gauss_type)
{
    if (gauss_type == 0)
        return exp(-fmin(200.0, x * x)) * UTILS_SQRT_INVPI;
    if (gauss_type == -99) {
        double arg = fmin(36.0, fabs(x));
        return 1.0 / (2.0 + 2.0 * cosh(arg));
    }
    double result, arg;
    return w0gauss(&x, &gauss_type, &result, &arg);
}

static int validate_superconductivity_gauss(long gauss_type)
{
    if (gauss_type == -99 || gauss_type == -1 ||
            (gauss_type >= 0 && gauss_type <= 10))
        return 1;
    PyErr_SetString(PyExc_ValueError,
                    "gauss_type must be -99, -1, or an integer from 0 to 10");
    return 0;
}

typedef struct {
    int icell;
    int jcell;
    int iatom;
    int idir;
    double *data_ptr;
    int *indices_ptr;
    int *indptr_ptr;
    double complex phase_product; // conj(phase_kpq[icell]) * phase_k[jcell]
} DictEntry;

/* Process-local prepared CSR metadata.  The referenced NumPy arrays are kept
 * alive so the raw pointers remain valid for the lifetime of the context. */
typedef struct EPCGradMatContext {
    PyObject *grad_mat_obj;
    Py_ssize_t n_entries;
    int released;
    int *icell_arr, *jcell_arr, *iatom_arr, *idir_arr;
    double **data_arr;
    int **indices_arr, **indptr_arr;
    PyObject **data_obj, **indices_obj, **indptr_obj;
    struct EPCGradMatContext *next;
} EPCGradMatContext;

/* Batch self-energy uses a compact cache with the same ownership semantics as
 * the historical QTrans20260721 SparseGradCache.  Keeping this separate from
 * EPCGradMatContext preserves the scalar helper API while allowing the batch
 * kernel to consume stable CSR pointers without touching Python dictionaries. */
#define SPARSE_GRAD_CACHE_CAPSULE "QTrans.SparseGradCache"
typedef struct SparseGradCache {
    Py_ssize_t n_entries;
    int norbs;
    int *icell, *jcell, *iatom, *idir;
    double **data;
    int **indices, **indptr;
    PyObject *owner;
} SparseGradCache;

static void sparse_grad_cache_release(SparseGradCache *cache)
{
    if (cache == NULL) return;
    free(cache->icell); free(cache->jcell); free(cache->iatom);
    free(cache->idir); free(cache->data); free(cache->indices);
    free(cache->indptr); Py_XDECREF(cache->owner); free(cache);
}

static void sparse_grad_cache_capsule_destructor(PyObject *capsule)
{
    SparseGradCache *cache = (SparseGradCache *)PyCapsule_GetPointer(
        capsule, SPARSE_GRAD_CACHE_CAPSULE);
    if (cache == NULL) { PyErr_Clear(); return; }
    sparse_grad_cache_release(cache);
}

static int sparse_cache_require_vector(PyObject *obj, int typenum,
                                       const char *name,
                                       PyArrayObject **array_out)
{
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return 0;
    }
    PyArrayObject *array = (PyArrayObject *)obj;
    if (PyArray_TYPE(array) != typenum || PyArray_NDIM(array) != 1 ||
        !PyArray_ISCARRAY_RO(array)) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a one-dimensional C-contiguous array with dtype %s",
                     name, typenum == NPY_DOUBLE ? "float64" : "int32");
        return 0;
    }
    *array_out = array;
    return 1;
}

static int require_c_array(PyObject *obj, int typenum, int ndim, int writable,
                           const char *name, PyArrayObject **array_out)
{
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return 0;
    }
    PyArrayObject *array = (PyArrayObject *)obj;
    if (PyArray_TYPE(array) != typenum || PyArray_NDIM(array) != ndim ||
        !PyArray_IS_C_CONTIGUOUS(array) || !PyArray_ISALIGNED(array) ||
        (writable && !PyArray_ISWRITEABLE(array))) {
        PyErr_Format(PyExc_ValueError,
                     "%s has the wrong dtype, dimensions, contiguity, alignment, or writability",
                     name);
        return 0;
    }
    *array_out = array;
    return 1;
}

static int checked_size_mul(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int checked_size_add(size_t left, size_t right, size_t *result)
{
    if (right > SIZE_MAX - left) return 0;
    *result = left + right;
    return 1;
}

static EPCGradMatContext *epc_contexts = NULL;
static int epc_profile_enabled = 0;
static volatile unsigned long long epc_profile_calls = 0;
static volatile unsigned long long epc_profile_parse_ns = 0;
static volatile unsigned long long epc_profile_alloc_ns = 0;
static volatile unsigned long long epc_profile_phase_ns = 0;
static volatile unsigned long long epc_profile_kernel_ns = 0;
static volatile unsigned long long epc_profile_free_ns = 0;

static unsigned long long epc_clock_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return 0;
    return (unsigned long long)ts.tv_sec * 1000000000ULL +
           (unsigned long long)ts.tv_nsec;
}

static void epc_profile_add(volatile unsigned long long *counter,
                            unsigned long long value)
{
    if (epc_profile_enabled) {
#pragma omp critical(qtrans_epc_profile)
        *counter += value;
    }
}

static void epc_profile_call(void)
{
    if (epc_profile_enabled) {
#pragma omp critical(qtrans_epc_profile)
        epc_profile_calls += 1;
    }
}

PyObject *epc_profile_reset(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    epc_profile_calls = 0;
    epc_profile_parse_ns = 0;
    epc_profile_alloc_ns = 0;
    epc_profile_phase_ns = 0;
    epc_profile_kernel_ns = 0;
    epc_profile_free_ns = 0;
    epc_profile_enabled = 1;
    Py_RETURN_NONE;
}

PyObject *epc_profile_snapshot(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K}",
        "calls", epc_profile_calls,
        "parse_ns", epc_profile_parse_ns,
        "alloc_ns", epc_profile_alloc_ns,
        "phase_ns", epc_profile_phase_ns,
        "kernel_ns", epc_profile_kernel_ns,
        "free_ns", epc_profile_free_ns);
}

static EPCGradMatContext *epc_context_find(PyObject *grad_mat_obj)
{
    EPCGradMatContext *ctx = epc_contexts;
    while (ctx != NULL) {
        if (ctx->grad_mat_obj == grad_mat_obj) return ctx;
        ctx = ctx->next;
    }
    return NULL;
}

static void epc_context_release_arrays(EPCGradMatContext *ctx)
{
    if (ctx == NULL || ctx->released) return;
    for (Py_ssize_t i = 0; i < ctx->n_entries; ++i) {
        Py_XDECREF(ctx->data_obj[i]);
        Py_XDECREF(ctx->indices_obj[i]);
        Py_XDECREF(ctx->indptr_obj[i]);
    }
    Py_XDECREF(ctx->grad_mat_obj);
    ctx->grad_mat_obj = NULL;
    free(ctx->icell_arr); free(ctx->jcell_arr); free(ctx->iatom_arr);
    free(ctx->idir_arr); free(ctx->data_arr); free(ctx->indices_arr);
    free(ctx->indptr_arr); free(ctx->data_obj); free(ctx->indices_obj);
    free(ctx->indptr_obj);
    ctx->released = 1;
}

static void epc_context_release(EPCGradMatContext *ctx)
{
    if (ctx == NULL) return;
    epc_context_release_arrays(ctx);
    free(ctx);
}

static void epc_capsule_destructor(PyObject *capsule)
{
    EPCGradMatContext *ctx = (EPCGradMatContext *)PyCapsule_GetPointer(
        capsule, "qtrans.epc_grad_mat");
    if (ctx == NULL) return;
    EPCGradMatContext **link = &epc_contexts;
    while (*link != NULL && *link != ctx) link = &(*link)->next;
    if (*link == ctx) *link = ctx->next;
    epc_context_release(ctx);
}

PyObject *prepare_sparse_grad_cache(PyObject *self, PyObject *args)
{
    import_array();
    (void)self;
    PyObject *grad_mat_obj;
    if (!PyArg_ParseTuple(args, "O:prepare_sparse_grad_cache", &grad_mat_obj))
        return NULL;
    if (!PyDict_Check(grad_mat_obj)) {
        PyErr_SetString(PyExc_TypeError, "grad_mat must be a dict");
        return NULL;
    }
    Py_ssize_t n_entries = PyDict_Size(grad_mat_obj);
    SparseGradCache *cache = (SparseGradCache *)calloc(1, sizeof(*cache));
    if (cache == NULL) return PyErr_NoMemory();
    cache->n_entries = n_entries;
    if (n_entries > 0) {
        cache->icell = (int *)malloc((size_t)n_entries * sizeof(int));
        cache->jcell = (int *)malloc((size_t)n_entries * sizeof(int));
        cache->iatom = (int *)malloc((size_t)n_entries * sizeof(int));
        cache->idir = (int *)malloc((size_t)n_entries * sizeof(int));
        cache->data = (double **)malloc((size_t)n_entries * sizeof(double *));
        cache->indices = (int **)malloc((size_t)n_entries * sizeof(int *));
        cache->indptr = (int **)malloc((size_t)n_entries * sizeof(int *));
        if (cache->icell == NULL || cache->jcell == NULL ||
            cache->iatom == NULL || cache->idir == NULL || cache->data == NULL ||
            cache->indices == NULL || cache->indptr == NULL) {
            sparse_grad_cache_release(cache); return PyErr_NoMemory();
        }
    }
    PyObject *key_obj, *value_obj;
    Py_ssize_t pos = 0, entry = 0;
    while (PyDict_Next(grad_mat_obj, &pos, &key_obj, &value_obj)) {
        if (!PyTuple_Check(key_obj) || PyTuple_GET_SIZE(key_obj) != 4 ||
            !PyTuple_Check(value_obj) || PyTuple_GET_SIZE(value_obj) != 3) {
            PyErr_SetString(PyExc_ValueError,
                            "grad_mat entries must be key=(icell,jcell,iatom,idir), value=(data,indices,indptr)");
            sparse_grad_cache_release(cache); return NULL;
        }
        long values[4];
        for (int i = 0; i < 4; ++i) {
            values[i] = PyLong_AsLong(PyTuple_GET_ITEM(key_obj, i));
            if (PyErr_Occurred() || values[i] < 0 || values[i] > INT_MAX) {
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_ValueError, "grad_mat key values must be non-negative C integers");
                sparse_grad_cache_release(cache); return NULL;
            }
        }
        if (values[3] > 2) {
            PyErr_SetString(PyExc_ValueError, "grad_mat idir must be 0, 1, or 2");
            sparse_grad_cache_release(cache); return NULL;
        }
        PyArrayObject *data_obj, *indices_obj, *indptr_obj;
        if (!sparse_cache_require_vector(PyTuple_GET_ITEM(value_obj, 0), NPY_DOUBLE,
                                         "grad_mat data", &data_obj) ||
            !sparse_cache_require_vector(PyTuple_GET_ITEM(value_obj, 1), NPY_INT32,
                                         "grad_mat indices", &indices_obj) ||
            !sparse_cache_require_vector(PyTuple_GET_ITEM(value_obj, 2), NPY_INT32,
                                         "grad_mat indptr", &indptr_obj)) {
            sparse_grad_cache_release(cache); return NULL;
        }
        npy_intp nnz = PyArray_SIZE(data_obj);
        npy_intp nrows = PyArray_SIZE(indptr_obj) - 1;
        if (nnz != PyArray_SIZE(indices_obj) || nnz > INT_MAX ||
            nrows <= 0 || nrows > INT_MAX) {
            PyErr_SetString(PyExc_ValueError, "invalid CSR sizes in grad_mat entry");
            sparse_grad_cache_release(cache); return NULL;
        }
        int *indices = INTP(indices_obj), *indptr = INTP(indptr_obj);
        if (cache->norbs == 0) cache->norbs = (int)nrows;
        if (cache->norbs != (int)nrows || indptr[0] != 0 ||
            indptr[nrows] != (int)nnz) {
            PyErr_SetString(PyExc_ValueError, "inconsistent CSR dimensions in grad_mat entry");
            sparse_grad_cache_release(cache); return NULL;
        }
        for (npy_intp i = 0; i < nrows; ++i) {
            if (indptr[i] < 0 || indptr[i] > indptr[i + 1] ||
                indptr[i + 1] > (int)nnz) {
                PyErr_SetString(PyExc_ValueError, "grad_mat CSR indptr must be monotonic");
                sparse_grad_cache_release(cache); return NULL;
            }
        }
        for (npy_intp i = 0; i < nnz; ++i) {
            if (indices[i] < 0 || indices[i] >= (int)nrows) {
                PyErr_SetString(PyExc_ValueError, "grad_mat CSR index out of range");
                sparse_grad_cache_release(cache); return NULL;
            }
        }
        cache->icell[entry] = (int)values[0]; cache->jcell[entry] = (int)values[1];
        cache->iatom[entry] = (int)values[2]; cache->idir[entry] = (int)values[3];
        cache->data[entry] = DOUBLEP(data_obj); cache->indices[entry] = indices;
        cache->indptr[entry] = indptr; ++entry;
    }
    Py_INCREF(grad_mat_obj); cache->owner = grad_mat_obj;
    PyObject *capsule = PyCapsule_New(cache, SPARSE_GRAD_CACHE_CAPSULE,
                                      sparse_grad_cache_capsule_destructor);
    if (capsule == NULL) { sparse_grad_cache_release(cache); return NULL; }
    return capsule;
}

PyObject *prepare_grad_mat(PyObject *self, PyObject *args)
{
    import_array();
    (void)self;
    PyObject *grad_mat_obj;
    if (!PyArg_ParseTuple(args, "O:prepare_grad_mat", &grad_mat_obj)) return NULL;
    if (!PyDict_Check(grad_mat_obj)) {
        PyErr_SetString(PyExc_TypeError, "grad_mat must be a dictionary");
        return NULL;
    }
    unsigned long long t_alloc = epc_clock_ns();
    EPCGradMatContext *ctx = (EPCGradMatContext *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) { PyErr_NoMemory(); return NULL; }
    ctx->grad_mat_obj = grad_mat_obj; Py_INCREF(grad_mat_obj);
    ctx->n_entries = PyDict_Size(grad_mat_obj);
    Py_ssize_t n = ctx->n_entries;
    ctx->icell_arr = (int *)malloc(n * sizeof(int));
    ctx->jcell_arr = (int *)malloc(n * sizeof(int));
    ctx->iatom_arr = (int *)malloc(n * sizeof(int));
    ctx->idir_arr = (int *)malloc(n * sizeof(int));
    ctx->data_arr = (double **)malloc(n * sizeof(double *));
    ctx->indices_arr = (int **)malloc(n * sizeof(int *));
    ctx->indptr_arr = (int **)malloc(n * sizeof(int *));
    ctx->data_obj = (PyObject **)calloc(n, sizeof(PyObject *));
    ctx->indices_obj = (PyObject **)calloc(n, sizeof(PyObject *));
    ctx->indptr_obj = (PyObject **)calloc(n, sizeof(PyObject *));
    if (n > 0 && (ctx->icell_arr == NULL || ctx->jcell_arr == NULL ||
        ctx->iatom_arr == NULL || ctx->idir_arr == NULL || ctx->data_arr == NULL ||
        ctx->indices_arr == NULL || ctx->indptr_arr == NULL || ctx->data_obj == NULL ||
        ctx->indices_obj == NULL || ctx->indptr_obj == NULL)) {
        epc_context_release(ctx); PyErr_NoMemory(); return NULL;
    }
    epc_profile_add(&epc_profile_alloc_ns, epc_clock_ns() - t_alloc);
    PyObject *key, *value; Py_ssize_t pos = 0; int entry = 0;
    unsigned long long t_parse = epc_clock_ns();
    while (PyDict_Next(grad_mat_obj, &pos, &key, &value)) {
        PyObject *d = PyTuple_GetItem(value, 0);
        PyObject *ix = PyTuple_GetItem(value, 1);
        PyObject *ip = PyTuple_GetItem(value, 2);
        if (!PyTuple_Check(key) || PyTuple_Size(key) < 4 || d == NULL || ix == NULL ||
            ip == NULL || !PyArray_Check(d) || !PyArray_Check(ix) || !PyArray_Check(ip)) {
            epc_context_release(ctx);
            PyErr_SetString(PyExc_TypeError, "invalid grad_mat sparse entry");
            return NULL;
        }
        ctx->icell_arr[entry] = (int)PyLong_AsLong(PyTuple_GetItem(key, 0));
        ctx->jcell_arr[entry] = (int)PyLong_AsLong(PyTuple_GetItem(key, 1));
        ctx->iatom_arr[entry] = (int)PyLong_AsLong(PyTuple_GetItem(key, 2));
        ctx->idir_arr[entry] = (int)PyLong_AsLong(PyTuple_GetItem(key, 3));
        ctx->data_obj[entry] = d; Py_INCREF(d);
        ctx->indices_obj[entry] = ix; Py_INCREF(ix);
        ctx->indptr_obj[entry] = ip; Py_INCREF(ip);
        ctx->data_arr[entry] = DOUBLEP((PyArrayObject *)d);
        ctx->indices_arr[entry] = INTP((PyArrayObject *)ix);
        ctx->indptr_arr[entry] = INTP((PyArrayObject *)ip);
        ++entry;
    }
    epc_profile_add(&epc_profile_parse_ns, epc_clock_ns() - t_parse);
    ctx->next = epc_contexts; epc_contexts = ctx;
    return PyCapsule_New(ctx, "qtrans.epc_grad_mat", epc_capsule_destructor);
}

PyObject *free_grad_mat_context(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O:free_grad_mat_context", &capsule)) return NULL;
    if (!PyCapsule_IsValid(capsule, "qtrans.epc_grad_mat")) return NULL;
    EPCGradMatContext *ctx = (EPCGradMatContext *)PyCapsule_GetPointer(
        capsule, "qtrans.epc_grad_mat");
    EPCGradMatContext **link = &epc_contexts;
    while (*link != NULL && *link != ctx) link = &(*link)->next;
    if (*link == ctx) *link = ctx->next;
    epc_context_release_arrays(ctx);
    Py_RETURN_NONE;
}

PyObject *epc_elements_cal_helper_sparse(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *ph_prefac_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_coeff_obj;
    PyArrayObject *epc_corr_obj;
    PyArrayObject *g_abs_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOOO", &ph_prefac_obj, &eigen_vec_phon_obj,
                          &phase_k_obj, &phase_kpq_obj, &grad_mat_obj, &wave_coeff_obj,
                          &epc_corr_obj, &g_abs_obj))
    {
        return NULL;
    }

    double *ph_prefac = DOUBLEP(ph_prefac_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_coeff = DCOMPLEXP(wave_coeff_obj);
    double complex *epc_corr = DCOMPLEXP(epc_corr_obj);
    double *g_abs = DOUBLEP(g_abs_obj);

    int nbranches = PyArray_DIMS(ph_prefac_obj)[0];
    int norbs = PyArray_DIMS(wave_coeff_obj)[0];
    int natoms = PyArray_DIMS(ph_prefac_obj)[1];

    int n_entries = (int)PyDict_Size(grad_mat_obj);
    int *icell_arr = (int *)malloc(n_entries * sizeof(int));
    int *jcell_arr = (int *)malloc(n_entries * sizeof(int));
    int *iatom_arr = (int *)malloc(n_entries * sizeof(int));
    int *idir_arr = (int *)malloc(n_entries * sizeof(int));
    double **data_arr = (double **)malloc(n_entries * sizeof(double *));
    int **indices_arr = (int **)malloc(n_entries * sizeof(int *));
    int **indptr_arr = (int **)malloc(n_entries * sizeof(int *));
    double complex *phase_product = (double complex *)malloc(n_entries * sizeof(double complex));

    if (icell_arr == NULL || jcell_arr == NULL || iatom_arr == NULL || idir_arr == NULL ||
        data_arr == NULL || indices_arr == NULL || indptr_arr == NULL || phase_product == NULL)
    {
        free(icell_arr); free(jcell_arr); free(iatom_arr); free(idir_arr);
        free(data_arr); free(indices_arr); free(indptr_arr);
        free(phase_product);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *cell_info_obj, *mat_sp_tuple;
    Py_ssize_t pos = 0;
    int entry_idx = 0;
    while (PyDict_Next(grad_mat_obj, &pos, &cell_info_obj, &mat_sp_tuple))
    {
        int icell = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 0));
        int jcell = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 1));
        icell_arr[entry_idx] = icell;
        jcell_arr[entry_idx] = jcell;
        iatom_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 2));
        idir_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 3));
        data_arr[entry_idx] = DOUBLEP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 0));
        indices_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 1));
        indptr_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 2));
        phase_product[entry_idx] = conj(phase_kpq[icell]) * phase_k[jcell];
        entry_idx++;
    }

    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel for
    for (int ibranch = 0; ibranch < nbranches; ibranch++)
    {
        double complex epc = 0.0 + 0.0 * I;
        for (int ie = 0; ie < n_entries; ie++)
        {
            int iatom = iatom_arr[ie];
            int idir = idir_arr[ie];
            double factor = ph_prefac[INDEX2D(natoms, ibranch, iatom)];
            double complex phase_phon = phase_product[ie] * factor *
                eigen_vec_phon[INDEX3D(natoms, 3, ibranch, iatom, idir)];
            double *data_ptr = data_arr[ie];
            int *indices_ptr = indices_arr[ie];
            int *indptr_ptr = indptr_arr[ie];

            for (int iorb = 0; iorb < norbs; iorb++)
            {
                for (int idx_val = indptr_ptr[iorb]; idx_val < indptr_ptr[iorb + 1]; idx_val++)
                {
                    int jorb = indices_ptr[idx_val];
                    epc += phase_phon * wave_coeff[INDEX2D(norbs, iorb, jorb)] * data_ptr[idx_val];
                }
            }
        }
        g_abs[ibranch] = cabs(epc + epc_corr[ibranch]);
    } // ibranch
    Py_END_ALLOW_THREADS;

    free(icell_arr);
    free(jcell_arr);
    free(iatom_arr);
    free(idir_arr);
    free(data_arr);
    free(indices_arr);
    free(indptr_arr);
    free(phase_product);
    Py_RETURN_NONE;
}

PyObject *eliashberg_spectrum_cal_EPC_helper_sparse(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *epc_strengths_obj;
    PyArrayObject *enk_match_table_obj;
    PyArrayObject *emkpq_match_table_obj;
    PyArrayObject *eig_k_obj;
    PyArrayObject *eig_kpq_obj;
    double phonon_frequency_cutoff, inv_degaussw, efermi, weight_k;
    int gauss_type = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOdddd|i", &freq_obj, &atomic_mass_obj, &eigen_vec_phon_obj,
                          &phase_k_obj, &phase_kpq_obj, &grad_mat_obj, &wave_k_obj, &wave_kpq_obj,
                          &enk_match_table_obj, &emkpq_match_table_obj, &eig_k_obj, &eig_kpq_obj, &epc_strengths_obj,
                          &phonon_frequency_cutoff, &inv_degaussw, &efermi, &weight_k, &gauss_type))
    {
        return NULL;
    }
    if (!validate_superconductivity_gauss(gauss_type)) return NULL;

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    double *epc_strengths = DOUBLEP(epc_strengths_obj);
    bool *enk_match_table = BOOLP(enk_match_table_obj);
    bool *emkpq_match_table = BOOLP(emkpq_match_table_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);

    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = 3 * natoms;
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(eig_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];

    // ====== 优化1: 预提取字典数据到 C 结构体数组（GIL 持有时） ======
    Py_ssize_t dict_size = PyDict_Size(grad_mat_obj);
    DictEntry *dict_entries = (DictEntry *)malloc(dict_size * sizeof(DictEntry));
    {
        PyObject *cell_info_obj, *mat_sp_tuple;
        Py_ssize_t pos = 0;
        int entry_idx = 0;
        while (PyDict_Next(grad_mat_obj, &pos, &cell_info_obj, &mat_sp_tuple))
        {
            int icell = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 0));
            int jcell = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 1));
            dict_entries[entry_idx].icell = icell;
            dict_entries[entry_idx].jcell = jcell;
            dict_entries[entry_idx].iatom = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 2));
            dict_entries[entry_idx].idir = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 3));

            PyArrayObject *data_obj = (PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 0);
            PyArrayObject *indices_obj = (PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 1);
            PyArrayObject *inptr_obj = (PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 2);
            dict_entries[entry_idx].data_ptr = DOUBLEP(data_obj);
            dict_entries[entry_idx].indices_ptr = INTP(indices_obj);
            dict_entries[entry_idx].indptr_ptr = INTP(inptr_obj);

            // 优化5: 预计算相位乘积
            dict_entries[entry_idx].phase_product = conj(phase_kpq[icell]) * phase_k[jcell];

            entry_idx++;
        }
    }

    // ====== 优化3: 预计算 freq 和 mass 因子 ======
    double *inv_sqrt_freq = (double *)malloc(nbranches * sizeof(double));
    for (int ib = 0; ib < nbranches; ib++)
    {
        if (freq[ib] >= phonon_frequency_cutoff)
            inv_sqrt_freq[ib] = 1.0 / sqrt(2.0 * cabs(freq[ib]));
        else
            inv_sqrt_freq[ib] = 0.0;
    }
    double *inv_sqrt_mass = (double *)malloc(natoms * sizeof(double));
    for (int ia = 0; ia < natoms; ia++)
    {
        inv_sqrt_mass[ia] = 1.0 / sqrt(atomic_mass[ia]);
    }

    // ====== 优化2: 预分配 wave function 乘积缓冲 ======
    // wfn_prod[iorb * norbs + jorb] = conj(wave_kpq[iband, iorb]) * wave_k[jband, jorb]
    double complex *wfn_prod = (double complex *)malloc(norbs * norbs * sizeof(double complex));

    int iorb, jorb, ibranch, iband, jband, e;
    double g2_tmp, delta_nk, delta_mkq, tmp_r1, tmp_r2;
    double complex epc;

    Py_BEGIN_ALLOW_THREADS;

    for (jband = 0; jband < NB; jband++)
    {
        if (!enk_match_table[jband])
            continue;

        /* Electronic delta function with the configured smearing. */
        tmp_r1 = (eig_k[jband] - efermi) * inv_degaussw;
        delta_nk = superconductivity_delta(tmp_r1, gauss_type) * inv_degaussw;

        for (iband = 0; iband < MB; iband++)
        {
            if (!emkpq_match_table[iband])
                continue;

            tmp_r2 = (eig_kpq[iband] - efermi) * inv_degaussw;
            delta_mkq = superconductivity_delta(tmp_r2, gauss_type) * inv_degaussw;

            double weight_factor = delta_nk * delta_mkq * weight_k;

            // 优化2: 预计算 wfn_prod 矩阵
            for (iorb = 0; iorb < norbs; iorb++)
            {
                double complex conj_wkpq = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]);
                for (jorb = 0; jorb < norbs; jorb++)
                {
                    wfn_prod[iorb * norbs + jorb] = conj_wkpq * wave_k[INDEX2D(norbs, jband, jorb)];
                }
            }

#pragma omp parallel for private(epc, e, iorb, jorb, g2_tmp) schedule(dynamic)
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (freq[ibranch] < phonon_frequency_cutoff)
                    continue;

                epc = 0.0 + 0.0 * I;
                for (e = 0; e < dict_size; e++)
                {
                    DictEntry *de = &dict_entries[e];
                    double factor = inv_sqrt_freq[ibranch] * inv_sqrt_mass[de->iatom];
                    double complex evp = factor * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, de->iatom, de->idir)];

                    int *indptr = de->indptr_ptr;
                    int *indices = de->indices_ptr;
                    double *data = de->data_ptr;
                    double complex pp = de->phase_product;

                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        int idx_start = indptr[iorb];
                        int idx_end = indptr[iorb + 1];
                        for (int idx_val = idx_start; idx_val < idx_end; idx_val++)
                        {
                            jorb = indices[idx_val];
                            epc += pp * evp * wfn_prod[iorb * norbs + jorb] * data[idx_val];
                        }
                    }
                }
                g2_tmp = pow(cabs(epc), 2);
                epc_strengths[ibranch] += g2_tmp * weight_factor;
            } // ibranch
        } // iband
    } // jband

    Py_END_ALLOW_THREADS;

    free(dict_entries);
    free(inv_sqrt_freq);
    free(inv_sqrt_mass);
    free(wfn_prod);

    Py_RETURN_NONE;
}

/*
 * Batch Eliashberg helper for the non-spin-polarized sparse path.
 *
 * The scalar helper above predates SparseGradCache and receives the Python
 * gradient dictionary on every call.  This entry point has the same
 * integrand, but consumes a prepared SparseGradCache capsule and evaluates a
 * bounded batch of k points in one Python/C transition.  The array layout is
 * deliberately explicit:
 *
 *   phase_k, phase_kpq : (B, ncell) complex128
 *   wave_k             : (B, NB, norbs) complex128
 *   wave_kpq           : (B, MB, norbs) complex128
 *   enk_match          : (B, NB) bool
 *   emkpq_match        : (B, MB) bool
 *   eig_k, eig_kpq     : (B, NB/MB) float64
 *   weight_k           : (B,) float64
 *   epc_strengths_out  : (B, nbranches) float64 (in/out)
 *
 * For every k, band pair, and phonon branch this is algebraically identical
 * to eliashberg_spectrum_cal_EPC_helper_sparse: the electronic matrix element
 * is contracted over the cached CSR gradient entries, multiplied by the
 * phase and phonon/mass/frequency factors, squared, and weighted by the two
 * Gaussian Fermi-surface factors and weight_k.  A batch is capped at 1024 k
 * points to make the intended memory bound part of the native contract.
 */
#define ELIASHBERG_BATCH_MAX_K 1024
#define ELIASHBERG_BATCH_SCRATCH_LIMIT ((size_t)64 * 1024 * 1024)

PyObject *eliashberg_spectrum_cal_EPC_helper_sparse_batch(PyObject *self, PyObject *args)
{
    import_array();
    (void)self;

    PyObject *freq_arg, *mass_arg, *phon_arg, *phase_k_arg, *phase_kpq_arg;
    PyObject *cache_arg, *wave_k_arg, *wave_kpq_arg, *enk_arg, *emkpq_arg;
    PyObject *eig_k_arg, *eig_kpq_arg, *weight_arg, *out_arg;
    PyObject *cutoff_arg, *inv_arg, *efermi_arg, *gauss_arg = NULL;
    if (!PyArg_UnpackTuple(args,
                           "eliashberg_spectrum_cal_EPC_helper_sparse_batch",
                           17, 18,
                           &freq_arg, &mass_arg, &phon_arg,
                           &phase_k_arg, &phase_kpq_arg, &cache_arg,
                           &wave_k_arg, &wave_kpq_arg, &enk_arg, &emkpq_arg,
                           &eig_k_arg, &eig_kpq_arg, &weight_arg, &out_arg,
                           &cutoff_arg, &inv_arg, &efermi_arg, &gauss_arg)) {
        return NULL;
    }

    double cutoff = PyFloat_AsDouble(cutoff_arg);
    double inv_degaussw = PyFloat_AsDouble(inv_arg);
    double efermi = PyFloat_AsDouble(efermi_arg);
    long gauss_value = gauss_arg == NULL ? 0 : PyLong_AsLong(gauss_arg);
    if (PyErr_Occurred()) return NULL;
    if (!validate_superconductivity_gauss(gauss_value)) return NULL;
    int gauss_type = (int)gauss_value;

    if (!PyCapsule_IsValid(cache_arg, SPARSE_GRAD_CACHE_CAPSULE)) {
        PyErr_SetString(PyExc_TypeError,
                        "cache must be a valid qtrans SparseGradCache capsule");
        return NULL;
    }
    SparseGradCache *cache = (SparseGradCache *)PyCapsule_GetPointer(
        cache_arg, SPARSE_GRAD_CACHE_CAPSULE);
    if (cache == NULL) return NULL;

    PyArrayObject *freq_obj, *mass_obj, *phon_obj, *phase_k_obj,
        *phase_kpq_obj, *wave_k_obj, *wave_kpq_obj, *enk_obj, *emkpq_obj,
        *eig_k_obj, *eig_kpq_obj, *weight_obj, *out_obj;
    if (!require_c_array(freq_arg, NPY_DOUBLE, 1, 0, "freq", &freq_obj) ||
        !require_c_array(mass_arg, NPY_DOUBLE, 1, 0, "atomic_mass", &mass_obj) ||
        !require_c_array(phon_arg, NPY_CDOUBLE, 3, 0, "eigen_vec_phon", &phon_obj) ||
        !require_c_array(phase_k_arg, NPY_CDOUBLE, 2, 0, "phase_k", &phase_k_obj) ||
        !require_c_array(phase_kpq_arg, NPY_CDOUBLE, 2, 0, "phase_kpq", &phase_kpq_obj) ||
        !require_c_array(wave_k_arg, NPY_CDOUBLE, 3, 0, "wave_k", &wave_k_obj) ||
        !require_c_array(wave_kpq_arg, NPY_CDOUBLE, 3, 0, "wave_kpq", &wave_kpq_obj) ||
        !require_c_array(enk_arg, NPY_BOOL, 2, 0, "enk_match", &enk_obj) ||
        !require_c_array(emkpq_arg, NPY_BOOL, 2, 0, "emkpq_match", &emkpq_obj) ||
        !require_c_array(eig_k_arg, NPY_DOUBLE, 2, 0, "eig_k", &eig_k_obj) ||
        !require_c_array(eig_kpq_arg, NPY_DOUBLE, 2, 0, "eig_kpq", &eig_kpq_obj) ||
        !require_c_array(weight_arg, NPY_DOUBLE, 1, 0, "weight_k", &weight_obj) ||
        !require_c_array(out_arg, NPY_DOUBLE, 2, 1, "epc_strengths_out", &out_obj)) {
        return NULL;
    }

    npy_intp nk_dim = PyArray_DIMS(wave_k_obj)[0];
    npy_intp nb_dim = PyArray_DIMS(wave_k_obj)[1];
    npy_intp norbs_dim = PyArray_DIMS(wave_k_obj)[2];
    npy_intp mb_dim = PyArray_DIMS(wave_kpq_obj)[1];
    npy_intp nbranches_dim = PyArray_DIMS(freq_obj)[0];
    npy_intp natoms_dim = PyArray_DIMS(mass_obj)[0];
    npy_intp ncell_dim = PyArray_DIMS(phase_k_obj)[1];
    if (nk_dim < 0 || nk_dim > ELIASHBERG_BATCH_MAX_K ||
        nb_dim <= 0 || nb_dim > INT_MAX || mb_dim <= 0 || mb_dim > INT_MAX ||
        norbs_dim <= 0 || norbs_dim > INT_MAX || nbranches_dim <= 0 ||
        nbranches_dim > INT_MAX || natoms_dim <= 0 || natoms_dim > INT_MAX ||
        ncell_dim <= 0 || ncell_dim > INT_MAX) {
        PyErr_SetString(PyExc_ValueError,
                        "batch dimensions are empty, too large, or exceed B=1024");
        return NULL;
    }
    int nk = (int)nk_dim;
    int nb = (int)nb_dim;
    int mb = (int)mb_dim;
    int norbs = (int)norbs_dim;
    int nbranches = (int)nbranches_dim;
    int natoms = (int)natoms_dim;
    int ncell = (int)ncell_dim;

#define ELI_CHECK1(a, d0, name) \
    if (PyArray_DIMS(a)[0] != (d0)) { \
        PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name); \
        return NULL; \
    }
#define ELI_CHECK2(a, d0, d1, name) \
    if (PyArray_DIMS(a)[0] != (d0) || PyArray_DIMS(a)[1] != (d1)) { \
        PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name); \
        return NULL; \
    }
#define ELI_CHECK3(a, d0, d1, d2, name) \
    if (PyArray_DIMS(a)[0] != (d0) || PyArray_DIMS(a)[1] != (d1) || \
        PyArray_DIMS(a)[2] != (d2)) { \
        PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name); \
        return NULL; \
    }
    ELI_CHECK1(mass_obj, natoms, "atomic_mass");
    ELI_CHECK1(freq_obj, nbranches, "freq");
    ELI_CHECK3(phon_obj, nbranches, natoms, 3, "eigen_vec_phon");
    ELI_CHECK2(phase_k_obj, nk, ncell, "phase_k");
    ELI_CHECK2(phase_kpq_obj, nk, ncell, "phase_kpq");
    ELI_CHECK3(wave_kpq_obj, nk, mb, norbs, "wave_kpq");
    ELI_CHECK2(enk_obj, nk, nb, "enk_match");
    ELI_CHECK2(emkpq_obj, nk, mb, "emkpq_match");
    ELI_CHECK2(eig_k_obj, nk, nb, "eig_k");
    ELI_CHECK2(eig_kpq_obj, nk, mb, "eig_kpq");
    ELI_CHECK1(weight_obj, nk, "weight_k");
    ELI_CHECK2(out_obj, nk, nbranches, "epc_strengths_out");
#undef ELI_CHECK1
#undef ELI_CHECK2
#undef ELI_CHECK3

    if (cache->norbs != 0 && cache->norbs != norbs) {
        PyErr_SetString(PyExc_ValueError,
                        "cache CSR row count does not match wave-function orbital count");
        return NULL;
    }
    for (Py_ssize_t entry = 0; entry < cache->n_entries; ++entry) {
        if (cache->icell[entry] < 0 || cache->icell[entry] >= ncell ||
            cache->jcell[entry] < 0 || cache->jcell[entry] >= ncell ||
            cache->iatom[entry] < 0 || cache->iatom[entry] >= natoms ||
            cache->idir[entry] < 0 || cache->idir[entry] >= 3) {
            PyErr_SetString(PyExc_ValueError,
                            "cache metadata is outside supplied cell/atom bounds");
            return NULL;
        }
    }
    if (nk == 0) {
        return Py_BuildValue("{s:i,s:d,s:i}",
                             "threads", 0, "kernel_seconds", 0.0,
                             "batch_size", 0);
    }

    size_t phon_count, phon_bytes;
    if (!checked_size_mul((size_t)nbranches, (size_t)natoms * 3, &phon_count) ||
        !checked_size_mul(phon_count, sizeof(double complex), &phon_bytes) ||
        phon_bytes > ELIASHBERG_BATCH_SCRATCH_LIMIT) {
        PyErr_SetString(PyExc_MemoryError,
                        "phonon factors exceed the sparse EPC scratch limit");
        return NULL;
    }
    int requested_threads = omp_get_max_threads();
    if (requested_threads > nk) requested_threads = nk;
    if (requested_threads < 1) requested_threads = 1;

    /* Cache phase products once per worker/k when that metadata fits the
     * bounded native scratch budget.  If the gradient dictionary is unusually
     * large, compute the phase product inline instead of turning the cache
     * into an unbounded allocation. */
    int phase_cached = 1;
    size_t phase_count = 0, phase_bytes = 0;
    if (!checked_size_mul((size_t)requested_threads,
                          (size_t)cache->n_entries, &phase_count) ||
        !checked_size_mul(phase_count, sizeof(double complex), &phase_bytes) ||
        phase_bytes > ELIASHBERG_BATCH_SCRATCH_LIMIT - phon_bytes) {
        phase_cached = 0;
        phase_count = 0;
        phase_bytes = 0;
    }

    size_t epc_count, epc_bytes;
    if (!checked_size_mul((size_t)requested_threads, (size_t)nbranches, &epc_count) ||
        !checked_size_mul(epc_count, sizeof(double complex), &epc_bytes) ||
        epc_bytes > ELIASHBERG_BATCH_SCRATCH_LIMIT - phon_bytes) {
        requested_threads = 1;
        if (!checked_size_mul((size_t)nbranches, sizeof(double complex), &epc_bytes) ||
            epc_bytes > ELIASHBERG_BATCH_SCRATCH_LIMIT - phon_bytes) {
            PyErr_SetString(PyExc_MemoryError,
                            "per-thread Eliashberg scratch exceeds the 64 MiB limit");
            return NULL;
        }
        epc_count = (size_t)nbranches;
    }

    double complex *phon_factor = (double complex *)malloc(phon_bytes);
    double complex *epc_thread = (double complex *)malloc(epc_count * sizeof(double complex));
    double complex *phase_thread = phase_count > 0 ?
        (double complex *)malloc(phase_bytes) : NULL;
    if ((phon_bytes > 0 && phon_factor == NULL) ||
        (epc_count > 0 && epc_thread == NULL) ||
        (phase_count > 0 && phase_thread == NULL)) {
        free(phon_factor); free(epc_thread); free(phase_thread);
        return PyErr_NoMemory();
    }
    double *freq = DOUBLEP(freq_obj);
    double *mass = DOUBLEP(mass_obj);
    double complex *phon = DCOMPLEXP(phon_obj);
    for (int ibranch = 0; ibranch < nbranches; ++ibranch) {
        for (int iatom = 0; iatom < natoms; ++iatom) {
            double factor = 0.0;
            if (freq[ibranch] >= cutoff && mass[iatom] > 0.0 &&
                fabs(freq[ibranch]) > 0.0) {
                factor = 1.0 / sqrt(2.0 * mass[iatom] * fabs(freq[ibranch]));
            }
            for (int idir = 0; idir < 3; ++idir) {
                size_t idx = ((size_t)ibranch * (size_t)natoms + (size_t)iatom) * 3u +
                             (size_t)idir;
                phon_factor[idx] = factor * phon[idx];
            }
        }
    }

    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk = BOOLP(enk_obj);
    bool *emkpq = BOOLP(emkpq_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double *weight_k = DOUBLEP(weight_obj);
    double *out = DOUBLEP(out_obj);

    int actual_threads = 0;
    double kernel_start = omp_get_wtime();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(requested_threads)
    {
        int tid = omp_get_thread_num();
        double complex *epc = epc_thread + (size_t)tid * (size_t)nbranches;
        double complex *phase = phase_cached ?
            phase_thread + (size_t)tid * (size_t)cache->n_entries : NULL;
#pragma omp single
        actual_threads = omp_get_num_threads();
#pragma omp for schedule(dynamic, 1)
        for (int ik = 0; ik < nk; ++ik) {
            size_t phase_base = (size_t)ik * (size_t)ncell;
            size_t eig_k_base = (size_t)ik * (size_t)nb;
            size_t eig_kpq_base = (size_t)ik * (size_t)mb;
            size_t wave_k_base = (size_t)ik * (size_t)nb * (size_t)norbs;
            size_t wave_kpq_base = (size_t)ik * (size_t)mb * (size_t)norbs;
            size_t out_base = (size_t)ik * (size_t)nbranches;
            if (phase_cached) {
                for (Py_ssize_t entry = 0; entry < cache->n_entries; ++entry) {
                    phase[entry] =
                        conj(phase_kpq[phase_base + (size_t)cache->icell[entry]]) *
                        phase_k[phase_base + (size_t)cache->jcell[entry]];
                }
            }
            for (int jband = 0; jband < nb; ++jband) {
                if (!enk[eig_k_base + (size_t)jband]) continue;
                double x_n = (eig_k[eig_k_base + (size_t)jband] - efermi) * inv_degaussw;
                double delta_n = superconductivity_delta(x_n, gauss_type) * inv_degaussw;
                for (int iband = 0; iband < mb; ++iband) {
                    if (!emkpq[eig_kpq_base + (size_t)iband]) continue;
                    double x_m = (eig_kpq[eig_kpq_base + (size_t)iband] - efermi) * inv_degaussw;
                    double delta_m = superconductivity_delta(x_m, gauss_type) * inv_degaussw;
                    double weight_factor = delta_n * delta_m * weight_k[ik];
                    for (int ibranch = 0; ibranch < nbranches; ++ibranch)
                        epc[ibranch] = 0.0 + 0.0 * I;
                    for (Py_ssize_t entry = 0; entry < cache->n_entries; ++entry) {
                        double complex electronic_element = 0.0 + 0.0 * I;
                        int *indptr = cache->indptr[entry];
                        int *indices = cache->indices[entry];
                        double *data = cache->data[entry];
                        size_t wave_m_base = wave_kpq_base + (size_t)iband * (size_t)norbs;
                        size_t wave_n_base = wave_k_base + (size_t)jband * (size_t)norbs;
                        for (int iorb = 0; iorb < norbs; ++iorb) {
                            for (int idx_val = indptr[iorb]; idx_val < indptr[iorb + 1]; ++idx_val) {
                                int jorb = indices[idx_val];
                                electronic_element +=
                                    conj(wave_kpq[wave_m_base + (size_t)iorb]) *
                                    wave_k[wave_n_base + (size_t)jorb] * data[idx_val];
                            }
                        }
                        double complex phase_product = phase_cached ? phase[entry] :
                            conj(phase_kpq[phase_base + (size_t)cache->icell[entry]]) *
                            phase_k[phase_base + (size_t)cache->jcell[entry]];
                        double complex base = phase_product * electronic_element;
                        for (int ibranch = 0; ibranch < nbranches; ++ibranch) {
                            size_t phon_base = ((size_t)cache->iatom[entry] * 3u) +
                                               (size_t)cache->idir[entry];
                            epc[ibranch] += base *
                                phon_factor[(size_t)ibranch * (size_t)natoms * 3u + phon_base];
                        }
                    }
                    for (int ibranch = 0; ibranch < nbranches; ++ibranch) {
                        double real_part = creal(epc[ibranch]);
                        double imag_part = cimag(epc[ibranch]);
                        out[out_base + (size_t)ibranch] +=
                            (real_part * real_part + imag_part * imag_part) * weight_factor;
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS;
    double kernel_seconds = omp_get_wtime() - kernel_start;
    free(phon_factor); free(epc_thread); free(phase_thread);
    return Py_BuildValue("{s:i,s:d,s:i,s:O}",
                         "threads", actual_threads,
                         "kernel_seconds", kernel_seconds,
                         "batch_size", nk,
                         "phase_cached", phase_cached ? Py_True : Py_False);
}

PyObject *eliashberg_spectrum_cal_EPC_helper_sparse_soc(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *epc_strengths_obj;
    PyArrayObject *enk_match_table_obj;
    PyArrayObject *emkpq_match_table_obj;
    PyArrayObject *eig_k_obj;
    PyArrayObject *eig_kpq_obj;
    double phonon_frequency_cutoff, inv_degaussw, efermi, weight_k;
    int gauss_type = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOdddd|i", &freq_obj, &atomic_mass_obj, &eigen_vec_phon_obj,
                          &phase_k_obj, &phase_kpq_obj, &grad_mat_obj, &wave_k_obj, &wave_kpq_obj,
                          &enk_match_table_obj, &emkpq_match_table_obj, &eig_k_obj, &eig_kpq_obj, &epc_strengths_obj,
                          &phonon_frequency_cutoff, &inv_degaussw, &efermi, &weight_k, &gauss_type))
    {
        return NULL;
    }
    if (!validate_superconductivity_gauss(gauss_type)) return NULL;

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    double *epc_strengths = DOUBLEP(epc_strengths_obj);
    bool *enk_match_table = BOOLP(enk_match_table_obj);
    bool *emkpq_match_table = BOOLP(emkpq_match_table_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);

    // 计算epc
    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = 3 * natoms;
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(eig_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];

    // ====== 优化1: 预提取字典数据 ======
    int n_entries = (int)PyDict_Size(grad_mat_obj);
    int *gm_iatom = (int *)malloc(n_entries * sizeof(int));
    int *gm_idir  = (int *)malloc(n_entries * sizeof(int));
    double complex **gm_data = (double complex **)malloc(n_entries * sizeof(double complex *));
    int **gm_indices = (int **)malloc(n_entries * sizeof(int *));
    int **gm_indptr  = (int **)malloc(n_entries * sizeof(int *));
    double complex *gm_phase_product = (double complex *)malloc(n_entries * sizeof(double complex));

    {
        PyObject *cell_info_obj, *mat_sp_tuple;
        Py_ssize_t pos = 0;
        int entry_idx = 0;
        while (PyDict_Next(grad_mat_obj, &pos, &cell_info_obj, &mat_sp_tuple))
        {
            int icell = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 0));
            int jcell = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 1));
            gm_iatom[entry_idx] = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 2));
            gm_idir[entry_idx] = PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 3));

            PyArrayObject *data_obj = (PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 0);
            PyArrayObject *indices_obj = (PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 1);
            PyArrayObject *inptr_obj = (PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 2);
            gm_data[entry_idx] = DCOMPLEXP(data_obj);
            gm_indices[entry_idx] = INTP(indices_obj);
            gm_indptr[entry_idx] = INTP(inptr_obj);

            gm_phase_product[entry_idx] = conj(phase_kpq[icell]) * phase_k[jcell];

            entry_idx++;
        }
    }

    // ====== 优化3: 预计算 freq 和 mass 因子 ======
    double *inv_sqrt_freq = (double *)malloc(nbranches * sizeof(double));
    for (int ib = 0; ib < nbranches; ib++)
    {
        if (freq[ib] >= phonon_frequency_cutoff)
            inv_sqrt_freq[ib] = 1.0 / sqrt(2.0 * cabs(freq[ib]));
        else
            inv_sqrt_freq[ib] = 0.0;
    }
    double *inv_sqrt_mass = (double *)malloc(natoms * sizeof(double));
    for (int ia = 0; ia < natoms; ia++)
    {
        inv_sqrt_mass[ia] = 1.0 / sqrt(atomic_mass[ia]);
    }

    // ====== 优化2: 预分配 wave function 乘积缓冲 ======
    double complex *wfn_prod = (double complex *)malloc(norbs * norbs * sizeof(double complex));

    int iorb, jorb, ibranch, iband, jband, e;
    double g2_tmp, delta_nk, delta_mkq, tmp_r1, tmp_r2;
    double complex epc;

    Py_BEGIN_ALLOW_THREADS;

    for (jband = 0; jband < NB; jband++)
    {
        if (!enk_match_table[jband])
            continue;

        tmp_r1 = (eig_k[jband] - efermi) * inv_degaussw;
        delta_nk = superconductivity_delta(tmp_r1, gauss_type) * inv_degaussw;

        for (iband = 0; iband < MB; iband++)
        {
            if (!emkpq_match_table[iband])
                continue;

            tmp_r2 = (eig_kpq[iband] - efermi) * inv_degaussw;
            delta_mkq = superconductivity_delta(tmp_r2, gauss_type) * inv_degaussw;

            double weight_factor = delta_nk * delta_mkq * weight_k;

            for (iorb = 0; iorb < norbs; iorb++)
            {
                double complex conj_wkpq = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]);
                for (jorb = 0; jorb < norbs; jorb++)
                {
                    wfn_prod[iorb * norbs + jorb] = conj_wkpq * wave_k[INDEX2D(norbs, jband, jorb)];
                }
            }

#pragma omp parallel for private(epc, e, iorb, jorb, g2_tmp) schedule(dynamic)
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (freq[ibranch] < phonon_frequency_cutoff)
                    continue;

                epc = 0.0 + 0.0 * I;
                for (e = 0; e < n_entries; e++)
                {
                    double factor = inv_sqrt_freq[ibranch] * inv_sqrt_mass[gm_iatom[e]];
                    double complex evp = factor * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, gm_iatom[e], gm_idir[e])];

                    int *indptr = gm_indptr[e];
                    int *indices = gm_indices[e];
                    double complex *data = gm_data[e];
                    double complex pp = gm_phase_product[e];

                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        int idx_start = indptr[iorb];
                        int idx_end = indptr[iorb + 1];
                        for (int idx_val = idx_start; idx_val < idx_end; idx_val++)
                        {
                            jorb = indices[idx_val];
                            epc += pp * evp * wfn_prod[iorb * norbs + jorb] * data[idx_val];
                        }
                    }
                }
                g2_tmp = pow(cabs(epc), 2);
                epc_strengths[ibranch] += g2_tmp * weight_factor;
            } // ibranch
        } // iband
    } // jband

    Py_END_ALLOW_THREADS;

    free(gm_iatom);
    free(gm_idir);
    free(gm_data);
    free(gm_indices);
    free(gm_indptr);
    free(gm_phase_product);
    free(inv_sqrt_freq);
    free(inv_sqrt_mass);
    free(wfn_prod);

    Py_RETURN_NONE;
}

PyObject *scattering_rate_cal_EPC_helper_sparse_soc(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *enk_inside_table_obj;
    PyArrayObject *emkpq_inside_table_obj;
    PyArrayObject *match_table_obj;
    PyArrayObject *eig_k_obj;
    PyArrayObject *eig_kpq_obj;
    PyArrayObject *bose_qvs_obj;
    PyArrayObject *fermi_kpqs_obj;
    PyArrayObject *epc_corr_obj;
    PyArrayObject *scatt_rate_obj;
    PyObject *trans_local_obj;
    PyArrayObject *fermi_ks_obj;
    double inv_degauss, weight_q;
    int is_rmp;
    int is_ibte;

    if (!PyArg_ParseTuple(args, 
            "OOOOOOOOOOOOOOOOOOOddii",
            &freq_obj,         // 1
            &atomic_mass_obj,  // 2
            &eigen_vec_phon_obj, // 3
            &phase_k_obj,      // 4
            &phase_kpq_obj,    // 5
            &grad_mat_obj,     // 6
            &wave_k_obj,       // 7
            &wave_kpq_obj,     // 8
            &enk_inside_table_obj, // 9
            &emkpq_inside_table_obj, // 10
            &match_table_obj,  // 11
            &eig_k_obj,        // 12
            &eig_kpq_obj,      // 13
            &bose_qvs_obj,     // 14
            &fermi_kpqs_obj,   // 15
            &epc_corr_obj,     // 16
            &scatt_rate_obj,   // 17
            &trans_local_obj,  // 18
            &fermi_ks_obj,     // 19
            &inv_degauss,      // 20
            &weight_q,         // 21
            &is_rmp,           // 22
            &is_ibte))         // 23
    {
        return NULL;
    }

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk_inside_table = BOOLP(enk_inside_table_obj);
    bool *emkpq_inside_table = BOOLP(emkpq_inside_table_obj);
    bool *match_table = BOOLP(match_table_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double *scatt_rate = DOUBLEP(scatt_rate_obj);
    double *bose_qvs = DOUBLEP(bose_qvs_obj);
    double *fermi_kpqs = DOUBLEP(fermi_kpqs_obj);
    double complex *epc_corr = DCOMPLEXP(epc_corr_obj);
    double *fermi_ks = DOUBLEP(fermi_ks_obj);

    // 计算epc
    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = PyArray_DIMS(freq_obj)[0];
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(eig_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];

    // 索引指标
    int iorb, jorb, icell, jcell, ibranch, iatom, idir, iband, jband, zero, idx_val;
    double factor, g2_tmp, delta_f1, delta_f2, result, arg, tmp_r1, tmp_r2;
    double complex epc, tmp1, tmp2;
    double scatt_rate_single;
    double trans_temp;

    // Pre-extract grad_mat dict into C arrays to avoid Python API calls in OpenMP region
    int n_entries = (int)PyDict_Size(grad_mat_obj);
    int *gm_icell = (int *)malloc(n_entries * sizeof(int));
    int *gm_jcell = (int *)malloc(n_entries * sizeof(int));
    int *gm_iatom = (int *)malloc(n_entries * sizeof(int));
    int *gm_idir  = (int *)malloc(n_entries * sizeof(int));
    double complex **gm_data = (double complex **)malloc(n_entries * sizeof(double complex *));
    int **gm_indices = (int **)malloc(n_entries * sizeof(int *));
    int **gm_indptr  = (int **)malloc(n_entries * sizeof(int *));
    {
        PyObject *key_tmp, *val_tmp;
        Py_ssize_t pos_tmp = 0;
        int entry_idx = 0;
        while (PyDict_Next(grad_mat_obj, &pos_tmp, &key_tmp, &val_tmp))
        {
            gm_icell[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 0));
            gm_jcell[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 1));
            gm_iatom[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 2));
            gm_idir[entry_idx]  = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 3));
            gm_data[entry_idx]    = DCOMPLEXP((PyArrayObject *)PyTuple_GetItem(val_tmp, 0));
            gm_indices[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(val_tmp, 1));
            gm_indptr[entry_idx]  = INTP((PyArrayObject *)PyTuple_GetItem(val_tmp, 2));
            entry_idx++;
        }
    }

    int ie;
    double complex *data_ptr;
    int *indices_ptr;
    int *indptr_ptr;
    double epc_re, epc_im, epc_corr_re, epc_corr_im, energy_diff;

    // Precompute factor_table[ibranch * natoms + iatom] = 1/sqrt(2*mass[iatom]*|freq[ibranch]|)
    double *factor_table = (double *)malloc(nbranches * natoms * sizeof(double));
    for (ibranch = 0; ibranch < nbranches; ibranch++)
    {
        double freq_abs = cabs(freq[ibranch]);
        for (iatom = 0; iatom < natoms; iatom++)
        {
            factor_table[ibranch * natoms + iatom] = 1.0 / sqrt(2.0 * atomic_mass[iatom] * freq_abs);
        }
    }
    // Pre-allocate cache-friendly wave function buffers
    double complex *wave_k_jband = (double complex *)malloc(norbs * sizeof(double complex));
    double complex *conj_wave_kpq_iband = (double complex *)malloc(norbs * sizeof(double complex));

    Py_BEGIN_ALLOW_THREADS;
    // first loop for n of |n,k>, with jband
    for (jband = 0; jband < NB; jband++)
    {
        if (!enk_inside_table[jband])
        {
            continue;
        }
        // Copy wave_k[jband,:] into contiguous buffer for cache locality
        for (iorb = 0; iorb < norbs; iorb++)
            wave_k_jband[iorb] = wave_k[INDEX2D(norbs, jband, iorb)];
        // second loop for m of |m,k+q>, with iband
        for (iband = 0; iband < MB; iband++)
        {
            if (!emkpq_inside_table[iband])
            {
                continue;
            }
            // Precompute conj(wave_kpq[iband,:]) into contiguous buffer
            for (iorb = 0; iorb < norbs; iorb++)
                conj_wave_kpq_iband[iorb] = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]);
            energy_diff = eig_k[jband] - eig_kpq[iband];
            scatt_rate_single = 0.0;
            trans_temp = 0.0;
#pragma omp parallel for reduction(+:scatt_rate_single, trans_temp) private(iorb, jorb, icell, jcell, ibranch, iatom, idir, epc, tmp1, tmp2, \
                                     g2_tmp, delta_f1, delta_f2, result, arg, tmp_r1, tmp_r2, zero, \
                                     ie, data_ptr, indices_ptr, indptr_ptr, idx_val, \
                                     epc_re, epc_im, epc_corr_re, epc_corr_im)
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (!match_table[INDEX3D(MB, nbranches, jband, iband, ibranch)])
                {
                    continue;
                }
                epc = 0.0 + 0.0 * I;
                for (ie = 0; ie < n_entries; ie++)
                {
                    icell = gm_icell[ie];
                    jcell = gm_jcell[ie];
                    iatom = gm_iatom[ie];
                    idir  = gm_idir[ie];
                    data_ptr = gm_data[ie];
                    indices_ptr = gm_indices[ie];
                    indptr_ptr = gm_indptr[ie];
                    // Precompute per-entry: phase * factor * phonon eigenvector (independent of iorb/jorb)
                    tmp2 = conj(phase_kpq[icell]) * phase_k[jcell] * factor_table[ibranch * natoms + iatom] * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, iatom, idir)];
                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        for (idx_val = indptr_ptr[iorb]; idx_val < indptr_ptr[iorb + 1]; idx_val++)
                        {
                            jorb = indices_ptr[idx_val];
                            tmp1 = conj_wave_kpq_iband[iorb] * wave_k_jband[jorb];
                            epc += tmp2 * tmp1 * data_ptr[idx_val];
                        }
                    }
                }
                epc = epc + epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)];
                // Avoid sqrt: |z|^2 = re^2 + im^2 directly, instead of pow(cabs(z), 2)
                epc_re = creal(epc); epc_im = cimag(epc);
                if (is_rmp == 0) {
                    g2_tmp = epc_re * epc_re + epc_im * epc_im;
                } else {
                    epc_corr_re = creal(epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)]);
                    epc_corr_im = cimag(epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)]);
                    g2_tmp = (epc_re * epc_re + epc_im * epc_im) - (epc_corr_re * epc_corr_re + epc_corr_im * epc_corr_im);
                }
                tmp_r1 = (energy_diff + freq[ibranch]) * inv_degauss;
                tmp_r2 = (energy_diff - freq[ibranch]) * inv_degauss;
                zero = 0;
                delta_f1 = w0gauss(&tmp_r1, &zero, &result, &arg) * inv_degauss;
                delta_f2 = w0gauss(&tmp_r2, &zero, &result, &arg) * inv_degauss;
                scatt_rate_single = scatt_rate_single + g2_tmp * ((bose_qvs[ibranch] + fermi_kpqs[iband]) * delta_f1 + (bose_qvs[ibranch] + 1.0 - fermi_kpqs[iband]) * delta_f2) * weight_q;
                if (is_ibte == 1) {
                    trans_temp = trans_temp + g2_tmp * ((bose_qvs[ibranch] + 1.0 - fermi_ks[jband]) * delta_f1 + (bose_qvs[ibranch] + fermi_ks[jband]) * delta_f2) * weight_q;
                }
            } // ibranch
            scatt_rate[jband] = scatt_rate[jband] + scatt_rate_single;

            if (is_ibte == 1){
                PyGILState_STATE gstate = PyGILState_Ensure();
                PyObject *idx_list = Py_BuildValue("(iid)",jband, iband, trans_temp);
                if (idx_list == NULL ||
                    PyList_Append(trans_local_obj, idx_list) < 0)
                {
                    Py_XDECREF(idx_list);
                    PyErr_SetString(PyExc_RuntimeError,
                                    "Failed to append to trans info");
                    PyGILState_Release(gstate);
                    return NULL;
                }
                Py_DECREF(idx_list);
                PyGILState_Release(gstate);
            }
        } // iband
    } // jband
    Py_END_ALLOW_THREADS;
    free(wave_k_jband); free(conj_wave_kpq_iband);
    free(factor_table);
    free(gm_icell); free(gm_jcell); free(gm_iatom); free(gm_idir);
    free(gm_data); free(gm_indices); free(gm_indptr);
    Py_RETURN_NONE;
}

PyObject *scattering_rate_cal_EPC_helper_sparse(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *enk_inside_table_obj;
    PyArrayObject *emkpq_inside_table_obj;
    PyArrayObject *match_table_obj;
    PyArrayObject *eig_k_obj;
    PyArrayObject *eig_kpq_obj;
    PyArrayObject *bose_qvs_obj;
    PyArrayObject *fermi_kpqs_obj;
    PyArrayObject *epc_corr_obj;
    PyArrayObject *scatt_rate_obj;
    PyObject *scatt_rate_res_obj;
    PyObject *trans_local_obj;
    PyArrayObject *fermi_ks_obj;
    double inv_degauss, weight_q;
    int is_rmp;
    int is_ibte;
    if (!PyArg_ParseTuple(args,
            "OOOOOOOOOOOOOOOOOOOOddii",
            &freq_obj,         // 1
            &atomic_mass_obj,  // 2
            &eigen_vec_phon_obj, // 3
            &phase_k_obj,      // 4
            &phase_kpq_obj,    // 5
            &grad_mat_obj,     // 6
            &wave_k_obj,       // 7
            &wave_kpq_obj,     // 8
            &enk_inside_table_obj, // 9
            &emkpq_inside_table_obj, // 10
            &match_table_obj,  // 11
            &eig_k_obj,        // 12
            &eig_kpq_obj,      // 13
            &bose_qvs_obj,     // 14
            &fermi_kpqs_obj,   // 15
            &epc_corr_obj,     // 16
            &scatt_rate_obj,   // 17
            &scatt_rate_res_obj, // 18
            &trans_local_obj,  // 19
            &fermi_ks_obj,     // 20
            &inv_degauss,      // 21
            &weight_q,         // 22
            &is_rmp,           // 23
            &is_ibte))         // 24
    {
        return NULL;
    }

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk_inside_table = BOOLP(enk_inside_table_obj);
    bool *emkpq_inside_table = BOOLP(emkpq_inside_table_obj);
    bool *match_table = BOOLP(match_table_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double *scatt_rate = DOUBLEP(scatt_rate_obj);
    int has_mode_resolved = (scatt_rate_res_obj != Py_None);
    double *scatt_rate_res = NULL;
    if (has_mode_resolved)
        scatt_rate_res = DOUBLEP((PyArrayObject *)scatt_rate_res_obj);
    double *bose_qvs = DOUBLEP(bose_qvs_obj);
    double *fermi_kpqs = DOUBLEP(fermi_kpqs_obj);
    double complex *epc_corr = DCOMPLEXP(epc_corr_obj);
    double *fermi_ks = DOUBLEP(fermi_ks_obj);

    // 计算epc
    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = PyArray_DIMS(freq_obj)[0];
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(eig_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];

    // 索引指标
    int iorb, jorb, icell, jcell, ibranch, iatom, idir, iband, jband, zero, idx_val;
    double factor, g2_tmp, delta_f1, delta_f2, result, arg, tmp_r1, tmp_r2;
    double complex epc, tmp1, tmp2;
    double scatt_rate_single, scatt_rate_single_tmp;
    double trans_temp;

    // Pre-extract grad_mat dict into C arrays to avoid Python API calls in OpenMP region
    int n_entries = (int)PyDict_Size(grad_mat_obj);
    int *gm_icell = (int *)malloc(n_entries * sizeof(int));
    int *gm_jcell = (int *)malloc(n_entries * sizeof(int));
    int *gm_iatom = (int *)malloc(n_entries * sizeof(int));
    int *gm_idir  = (int *)malloc(n_entries * sizeof(int));
    double **gm_data = (double **)malloc(n_entries * sizeof(double *));
    int **gm_indices = (int **)malloc(n_entries * sizeof(int *));
    int **gm_indptr  = (int **)malloc(n_entries * sizeof(int *));
    {
        PyObject *key_tmp, *val_tmp;
        Py_ssize_t pos_tmp = 0;
        int entry_idx = 0;
        while (PyDict_Next(grad_mat_obj, &pos_tmp, &key_tmp, &val_tmp))
        {
            gm_icell[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 0));
            gm_jcell[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 1));
            gm_iatom[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 2));
            gm_idir[entry_idx]  = (int)PyLong_AsLong(PyTuple_GetItem(key_tmp, 3));
            gm_data[entry_idx]    = DOUBLEP((PyArrayObject *)PyTuple_GetItem(val_tmp, 0));
            gm_indices[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(val_tmp, 1));
            gm_indptr[entry_idx]  = INTP((PyArrayObject *)PyTuple_GetItem(val_tmp, 2));
            entry_idx++;
        }
    }

    int ie;
    double *data_ptr;
    int *indices_ptr;
    int *indptr_ptr;
    double epc_re, epc_im, epc_corr_re, epc_corr_im, energy_diff;

    // Precompute factor_table[ibranch * natoms + iatom] = 1/sqrt(2*mass[iatom]*|freq[ibranch]|)
    double *factor_table = (double *)malloc(nbranches * natoms * sizeof(double));
    for (ibranch = 0; ibranch < nbranches; ibranch++)
    {
        double freq_abs = cabs(freq[ibranch]);
        for (iatom = 0; iatom < natoms; iatom++)
        {
            factor_table[ibranch * natoms + iatom] = 1.0 / sqrt(2.0 * atomic_mass[iatom] * freq_abs);
        }
    }
    // Pre-allocate cache-friendly wave function buffers
    double complex *wave_k_jband = (double complex *)malloc(norbs * sizeof(double complex));
    double complex *conj_wave_kpq_iband = (double complex *)malloc(norbs * sizeof(double complex));

    Py_BEGIN_ALLOW_THREADS;
    // first loop for n of |n,k>, with jband
    for (jband = 0; jband < NB; jband++)
    {
        if (!enk_inside_table[jband])
        {
            continue;
        }
        // Copy wave_k[jband,:] into contiguous buffer for cache locality
        for (iorb = 0; iorb < norbs; iorb++)
            wave_k_jband[iorb] = wave_k[INDEX2D(norbs, jband, iorb)];
        // second loop for m of |m,k+q>, with iband
        for (iband = 0; iband < MB; iband++)
        {
            if (!emkpq_inside_table[iband])
            {
                continue;
            }
            // Precompute conj(wave_kpq[iband,:]) into contiguous buffer
            for (iorb = 0; iorb < norbs; iorb++)
                conj_wave_kpq_iband[iorb] = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]);
            energy_diff = eig_k[jband] - eig_kpq[iband];
            scatt_rate_single = 0.0;
            trans_temp = 0.0;
#pragma omp parallel for reduction(+:scatt_rate_single, trans_temp) private(iorb, jorb, icell, jcell, ibranch, iatom, idir, epc, scatt_rate_single_tmp, tmp1, tmp2, g2_tmp, delta_f1, delta_f2, result, arg, tmp_r1, tmp_r2, zero, \
                                     ie, data_ptr, indices_ptr, indptr_ptr, idx_val, \
                                     epc_re, epc_im, epc_corr_re, epc_corr_im)
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (!match_table[INDEX3D(MB, nbranches, jband, iband, ibranch)])
                {
                    continue;
                }
                scatt_rate_single_tmp = 0;
                epc = 0.0 + 0.0 * I;
                for (ie = 0; ie < n_entries; ie++)
                {
                    icell = gm_icell[ie];
                    jcell = gm_jcell[ie];
                    iatom = gm_iatom[ie];
                    idir  = gm_idir[ie];
                    data_ptr = gm_data[ie];
                    indices_ptr = gm_indices[ie];
                    indptr_ptr = gm_indptr[ie];
                    // Precompute per-entry: phase * factor * phonon eigenvector (independent of iorb/jorb)
                    tmp2 = conj(phase_kpq[icell]) * phase_k[jcell] * factor_table[ibranch * natoms + iatom] * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, iatom, idir)];
                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        for (idx_val = indptr_ptr[iorb]; idx_val < indptr_ptr[iorb + 1]; idx_val++)
                        {
                            jorb = indices_ptr[idx_val];
                            tmp1 = conj_wave_kpq_iband[iorb] * wave_k_jband[jorb];
                            epc += tmp2 * tmp1 * data_ptr[idx_val];
                        }
                    }
                }
                epc = epc + epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)];
                // Avoid sqrt: |z|^2 = re^2 + im^2 directly, instead of pow(cabs(z), 2)
                epc_re = creal(epc); epc_im = cimag(epc);
                if (is_rmp == 0) {
                    g2_tmp = epc_re * epc_re + epc_im * epc_im;
                } else {
                    epc_corr_re = creal(epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)]);
                    epc_corr_im = cimag(epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)]);
                    g2_tmp = (epc_re * epc_re + epc_im * epc_im) - (epc_corr_re * epc_corr_re + epc_corr_im * epc_corr_im);
                }
                tmp_r1 = (energy_diff + freq[ibranch]) * inv_degauss;
                tmp_r2 = (energy_diff - freq[ibranch]) * inv_degauss;
                zero = 0;
                delta_f1 = w0gauss(&tmp_r1, &zero, &result, &arg) * inv_degauss;
                delta_f2 = w0gauss(&tmp_r2, &zero, &result, &arg) * inv_degauss;
                scatt_rate_single_tmp = g2_tmp * ((bose_qvs[ibranch] + fermi_kpqs[iband]) * delta_f1 + (bose_qvs[ibranch] + 1.0 - fermi_kpqs[iband]) * delta_f2) * weight_q;
                scatt_rate_single = scatt_rate_single + scatt_rate_single_tmp;
                if (is_ibte == 1) {
                    trans_temp = trans_temp + g2_tmp * ((bose_qvs[ibranch] + 1.0 - fermi_ks[jband]) * delta_f1 + (bose_qvs[ibranch] + fermi_ks[jband]) * delta_f2) * weight_q;
                }
                if (has_mode_resolved) {
                    scatt_rate_res[INDEX2D(nbranches, jband, ibranch)] += scatt_rate_single_tmp;
                }
            } // ibranch
            scatt_rate[jband] = scatt_rate[jband] + scatt_rate_single;
            if (is_ibte == 1){
                PyGILState_STATE gstate = PyGILState_Ensure();
                PyObject *idx_list = Py_BuildValue("(iid)",jband, iband, trans_temp);
                if (idx_list == NULL ||
                    PyList_Append(trans_local_obj, idx_list) < 0)
                {
                    Py_XDECREF(idx_list);
                    PyErr_SetString(PyExc_RuntimeError,
                                    "Failed to append to trans info");
                    PyGILState_Release(gstate);
                    return NULL;
                }
                Py_DECREF(idx_list);
                PyGILState_Release(gstate);
            }
        } // iband
    } // jband
    Py_END_ALLOW_THREADS;
    free(wave_k_jband); free(conj_wave_kpq_iband);
    free(factor_table);
    free(gm_icell); free(gm_jcell); free(gm_iatom); free(gm_idir);
    free(gm_data); free(gm_indices); free(gm_indptr);
    Py_RETURN_NONE;
}

/* Non-spin sparse mobility: one native call for a bounded group of k points.
 * Positions match the scalar helper, with a cache at position 6 and a dense
 * transfer output at position 19.  Each k owns disjoint output slices. */
#define MOBILITY_BATCH_MAX_K 1024

PyObject *scattering_rate_cal_EPC_helper_sparse_batch(PyObject *self, PyObject *args)
{
    import_array();
    (void)self;
    PyObject *arg[24];
    if (!PyArg_UnpackTuple(args, "scattering_rate_cal_EPC_helper_sparse_batch",
                           24, 24, &arg[0], &arg[1], &arg[2], &arg[3],
                           &arg[4], &arg[5], &arg[6], &arg[7], &arg[8],
                           &arg[9], &arg[10], &arg[11], &arg[12], &arg[13],
                           &arg[14], &arg[15], &arg[16], &arg[17], &arg[18],
                           &arg[19], &arg[20], &arg[21], &arg[22], &arg[23]))
        return NULL;

    double inv_degauss = PyFloat_AsDouble(arg[20]);
    double weight_q = PyFloat_AsDouble(arg[21]);
    long is_rmp = PyLong_AsLong(arg[22]);
    long is_ibte = PyLong_AsLong(arg[23]);
    if (PyErr_Occurred()) return NULL;
    if (!PyCapsule_IsValid(arg[5], SPARSE_GRAD_CACHE_CAPSULE)) {
        PyErr_SetString(PyExc_TypeError, "cache must be a valid SparseGradCache capsule");
        return NULL;
    }
    SparseGradCache *cache = (SparseGradCache *)PyCapsule_GetPointer(
        arg[5], SPARSE_GRAD_CACHE_CAPSULE);
    if (cache == NULL) return NULL;

    PyArrayObject *a[24] = {NULL};
    if (!require_c_array(arg[0], NPY_DOUBLE, 1, 0, "freq", &a[0]) ||
        !require_c_array(arg[1], NPY_DOUBLE, 1, 0, "atomic_mass", &a[1]) ||
        !require_c_array(arg[2], NPY_CDOUBLE, 3, 0, "eigen_vec_phon", &a[2]) ||
        !require_c_array(arg[3], NPY_CDOUBLE, 2, 0, "phase_k", &a[3]) ||
        !require_c_array(arg[4], NPY_CDOUBLE, 2, 0, "phase_kpq", &a[4]) ||
        !require_c_array(arg[6], NPY_CDOUBLE, 3, 0, "wave_k", &a[6]) ||
        !require_c_array(arg[7], NPY_CDOUBLE, 3, 0, "wave_kpq", &a[7]) ||
        !require_c_array(arg[8], NPY_BOOL, 2, 0, "enk_inside_table", &a[8]) ||
        !require_c_array(arg[9], NPY_BOOL, 2, 0, "emkpq_inside_table", &a[9]) ||
        !require_c_array(arg[10], NPY_BOOL, 4, 0, "match_table", &a[10]) ||
        !require_c_array(arg[11], NPY_DOUBLE, 2, 0, "eig_k", &a[11]) ||
        !require_c_array(arg[12], NPY_DOUBLE, 2, 0, "eig_kpq", &a[12]) ||
        !require_c_array(arg[13], NPY_DOUBLE, 1, 0, "bose_qvs", &a[13]) ||
        !require_c_array(arg[14], NPY_DOUBLE, 2, 0, "fermi_kpqs", &a[14]) ||
        !require_c_array(arg[15], NPY_CDOUBLE, 4, 0, "epc_corr", &a[15]) ||
        !require_c_array(arg[16], NPY_DOUBLE, 2, 1, "scatt_rate", &a[16]) ||
        !require_c_array(arg[19], NPY_DOUBLE, 2, 0, "fermi_ks", &a[19]))
        return NULL;
    if (arg[17] != Py_None &&
        !require_c_array(arg[17], NPY_DOUBLE, 3, 1, "scatt_rate_res", &a[17]))
        return NULL;
    if (arg[18] != Py_None &&
        !require_c_array(arg[18], NPY_DOUBLE, 3, 1, "trans_out", &a[18]))
        return NULL;
    if (is_ibte == 1 && a[18] == NULL) {
        PyErr_SetString(PyExc_ValueError, "trans_out is required when is_ibte is 1");
        return NULL;
    }

    npy_intp nk_d = PyArray_DIMS(a[6])[0];
    npy_intp nb_d = PyArray_DIMS(a[6])[1];
    npy_intp mb_d = PyArray_DIMS(a[7])[1];
    npy_intp norbs_d = PyArray_DIMS(a[6])[2];
    npy_intp ncell_d = PyArray_DIMS(a[3])[1];
    npy_intp natoms_d = PyArray_DIMS(a[1])[0];
    npy_intp nbranches_d = PyArray_DIMS(a[0])[0];
    if (nk_d > MOBILITY_BATCH_MAX_K || nb_d <= 0 || nb_d > INT_MAX ||
        mb_d <= 0 || mb_d > INT_MAX || norbs_d <= 0 || norbs_d > INT_MAX ||
        ncell_d <= 0 || ncell_d > INT_MAX || natoms_d <= 0 || natoms_d > INT_MAX ||
        nbranches_d <= 0 || nbranches_d > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "invalid mobility batch dimensions or B exceeds 1024");
        return NULL;
    }
    int nk = (int)nk_d, nb = (int)nb_d, mb = (int)mb_d;
    int norbs = (int)norbs_d, ncell = (int)ncell_d;
    int natoms = (int)natoms_d, nbranches = (int)nbranches_d;
#define MOB_CHECK_SHAPE(array, name, ...) do { \
        const npy_intp expected[] = {__VA_ARGS__}; \
        for (int dimension = 0; dimension < PyArray_NDIM(array); ++dimension) { \
            if (PyArray_DIMS(array)[dimension] != expected[dimension]) { \
                PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name); \
                return NULL; \
            } \
        } \
    } while (0)
    MOB_CHECK_SHAPE(a[2], "eigen_vec_phon", nbranches_d, natoms_d, 3);
    MOB_CHECK_SHAPE(a[3], "phase_k", nk_d, ncell_d);
    MOB_CHECK_SHAPE(a[4], "phase_kpq", nk_d, ncell_d);
    MOB_CHECK_SHAPE(a[6], "wave_k", nk_d, nb_d, norbs_d);
    MOB_CHECK_SHAPE(a[7], "wave_kpq", nk_d, mb_d, norbs_d);
    MOB_CHECK_SHAPE(a[8], "enk_inside_table", nk_d, nb_d);
    MOB_CHECK_SHAPE(a[9], "emkpq_inside_table", nk_d, mb_d);
    MOB_CHECK_SHAPE(a[10], "match_table", nk_d, nb_d, mb_d, nbranches_d);
    MOB_CHECK_SHAPE(a[11], "eig_k", nk_d, nb_d);
    MOB_CHECK_SHAPE(a[12], "eig_kpq", nk_d, mb_d);
    MOB_CHECK_SHAPE(a[13], "bose_qvs", nbranches_d);
    MOB_CHECK_SHAPE(a[14], "fermi_kpqs", nk_d, mb_d);
    MOB_CHECK_SHAPE(a[15], "epc_corr", nk_d, nb_d, mb_d, nbranches_d);
    MOB_CHECK_SHAPE(a[16], "scatt_rate", nk_d, nb_d);
    MOB_CHECK_SHAPE(a[19], "fermi_ks", nk_d, nb_d);
    if (a[17]) MOB_CHECK_SHAPE(a[17], "scatt_rate_res", nk_d, nb_d, nbranches_d);
    if (a[18]) MOB_CHECK_SHAPE(a[18], "trans_out", nk_d, nb_d, mb_d);
#undef MOB_CHECK_SHAPE
    if (cache->n_entries > 0 && cache->norbs != norbs) {
        PyErr_SetString(PyExc_ValueError, "cache orbital dimension does not match wave arrays");
        return NULL;
    }
    for (Py_ssize_t ie = 0; ie < cache->n_entries; ++ie) {
        if (cache->icell[ie] >= ncell || cache->jcell[ie] >= ncell ||
            cache->iatom[ie] >= natoms || cache->idir[ie] >= 3) {
            PyErr_SetString(PyExc_ValueError, "cache cell or atom index exceeds array shape");
            return NULL;
        }
    }
    if (nk == 0) Py_RETURN_NONE;

    size_t factor_count, factor_bytes;
    if (!checked_size_mul((size_t)nbranches, (size_t)natoms, &factor_count) ||
        !checked_size_mul(factor_count, sizeof(double), &factor_bytes)) {
        PyErr_SetString(PyExc_OverflowError, "phonon factor allocation overflows");
        return NULL;
    }
    double *factor_table = (double *)malloc(factor_bytes);
    if (factor_table == NULL) return PyErr_NoMemory();
    const double *freq = DOUBLEP(a[0]), *mass = DOUBLEP(a[1]);
    for (int branch = 0; branch < nbranches; ++branch)
        for (int atom = 0; atom < natoms; ++atom)
            factor_table[(size_t)branch * (size_t)natoms + (size_t)atom] =
                1.0 / sqrt(2.0 * mass[atom] * cabs(freq[branch]));

    const double complex *phon = DCOMPLEXP(a[2]);
    const double complex *phase_k = DCOMPLEXP(a[3]);
    const double complex *phase_kpq = DCOMPLEXP(a[4]);
    const double complex *wave_k = DCOMPLEXP(a[6]);
    const double complex *wave_kpq = DCOMPLEXP(a[7]);
    const bool *enk = BOOLP(a[8]), *emkpq = BOOLP(a[9]), *match = BOOLP(a[10]);
    const double *eig_k = DOUBLEP(a[11]), *eig_kpq = DOUBLEP(a[12]);
    const double *bose = DOUBLEP(a[13]), *fermi_kpq = DOUBLEP(a[14]);
    const double complex *corr = DCOMPLEXP(a[15]);
    double *rate = DOUBLEP(a[16]);
    double *resolved = a[17] ? DOUBLEP(a[17]) : NULL;
    double *trans = a[18] ? DOUBLEP(a[18]) : NULL;
    const double *fermi_k = DOUBLEP(a[19]);

    int requested_threads = omp_get_max_threads();
    if (requested_threads > nk) requested_threads = nk;
    if (requested_threads < 1) requested_threads = 1;
    size_t scratch_count, scratch_bytes;
    if (!checked_size_mul((size_t)requested_threads,
                          (size_t)cache->n_entries, &scratch_count) ||
        !checked_size_mul(scratch_count, sizeof(double complex),
                          &scratch_bytes) ||
        scratch_bytes > (size_t)64 * 1024 * 1024) {
        requested_threads = 1;
        if (!checked_size_mul((size_t)cache->n_entries,
                              sizeof(double complex), &scratch_bytes) ||
            scratch_bytes > (size_t)64 * 1024 * 1024) {
            free(factor_table);
            PyErr_SetString(PyExc_MemoryError,
                            "mobility sparse batch scratch exceeds 64 MiB");
            return NULL;
        }
        scratch_count = (size_t)cache->n_entries;
    }
    double complex *phase_scratch = scratch_count > 0 ?
        (double complex *)malloc(scratch_bytes) : NULL;
    double complex *entry_scratch = scratch_count > 0 ?
        (double complex *)malloc(scratch_bytes) : NULL;
    if ((scratch_count > 0 && phase_scratch == NULL) ||
        (scratch_count > 0 && entry_scratch == NULL)) {
        free(factor_table);
        free(phase_scratch);
        free(entry_scratch);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(requested_threads)
    {
        int tid = omp_get_thread_num();
        double complex *phase_entry = cache->n_entries > 0 ?
            phase_scratch + (size_t)tid * (size_t)cache->n_entries : NULL;
        double complex *entry_element = cache->n_entries > 0 ?
            entry_scratch + (size_t)tid * (size_t)cache->n_entries : NULL;
#pragma omp for schedule(static)
        for (int ik = 0; ik < nk; ++ik) {
        size_t k_nb = (size_t)ik * (size_t)nb;
        size_t k_mb = (size_t)ik * (size_t)mb;
        size_t k_phase = (size_t)ik * (size_t)ncell;
        size_t k_wave_n = k_nb * (size_t)norbs;
        size_t k_wave_m = k_mb * (size_t)norbs;
        size_t k_pair = k_nb * (size_t)mb;
        size_t k_branch = k_pair * (size_t)nbranches;
        for (Py_ssize_t ie = 0; ie < cache->n_entries; ++ie) {
            phase_entry[ie] =
                conj(phase_kpq[k_phase + (size_t)cache->icell[ie]]) *
                phase_k[k_phase + (size_t)cache->jcell[ie]];
        }
        for (int jband = 0; jband < nb; ++jband) {
            for (int iband = 0; iband < mb; ++iband) {
                size_t pair = k_pair + (size_t)jband * (size_t)mb + (size_t)iband;
                if (trans) trans[pair] = 0.0;
                if (!enk[k_nb + (size_t)jband] || !emkpq[k_mb + (size_t)iband])
                    continue;
                size_t pair_branch = k_branch +
                    ((size_t)jband * (size_t)mb + (size_t)iband) *
                    (size_t)nbranches;
                int pair_has_match = 0;
                for (int branch = 0; branch < nbranches; ++branch) {
                    if (match[pair_branch + (size_t)branch]) {
                        pair_has_match = 1;
                        break;
                    }
                }
                if (!pair_has_match) continue;
                size_t wave_n_base = k_wave_n +
                    (size_t)jband * (size_t)norbs;
                size_t wave_m_base = k_wave_m +
                    (size_t)iband * (size_t)norbs;
                for (Py_ssize_t ie = 0; ie < cache->n_entries; ++ie) {
                    const int *indptr = cache->indptr[ie];
                    const int *indices = cache->indices[ie];
                    const double *data = cache->data[ie];
                    double complex electronic_element = 0.0 + 0.0 * I;
                    for (int iorb = 0; iorb < norbs; ++iorb) {
                        double complex wave_m = conj(
                            wave_kpq[wave_m_base + (size_t)iorb]);
                        for (int idx = indptr[iorb];
                             idx < indptr[iorb + 1]; ++idx) {
                            int jorb = indices[idx];
                            electronic_element +=
                                phase_entry[ie] * wave_m *
                                wave_k[wave_n_base + (size_t)jorb] *
                                data[idx];
                        }
                    }
                    entry_element[ie] = electronic_element;
                }
                double energy_diff = eig_k[k_nb + (size_t)jband] -
                                     eig_kpq[k_mb + (size_t)iband];
                double rate_pair = 0.0, trans_pair = 0.0;
                for (int branch = 0; branch < nbranches; ++branch) {
                    size_t branch_idx = k_branch +
                        ((size_t)jband * (size_t)mb + (size_t)iband) *
                        (size_t)nbranches + (size_t)branch;
                    if (!match[branch_idx]) continue;
                    double complex epc = 0.0 + 0.0 * I;
                    for (Py_ssize_t ie = 0; ie < cache->n_entries; ++ie) {
                        int atom = cache->iatom[ie];
                        epc += entry_element[ie] *
                            factor_table[(size_t)branch * (size_t)natoms +
                                         (size_t)atom] *
                            phon[((size_t)branch * (size_t)natoms +
                                  (size_t)atom) * 3u +
                                 (size_t)cache->idir[ie]];
                    }
                    double complex correction = corr[branch_idx];
                    epc += correction;
                    double re = creal(epc), im = cimag(epc);
                    double g2 = re * re + im * im;
                    if (is_rmp != 0) {
                        double corr_re = creal(correction), corr_im = cimag(correction);
                        g2 -= corr_re * corr_re + corr_im * corr_im;
                    }
                    double r1 = (energy_diff + freq[branch]) * inv_degauss;
                    double r2 = (energy_diff - freq[branch]) * inv_degauss;
                    int zero = 0;
                    double result, gauss_arg;
                    double delta1 = w0gauss(&r1, &zero, &result, &gauss_arg) * inv_degauss;
                    double delta2 = w0gauss(&r2, &zero, &result, &gauss_arg) * inv_degauss;
                    double occupation_m = fermi_kpq[k_mb + (size_t)iband];
                    double contribution = g2 *
                        ((bose[branch] + occupation_m) * delta1 +
                         (bose[branch] + 1.0 - occupation_m) * delta2) * weight_q;
                    rate_pair += contribution;
                    if (resolved)
                        resolved[(k_nb + (size_t)jband) * (size_t)nbranches +
                                 (size_t)branch] += contribution;
                    if (is_ibte == 1) {
                        double occupation_n = fermi_k[k_nb + (size_t)jband];
                        trans_pair += g2 *
                            ((bose[branch] + 1.0 - occupation_n) * delta1 +
                             (bose[branch] + occupation_n) * delta2) * weight_q;
                    }
                }
                rate[k_nb + (size_t)jband] += rate_pair;
                if (is_ibte == 1) trans[pair] = trans_pair;
            }
        }
        }
    }
    Py_END_ALLOW_THREADS;
    free(phase_scratch);
    free(entry_scratch);
    free(factor_table);
    Py_RETURN_NONE;
}

PyObject *breakdown_rate_cal_EPC_helper_sparse(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *enk_inside_table_obj;
    PyArrayObject *match_table_obj;
    PyArrayObject *eig_k_obj;
    PyArrayObject *eig_kpq_obj;
    PyArrayObject *bose_qvs_obj;
    PyArrayObject *epc_corr_obj;
    PyArrayObject *breakdown_rate_obj;
    PyObject *breakdown_rate_mode_obj;
    double inv_degauss, weight_q;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOdd",
                        &freq_obj, &atomic_mass_obj, &eigen_vec_phon_obj,
                        &phase_k_obj, &phase_kpq_obj, &grad_mat_obj,
                        &wave_k_obj, &wave_kpq_obj,
                        &enk_inside_table_obj, &match_table_obj,
                        &eig_k_obj, &eig_kpq_obj, &bose_qvs_obj,
                        &epc_corr_obj, &breakdown_rate_obj,
                        &breakdown_rate_mode_obj,
                        &inv_degauss, &weight_q))
    {
        return NULL;
    }

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk_inside_table = BOOLP(enk_inside_table_obj);
    bool *match_table = BOOLP(match_table_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double *breakdown_rate = DOUBLEP(breakdown_rate_obj);
    double *bose_qvs = DOUBLEP(bose_qvs_obj);
    double complex *epc_corr = DCOMPLEXP(epc_corr_obj);

    int has_mode_resolved = (breakdown_rate_mode_obj != Py_None);
    double *breakdown_rate_mode = NULL;
    if (has_mode_resolved)
        breakdown_rate_mode = DOUBLEP((PyArrayObject *)breakdown_rate_mode_obj);

    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = 3 * natoms;
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(eig_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];

    int iorb, jorb, ibranch, iatom, idir, iband, jband, zero, idx_val, ie;
    double g2_tmp, delta_f1, delta_f2, result, arg, tmp_r1, tmp_r2;
    double complex epc, tmp1, tmp2;
    double scatt_rate_single, eloss_rate_single;

    // --- Pre-extract Python dict data into C arrays (GIL-safe) ---
    int n_entries = (int)PyDict_Size(grad_mat_obj);
    int *icell_arr = (int *)malloc(n_entries * sizeof(int));
    int *jcell_arr = (int *)malloc(n_entries * sizeof(int));
    int *iatom_arr = (int *)malloc(n_entries * sizeof(int));
    int *idir_arr  = (int *)malloc(n_entries * sizeof(int));
    double **data_arr    = (double **)malloc(n_entries * sizeof(double *));
    int **indices_arr    = (int **)malloc(n_entries * sizeof(int *));
    int **indptr_arr     = (int **)malloc(n_entries * sizeof(int *));

    PyObject *cell_info_obj, *mat_sp_tuple;
    Py_ssize_t pos = 0;
    int entry_idx = 0;
    while (PyDict_Next(grad_mat_obj, &pos, &cell_info_obj, &mat_sp_tuple))
    {
        icell_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 0));
        jcell_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 1));
        iatom_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 2));
        idir_arr[entry_idx]  = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 3));
        data_arr[entry_idx]    = DOUBLEP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 0));
        indices_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 1));
        indptr_arr[entry_idx]  = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 2));
        entry_idx++;
    }

    // --- Precompute phon_factor = eigen_vec_phon / sqrt(2 * mass * |freq|) ---
    double complex *phon_factor = (double complex *)malloc(nbranches * natoms * 3 * sizeof(double complex));
    for (ibranch = 0; ibranch < nbranches; ibranch++)
    {
        double inv_sqrt_2w = 1.0 / sqrt(2.0 * cabs(freq[ibranch]));
        for (iatom = 0; iatom < natoms; iatom++)
        {
            double fac = inv_sqrt_2w / sqrt(atomic_mass[iatom]);
            for (idir = 0; idir < 3; idir++)
            {
                phon_factor[INDEX3D(natoms, 3, ibranch, iatom, idir)] =
                    fac * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, iatom, idir)];
            }
        }
    }

    double complex *tmp1_mat = (double complex *)malloc(norbs * norbs * sizeof(double complex));

    Py_BEGIN_ALLOW_THREADS;
    for (jband = 0; jband < NB; jband++)
    {
        if (!enk_inside_table[jband])
        {
            continue;
        }
        for (iband = 0; iband < MB; iband++)
        {
            for (iorb = 0; iorb < norbs; iorb++)
            {
                double complex wkpq_conj = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]);
                for (jorb = 0; jorb < norbs; jorb++)
                {
                    tmp1_mat[iorb * norbs + jorb] = wkpq_conj * wave_k[INDEX2D(norbs, jband, jorb)];
                }
            }

            scatt_rate_single = 0.0;
            eloss_rate_single = 0.0;
#pragma omp parallel for reduction(+:scatt_rate_single,eloss_rate_single) \
            private(iorb, jorb, ie, epc, tmp2, g2_tmp, delta_f1, delta_f2, result, arg, tmp_r1, tmp_r2, zero, idx_val)
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (!match_table[INDEX3D(MB, nbranches, jband, iband, ibranch)])
                {
                    continue;
                }
                epc = 0.0 + 0.0 * I;

                for (ie = 0; ie < n_entries; ie++)
                {
                    double complex phase_phon = conj(phase_kpq[icell_arr[ie]]) * phase_k[jcell_arr[ie]]
                        * phon_factor[INDEX3D(natoms, 3, ibranch, iatom_arr[ie], idir_arr[ie])];

                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        for (idx_val = indptr_arr[ie][iorb]; idx_val < indptr_arr[ie][iorb + 1]; idx_val++)
                        {
                            jorb = indices_arr[ie][idx_val];
                            epc += phase_phon * tmp1_mat[iorb * norbs + jorb] * data_arr[ie][idx_val];
                        }
                    }
                }
                g2_tmp = pow(cabs(epc + epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)]), 2);
                tmp_r1 = (eig_k[jband] - eig_kpq[iband] + freq[ibranch]) * inv_degauss;
                tmp_r2 = (eig_k[jband] - eig_kpq[iband] - freq[ibranch]) * inv_degauss;
                zero = 0;
                delta_f1 = w0gauss(&tmp_r1, &zero, &result, &arg) * inv_degauss;
                delta_f2 = w0gauss(&tmp_r2, &zero, &result, &arg) * inv_degauss;

                double scatt_contrib = g2_tmp *
                    (bose_qvs[ibranch] * delta_f1 + (bose_qvs[ibranch] + 1.0) * delta_f2) * weight_q;
                double eloss_contrib = g2_tmp * freq[ibranch] *
                    ((bose_qvs[ibranch] + 1.0) * delta_f2 - bose_qvs[ibranch] * delta_f1) * weight_q;

                scatt_rate_single = scatt_rate_single + scatt_contrib;
                eloss_rate_single = eloss_rate_single + eloss_contrib;

                if (has_mode_resolved)
                {
                    // shape: (2, nbranches, NB), each ibranch writes to a distinct location — thread-safe
                    breakdown_rate_mode[(0 * nbranches + ibranch) * NB + jband] += scatt_contrib;
                    breakdown_rate_mode[(1 * nbranches + ibranch) * NB + jband] += eloss_contrib;
                }
            } // ibranch
            breakdown_rate[INDEX2D(NB, 0, jband)] = breakdown_rate[INDEX2D(NB, 0, jband)] + scatt_rate_single;
            breakdown_rate[INDEX2D(NB, 1, jband)] = breakdown_rate[INDEX2D(NB, 1, jband)] + eloss_rate_single;
        } // iband
    } // jband
    Py_END_ALLOW_THREADS;

    free(icell_arr);
    free(jcell_arr);
    free(iatom_arr);
    free(idir_arr);
    free(data_arr);
    free(indices_arr);
    free(indptr_arr);
    free(phon_factor);
    free(tmp1_mat);

    Py_RETURN_NONE;
}

PyObject *transport_eliashberg_function_cal_EPC_iso_helper_sparse(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *enk_match_table_obj;
    PyArrayObject *emkpq_match_table_obj;
    PyArrayObject *eig_k_obj;
    PyArrayObject *eig_kpq_obj;
    PyArrayObject *vnk_obj;
    PyArrayObject *vmkq_obj;
    PyArrayObject *epc_strengths_obj;
    PyArrayObject *epc_strengths_tr_obj;
    double phonon_frequency_cutoff, inv_degaussw, efermi;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOddd",
                        &freq_obj, &atomic_mass_obj, &eigen_vec_phon_obj,
                        &phase_k_obj, &phase_kpq_obj, &grad_mat_obj, &wave_k_obj,
                        &wave_kpq_obj, &enk_match_table_obj, &emkpq_match_table_obj, &eig_k_obj,
                        &eig_kpq_obj, &vnk_obj, &vmkq_obj, &epc_strengths_obj, &epc_strengths_tr_obj,
                        &phonon_frequency_cutoff, &inv_degaussw, &efermi))
    {
        return NULL;
    }

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk_match_table = BOOLP(enk_match_table_obj);
    bool *emkpq_match_table = BOOLP(emkpq_match_table_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double *vnk = DOUBLEP(vnk_obj);
    double *vmkq = DOUBLEP(vmkq_obj);
    double *epc_strengths = DOUBLEP(epc_strengths_obj);
    double *epc_strengths_tr = DOUBLEP(epc_strengths_tr_obj);

    // 计算epc
    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = 3 * natoms;
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(eig_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];

    // 索引指标
    int iorb, jorb, icell, jcell, ibranch, iatom, idir, iband, jband, zero, idx_val;
    double factor, g2_tmp, delta_nk, delta_mkq, result, arg, tmp_r1, tmp_r2, vel_factor;
    double complex epc, tmp1, tmp2;

    int n_entries = (int)PyDict_Size(grad_mat_obj);
    int *icell_arr = (int *)malloc(n_entries * sizeof(int));
    int *jcell_arr = (int *)malloc(n_entries * sizeof(int));
    int *iatom_arr = (int *)malloc(n_entries * sizeof(int));
    int *idir_arr = (int *)malloc(n_entries * sizeof(int));
    double **data_arr = (double **)malloc(n_entries * sizeof(double *));
    int **indices_arr = (int **)malloc(n_entries * sizeof(int *));
    int **indptr_arr = (int **)malloc(n_entries * sizeof(int *));
    double complex *phase_product = (double complex *)malloc(n_entries * sizeof(double complex));

    if (icell_arr == NULL || jcell_arr == NULL || iatom_arr == NULL || idir_arr == NULL ||
        data_arr == NULL || indices_arr == NULL || indptr_arr == NULL || phase_product == NULL)
    {
        free(icell_arr);
        free(jcell_arr);
        free(iatom_arr);
        free(idir_arr);
        free(data_arr);
        free(indices_arr);
        free(indptr_arr);
        free(phase_product);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *cell_info_obj, *mat_sp_tuple;
    Py_ssize_t pos = 0;
    int entry_idx = 0;
    while (PyDict_Next(grad_mat_obj, &pos, &cell_info_obj, &mat_sp_tuple))
    {
        int icell_tmp = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 0));
        int jcell_tmp = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 1));
        icell_arr[entry_idx] = icell_tmp;
        jcell_arr[entry_idx] = jcell_tmp;
        iatom_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 2));
        idir_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 3));
        data_arr[entry_idx] = DOUBLEP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 0));
        indices_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 1));
        indptr_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 2));
        phase_product[entry_idx] = conj(phase_kpq[icell_tmp]) * phase_k[jcell_tmp];
        entry_idx++;
    }

    Py_BEGIN_ALLOW_THREADS;
    for (iband = 0; iband < MB; iband++)
    {
        if (!emkpq_match_table[iband])
        {
            continue;
        }
        for (jband = 0; jband < NB; jband++)
        {
            if (!enk_match_table[jband])
            {
                continue;
            }
            // |vnk|^2
            double norm_sq = vnk[INDEX2D(3, jband, 0)] * vnk[INDEX2D(3, jband, 0)] +
                 vnk[INDEX2D(3, jband, 1)] * vnk[INDEX2D(3, jband, 1)] +
                 vnk[INDEX2D(3, jband, 2)] * vnk[INDEX2D(3, jband, 2)];
            // sanity check!!!
            if (norm_sq > 1e-4) {
                vel_factor = 1.0 - (vnk[INDEX2D(3, jband, 0)] * vmkq[INDEX2D(3, iband, 0)] + \
                                    vnk[INDEX2D(3, jband, 1)] * vmkq[INDEX2D(3, iband, 1)] + \
                                    vnk[INDEX2D(3, jband, 2)] * vmkq[INDEX2D(3, iband, 2)] \
                                ) / norm_sq;
            } else {
                vel_factor = 1.0;
            }
#pragma omp parallel for private(iorb, jorb, ibranch, epc, tmp1, factor, tmp2, g2_tmp, delta_nk, delta_mkq, result, arg, tmp_r1, tmp_r2, zero, idx_val)
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (freq[ibranch] < phonon_frequency_cutoff)
                {
                    continue;
                }
                epc = 0.0 + 0.0 * I;

                // 遍历字
                for (int ie = 0; ie < n_entries; ie++)
                {
                    // 获取 icell, jcell, iatom, idir
                    factor = 1.0 / sqrt(2.0 * atomic_mass[iatom_arr[ie]] * cabs(freq[ibranch]));
                    tmp2 = phase_product[ie] * factor * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, iatom_arr[ie], idir_arr[ie])];

                    // 将数据传递给C函数
                    // 在这里可以使用 data_ptr, indices_ptr, indptr_ptr 来访问稀疏矩阵的数据
                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        // 每一行各非零元素的索引
                        for (idx_val = indptr_arr[ie][iorb]; idx_val < indptr_arr[ie][iorb + 1]; idx_val++)
                        {
                            jorb = indices_arr[ie][idx_val];
                            tmp1 = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]) * wave_k[INDEX2D(norbs, jband, jorb)];
                            epc += tmp2 * tmp1 * data_arr[ie][idx_val];
                        }
                    }
                }
                g2_tmp = pow(cabs(epc), 2);
                tmp_r1 = (eig_k[jband] - efermi) * inv_degaussw;
                tmp_r2 = (eig_kpq[iband] - efermi) * inv_degaussw;
                zero = 0;
                delta_nk = w0gauss(&tmp_r1, &zero, &result, &arg) * inv_degaussw;
                delta_mkq = w0gauss(&tmp_r2, &zero, &result, &arg) * inv_degaussw;
                epc_strengths[ibranch] = epc_strengths[ibranch] + g2_tmp * delta_nk * delta_mkq;
                epc_strengths_tr[ibranch] = epc_strengths_tr[ibranch] + g2_tmp * delta_nk * delta_mkq * vel_factor;
            } // ibranch
        } // jb
    } // ib
    Py_END_ALLOW_THREADS;
    free(icell_arr);
    free(jcell_arr);
    free(iatom_arr);
    free(idir_arr);
    free(data_arr);
    free(indices_arr);
    free(indptr_arr);
    free(phase_product);
    Py_RETURN_NONE;
}

PyObject *dynamics_setup_helper_sparse(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *enk_inside_table_obj;
    PyArrayObject *emkpq_inside_table_obj;
    PyArrayObject *match_table_obj;
    PyArrayObject *eig_k_obj;
    PyArrayObject *eig_kpq_obj;
    PyArrayObject *epc_corr_obj;
    PyObject *scat_data_obj;
    int ik;
    int ikpq;
    int is_rmp;
    if (!PyArg_ParseTuple(args,
            "OOOOOOOOOOOOOOOiii",
            &freq_obj,         // 1
            &atomic_mass_obj,  // 2
            &eigen_vec_phon_obj, // 3
            &phase_k_obj,      // 4
            &phase_kpq_obj,    // 5
            &grad_mat_obj,     // 6
            &wave_k_obj,       // 7
            &wave_kpq_obj,     // 8
            &enk_inside_table_obj, // 9
            &emkpq_inside_table_obj, // 10
            &match_table_obj,  // 11
            &eig_k_obj,        // 12
            &eig_kpq_obj,      // 13
            &epc_corr_obj,     // 14
            &scat_data_obj,   // 15
            &ik,   // 16
            &ikpq,   // 17
            &is_rmp))         // 18
    {
        return NULL;
    }

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk_inside_table = BOOLP(enk_inside_table_obj);
    bool *emkpq_inside_table = BOOLP(emkpq_inside_table_obj);
    bool *match_table = BOOLP(match_table_obj);
    double *eig_k = DOUBLEP(eig_k_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double complex *epc_corr = DCOMPLEXP(epc_corr_obj);

    // 计算epc
    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = PyArray_DIMS(freq_obj)[0];
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(eig_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];

    // 索引指标
    int iorb, jorb, icell, jcell, ibranch, iatom, idir, iband, jband, idx_val;
    double factor, g2_tmp;
    double complex epc, tmp1, tmp2;

    int n_entries = (int)PyDict_Size(grad_mat_obj);
    int *icell_arr = (int *)malloc(n_entries * sizeof(int));
    int *jcell_arr = (int *)malloc(n_entries * sizeof(int));
    int *iatom_arr = (int *)malloc(n_entries * sizeof(int));
    int *idir_arr = (int *)malloc(n_entries * sizeof(int));
    double **data_arr = (double **)malloc(n_entries * sizeof(double *));
    int **indices_arr = (int **)malloc(n_entries * sizeof(int *));
    int **indptr_arr = (int **)malloc(n_entries * sizeof(int *));
    double complex *phase_product = (double complex *)malloc(n_entries * sizeof(double complex));

    if (icell_arr == NULL || jcell_arr == NULL || iatom_arr == NULL || idir_arr == NULL ||
        data_arr == NULL || indices_arr == NULL || indptr_arr == NULL || phase_product == NULL)
    {
        free(icell_arr);
        free(jcell_arr);
        free(iatom_arr);
        free(idir_arr);
        free(data_arr);
        free(indices_arr);
        free(indptr_arr);
        free(phase_product);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *cell_info_obj, *mat_sp_tuple;
    Py_ssize_t pos = 0;
    int entry_idx = 0;
    while (PyDict_Next(grad_mat_obj, &pos, &cell_info_obj, &mat_sp_tuple))
    {
        int icell_tmp = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 0));
        int jcell_tmp = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 1));
        icell_arr[entry_idx] = icell_tmp;
        jcell_arr[entry_idx] = jcell_tmp;
        iatom_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 2));
        idir_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 3));
        data_arr[entry_idx] = DOUBLEP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 0));
        indices_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 1));
        indptr_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 2));
        phase_product[entry_idx] = conj(phase_kpq[icell_tmp]) * phase_k[jcell_tmp];
        entry_idx++;
    }

    // Py_BEGIN_ALLOW_THREADS;
    PyGILState_STATE gstate = PyGILState_Ensure();
    // first loop for n of |n,k>, with jband
    for (jband = 0; jband < NB; jband++)
    {
        if (!enk_inside_table[jband])
        {
            continue;
        }
        // second loop for m of |m,k+q>, with iband
        for (iband = 0; iband < MB; iband++)
        {
            if (!emkpq_inside_table[iband])
            {
                continue;
            }
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (!match_table[INDEX3D(MB, nbranches, jband, iband, ibranch)])
                {
                    continue;
                }
                epc = 0.0 + 0.0 * I;
                for (int ie = 0; ie < n_entries; ie++)
                {
                    // 获取 icell, jcell, iatom, idir
                    factor = 1.0 / sqrt(2.0 * atomic_mass[iatom_arr[ie]] * cabs(freq[ibranch]));
                    tmp2 = phase_product[ie] * factor * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, iatom_arr[ie], idir_arr[ie])];
                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        for (idx_val = indptr_arr[ie][iorb]; idx_val < indptr_arr[ie][iorb + 1]; idx_val++)
                        {
                            jorb = indices_arr[ie][idx_val];
                            tmp1 = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]) * wave_k[INDEX2D(norbs, jband, jorb)];
                            epc += tmp2 * tmp1 * data_arr[ie][idx_val];
                        }
                    }
                }
                epc = epc + epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)];
                if (is_rmp == 0) {
                    g2_tmp = pow(cabs(epc), 2);
                } else {
                    g2_tmp = pow(cabs(epc), 2) - pow(cabs(epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)]), 2);
                }
                PyObject *idx_list = Py_BuildValue("(iiiiddd)",ik, ikpq, jband, iband, freq[ibranch], g2_tmp, eig_k[jband] - eig_kpq[iband]);
                if (idx_list == NULL ||
                    PyList_Append(scat_data_obj, idx_list) < 0)
                {
                    Py_XDECREF(idx_list);
                    PyErr_SetString(PyExc_RuntimeError,
                                    "Failed to append to scat_data");
                    PyGILState_Release(gstate);
                    free(icell_arr);
                    free(jcell_arr);
                    free(iatom_arr);
                    free(idir_arr);
                    free(data_arr);
                    free(indices_arr);
                    free(indptr_arr);
                    free(phase_product);
                    return NULL;
                }
                Py_DECREF(idx_list);
            } // ibranch
        } // iband
    } // jband
    PyGILState_Release(gstate);
    // Py_END_ALLOW_THREADS;
    free(icell_arr);
    free(jcell_arr);
    free(iatom_arr);
    free(idir_arr);
    free(data_arr);
    free(indices_arr);
    free(indptr_arr);
    free(phase_product);
    Py_RETURN_NONE;
}

PyObject *electron_self_energy_cal_EPC_helper_sparse(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyObject *grad_mat_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *enk_inside_table_obj;
    PyArrayObject *match_table_obj;
    PyArrayObject *eig_kpq_obj;
    PyArrayObject *bose_qvs_obj;
    PyArrayObject *fermi_kpqs_obj;
    PyArrayObject *epc_corr_obj;
    PyArrayObject *self_energy_obj;
    PyArrayObject *omega_list_obj;
    double weight_q;
    double eta;

    if (!PyArg_ParseTuple(args,
            "OOOOOOOOOOOOOOOOdd",
            &freq_obj,         // 1
            &atomic_mass_obj,  // 2
            &eigen_vec_phon_obj, // 3
            &phase_k_obj,      // 4
            &phase_kpq_obj,    // 5
            &grad_mat_obj,     // 6
            &wave_k_obj,       // 7
            &wave_kpq_obj,     // 8
            &enk_inside_table_obj, // 9
            &match_table_obj,  // 10
            &eig_kpq_obj,      // 11
            &bose_qvs_obj,     // 12
            &fermi_kpqs_obj,   // 13
            &epc_corr_obj,     // 14
            &self_energy_obj,   // 15
            &omega_list_obj,  // 16
            &weight_q,         // 17
            &eta             // 18
        ))
    {
        return NULL;
    }

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk_inside_table = BOOLP(enk_inside_table_obj);
    bool *match_table = BOOLP(match_table_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double *bose_qvs = DOUBLEP(bose_qvs_obj);
    double *fermi_kpqs = DOUBLEP(fermi_kpqs_obj);
    double complex *epc_corr = DCOMPLEXP(epc_corr_obj);
    // self_energy: shape (nbands, nbranches, nomegas)
    double complex *self_energy = DCOMPLEXP(self_energy_obj);
    double *omega_list = DOUBLEP(omega_list_obj);
    double complex ieta = 0.0 + eta * I;
    epc_profile_call();

    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = PyArray_DIMS(freq_obj)[0];
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(wave_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];
    int nomegas = PyArray_DIMS(omega_list_obj)[0];

    int iorb, jorb, icell, jcell, ibranch, iatom, idir, iband, jband, idx_val;
    double factor, g2_tmp;
    double occ_factor1, occ_factor2, se0_factor;
    double complex epc, tmp1, tmp2;
    double complex e_pole1, e_pole2, se_factor;

    unsigned long long t_alloc = epc_clock_ns();
    EPCGradMatContext *prepared_ctx = epc_context_find(grad_mat_obj);
    int n_entries = prepared_ctx ? (int)prepared_ctx->n_entries : (int)PyDict_Size(grad_mat_obj);
    int *icell_arr = prepared_ctx ? prepared_ctx->icell_arr : (int *)malloc(n_entries * sizeof(int));
    int *jcell_arr = prepared_ctx ? prepared_ctx->jcell_arr : (int *)malloc(n_entries * sizeof(int));
    int *iatom_arr = prepared_ctx ? prepared_ctx->iatom_arr : (int *)malloc(n_entries * sizeof(int));
    int *idir_arr = prepared_ctx ? prepared_ctx->idir_arr : (int *)malloc(n_entries * sizeof(int));
    double **data_arr = prepared_ctx ? prepared_ctx->data_arr : (double **)malloc(n_entries * sizeof(double *));
    int **indices_arr = prepared_ctx ? prepared_ctx->indices_arr : (int **)malloc(n_entries * sizeof(int *));
    int **indptr_arr = prepared_ctx ? prepared_ctx->indptr_arr : (int **)malloc(n_entries * sizeof(int *));
    double complex *phase_product = (double complex *)malloc(n_entries * sizeof(double complex));

    if (icell_arr == NULL || jcell_arr == NULL || iatom_arr == NULL || idir_arr == NULL ||
        data_arr == NULL || indices_arr == NULL || indptr_arr == NULL || phase_product == NULL)
    {
        if (!prepared_ctx) {
            free(icell_arr); free(jcell_arr); free(iatom_arr); free(idir_arr);
            free(data_arr); free(indices_arr); free(indptr_arr);
        }
        free(phase_product);
        PyErr_NoMemory();
        return NULL;
    }
    epc_profile_add(&epc_profile_alloc_ns, epc_clock_ns() - t_alloc);

    int entry_idx = 0;
    if (prepared_ctx) {
        unsigned long long t_phase = epc_clock_ns();
        for (entry_idx = 0; entry_idx < n_entries; ++entry_idx)
            phase_product[entry_idx] = conj(phase_kpq[icell_arr[entry_idx]]) * phase_k[jcell_arr[entry_idx]];
        epc_profile_add(&epc_profile_phase_ns, epc_clock_ns() - t_phase);
    } else {
        PyObject *cell_info_obj, *mat_sp_tuple;
        Py_ssize_t pos = 0;
        unsigned long long t_parse = epc_clock_ns();
        while (PyDict_Next(grad_mat_obj, &pos, &cell_info_obj, &mat_sp_tuple))
        {
            unsigned long long t_phase = epc_clock_ns();
            int icell_tmp = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 0));
            int jcell_tmp = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 1));
            icell_arr[entry_idx] = icell_tmp;
            jcell_arr[entry_idx] = jcell_tmp;
            iatom_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 2));
            idir_arr[entry_idx] = (int)PyLong_AsLong(PyTuple_GetItem(cell_info_obj, 3));
            data_arr[entry_idx] = DOUBLEP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 0));
            indices_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 1));
            indptr_arr[entry_idx] = INTP((PyArrayObject *)PyTuple_GetItem(mat_sp_tuple, 2));
            phase_product[entry_idx] = conj(phase_kpq[icell_tmp]) * phase_k[jcell_tmp];
            epc_profile_add(&epc_profile_phase_ns, epc_clock_ns() - t_phase);
            entry_idx++;
        }
        epc_profile_add(&epc_profile_parse_ns, epc_clock_ns() - t_parse);
    }

    unsigned long long t_kernel = epc_clock_ns();
    Py_BEGIN_ALLOW_THREADS;
    // first loop for n of |n,k>, with jband
    for (jband = 0; jband < NB; jband++)
    {
        if (!enk_inside_table[jband])
        {
            continue;
        }
        // second loop for m of |m,k+q>, with iband
        for (iband = 0; iband < MB; iband++)
        {
#pragma omp parallel for private(iorb, jorb, ibranch, epc, tmp1, factor, tmp2, g2_tmp, \
                                e_pole1, e_pole2, occ_factor1, occ_factor2, se_factor, se0_factor, \
                                idx_val)
            for (ibranch = 0; ibranch < nbranches; ibranch++)
            {
                if (!match_table[INDEX3D(MB, nbranches, jband, iband, ibranch)])
                {
                    continue;
                }
                epc = 0.0 + 0.0 * I;
                for (int ie = 0; ie < n_entries; ie++)
                {
                    // 获取 icell, jcell, iatom, idir
                    factor = 1.0 / sqrt(2.0 * atomic_mass[iatom_arr[ie]] * cabs(freq[ibranch]));
                    tmp2 = phase_product[ie] * factor * eigen_vec_phon[INDEX3D(natoms, 3, ibranch, iatom_arr[ie], idir_arr[ie])];
                    for (iorb = 0; iorb < norbs; iorb++)
                    {
                        for (idx_val = indptr_arr[ie][iorb]; idx_val < indptr_arr[ie][iorb + 1]; idx_val++)
                        {
                            jorb = indices_arr[ie][idx_val];
                            tmp1 = conj(wave_kpq[INDEX2D(norbs, iband, iorb)]) * wave_k[INDEX2D(norbs, jband, jorb)];
                            epc += tmp2 * tmp1 * data_arr[ie][idx_val];
                        }
                    }
                }
                epc = epc + epc_corr[INDEX3D(MB, nbranches, jband, iband, ibranch)];
                g2_tmp = pow(cabs(epc), 2);

                occ_factor1 = fermi_kpqs[iband] + bose_qvs[ibranch];
                occ_factor2 = 1.0 - fermi_kpqs[iband] + bose_qvs[ibranch];
                e_pole1 = eig_kpq[iband] - freq[ibranch] - ieta;
                e_pole2 = eig_kpq[iband] + freq[ibranch] - ieta;
                se0_factor = -1.0 * creal(occ_factor1 / e_pole1 + occ_factor2 / e_pole2);
                for (int iomega = 0; iomega < nomegas; iomega++) {
                    se_factor = occ_factor1 / (omega_list[iomega] - e_pole1) + occ_factor2 / (omega_list[iomega] - e_pole2);
                    self_energy[INDEX3D(nbranches, nomegas, jband, ibranch, iomega)] = self_energy[INDEX3D(nbranches, nomegas, jband, ibranch, iomega)] + g2_tmp * (se_factor - se0_factor) * weight_q;
                }
            } // ibranch
        } // iband
    } // jband
    Py_END_ALLOW_THREADS;
    epc_profile_add(&epc_profile_kernel_ns, epc_clock_ns() - t_kernel);
    unsigned long long t_free = epc_clock_ns();
    if (!prepared_ctx) {
        free(icell_arr);
        free(jcell_arr);
        free(iatom_arr);
        free(idir_arr);
        free(data_arr);
        free(indices_arr);
        free(indptr_arr);
    }
    free(phase_product);
    epc_profile_add(&epc_profile_free_ns, epc_clock_ns() - t_free);
    Py_RETURN_NONE;
}
#define SPARSE_EPC_SCRATCH_LIMIT ((size_t)64 * 1024 * 1024)

PyObject *electron_self_energy_cal_EPC_helper_sparse_batch(PyObject *self, PyObject *args)
{
    import_array();
    PyObject *freq_arg;
    PyObject *atomic_mass_arg;
    PyObject *eigen_vec_phon_arg;
    PyObject *phase_k_arg;
    PyObject *phase_kpq_arg;
    PyObject *cache_arg;
    PyObject *wave_k_arg;
    PyObject *wave_kpq_arg;
    PyObject *enk_inside_arg;
    PyObject *match_table_arg;
    PyObject *eig_kpq_arg;
    PyObject *bose_qvs_arg;
    PyObject *fermi_kpqs_arg;
    PyObject *self_energy_arg;
    PyObject *omega_list_arg;
    PyObject *weight_q_arg;
    PyObject *eta_arg;
    PyObject *sigma_mode_arg = Py_None;

    if (!PyArg_UnpackTuple(args, "electron_self_energy_cal_EPC_helper_sparse_batch",
                           17, 18,
                           &freq_arg, &atomic_mass_arg, &eigen_vec_phon_arg,
                           &phase_k_arg, &phase_kpq_arg, &cache_arg,
                           &wave_k_arg, &wave_kpq_arg, &enk_inside_arg,
                           &match_table_arg, &eig_kpq_arg, &bose_qvs_arg,
                           &fermi_kpqs_arg, &self_energy_arg, &omega_list_arg,
                           &weight_q_arg, &eta_arg, &sigma_mode_arg)) {
        return NULL;
    }

    double weight_q = PyFloat_AsDouble(weight_q_arg);
    if (PyErr_Occurred()) {
        return NULL;
    }
    double eta = PyFloat_AsDouble(eta_arg);
    if (PyErr_Occurred()) {
        return NULL;
    }

    SparseGradCache *cache = (SparseGradCache *)PyCapsule_GetPointer(
        cache_arg, SPARSE_GRAD_CACHE_CAPSULE);
    if (cache == NULL) {
        return NULL;
    }

    PyArrayObject *freq_obj;
    PyArrayObject *atomic_mass_obj;
    PyArrayObject *eigen_vec_phon_obj;
    PyArrayObject *phase_k_obj;
    PyArrayObject *phase_kpq_obj;
    PyArrayObject *wave_k_obj;
    PyArrayObject *wave_kpq_obj;
    PyArrayObject *enk_inside_obj;
    PyArrayObject *match_table_obj;
    PyArrayObject *eig_kpq_obj;
    PyArrayObject *bose_qvs_obj;
    PyArrayObject *fermi_kpqs_obj;
    PyArrayObject *self_energy_obj;
    PyArrayObject *omega_list_obj;
    PyArrayObject *sigma_mode_obj = NULL;

    if (!require_c_array(freq_arg, NPY_DOUBLE, 1, 0, "freq", &freq_obj) ||
        !require_c_array(atomic_mass_arg, NPY_DOUBLE, 1, 0, "atomic_mass", &atomic_mass_obj) ||
        !require_c_array(eigen_vec_phon_arg, NPY_CDOUBLE, 3, 0, "eigen_vec_phon", &eigen_vec_phon_obj) ||
        !require_c_array(phase_k_arg, NPY_CDOUBLE, 2, 0, "phase_k", &phase_k_obj) ||
        !require_c_array(phase_kpq_arg, NPY_CDOUBLE, 2, 0, "phase_kpq", &phase_kpq_obj) ||
        !require_c_array(wave_k_arg, NPY_CDOUBLE, 3, 0, "wave_k", &wave_k_obj) ||
        !require_c_array(wave_kpq_arg, NPY_CDOUBLE, 3, 0, "wave_kpq", &wave_kpq_obj) ||
        !require_c_array(enk_inside_arg, NPY_BOOL, 2, 0, "enk_inside", &enk_inside_obj) ||
        !require_c_array(match_table_arg, NPY_BOOL, 3, 0, "match_table", &match_table_obj) ||
        !require_c_array(eig_kpq_arg, NPY_DOUBLE, 2, 0, "eig_kpq", &eig_kpq_obj) ||
        !require_c_array(bose_qvs_arg, NPY_DOUBLE, 1, 0, "bose_qvs", &bose_qvs_obj) ||
        !require_c_array(fermi_kpqs_arg, NPY_DOUBLE, 2, 0, "fermi_kpqs", &fermi_kpqs_obj) ||
        !require_c_array(self_energy_arg, NPY_CDOUBLE, 3, 1, "self_energy_out", &self_energy_obj) ||
        !require_c_array(omega_list_arg, NPY_DOUBLE, 1, 0, "omega_list", &omega_list_obj)) {
        return NULL;
    }

    npy_intp nk_dim = PyArray_DIMS(wave_k_obj)[0];
    npy_intp nb_dim = PyArray_DIMS(wave_k_obj)[1];
    npy_intp norbs_dim = PyArray_DIMS(wave_k_obj)[2];
    npy_intp nbranches_dim = PyArray_DIMS(freq_obj)[0];
    npy_intp natoms_dim = PyArray_DIMS(atomic_mass_obj)[0];
    npy_intp ncell_dim = PyArray_DIMS(phase_k_obj)[1];
    npy_intp nomegas_dim = PyArray_DIMS(omega_list_obj)[0];
    if (nk_dim < 0 || nk_dim > INT_MAX || nb_dim <= 0 || nb_dim > INT_MAX ||
        norbs_dim <= 0 || norbs_dim > INT_MAX || nbranches_dim <= 0 ||
        nbranches_dim > INT_MAX || natoms_dim <= 0 || natoms_dim > INT_MAX ||
        ncell_dim <= 0 || ncell_dim > INT_MAX || nomegas_dim <= 0 ||
        nomegas_dim > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "batch array dimensions are empty or too large");
        return NULL;
    }

    int nk = (int)nk_dim;
    int nbands = (int)nb_dim;
    int norbs = (int)norbs_dim;
    int nbranches = (int)nbranches_dim;
    int natoms = (int)natoms_dim;
    int ncell = (int)ncell_dim;
    int nomegas = (int)nomegas_dim;

    if (sigma_mode_arg != Py_None) {
        if (!require_c_array(sigma_mode_arg, NPY_CDOUBLE, 4, 1,
                             "sigma_mode_out", &sigma_mode_obj)) {
            return NULL;
        }
        if (!PyArray_ISNOTSWAPPED(sigma_mode_obj)) {
            PyErr_SetString(PyExc_ValueError, "sigma_mode_out must use native byte order");
            return NULL;
        }
        if (PyArray_DIMS(sigma_mode_obj)[0] != nk_dim ||
            PyArray_DIMS(sigma_mode_obj)[1] != nb_dim ||
            PyArray_DIMS(sigma_mode_obj)[2] != nbranches_dim ||
            PyArray_DIMS(sigma_mode_obj)[3] != nomegas_dim) {
            PyErr_SetString(PyExc_ValueError, "sigma_mode_out has an incompatible shape");
            return NULL;
        }
        size_t mode_bytes;
        if (!checked_size_mul((size_t)nk, (size_t)nbands, &mode_bytes) ||
            !checked_size_mul(mode_bytes, (size_t)nbranches, &mode_bytes) ||
            !checked_size_mul(mode_bytes, (size_t)nomegas, &mode_bytes) ||
            !checked_size_mul(mode_bytes, sizeof(double complex), &mode_bytes)) {
            PyErr_SetString(PyExc_OverflowError, "sigma_mode_out size overflow");
            return NULL;
        }
        // Both outputs must own disjoint storage; otherwise the extra writes
        // could corrupt the independent total accumulator.
        uintptr_t mode_start = (uintptr_t)PyArray_DATA(sigma_mode_obj);
        uintptr_t total_start = (uintptr_t)PyArray_DATA(self_energy_obj);
        size_t total_bytes = (size_t)PyArray_NBYTES(self_energy_obj);
        if ((mode_start <= total_start && total_start - mode_start < mode_bytes) ||
            (total_start < mode_start && mode_start - total_start < total_bytes)) {
            PyErr_SetString(PyExc_ValueError, "sigma_mode_out overlaps self_energy_out");
            return NULL;
        }
    }

#define CHECK_SHAPE_1(array, d0, name) \
    if (PyArray_DIMS(array)[0] != (d0)) { \
        PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name); \
        return NULL; \
    }
#define CHECK_SHAPE_2(array, d0, d1, name) \
    if (PyArray_DIMS(array)[0] != (d0) || PyArray_DIMS(array)[1] != (d1)) { \
        PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name); \
        return NULL; \
    }
#define CHECK_SHAPE_3(array, d0, d1, d2, name) \
    if (PyArray_DIMS(array)[0] != (d0) || PyArray_DIMS(array)[1] != (d1) || \
        PyArray_DIMS(array)[2] != (d2)) { \
        PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name); \
        return NULL; \
    }

    CHECK_SHAPE_1(bose_qvs_obj, nbranches, "bose_qvs");
    CHECK_SHAPE_3(eigen_vec_phon_obj, nbranches, natoms, 3, "eigen_vec_phon");
    CHECK_SHAPE_2(phase_k_obj, nk, ncell, "phase_k");
    CHECK_SHAPE_2(phase_kpq_obj, nk, ncell, "phase_kpq");
    CHECK_SHAPE_3(wave_kpq_obj, nk, nbands, norbs, "wave_kpq");
    CHECK_SHAPE_2(enk_inside_obj, nk, nbands, "enk_inside");
    CHECK_SHAPE_3(match_table_obj, nk, nbands, nbranches, "match_table");
    CHECK_SHAPE_2(eig_kpq_obj, nk, nbands, "eig_kpq");
    CHECK_SHAPE_2(fermi_kpqs_obj, nk, nbands, "fermi_kpqs");
    CHECK_SHAPE_3(self_energy_obj, nk, nbands, nomegas, "self_energy_out");
#undef CHECK_SHAPE_1
#undef CHECK_SHAPE_2
#undef CHECK_SHAPE_3

    if (cache->norbs != 0 && cache->norbs != norbs) {
        PyErr_SetString(PyExc_ValueError,
                        "cache CSR row count does not match wave-function orbital count");
        return NULL;
    }
    for (Py_ssize_t entry = 0; entry < cache->n_entries; entry++) {
        if (cache->icell[entry] >= ncell || cache->jcell[entry] >= ncell ||
            cache->iatom[entry] >= natoms || cache->idir[entry] >= 3) {
            PyErr_SetString(PyExc_ValueError,
                            "cache metadata is outside the supplied cell/atom/direction bounds");
            return NULL;
        }
    }

    if (nk == 0) {
        return Py_BuildValue("{s:i,s:K,s:K,s:d,s:i,s:i,s:O}",
                             "threads", 0,
                             "scratch_per_thread", (unsigned long long)0,
                             "scratch_total", (unsigned long long)0,
                             "kernel_seconds", 0.0,
                             "omega_tile", 0,
                             "band_block", 0,
                             "phase_cached", Py_False);
    }

    size_t phon_count;
    size_t phon_bytes;
    if (!checked_size_mul((size_t)nbranches, (size_t)natoms * 3, &phon_count) ||
        !checked_size_mul(phon_count, sizeof(double complex), &phon_bytes) ||
        phon_bytes > SPARSE_EPC_SCRATCH_LIMIT) {
        PyErr_SetString(PyExc_MemoryError,
                        "shared phonon factors exceed the 64 MiB sparse EPC scratch limit");
        return NULL;
    }

    int omega_tile = nomegas < 64 ? nomegas : 64;
    int band_block = nbands;
    int phase_cached = 1;
    size_t phase_bytes = 0;
    size_t g_branch_bytes = 0;
    size_t g2_bytes = 0;
    size_t kernel_bytes = 0;
    size_t scratch_per_thread = 0;
    size_t memory_threads = 0;

    while (1) {
        size_t temp;
        if (!checked_size_mul((size_t)cache->n_entries, sizeof(double complex), &phase_bytes) ||
            !checked_size_mul((size_t)nbranches, sizeof(double complex), &g_branch_bytes) ||
            !checked_size_mul((size_t)band_block * (size_t)nbranches, sizeof(double), &g2_bytes) ||
            !checked_size_mul((size_t)nbranches * (size_t)omega_tile,
                              sizeof(double complex), &kernel_bytes)) {
            PyErr_SetString(PyExc_MemoryError, "sparse EPC scratch size overflow");
            return NULL;
        }
        scratch_per_thread = phase_cached ? phase_bytes : 0;
        if (!checked_size_add(scratch_per_thread, g_branch_bytes, &temp) ||
            !checked_size_add(temp, g2_bytes, &temp) ||
            !checked_size_add(temp, kernel_bytes, &scratch_per_thread)) {
            PyErr_SetString(PyExc_MemoryError, "sparse EPC scratch size overflow");
            return NULL;
        }
        memory_threads = scratch_per_thread == 0 ? (size_t)nk :
            (SPARSE_EPC_SCRATCH_LIMIT - phon_bytes) / scratch_per_thread;
        if (memory_threads > 0) {
            break;
        }
        if (phase_cached) {
            phase_cached = 0;
        } else if (omega_tile > 1) {
            omega_tile = (omega_tile + 1) / 2;
        } else if (band_block > 1) {
            band_block = (band_block + 1) / 2;
        } else {
            PyErr_SetString(PyExc_MemoryError,
                            "one sparse EPC worker exceeds the 64 MiB scratch limit");
            return NULL;
        }
    }

    int requested_threads = omp_get_max_threads();
    if (requested_threads > nk) {
        requested_threads = nk;
    }
    if ((size_t)requested_threads > memory_threads) {
        requested_threads = (int)memory_threads;
    }
    if (requested_threads < 1) {
        requested_threads = 1;
    }

    size_t phase_total_count = phase_cached ?
        (size_t)requested_threads * (size_t)cache->n_entries : 0;
    size_t g_branch_total_count = (size_t)requested_threads * (size_t)nbranches;
    size_t g2_total_count = (size_t)requested_threads * (size_t)band_block * (size_t)nbranches;
    size_t kernel_total_count = (size_t)requested_threads * (size_t)nbranches * (size_t)omega_tile;
    double complex *phon_factor = phon_count > 0 ?
        (double complex *)malloc(phon_count * sizeof(double complex)) : NULL;
    double complex *phase_product = phase_total_count > 0 ?
        (double complex *)malloc(phase_total_count * sizeof(double complex)) : NULL;
    double complex *g_branch = g_branch_total_count > 0 ?
        (double complex *)malloc(g_branch_total_count * sizeof(double complex)) : NULL;
    double *g2_table = g2_total_count > 0 ?
        (double *)malloc(g2_total_count * sizeof(double)) : NULL;
    double complex *kernel_tile = kernel_total_count > 0 ?
        (double complex *)malloc(kernel_total_count * sizeof(double complex)) : NULL;
    if ((phon_count > 0 && phon_factor == NULL) ||
        (phase_total_count > 0 && phase_product == NULL) ||
        (g_branch_total_count > 0 && g_branch == NULL) ||
        (g2_total_count > 0 && g2_table == NULL) ||
        (kernel_total_count > 0 && kernel_tile == NULL)) {
        free(phon_factor);
        free(phase_product);
        free(g_branch);
        free(g2_table);
        free(kernel_tile);
        return PyErr_NoMemory();
    }

    double *freq = DOUBLEP(freq_obj);
    double *atomic_mass = DOUBLEP(atomic_mass_obj);
    double complex *eigen_vec_phon = DCOMPLEXP(eigen_vec_phon_obj);
    double complex *phase_k = DCOMPLEXP(phase_k_obj);
    double complex *phase_kpq = DCOMPLEXP(phase_kpq_obj);
    double complex *wave_k = DCOMPLEXP(wave_k_obj);
    double complex *wave_kpq = DCOMPLEXP(wave_kpq_obj);
    bool *enk_inside = BOOLP(enk_inside_obj);
    bool *match_table = BOOLP(match_table_obj);
    double *eig_kpq = DOUBLEP(eig_kpq_obj);
    double *bose_qvs = DOUBLEP(bose_qvs_obj);
    double *fermi_kpqs = DOUBLEP(fermi_kpqs_obj);
    double complex *self_energy = DCOMPLEXP(self_energy_obj);
    double *omega_list = DOUBLEP(omega_list_obj);

    double complex *sigma_mode = sigma_mode_obj ? DCOMPLEXP(sigma_mode_obj) : NULL;
    for (int ibranch = 0; ibranch < nbranches; ibranch++) {
        double factor = 1.0 / sqrt(2.0 * atomic_mass[0] * fabs(freq[ibranch]));
        for (int iatom = 0; iatom < natoms; iatom++) {
            if (iatom != 0) {
                factor = 1.0 / sqrt(2.0 * atomic_mass[iatom] * fabs(freq[ibranch]));
            }
            for (int idir = 0; idir < 3; idir++) {
                size_t index = ((size_t)ibranch * (size_t)natoms + (size_t)iatom) * 3 + (size_t)idir;
                phon_factor[index] = factor * eigen_vec_phon[index];
            }
        }
    }

    int actual_threads = 0;
    double kernel_start = omp_get_wtime();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(requested_threads)
    {
        int thread_id = omp_get_thread_num();
        double complex *phase_thread = phase_cached ?
            phase_product + (size_t)thread_id * (size_t)cache->n_entries : NULL;
        double complex *g_branch_thread =
            g_branch + (size_t)thread_id * (size_t)nbranches;
        double *g2_thread =
            g2_table + (size_t)thread_id * (size_t)band_block * (size_t)nbranches;
        double complex *kernel_thread =
            kernel_tile + (size_t)thread_id * (size_t)nbranches * (size_t)omega_tile;

#pragma omp single
        actual_threads = omp_get_num_threads();

#pragma omp for schedule(dynamic, 1)
        for (int ik = 0; ik < nk; ik++) {
            if (phase_cached) {
                for (Py_ssize_t entry = 0; entry < cache->n_entries; entry++) {
                    phase_thread[entry] =
                        conj(phase_kpq[(size_t)ik * (size_t)ncell + (size_t)cache->icell[entry]]) *
                        phase_k[(size_t)ik * (size_t)ncell + (size_t)cache->jcell[entry]];
                }
            }

            for (int mband = 0; mband < nbands; mband++) {
                bool has_match = false;
                size_t match_base = ((size_t)ik * (size_t)nbands + (size_t)mband) * (size_t)nbranches;
                for (int ibranch = 0; ibranch < nbranches; ibranch++) {
                    if (match_table[match_base + (size_t)ibranch]) {
                        has_match = true;
                        break;
                    }
                }
                if (!has_match) {
                    continue;
                }

                for (int nstart = 0; nstart < nbands; nstart += band_block) {
                    int ncount = nbands - nstart;
                    if (ncount > band_block) {
                        ncount = band_block;
                    }
                    memset(g2_thread, 0,
                           (size_t)ncount * (size_t)nbranches * sizeof(double));

                    for (int nlocal = 0; nlocal < ncount; nlocal++) {
                        int nband = nstart + nlocal;
                        if (!enk_inside[(size_t)ik * (size_t)nbands + (size_t)nband]) {
                            continue;
                        }
                        memset(g_branch_thread, 0,
                               (size_t)nbranches * sizeof(double complex));

                        for (Py_ssize_t entry = 0; entry < cache->n_entries; entry++) {
                            double complex electronic_element = 0.0 + 0.0 * I;
                            int *indptr = cache->indptr[entry];
                            int *indices = cache->indices[entry];
                            double *data = cache->data[entry];
                            for (int iorb = 0; iorb < norbs; iorb++) {
                                for (int idx_val = indptr[iorb]; idx_val < indptr[iorb + 1]; idx_val++) {
                                    int jorb = indices[idx_val];
                                    size_t wave_m_index =
                                        ((size_t)ik * (size_t)nbands + (size_t)mband) * (size_t)norbs + (size_t)iorb;
                                    size_t wave_n_index =
                                        ((size_t)ik * (size_t)nbands + (size_t)nband) * (size_t)norbs + (size_t)jorb;
                                    electronic_element += conj(wave_kpq[wave_m_index]) *
                                        wave_k[wave_n_index] * data[idx_val];
                                }
                            }
                            double complex phase = phase_cached ? phase_thread[entry] :
                                conj(phase_kpq[(size_t)ik * (size_t)ncell + (size_t)cache->icell[entry]]) *
                                phase_k[(size_t)ik * (size_t)ncell + (size_t)cache->jcell[entry]];
                            double complex base = phase * electronic_element;
                            size_t phon_base =
                                ((size_t)cache->iatom[entry] * 3 + (size_t)cache->idir[entry]);
                            for (int ibranch = 0; ibranch < nbranches; ibranch++) {
                                if (match_table[match_base + (size_t)ibranch]) {
                                    g_branch_thread[ibranch] += base *
                                        phon_factor[(size_t)ibranch * (size_t)natoms * 3 + phon_base];
                                }
                            }
                        }

                        for (int ibranch = 0; ibranch < nbranches; ibranch++) {
                            double real_part = creal(g_branch_thread[ibranch]);
                            double imag_part = cimag(g_branch_thread[ibranch]);
                            g2_thread[(size_t)nlocal * (size_t)nbranches + (size_t)ibranch] =
                                real_part * real_part + imag_part * imag_part;
                        }
                    }

                    for (int ostart = 0; ostart < nomegas; ostart += omega_tile) {
                        int ocount = nomegas - ostart;
                        if (ocount > omega_tile) {
                            ocount = omega_tile;
                        }
                        memset(kernel_thread, 0,
                               (size_t)nbranches * (size_t)omega_tile * sizeof(double complex));
                        for (int ibranch = 0; ibranch < nbranches; ibranch++) {
                            if (!match_table[match_base + (size_t)ibranch]) {
                                continue;
                            }
                            double occ_factor1 =
                                fermi_kpqs[(size_t)ik * (size_t)nbands + (size_t)mband] + bose_qvs[ibranch];
                            double occ_factor2 = 1.0 -
                                fermi_kpqs[(size_t)ik * (size_t)nbands + (size_t)mband] + bose_qvs[ibranch];
                            double complex e_pole1 =
                                eig_kpq[(size_t)ik * (size_t)nbands + (size_t)mband] - freq[ibranch] - eta * I;
                            double complex e_pole2 =
                                eig_kpq[(size_t)ik * (size_t)nbands + (size_t)mband] + freq[ibranch] - eta * I;
                            double se0_factor = -creal(
                                occ_factor1 / e_pole1 + occ_factor2 / e_pole2);
                            for (int iomega = 0; iomega < ocount; iomega++) {
                                double complex se_factor =
                                    occ_factor1 / (omega_list[ostart + iomega] - e_pole1) +
                                    occ_factor2 / (omega_list[ostart + iomega] - e_pole2);
                                kernel_thread[(size_t)ibranch * (size_t)omega_tile + (size_t)iomega] =
                                    (se_factor - se0_factor) * weight_q;
                            }
                        }

                        for (int nlocal = 0; nlocal < ncount; nlocal++) {
                            int nband = nstart + nlocal;
                            if (!enk_inside[(size_t)ik * (size_t)nbands + (size_t)nband]) {
                                continue;
                            }
                            for (int iomega = 0; iomega < ocount; iomega++) {
                                double complex contribution = 0.0 + 0.0 * I;
                                for (int ibranch = 0; ibranch < nbranches; ibranch++) {
                                    double complex term =
                                        g2_thread[(size_t)nlocal * (size_t)nbranches + (size_t)ibranch] *
                                        kernel_thread[(size_t)ibranch * (size_t)omega_tile + (size_t)iomega];
                                    contribution += term;
                                    if (sigma_mode != NULL) {
                                        // The outer OpenMP loop owns this entire k slice.
                                        size_t mode_index =
                                            (((size_t)ik * (size_t)nbands + (size_t)nband) *
                                             (size_t)nbranches + (size_t)ibranch) * (size_t)nomegas +
                                            (size_t)(ostart + iomega);
                                        sigma_mode[mode_index] += term;
                                    }
                                }
                                size_t out_index =
                                    ((size_t)ik * (size_t)nbands + (size_t)nband) * (size_t)nomegas +
                                    (size_t)(ostart + iomega);
                                self_energy[out_index] += contribution;
                            }
                        }
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS;
    double kernel_seconds = omp_get_wtime() - kernel_start;

    size_t scratch_total = phon_bytes + (size_t)requested_threads * scratch_per_thread;
    free(phon_factor);
    free(phase_product);
    free(g_branch);
    free(g2_table);
    free(kernel_tile);

    return Py_BuildValue("{s:i,s:K,s:K,s:d,s:i,s:i,s:O}",
                         "threads", actual_threads,
                         "scratch_per_thread", (unsigned long long)scratch_per_thread,
                         "scratch_total", (unsigned long long)scratch_total,
                         "kernel_seconds", kernel_seconds,
                         "omega_tile", omega_tile,
                         "band_block", band_block,
                         "phase_cached", phase_cached ? Py_True : Py_False);
}
