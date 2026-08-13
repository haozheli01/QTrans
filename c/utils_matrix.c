#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdbool.h>

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

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOdddd", &freq_obj, &atomic_mass_obj, &eigen_vec_phon_obj,
                          &phase_k_obj, &phase_kpq_obj, &grad_mat_obj, &wave_k_obj, &wave_kpq_obj,
                          &enk_match_table_obj, &emkpq_match_table_obj, &eig_k_obj, &eig_kpq_obj, &epc_strengths_obj,
                          &phonon_frequency_cutoff, &inv_degaussw, &efermi, &weight_k))
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

        // 内联 w0gauss (ngauss=0): result = exp(-fmin(200, x*x)) * UTILS_SQRT_INVPI
        tmp_r1 = (eig_k[jband] - efermi) * inv_degaussw;
        delta_nk = exp(-fmin(200.0, tmp_r1 * tmp_r1)) * UTILS_SQRT_INVPI * inv_degaussw;

        for (iband = 0; iband < MB; iband++)
        {
            if (!emkpq_match_table[iband])
                continue;

            // 内联 w0gauss (ngauss=0): result = exp(-fmin(200, x*x)) * UTILS_SQRT_INVPI
            tmp_r2 = (eig_kpq[iband] - efermi) * inv_degaussw;
            delta_mkq = exp(-fmin(200.0, tmp_r2 * tmp_r2)) * UTILS_SQRT_INVPI * inv_degaussw;

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

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOdddd", &freq_obj, &atomic_mass_obj, &eigen_vec_phon_obj,
                          &phase_k_obj, &phase_kpq_obj, &grad_mat_obj, &wave_k_obj, &wave_kpq_obj,
                          &enk_match_table_obj, &emkpq_match_table_obj, &eig_k_obj, &eig_kpq_obj, &epc_strengths_obj,
                          &phonon_frequency_cutoff, &inv_degaussw, &efermi, &weight_k))
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
        delta_nk = exp(-fmin(200.0, tmp_r1 * tmp_r1)) * UTILS_SQRT_INVPI * inv_degaussw;

        for (iband = 0; iband < MB; iband++)
        {
            if (!emkpq_match_table[iband])
                continue;

            tmp_r2 = (eig_kpq[iband] - efermi) * inv_degaussw;
            delta_mkq = exp(-fmin(200.0, tmp_r2 * tmp_r2)) * UTILS_SQRT_INVPI * inv_degaussw;

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

    int natoms = PyArray_DIMS(atomic_mass_obj)[0];
    int nbranches = PyArray_DIMS(freq_obj)[0];
    int norbs = PyArray_DIMS(wave_k_obj)[1];
    int NB = PyArray_DIMS(wave_k_obj)[0];
    int MB = PyArray_DIMS(eig_kpq_obj)[0];
    int nomegas = PyArray_DIMS(omega_list_obj)[0];

    int iorb, jorb, icell, jcell, ibranch, iatom, idir, iband, jband, idx_val, iomega;
    double factor, g2_tmp;
    double occ_factor1, occ_factor2, se0_factor;
    double complex epc, tmp1, tmp2;
    double complex e_pole1, e_pole2, se_factor;

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
                for (iomega = 0; iomega < nomegas; iomega++) {
                    se_factor = occ_factor1 / (omega_list[iomega] - e_pole1) + occ_factor2 / (omega_list[iomega] - e_pole2);
                    self_energy[INDEX3D(nbranches, nomegas, jband, ibranch, iomega)] = self_energy[INDEX3D(nbranches, nomegas, jband, ibranch, iomega)] + g2_tmp * (se_factor - se0_factor) * weight_q;
                }
            } // ibranch
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
    free(phase_product);
    Py_RETURN_NONE;
}
