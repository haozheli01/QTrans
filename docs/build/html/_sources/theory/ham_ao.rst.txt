=========================================
Hamiltonian in an Atomic Orbital Basis
=========================================

Bloch States
------------

In a localized atomic orbital basis, a Bloch state can be expanded as
:cite:`zhongAcceleratingCalculationElectron2024`

.. math::

    |\nu\mathbf{k}\rangle = \frac{1}{\sqrt{N}}
    \sum_{n=1}^{N} \mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_{n}}
    \sum_{j\beta} C^{\mathbf{k},\nu}_{j\beta}|\phi_{nj\beta}\rangle,

where :math:`\nu` is the band index, :math:`\mathbf{k}` is the crystal wave vector,
and :math:`N` is the number of unit cells in the Born-von Kármán supercell.
The lattice vector :math:`\mathbf{R}_{n}` locates the :math:`n`-th unit cell,
:math:`j` labels an atom within that cell, and :math:`\beta` labels an orbital
on that atom. The coefficients :math:`C^{\mathbf{k},\nu}_{j\beta}` specify the
expansion of the Bloch state. The localized orbitals are defined by

.. math::

    \langle\mathbf{r}|\phi_{nj\beta}\rangle
    = \phi_{j\beta}(\mathbf{r}-\mathbf{R}_{n}-\boldsymbol{\tau}_{j}),

where :math:`\boldsymbol{\tau}_{j}` is the position of atom :math:`j` relative
to the origin of its unit cell. Here, the Bloch phase contains only the lattice
vector :math:`\mathbf{R}_{n}`; the atomic position :math:`\boldsymbol{\tau}_{j}`
is included in the orbital definition. The reciprocal-space matrices below
use this same phase convention.

Hamiltonian and Overlap Matrices
--------------------------------

Atomic orbitals are generally nonorthogonal. The real-space Hamiltonian and
overlap matrix elements are defined as

.. math::

    H_{mi\alpha,nj\beta}
    &= \langle\phi_{mi\alpha}|\hat{\mathcal{H}}|\phi_{nj\beta}\rangle, \\
    S_{mi\alpha,nj\beta}
    &= \langle\phi_{mi\alpha}|\phi_{nj\beta}\rangle.

Assuming that the basis is linearly independent, the overlap matrix :math:`S`
is invertible. Within the space spanned by this basis, the Hamiltonian operator
can be written as

.. math::

    \hat{\mathcal{H}}_{\mathrm{AO}}
    = \sum_{mi\alpha,nj\beta}
    |\phi_{mi\alpha}\rangle
    \widetilde{H}_{mi\alpha,nj\beta}
    \langle\phi_{nj\beta}|,
    \qquad \widetilde{H}=S^{-1}HS^{-1}.

The matrix :math:`\widetilde{H}` gives the coefficients of the operator in this
outer-product expansion; it differs from the matrix of Hamiltonian elements
:math:`H`. The two factors of :math:`S^{-1}` account for the nonorthogonality
of the basis. For an orthonormal basis, :math:`S=I` and
:math:`\widetilde{H}=H`.

Generalized Eigenvalue Problem
------------------------------

For a periodic system, translational symmetry allows the real-space matrices
to be Fourier transformed into reciprocal space:

.. math::

    H^{\mathbf{k}}_{i\alpha,j\beta}
    &= \sum_n \mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_{n}}
    H_{0i\alpha,nj\beta}, \\
    S^{\mathbf{k}}_{i\alpha,j\beta}
    &= \sum_n \mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_{n}}
    S_{0i\alpha,nj\beta},

where the index :math:`0` denotes the reference cell at
:math:`\mathbf{R}_0=\mathbf{0}`, and the sum runs over relative lattice vectors
with Born–von Kármán periodic boundary conditions.

The electronic energies and expansion coefficients are obtained by solving
the generalized eigenvalue problem :cite:`zhongTransferableEquivariantGraph2023`

.. math::

    H^{\mathbf{k}}\mathbf{C}_{\nu\mathbf{k}}
    = \varepsilon_{\nu\mathbf{k}}
    S^{\mathbf{k}}\mathbf{C}_{\nu\mathbf{k}},

where :math:`\varepsilon_{\nu\mathbf{k}}` is the energy of the Bloch state
:math:`|\nu\mathbf{k}\rangle` and the components of
:math:`\mathbf{C}_{\nu\mathbf{k}}` are :math:`C^{\mathbf{k},\nu}_{j\beta}`.
The eigenvectors are chosen to satisfy the overlap-weighted orthonormality
condition

.. math::

    \mathbf{C}_{\nu\mathbf{k}}^{\dagger}
    S^{\mathbf{k}}\mathbf{C}_{\mu\mathbf{k}}
    = \delta_{\nu\mu},

which ensures that the corresponding Bloch states are normalized and mutually
orthogonal at each :math:`\mathbf{k}`.
