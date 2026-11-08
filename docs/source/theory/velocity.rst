.. _theory_velocity:

===================
Electronic Velocity
===================

**QTrans** evaluates electronic band velocities analytically from the
real-space Hamiltonian and overlap matrices in an atomic orbital basis.
These velocities enter the :ref:`conductivity <theory_conductivity>` and
:ref:`transport spectral function <theory_resistivity>` calculations.

Group Velocity in a Nonorthogonal Basis
---------------------------------------

For a differentiable band :math:`\varepsilon_{\nu\mathbf{k}}`, the group
velocity is

.. math::

    v_{\nu\mathbf{k},a}
    = \frac{1}{\hbar}\frac{\partial\varepsilon_{\nu\mathbf{k}}}{\partial k_a},

where :math:`a` labels a Cartesian direction and :math:`\mathbf{k}` is the
physical wave vector, with units of inverse length. In the atomic orbital
basis, the band energies and expansion coefficients satisfy

.. math::

    H^{\mathbf{k}}\mathbf{C}_{\nu\mathbf{k}}
    = \varepsilon_{\nu\mathbf{k}}S^{\mathbf{k}}\mathbf{C}_{\nu\mathbf{k}},
    \qquad
    \mathbf{C}_{\nu\mathbf{k}}^\dagger S^{\mathbf{k}}
    \mathbf{C}_{\nu\mathbf{k}}=1.

Differentiating this generalized eigenvalue equation and using the
normalization gives the generalized Hellmann–Feynman expression

.. math::

   v_{\nu\mathbf{k},a}
   = \frac{1}{\hbar}\mathbf{C}_{\nu\mathbf{k}}^\dagger
   \left(
   \frac{\partial H^{\mathbf{k}}}{\partial k_a}
   -\varepsilon_{\nu\mathbf{k}}
   \frac{\partial S^{\mathbf{k}}}{\partial k_a}
   \right)
   \mathbf{C}_{\nu\mathbf{k}}.

The derivatives of the expansion coefficients cancel, so they do not need
to be evaluated explicitly. The overlap derivative accounts for the
nonorthogonality of the basis. Omitting it generally gives an incorrect
band velocity. For a :math:`\mathbf{k}`-independent orthonormal basis,
:math:`S^{\mathbf{k}}=I` and this term vanishes.

Analytic Derivatives of the Bloch Matrices
------------------------------------------

Using the same Bloch phase convention as :doc:`ham_ao`, write

.. math::

    H^{\mathbf{k}}_{i\alpha,j\beta}
    &= \sum_n\mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_n}
    H_{0i\alpha,nj\beta}, \\
    S^{\mathbf{k}}_{i\alpha,j\beta}
    &= \sum_n\mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_n}
    S_{0i\alpha,nj\beta}.

Here, :math:`\mathbf{R}_n` is a lattice translation, :math:`i,j` label atoms
in the unit cell, and :math:`\alpha,\beta` label their orbitals. The real-space
matrices are independent of :math:`\mathbf{k}`. Their Bloch derivatives
therefore follow directly from the phase factors:

.. math::

    \frac{\partial H^{\mathbf{k}}_{i\alpha,j\beta}}{\partial k_a}
    &= \mathrm{i}\sum_n R_{n,a}
    \mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_n}
    H_{0i\alpha,nj\beta}, \\
    \frac{\partial S^{\mathbf{k}}_{i\alpha,j\beta}}{\partial k_a}
    &= \mathrm{i}\sum_n R_{n,a}
    \mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_n}
    S_{0i\alpha,nj\beta}.

Substitution into above formula gives

.. math::

    v_{\nu\mathbf{k},a}
    = \frac{1}{\hbar}\operatorname{Re}\Bigg[
    \mathrm{i}\sum_n\sum_{i\alpha,j\beta}
    R_{n,a}\mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_n}
    (C^{\mathbf{k},\nu}_{i\alpha})^*
    \left(H_{0i\alpha,nj\beta}
    -\varepsilon_{\nu\mathbf{k}}S_{0i\alpha,nj\beta}\right)
    C^{\mathbf{k},\nu}_{j\beta}\Bigg].

For Hermitian Bloch matrices, the result is real; taking the real part
removes numerical imaginary residuals. This expression is analytic for the
supplied real-space matrices and requires no finite differences between
neighboring :math:`\mathbf{k}` points.

.. note::

    The factors :math:`R_{n,a}H_{0i\alpha,nj\beta}` and
    :math:`R_{n,a}S_{0i\alpha,nj\beta}` arise from differentiation with respect
    to the wave vector. They are not derivatives with respect to atomic
    displacements. The latter describe a different response, such as the
    electron-phonon interaction.

Implementation and Units
------------------------

The implementation precomputes the translation-weighted real-space matrices,
Fourier transforms them at each requested wave vector, and contracts the
results with the normalized band coefficients. In collinear spin-polarized
calculations, this procedure is applied separately to each spin channel.
For spinor calculations, the Hamiltonian and overlap matrices are represented
in the corresponding spinor basis.

Internally, reciprocal Cartesian coordinates :math:`\boldsymbol{\kappa}`
omit the factor of :math:`2\pi`, so

.. math::

    \mathbf{k}=2\pi\boldsymbol{\kappa},
    \qquad
    \mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_n}
    =\mathrm{e}^{2\pi\mathrm{i}\boldsymbol{\kappa}\cdot\mathbf{R}_n}.

Consequently,

.. math::

    \frac{\partial}{\partial k_a}
    = \frac{1}{2\pi}\frac{\partial}{\partial\kappa_a}.

The factor :math:`2\pi` from differentiating the internal phase cancels this
conversion factor. The velocity contraction therefore contains
:math:`\mathrm{i}R_{n,a}`, without an additional :math:`2\pi`.

With energies in Hartree and translations in Bohr, the routine returns
Cartesian velocities in atomic units. Conversion to SI units uses

.. math::

    v_{\mathrm{SI}}
    = v_{\mathrm{au}}\frac{E_{\mathrm{h}}a_0}{\hbar},

where :math:`E_{\mathrm{h}}` is the Hartree energy and :math:`a_0` is the Bohr
radius. One atomic unit of velocity is approximately
:math:`2.1877\times10^6\,\mathrm{m\,s^{-1}}`.

.. note::

    At an exact band degeneracy, individual band slopes require care. For a
    chosen direction, the first-order slopes are the eigenvalues of
    :math:`\partial_{k_a}H-\varepsilon\partial_{k_a}S` projected onto an
    overlap-orthonormal basis of the degenerate subspace. The current velocity
    routine evaluates diagonal expectations in the eigenvectors returned by
    the band solver; it does not perform this additional diagonalization.
    Such expectations can depend on the chosen basis within a degenerate
    subspace.
