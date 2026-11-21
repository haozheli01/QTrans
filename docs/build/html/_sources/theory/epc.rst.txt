.. _theory_epc:

========================
Electron-Phonon Coupling
========================

**QTrans** evaluates electron-phonon coupling (EPC) by combining real-space
Hamiltonian derivatives, atomic orbital expansion coefficients, and phonon
eigenvectors. The formulation accounts for the motion and nonorthogonality
of the atomic orbital basis :cite:`zhongAcceleratingCalculationElectron2024`.
Long-range electrostatic contributions can be added for polar and
quadrupolar interactions.

Matrix Element and Phonon Convention
------------------------------------

The EPC matrix element connects an initial Bloch state
:math:`|\nu\mathbf{k}\rangle` to a final state
:math:`|\mu\mathbf{k+q}\rangle` through phonon branch :math:`b`:

.. math::

    g_{\mu\nu b}(\mathbf{k},\mathbf{q})
    = \langle\mu\mathbf{k+q}|\Delta_{\mathbf{q}b}\hat{\mathcal{H}}
    |\nu\mathbf{k}\rangle.

We use the electronic Bloch convention of :doc:`ham_ao`. For a phonon with
angular frequency :math:`\omega_{\mathbf{q}b}>0`, define

.. math::

    \ell_{\kappa,\mathbf{q}b}
    = \sqrt{\frac{\hbar}{2M_\kappa\omega_{\mathbf{q}b}}},
    \qquad
    \widetilde{e}_{\kappa\sigma}^{\mathbf{q}b}
    = \mathrm{e}^{\mathrm{i}\mathbf{q}\cdot\boldsymbol{\tau}_\kappa}
    e_{\kappa\sigma}^{\mathbf{q}b},

where :math:`M_\kappa` and :math:`\boldsymbol{\tau}_\kappa` are the mass and
unit-cell position of atom :math:`\kappa`, and :math:`\sigma` is a Cartesian
direction. The vectors :math:`e^{\mathbf{q}b}` are the normalized eigenvectors
of the mass-weighted dynamical matrix returned by Phonopy, with
:math:`\sum_{\kappa\sigma}|e_{\kappa\sigma}^{\mathbf{q}b}|^2=1`. The explicit
atomic phase in :math:`\widetilde{e}` matches the code's phonon convention.

Let :math:`u_{p\kappa\sigma}` denote the displacement of atom :math:`\kappa`
in cell :math:`p`. The phonon perturbation is

.. math::

    \Delta_{\mathbf{q}b}\hat{\mathcal{H}}
    = \sum_{p\kappa\sigma}
    \ell_{\kappa,\mathbf{q}b}
    \widetilde{e}_{\kappa\sigma}^{\mathbf{q}b}
    \mathrm{e}^{\mathrm{i}\mathbf{q}\cdot\mathbf{R}_p}
    \frac{\partial\hat{\mathcal{H}}}{\partial u_{p\kappa\sigma}}.

This definition gives :math:`g` units of energy. With Bloch states normalized
over :math:`N` cells, the corresponding second-quantized interaction contains
the conventional overall factor :math:`1/\sqrt{N}`:

.. math::

    \hat{\mathcal{H}}_{\mathrm{ep}}
    = \frac{1}{\sqrt{N}}\sum_{\mathbf{kq}\mu\nu b}
    g_{\mu\nu b}(\mathbf{k},\mathbf{q})
    \hat{c}_{\mu\mathbf{k+q}}^\dagger\hat{c}_{\nu\mathbf{k}}
    (\hat{b}_{\mathbf{q}b}+\hat{b}_{-\mathbf{q}b}^\dagger).

Here, :math:`\hat c` and :math:`\hat b` are electron and phonon annihilation
operators, respectively, with consistent phases for the modes at
:math:`\mathbf{q}` and :math:`-\mathbf{q}`.

Real-Space Interpolation
------------------------

Translational symmetry allows all displacement derivatives to be referred
to atoms in the central unit cell. Define

.. math::

    G^{\kappa\sigma}_{mi\alpha,nj\beta}
    = \left\langle\phi_{mi\alpha}\left|
    \frac{\partial\hat{\mathcal{H}}}{\partial u_{0\kappa\sigma}}
    \right|\phi_{nj\beta}\right\rangle,

where :math:`m,n` label lattice cells, :math:`i,j` label atoms, and
:math:`\alpha,\beta` label orbitals. The EPC is then

.. math::

    g_{\mu\nu b}(\mathbf{k},\mathbf{q})
    ={}& \sum_{mn}\sum_{i\alpha,j\beta}\sum_{\kappa\sigma}
    \ell_{\kappa,\mathbf{q}b}
    \widetilde{e}_{\kappa\sigma}^{\mathbf{q}b}
    \mathrm{e}^{-\mathrm{i}(\mathbf{k+q})\cdot\mathbf{R}_m}
    \mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_n} \\
    &\times(C_{i\alpha}^{\mathbf{k+q},\mu})^*
    G^{\kappa\sigma}_{mi\alpha,nj\beta}
    C_{j\beta}^{\mathbf{k},\nu}.

The conjugated coefficient and negative Bloch phase belong to the final
state, while the unconjugated coefficient and positive phase belong to the
initial state. Both cell indices are retained because displacing one atom
breaks translational symmetry locally.

Unlike the :doc:`band velocity <velocity>`, which uses analytic wave-vector
derivatives, EPC requires derivatives with respect to atomic displacements.
Once the real-space gradient tensor is prepared, the interpolation above
can be evaluated at arbitrary electronic and phonon wave vectors.

Moving-Basis Corrections
------------------------

The derivative of an atomic orbital Hamiltonian matrix element is not equal
to the matrix element of the derivative operator, because the orbitals move
with their parent atoms. For compact notation, use composite orbital indices
:math:`I=(m,i,\alpha)` and :math:`J=(n,j,\beta)`, and write
:math:`\partial_{\kappa\sigma}=\partial/\partial u_{0\kappa\sigma}`. Then

.. math::

    \partial_{\kappa\sigma}H_{IJ}
    = \langle\partial_{\kappa\sigma}\phi_I|
    \hat{\mathcal{H}}|\phi_J\rangle
    +G^{\kappa\sigma}_{IJ}
    +\langle\phi_I|\hat{\mathcal{H}}|
    \partial_{\kappa\sigma}\phi_J\rangle.

Define the one-sided overlap derivative and auxiliary matrices by

.. math::

    D^{\kappa\sigma}_{IJ}
    &= \langle\phi_I|\partial_{\kappa\sigma}\phi_J\rangle, \\
    A^{\kappa\sigma}
    &= D^{\kappa\sigma}-\partial_{\kappa\sigma}S
    = -(D^{\kappa\sigma})^\dagger, \\
    B^{\kappa\sigma}
    &= (A^{\kappa\sigma})^\dagger=-D^{\kappa\sigma},

where
:math:`\partial_{\kappa\sigma}S=D^{\kappa\sigma}+(D^{\kappa\sigma})^\dagger`.
Using the inverse overlap matrix within the finite atomic orbital space gives

.. math::

    G^{\kappa\sigma}
    = \partial_{\kappa\sigma}H
    + A^{\kappa\sigma}S^{-1}H
    + HS^{-1}(A^{\kappa\sigma})^\dagger.

The matrices in this expression act on the full set of supercell orbitals.
The overlap matrix must be invertible. This is the moving-basis correction
implemented when constructing the gradient tensor; using
:math:`\partial_{\kappa\sigma}H` alone omits the two basis-response terms.

For clarity, the matrix element of :math:`B` has the ordered indices

.. math::

    B^{\kappa\sigma}_{IJ}
    = (A^{\kappa\sigma}_{JI})^*,
    \qquad
    \langle\partial_{\kappa\sigma}\phi_I|\phi_J\rangle
    = \partial_{\kappa\sigma}S_{IJ}+B^{\kappa\sigma}_{IJ}.

The overlap on the right has the same bra and ket ordering as the left-hand
side. For real matrices, the adjoint reduces to a transpose; complex spinor
matrices require the conjugate transpose.

Finite-Displacement Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The preparation workflow obtains :math:`\partial H` and :math:`\partial S`
from displaced supercells, using either forward differences,

.. math::

    \partial_{\kappa\sigma}H
    \simeq\frac{H(+\Delta u_{\kappa\sigma})-H(0)}{\Delta u},

or central differences,

.. math::

    \partial_{\kappa\sigma}H
    \simeq\frac{H(+\Delta u_{\kappa\sigma})
    -H(-\Delta u_{\kappa\sigma})}{2\Delta u},

with analogous expressions for :math:`\partial S`. The input ``fd_method``
selects ``forward`` or ``central``, and
:ref:`epc_sc_disp <prep_epc_sc_disp>` sets the displacement in angstrom.
Forward differences are first-order accurate in :math:`\Delta u`; central
differences are second-order accurate for smooth matrix elements.

The auxiliary matrix :math:`A` is assembled from overlap derivatives and
additional on-site orbital derivative overlaps. On-site terms require
separate treatment because moving both orbitals together leaves their
overlap unchanged even when a one-sided derivative is nonzero. The corrected
:math:`G` tensor is stored for subsequent EPC calculations. Its units are
energy per length, internally Hartree per Bohr. Dense and sparse storage
represent the same contraction; sparse thresholding introduces an additional
numerical approximation.

Long-Range Electrostatic Correction
-----------------------------------

A finite real-space interpolation does not reproduce the nonanalytic
long-wavelength field of a polar displacement. **QTrans** can add analytic
long-range terms to the local EPC:

.. math::

    g_{\mu\nu b}=g^{\mathrm{local}}_{\mu\nu b}
    +g^{\mathrm{LR}}_{\mu\nu b}.

The contributions are added as complex amplitudes before taking
:math:`|g|^2`, so their interference is retained. This decomposition assumes
that the supplied local gradient and the analytic correction use a
consistent separation of local and long-range contributions.

Three-Dimensional Correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In atomic units, the implemented three-dimensional dipole and quadrupole
correction can be expressed as

.. math::

    g^{\mathrm{LR}}_{\mu\nu b}
    ={}& \frac{4\pi}{V_{\mathrm{uc}}}
    \sum_{\kappa\sigma}\ell_{\kappa,\mathbf{q}b}
    \widetilde e_{\kappa\sigma}^{\mathbf{q}b}
    \sum_{\mathbf G}^{\mathbf Q\ne0}
    \frac{\exp[-d(\mathbf Q)/\eta_{\mathrm E}]}{d(\mathbf Q)}
    \mathrm{e}^{-\mathrm{i}\mathbf Q\cdot\boldsymbol\tau_\kappa}
    \mathcal{M}^{(t)}_{\mu\nu}(\mathbf{k},\mathbf Q) \\
    &\times\left[
    \mathrm{i}\sum_a Q_a Z^*_{\kappa,a\sigma}
    +\frac{1}{2}\sum_{ac}Q_a
    \mathcal Q_{\kappa,\sigma ac}Q_c\right],

where :math:`\mathbf Q=\mathbf q+\mathbf G` contains the physical
:math:`2\pi` factor, :math:`V_{\mathrm{uc}}` is the unit-cell volume,
:math:`Z^*` is the Born effective charge tensor, and :math:`\mathcal Q` is the
dynamical quadrupole tensor in the code's convention. The screened
denominator is

.. math::

    d(\mathbf Q)=\mathbf Q\cdot\boldsymbol\epsilon_\infty\cdot\mathbf Q,

with the electronic dielectric tensor :math:`\boldsymbol\epsilon_\infty`.
The positive parameter :math:`\eta_{\mathrm E}` controls the Ewald damping;
its units match :math:`d(\mathbf Q)`. The dipole and quadrupole terms are
included only when their respective correction flags are enabled. With
:math:`\mathcal Q=0`, the expression reduces to the dipole correction.

The electronic factor uses a Taylor approximation of the plane-wave
operator in the atomic orbital basis. At order zero,

.. math::

    \mathcal M^{(0)}_{\mu\nu}
    = \mathbf C_{\mu\mathbf{k+q}}^\dagger
    S^{\mathbf k}\mathbf C_{\nu\mathbf k}.

At order one, the code includes the position-matrix contribution:

.. math::

    \mathcal M^{(1)}_{\mu\nu}
    = \mathbf C_{\mu\mathbf{k+q}}^\dagger
    \left[\sum_n\mathrm{e}^{\mathrm{i}\mathbf k\cdot\mathbf R_n}
    \left(S_{0,n}+\mathrm{i}\sum_aQ_a P^a_{0,n}\right)\right]
    \mathbf C_{\nu\mathbf k},

where :math:`P^a_{0,n}` denotes the supplied atomic orbital position matrix.
Orbital indices are suppressed in this expression. The parameter
:ref:`LRC_taylor_order <cal_LRC_taylor_order>` selects order zero or one.
The current three-dimensional order-one path supports only nonmagnetic
calculations without spin-orbit coupling.

Numerical Settings and Output
-----------------------------

The calculator reads the prepared gradient tensor from
:ref:`grad_mat_path <cal_grad_mat_path>`. The real-space range is selected by
:ref:`cell_range_cut <cal_cell_range_cut>`, and
:ref:`read_sparse_grad_mat <cal_read_sparse_grad_mat>` chooses its storage
format. Convergence should be checked with respect to the displacement,
preparation supercell, retained real-space range, and sparse threshold.

The dipole and quadrupole corrections are controlled by
:ref:`apply_correction <cal_apply_correction>` and
:ref:`quadrupole_correction <cal_quadrupole_correction>`. They require the
corresponding Born charge, dielectric, and quadrupole data. The
:ref:`q_cut <cal_q_cut>` parameter limits where corrections are applied in
the calling workflows.

.. note::

    Phonon frequencies enter the factor
    :math:`\sqrt{\hbar/(2M_\kappa\omega_{\mathbf qb})}`. The code stores the
    phonon energy :math:`\hbar\omega` in Hartree and masses in atomic units,
    giving the internal prefactor :math:`1/\sqrt{2M_\kappa E_{\mathbf qb}}`.
    Low-frequency branches are filtered using
    :ref:`phonon_frequency_cutoff <cal_phonon_frequency_cutoff>`, entered in
    meV. Imaginary phonon modes require a structural-stability assessment;
    taking their absolute frequencies does not establish physical stability.

.. note::

    The EPC path routine returns magnitudes :math:`|g|` in Hartree. Its
    current small-wave-vector guard returns zero when the norm of the
    internal Cartesian wave vector is below :math:`10^{-6}`. This is a
    numerical convention, not the physical long-wavelength limit of a polar
    coupling. If path symmetrization is requested, the current implementation
    averages :math:`|g|^2` over nearly degenerate phonon branches and returns
    its square root; it does not average over electronic degeneracies.
