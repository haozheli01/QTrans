.. _theory_resistivity:

===========
Resistivity
===========

**QTrans** evaluates phonon-limited metallic resistivity using either a
transport Eliashberg spectral function or the conductivity tensor obtained
from the Boltzmann transport equation (BTE).

Transport Eliashberg Spectral Function
--------------------------------------

The scalar spectral approach uses a Fermi-surface approximation and an
isotropic description of transport. To fix the normalization, the formulas
in this section refer to a nonmagnetic, spin-degenerate metal:
:math:`N_F=N(\varepsilon_F)` is the density of electronic states per unit cell,
per unit energy, and per spin, and the band sums include one copy of each
spin-degenerate band. The variable :math:`\omega` denotes angular frequency,
so the phonon energy is :math:`\hbar\omega`.

The Eliashberg spectral function is

.. math::

    \alpha^2F(\omega)
    = \frac{1}{2}\sum_b\int_{\mathrm{BZ}}
    \frac{d^3q}{\Omega_{\mathrm{BZ}}}
    \omega_{\mathbf{q}b}\lambda_{\mathbf{q}b}
    \delta(\omega-\omega_{\mathbf{q}b}),

where the mode-resolved electron-phonon coupling constant is

.. math::

    \lambda_{\mathbf{q}b}
    = \frac{2}{\hbar N_F\omega_{\mathbf{q}b}}
    \sum_{\mu\nu}\int_{\mathrm{BZ}}\frac{d^3k}{\Omega_{\mathrm{BZ}}}
    |g_{\mu\nu b}(\mathbf{k},\mathbf{q})|^2
    \delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_F)
    \delta(\varepsilon_{\mu\mathbf{k+q}}-\varepsilon_F).

Here, :math:`\varepsilon_F` is the Fermi energy,
:math:`g_{\mu\nu b}(\mathbf{k},\mathbf{q})` is an electron-phonon coupling
matrix element with units of energy, and :math:`\Omega_{\mathrm{BZ}}` is the
Brillouin-zone volume. The factor of 2 accounts for spin degeneracy in a nonmagnetic system. 
For collinear spin-polarized calculations, each spin channel is treated separately, 
so this factor is 1. This distinction is handled explicitly in the code.
These conventions give the dimensionless total coupling

.. math::

    \lambda
    = \sum_b\int_{\mathrm{BZ}}\frac{d^3q}{\Omega_{\mathrm{BZ}}}
    \lambda_{\mathbf{q}b}
    = 2\int_0^\infty\frac{d\omega}{\omega}\alpha^2F(\omega).

To describe current relaxation, **QTrans** uses the velocity factor associated
with the momentum relaxation time approximation (MRTA):

.. math::

    \eta_{\nu\mathbf{k},\mu\mathbf{k+q}}
    = 1-\frac{\mathbf{v}_{\nu\mathbf{k}}\cdot
    \mathbf{v}_{\mu\mathbf{k+q}}}{|\mathbf{v}_{\nu\mathbf{k}}|^2}.

For equal initial and final speed, this factor reduces to
:math:`1-\cos\theta`, where :math:`\theta` is the angle between the two
velocities. It suppresses forward scattering and emphasizes processes that
relax the current. Inserting this factor gives

.. math::

    \lambda_{\mathrm{tr},\mathbf{q}b}
    ={}& \frac{2}{\hbar N_F\omega_{\mathbf{q}b}}
    \sum_{\mu\nu}\int_{\mathrm{BZ}}\frac{d^3k}{\Omega_{\mathrm{BZ}}}
    |g_{\mu\nu b}(\mathbf{k},\mathbf{q})|^2 \\
    &\times\delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_F)
    \delta(\varepsilon_{\mu\mathbf{k+q}}-\varepsilon_F)
    \eta_{\nu\mathbf{k},\mu\mathbf{k+q}},

and the transport spectral function is

.. math::

    \alpha_{\mathrm{tr}}^2F(\omega)
    = \frac{1}{2}\sum_b\int_{\mathrm{BZ}}
    \frac{d^3q}{\Omega_{\mathrm{BZ}}}
    \omega_{\mathbf{q}b}\lambda_{\mathrm{tr},\mathbf{q}b}
    \delta(\omega-\omega_{\mathbf{q}b}).

Ziman and Allen Estimates
^^^^^^^^^^^^^^^^^^^^^^^^^

The Ziman-type estimate :cite:`zimanRhoCalculation1960` uses a free-electron
Drude prefactor:

.. math::

    \rho_{\mathrm{Ziman}}(T)
    = \frac{4\pi m_e}{n_c e^2 k_{\mathrm{B}}T}
    \int_0^\infty d\omega\,\hbar\omega\,
    \alpha_{\mathrm{tr}}^2F(\omega)
    n(\omega,T)[1+n(\omega,T)],

where :math:`m_e` is the free-electron mass, :math:`e>0` is the elementary
charge, :math:`n_c` is the total carrier number density per unit volume
including both spins, and

.. math::

    n(\omega,T)=\frac{1}{\exp[\hbar\omega/(k_{\mathrm{B}}T)]-1}.

The Allen-type estimate :cite:`allenRhoCalculation1978` replaces the
free-electron prefactor with a Fermi-surface velocity average:

.. math::

    \rho_{\mathrm{Allen}}(T)
    = \frac{2\pi V_{\mathrm{uc}}}
    {e^2 N_F\langle v_z^2\rangle k_{\mathrm{B}}T}
    \int_0^\infty d\omega\,\hbar\omega\,
    \alpha_{\mathrm{tr}}^2F(\omega)
    n(\omega,T)[1+n(\omega,T)],

where :math:`V_{\mathrm{uc}}` is the unit-cell volume and

.. math::

    \langle v_z^2\rangle
    = \frac{1}{N_F}\sum_\nu\int_{\mathrm{BZ}}
    \frac{d^3k}{\Omega_{\mathrm{BZ}}}
    v_{\nu\mathbf{k}z}^2\delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_F).

For an isotropic metal, this average is independent of the chosen Cartesian
direction. Using :math:`x=\hbar\omega/(2k_{\mathrm{B}}T)` and
:math:`n(1+n)=1/(4\sinh^2x)`, an equivalent expression is

.. math::

    \rho_{\mathrm{Allen}}(T)
    = \frac{2\pi V_{\mathrm{uc}}k_{\mathrm{B}}T}
    {e^2\hbar N_F\langle v_z^2\rangle}
    \int_0^\infty\frac{d\omega}{\omega}
    \frac{x^2}{\sinh^2x}\alpha_{\mathrm{tr}}^2F(\omega).

For a three-dimensional parabolic free-electron band,
:math:`2N_F\langle v_z^2\rangle/V_{\mathrm{uc}}=n_c/m_e`, so the two
prefactors agree.

.. note::

    These scalar estimates use an isotropic transport approximation. The MRTA
    velocity factor is also an approximation for general multiband Fermi
    surfaces. Anisotropic transport is better described using the BTE
    conductivity tensor below. Spin-resolved calculations require consistent
    spin counting in the spectral function, density of states, and carrier
    density; the prefactors above assume two equivalent spin channels.

Resistivity from the Conductivity Tensor
----------------------------------------

The :ref:`BTE approach <theory_conductivity>` gives the conductivity tensor as

.. math::

    \sigma_{\alpha\beta}(T)
    = -\frac{e}{V_{\mathrm{uc}}}\sum_\nu
    \int_{\mathrm{BZ}}\frac{d^3k}{\Omega_{\mathrm{BZ}}}
    v_{\nu\mathbf{k}\alpha}\partial_{E_\beta}f_{\nu\mathbf{k}},

where :math:`\partial_{E_\beta}f_{\nu\mathbf{k}}` is the distribution response
at zero electric field. The band sum must include all spin states, or the
corresponding spin degeneracy factor must be included explicitly.

The resistivity tensor is the matrix inverse of the conductivity tensor:

.. math::

    \boldsymbol{\rho}(T)=\boldsymbol{\sigma}(T)^{-1},
    \qquad
    \rho_{\alpha\beta}(T)=[\boldsymbol{\sigma}(T)^{-1}]_{\alpha\beta}.

This relation applies when the conductivity matrix is invertible. Only when
that matrix is diagonal does :math:`\rho_{\alpha\alpha}=1/\sigma_{\alpha\alpha}`
hold component by component. The tensor approach can describe anisotropic
metals within the scattering mechanisms and approximations included in the
BTE calculation.
