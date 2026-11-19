.. _theory_conductivity:

============
Conductivity
============

Boltzmann Transport Equation (BTE)
----------------------------------

**QTrans** calculates electrical conductivity and carrier mobility using the
linearized Boltzmann transport equation (BTE) :cite:`ponceBTE2020`. The equations
below describe a spatially homogeneous system in a weak, static electric field,
with phonons in thermal equilibrium and electron-phonon scattering as the
collision mechanism.

We define the linear response of the electron distribution as

.. math::

    \partial_{E_\beta} f_{\nu\mathbf{k}}
    \equiv \left.\frac{\partial f_{\nu\mathbf{k}}}{\partial E_\beta}
    \right|_{\mathbf{E}=\mathbf{0}},

where :math:`\nu` is the electron band index, :math:`\mathbf{k}` is the electron
wave vector, and :math:`\beta` labels a Cartesian direction. Throughout this
page, :math:`e>0` is the elementary charge, so the electron charge is :math:`-e`.
The equilibrium electron and phonon occupations are denoted by
:math:`f^0_{\nu\mathbf{k}}` and :math:`n^0_{\mathbf{q}b}`, respectively, where
:math:`b` is the phonon branch index.

The linearized BTE can be written as

.. math::

    \partial_{E_\beta} f_{\nu\mathbf{k}}
    = e\tau_{\nu\mathbf{k}}v_{\nu\mathbf{k}\beta}
    \frac{\partial f^0_{\nu\mathbf{k}}}{\partial\varepsilon_{\nu\mathbf{k}}}
    + \tau_{\nu\mathbf{k}}\sum_{\mu b}
    \int_{\mathrm{BZ}}\frac{d^3q}{\Omega_{\mathrm{BZ}}}
    K_{\nu\mathbf{k},\mu\mathbf{k+q}}^{b}
    \partial_{E_\beta} f_{\mu\mathbf{k+q}},

where :math:`\varepsilon_{\nu\mathbf{k}}` is the electron energy,
:math:`v_{\nu\mathbf{k}\beta}` is the corresponding group velocity component,
and :math:`\Omega_{\mathrm{BZ}}` is the Brillouin-zone volume. The total
relaxation time :math:`\tau_{\nu\mathbf{k}}` is defined by

.. math::

    \tau_{\nu\mathbf{k}}^{-1}
    ={}& \frac{2\pi}{\hbar}\sum_{\mu b}
    \int_{\mathrm{BZ}}\frac{d^3q}{\Omega_{\mathrm{BZ}}}
    |g_{\mu\nu b}(\mathbf{k},\mathbf{q})|^2 \\
    &\times\Big[
    (n^0_{\mathbf{q}b}+1-f^0_{\mu\mathbf{k+q}})
    \delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_{\mu\mathbf{k+q}}
    -\hbar\omega_{\mathbf{q}b}) \\
    &\qquad+
    (n^0_{\mathbf{q}b}+f^0_{\mu\mathbf{k+q}})
    \delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_{\mu\mathbf{k+q}}
    +\hbar\omega_{\mathbf{q}b})\Big].

Here, :math:`g_{\mu\nu b}(\mathbf{k},\mathbf{q})` is the electron–phonon
coupling matrix element, and :math:`\omega_{\mathbf{q}b}` is the phonon angular
frequency. The terms with :math:`-\hbar\omega_{\mathbf{q}b}` and
:math:`+\hbar\omega_{\mathbf{q}b}` in the delta functions describe phonon
emission and absorption, respectively. This expression is summed over all
final electronic states and phonon modes; it is a total rate for the initial
state :math:`\nu\mathbf{k}`, rather than a rate for a single transition.

The scattering-in kernel in the linearized BTE is

.. math::

    K_{\nu\mathbf{k},\mu\mathbf{k+q}}^{b}
    ={}& \frac{2\pi}{\hbar}|g_{\mu\nu b}(\mathbf{k},\mathbf{q})|^2 \\
    &\times\Big[
    (n^0_{\mathbf{q}b}+1-f^0_{\nu\mathbf{k}})
    \delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_{\mu\mathbf{k+q}}
    +\hbar\omega_{\mathbf{q}b}) \\
    &\qquad+
    (n^0_{\mathbf{q}b}+f^0_{\nu\mathbf{k}})
    \delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_{\mu\mathbf{k+q}}
    -\hbar\omega_{\mathbf{q}b})\Big].

Unlike the expression for :math:`\tau_{\nu\mathbf{k}}^{-1}`, this kernel
contains the equilibrium occupation of the initial state
:math:`f^0_{\nu\mathbf{k}}`. It couples the distribution response at
:math:`\mu\mathbf{k+q}` to that at :math:`\nu\mathbf{k}`.

Electrical Conductivity and Mobility
------------------------------------

The electrical conductivity tensor relates the induced current density to
the applied field, :math:`j_\alpha=\sum_\beta\sigma_{\alpha\beta}E_\beta`:

.. math::

    \sigma_{\alpha\beta}
    = -\frac{e}{V_{\mathrm{uc}}}\sum_\nu
    \int_{\mathrm{BZ}}\frac{d^3k}{\Omega_{\mathrm{BZ}}}
    v_{\nu\mathbf{k}\alpha}\partial_{E_\beta}f_{\nu\mathbf{k}},

where :math:`V_{\mathrm{uc}}` is the unit-cell volume. For a single carrier type
in a semiconductor, the mobility tensor is obtained from its contribution to
the conductivity:

.. math::

    \mu_{\alpha\beta}=\frac{\sigma_{\alpha\beta}}{e n_c},

where :math:`n_c>0` is the carrier number density per unit volume. Electron and
hole mobilities use their respective conductivity contributions and densities.

**QTrans** provides two methods for solving the BTE.

Self-Energy Relaxation Time Approximation (ERTA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The self-energy relaxation time approximation, called ERTA in **QTrans** and
also commonly abbreviated as SERTA, neglects the scattering-in term on the
right-hand side of the linearized BTE. The distribution response is then

.. math::

    [\partial_{E_\beta}f_{\nu\mathbf{k}}]^{\mathrm{ERTA}}
    = e\tau_{\nu\mathbf{k}}v_{\nu\mathbf{k}\beta}
    \frac{\partial f^0_{\nu\mathbf{k}}}{\partial\varepsilon_{\nu\mathbf{k}}},

which gives

.. math::

    \sigma_{\alpha\beta}^{\mathrm{ERTA}}
    = \frac{e^2}{V_{\mathrm{uc}}}\sum_\nu
    \int_{\mathrm{BZ}}\frac{d^3k}{\Omega_{\mathrm{BZ}}}
    \left(-\frac{\partial f^0_{\nu\mathbf{k}}}
    {\partial\varepsilon_{\nu\mathbf{k}}}\right)
    v_{\nu\mathbf{k}\alpha}v_{\nu\mathbf{k}\beta}\tau_{\nu\mathbf{k}}.

Iterative Boltzmann Transport Equation (IBTE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The iterative method retains the scattering-in term. Starting from the ERTA
solution,

.. math::

    [\partial_{E_\beta}f_{\nu\mathbf{k}}]^{(0)}
    = [\partial_{E_\beta}f_{\nu\mathbf{k}}]^{\mathrm{ERTA}},

successive approximations are obtained from

.. math::

    [\partial_{E_\beta}f_{\nu\mathbf{k}}]^{(i+1)}
    = [\partial_{E_\beta}f_{\nu\mathbf{k}}]^{(0)}
    + \tau_{\nu\mathbf{k}}\sum_{\mu b}
    \int_{\mathrm{BZ}}\frac{d^3q}{\Omega_{\mathrm{BZ}}}
    K_{\nu\mathbf{k},\mu\mathbf{k+q}}^{b}
    [\partial_{E_\beta}f_{\mu\mathbf{k+q}}]^{(i)}.

The converged distribution response is used to evaluate the conductivity
and, for a specified carrier density, the mobility.

.. note::

    The equations above are written for one spin channel, or for a band index
    that explicitly includes all spin states. If a spin-degenerate system is
    represented using only one copy of each band, its total conductivity
    includes an additional factor of two. No extra factor is needed when spin
    states are counted explicitly. The carrier density used for mobility must
    follow the same spin-counting convention. **QTrans** includes the
    appropriate spin factor in its transport calculations.
