.. _theory_superconductivity:

=================
Superconductivity
=================

**QTrans** estimates the superconducting transition temperature using a
simplified Allen-Dynes formula. 
The spectral definitions below use the isotropic, nonmagnetic,
spin-degenerate case as the reference convention.

Eliashberg Spectral Function
----------------------------

Let :math:`N_F=N(\varepsilon_F)` denote the electronic density of states per
unit cell, per unit energy, and per spin at the Fermi energy
:math:`\varepsilon_F`. The band sums below include one copy of each
spin-degenerate band, and Brillouin-zone integrals are normalized by the
Brillouin-zone volume :math:`\Omega_{\mathrm{BZ}}`. The variable
:math:`\omega` denotes angular frequency, and the electron-phonon coupling
matrix element :math:`g_{\mu\nu b}(\mathbf{k},\mathbf{q})` has units of energy.

In this convention, define the phonon linewidth parameter in angular-frequency
units as

.. math::

    \gamma_{\mathbf{q}b}
    = 2\pi\omega_{\mathbf{q}b}\sum_{\mu\nu}
    \int_{\mathrm{BZ}}\frac{d^3k}{\Omega_{\mathrm{BZ}}}
    |g_{\mu\nu b}(\mathbf{k},\mathbf{q})|^2
    \delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_F)
    \delta(\varepsilon_{\mu\mathbf{k+q}}-\varepsilon_F).

The corresponding energy linewidth parameter is
:math:`\hbar\gamma_{\mathbf{q}b}`. The Eliashberg spectral function is

.. math::

    \alpha^2F(\omega)
    = \frac{1}{2\pi\hbar N_F}\sum_b
    \int_{\mathrm{BZ}}\frac{d^3q}{\Omega_{\mathrm{BZ}}}
    \frac{\gamma_{\mathbf{q}b}}{\omega_{\mathbf{q}b}}
    \delta(\omega-\omega_{\mathbf{q}b}).

This normalization agrees with the ordinary Eliashberg spectral function
in :ref:`theory_resistivity`. The dimensionless electron–phonon coupling
constant is

.. math::

    \lambda=2\int_0^\infty\frac{d\omega}{\omega}\alpha^2F(\omega).

The factor of 2 in this frequency integral is part of the definition of
:math:`\lambda`; it is not an additional spin degeneracy factor. Spin counting
must instead be consistent between the electronic band sums and :math:`N_F`.

For :math:`\lambda>0`, the logarithmic average phonon frequency is

.. math::

    \omega_{\mathrm{log}}
    = \exp\left[
    \frac{2}{\lambda}\int_0^\infty\frac{d\omega}{\omega}
    \alpha^2F(\omega)\log\left(\omega\right)
    \right].

These integrals assume a dynamically stable phonon spectrum with positive
frequencies; imaginary modes require a separate assessment of structural
stability.

Allen-Dynes Critical Temperature
--------------------------------

The simplified Allen-Dynes expression implemented in **QTrans** is

.. math::

    T_{\mathrm{c}}
    = \frac{\hbar\omega_{\mathrm{log}}}{1.2k_{\mathrm{B}}}
    \exp\left[-\frac{1.04(1+\lambda)}
    {\lambda-\mu^*(1+0.62\lambda)}\right],

where :math:`\mu^*` is the dimensionless effective Coulomb pseudopotential.
This semi-empirical estimate describes isotropic, phonon-mediated pairing.
The full Allen-Dynes formula contains additional strong-coupling and
spectral-shape correction factors :math:`f_1f_2`; these factors are set to
unity in the current implementation. The simplified estimate should therefore
be used with care for strongly coupled or strongly anisotropic systems.

**QTrans** returns :math:`T_{\mathrm{c}}=0` when
:math:`\lambda-\mu^*(1+0.62\lambda)\leq0`. This is an implementation convention
outside the attractive regime of the estimate, rather than a general
criterion for the absence of superconductivity. The exponential is evaluated
only when the denominator is positive.

.. note::

    Internally, **QTrans** uses phonon energies in Hartree. The logarithmic
    average returned by the code is therefore the energy
    :math:`E_{\mathrm{log}}=\hbar\omega_{\mathrm{log}}`. The code evaluates
    :math:`k_{\mathrm{B}}T_{\mathrm{c}}` in Hartree and converts the result to
    kelvin for output. The spectral coordinate in ``a2f.dat`` is an energy
    in meV, despite being labeled ``omega``.

Anisotropic Superconductivity
-----------------------------

Under development.

