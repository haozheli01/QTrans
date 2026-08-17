.. _theory-breakdown:

===============================
Intrinsic Dielectric Breakdown
===============================

.. image:: ../_static/breakdown.png
   :alt: breakdown.png
   :width: 419px
   :height: 200px
   :scale: 100 %
   :align: center
   :class: custom-class

The energy gain rate of the electron with energy :math:`E` in
a electric field :math:`F` can be written as :cite:`sunIntrinsicElectricalBreakdown2012, liuIntrinsicBreakdownStrength2025`:

.. math::
    \label{eq:energy-gain-rate}
    A(E, F) = \frac{e^2\tau(E)F^2}{3m^{*}},

where :math:`e` is the electronic charge and :math:`m^{*}` is the effective electronic mass. The average electron relaxation time
:math:`\tau(E)` is the reciprocal of the average electron scattering rate, given by:

.. math::
    \frac{1}{\tau(E)} = \frac{1}{D(E)}\sum_{\nu\mathbf{k}}\frac{1}{\tau_{\nu\mathbf{k}}}\delta(\varepsilon_{\nu\mathbf{k}}-E),

where :math:`D(E)` is the density of states. Using Fermi's golden rule, :math:`\tau_{\nu\mathbf{k}}` can be evaluated as:

.. math::
    \begin{align}
    \frac{1}{\tau_{\nu\mathbf{k}}} =& \frac{2\pi}{\hbar} \sum_{\mu\mathbf{q}b} {\left|g^{\mathbf{q}b}_{\mu\mathbf{k+q},\nu\mathbf{k}}\right|}^{2} \\ \notag
    &\left[ n_{\mathbf{q}b} \delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_{\mu\mathbf{k+q}} + \hbar\omega_{\mathbf{q}b}) + 
    (n_{\mathbf{q}b} + 1) \delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_{\mu\mathbf{k+q}} - \hbar\omega_{\mathbf{q}b})
    \right],
    \end{align}

where :math:`\varepsilon_{\nu\mathbf{k}}` and :math:`\varepsilon_{\mu\mathbf{k+q}}` are the electronic energies of initial state
:math:`\left| \nu\mathbf{k} \right\rangle` and final state :math:`\left| \mu\mathbf{k+q} \right\rangle`, respectively.
The :math:`n_{\mathbf{q}b}` is the Bose-Einstein distribution for the phonon with energy :math:`\hbar\omega_{\mathbf{q}b}`.

Meanwhile the energy loss rate of the electron during the scattering processes with phonon can be evaluated by:

.. math::
    \begin{align}
    B(E) =& \frac{2\pi}{\hbar D(E)}\sum_{\nu\mathbf{k}}\sum_{\mu\mathbf{q}b}
    (\hbar\omega_{\mathbf{q}b}){\left|g^{\mathbf{q}b}_{\mu\mathbf{k+q},\nu\mathbf{k}}\right|}^{2} \\ \notag
    &\left[ (n_{\mathbf{q}b} + 1) \delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_{\mu\mathbf{k+q}} - \hbar\omega_{\mathbf{q}b}) -
    n_{\mathbf{q}b} \delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_{\mu\mathbf{k+q}} + \hbar\omega_{\mathbf{q}b})
    \right]\delta(\varepsilon_{\nu\mathbf{k}}-E),
    \end{align}

Then, using the von Hipple breakdown criterion :cite:`vonhippelElectricBreakdownSolid1937`:

.. math::
    A(E,F) > B(E),\;\; E\in\left[ \text{CBM}, \text{CBM}+E_\text{g} \right],

where, CBM represents the conductive band minimum and :math:`E_\text{g}` is the band gap, we obatin:

.. math::
    F_{\text{bd}} = \text{Max}\left[ \frac{\sqrt{3m^{*}}}{e}\sqrt{\frac{1}{\tau(E)}B(E)} \right], \;\; E\in\left[ \text{CBM}, \text{CBM}+E_\text{g} \right].

