.. _theory_conductivity:

==============
Conductivity
==============

Boltzmann Transport Equation (BTE)
----------------------------------

In **QTrans**, we calculate the conductivity (mobility) based on the linearised Boltzmann transport equation :cite:`ponceBTE2020` :

.. math::
    &\partial_{E_\beta} f_{\nu\mathbf{k}} = e v_{\nu\mathbf{k}\beta}\frac{\partial{f_{\nu\mathbf{k}}^{0}}}{\partial{\varepsilon_{\nu\mathbf{k}}}}
    + \frac{2\pi\tau_{\nu\mathbf{k} \rightarrow \mu\mathbf{k+q}}}{\hbar}\sum_{\mu b} \int \frac{d^3q}{\Omega_{\text{BZ}}} \left| g_{\mu\nu b}(\mathbf{k},\mathbf{q}) \right|^2 \\
    &\times [(n_{\mathbf{q}b}^0 + 1 - f_{\nu\mathbf{k}}^0)\delta( \varepsilon_{\nu\mathbf{k}} - \varepsilon_{\mu\mathbf{k+q}}+ \hbar\omega_{\mathbf{q} b}) +
    (n_{\mathbf{q} b}^0 + f_{\nu\mathbf{k}}^0)\delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_{\mu\mathbf{k+q}} - \hbar\omega_{\mathbf{q} b})]
    \partial_{E_\beta} f_{\mu\mathbf{k+q}},
    \label{eq:bte}

where the scattering rate :math:`\tau_{\nu\mathbf{k} \rightarrow \mu\mathbf{k+q}}^{-1}` is defined as:

.. math::
    &\tau_{\nu\mathbf{k} \rightarrow \mu\mathbf{k+q}}^{-1} = \frac{2\pi}{\hbar}\sum_{\mu b} \int \frac{d^3q}{\Omega_{\text{BZ}}} \left| g_{\mu\nu b}(\mathbf{k},\mathbf{q}) \right|^2\\
    &\times [(n_{\mathbf{q}b}^0 + 1 - f_{\mu\mathbf{k+q}}^0)\delta( \varepsilon_{\nu\mathbf{k}}-\varepsilon_{\mu\mathbf{k+q}}-\hbar\omega_{\mathbf{q} b}) +
    (n_{\mathbf{q} b}^0 + f_{\mu\mathbf{k+q}}^0)\delta(\varepsilon_{\nu\mathbf{k}}-\varepsilon_{\mu\mathbf{k+q}}+\hbar\omega_{\mathbf{q} b})].

Here, we introduced the short-hand notation :math:`\partial_{E_\beta} f_{\nu\mathbf{k}} = (\partial f_{\nu\mathbf{k}} / \partial_{E_\beta})|_\mathbf{E = 0}`, 
and :math:`f_{\nu\mathbf{k}}` is the electron distribution function of band :math:`n` with the wavevector :math:`\mathbf{k}`. 
:math:`f_{\nu\mathbf{k}}^0` and :math:`n_{\mathbf{q} b}^0` denote the Fermi-Dirac and Bose-Einstein equilibrium distribution functions, respectively.

Calculation of conductivity
----------------------------
The electrical conductivity tensor is defined as:

.. math::
    \sigma_{\alpha\beta} = - \frac{e}{V_{uc}}\sum_{\nu}\int \frac{d^3k}{\Omega_{\text{BZ}}}
     v_{\nu\mathbf{k}\alpha}\partial_{E_\beta} f_{\nu\mathbf{k}}.

Two methods for computing conductivity using the Boltzmann Transport Equation (BTE) have been implemented in **QTrans**:

1. Self-energy relaxation time approximation (ERTA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the most common approximations for BTE is the self-energy relaxation time approximation (ERTA), which neglects the second term in
the right-hand of BTE and gives the conductivity as:

.. math::
    \sigma_{\alpha\beta}^\text{ERTA} = - \frac{e}{V_{uc}}\sum_{\nu}\int \frac{d^3k}{\Omega_{\text{BZ}}} 
    \frac{\partial{f_{\nu\mathbf{k}}^{0}}}{\partial{\varepsilon_{\nu\mathbf{k}}}} v_{\nu\mathbf{k}\alpha}v_{\nu\mathbf{k}\beta}\tau_{\nu\mathbf{k}}.

For metals, we can calculate the conductivity, while for semiconductors, we calculate mobility instead. 
The mobility is obtained from the conductivity upon dividing by the carrier density :math:`n_c` :

.. math::
    \mu_{\alpha\beta}^\text{ERTA} = \frac{\sigma_{\alpha\beta}^\text{ERTA}}{e n_c}.

2. Iterative Boltzmann transport equation (IBTE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**QTrans** also implements an iterative solver to obtain :math:`\partial_{E_\beta} f_{\nu\mathbf{k}}` starting with a zero-order solution
:math:`[\partial_{E_\beta} f_{\nu\mathbf{k}}]^{(0)} = [\partial_{E_\beta} f_{\nu\mathbf{k}}]^\text{ERTA} = e v_{\nu\mathbf{k}\beta}\frac{\partial{f_{\nu\mathbf{k}}^{0}}}{\partial{\varepsilon_{\nu\mathbf{k}}}}`:

.. math::
    [\partial_{E_\beta} f_{\nu\mathbf{k}}]^{(i+1)} = [\partial_{E_\beta} f_{\nu\mathbf{k}}]^{(0)} + 
    \frac{2\pi\tau_{\nu\mathbf{k} \rightarrow \mu\mathbf{k+q}}}{\hbar}\sum_{\mu b} \int \frac{d^3q}{\Omega_{\text{BZ}}}
    \tau_{\mu\mathbf{k+q} \rightarrow \nu\mathbf{k}}[\partial_{E_\beta} f_{\mu\mathbf{k+q}}]^{(i)},

where :math:`\tau_{\mu\mathbf{k+q} \rightarrow \nu\mathbf{k}}` is defined in a similar way as 
:math:`\tau_{\nu\mathbf{k} \rightarrow \mu\mathbf{k+q}}`.

By iteratively solving the above equation, a converged :math:`\partial_{E_\beta} f_{\nu\mathbf{k}}` can be obtained, 
from which the conductivity or mobility can be computed.

.. note::
    The above equation does not account for spin degeneracy, but it is correctly included in the code.