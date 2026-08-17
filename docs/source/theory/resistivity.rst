.. _theory_resistivity:

==============
Resistivity
==============

Calculation of resistivity for metals
--------------------------------------
There are two methods for computing resistivity in **QTrans**:

1. Using the Eliashberg transport spectral function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Firstly, the isotropic Eliashberg spectral function can be obtained as:

.. math::
    \alpha^2F(\omega) = \frac{1}{2} \sum_b \int_{\text{BZ}} \frac{d^3q}{\Omega_{\text{BZ}}} \,
     \omega_{\mathbf{q}b} \lambda_{\mathbf{q}b} \, \delta(\omega - \omega_{\mathbf{q}b}),

where the electron-phonon coupling strength associated with a specific phonon mode :math:`b` and wavevector :math:`\mathbf{q}` is

.. math::
    \lambda_{\mathbf{q}b} = \frac{1}{N(\varepsilon_F)\, \omega_{\mathbf{q}b}} \sum_{\mu\nu} \int_{\text{BZ}} \frac{d\mathbf{k}}{\Omega_{\text{BZ}}} \, 
    \left| g_{\mu\nu b}(\mathbf{k}, \mathbf{q}) \right|^2 
    \delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_F) 
    \delta(\varepsilon_{\mu\mathbf{k+q}} - \varepsilon_F).

Here :math:`\varepsilon_F` is the Fermi level.

The Eliashberg transport spectral function is an extension of the isotropic Eliashberg spectral function, incorporating the relative change 
in electron velocity for each scattering process. This approach is also known as the momentum relaxation time approximation (MRTA):

.. math::
    \alpha_{tr}^2F(\omega) = \frac{1}{2} \sum_b \int_{\text{BZ}} \frac{d^3q}{\Omega_{\text{BZ}}} \,
     \omega_{\mathbf{q}b} \lambda_{tr,\mathbf{q}b} \, \delta(\omega - \omega_{\mathbf{q}b}),

where an efficiency factor is introduced to characterize the effective change in electron velocity resulting from scattering:

.. math::
    \lambda_{tr,\mathbf{q} b} = \frac{1}{N(\varepsilon_F)\, \omega_{\mathbf{q}b}} \sum_{\mu\nu} \int_{\text{BZ}} \frac{d\mathbf{k}}{\Omega_{\text{BZ}}} \, 
    \left| g_{\mu\nu b}(\mathbf{k}, \mathbf{q}) \right|^2 
    \delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_F) 
    \delta(\varepsilon_{\mu\mathbf{k+q}} - \varepsilon_F)
    (1 - \frac{ \mathbf{v}_{\nu\mathbf{k}} \cdot \mathbf{v}_{\mu\mathbf{k+q}} }{ |\mathbf{v}_{\nu\mathbf{k}}|^2 }).

We can obtain the resistivity by using the most popular approximation--Ziman's resistivity formula :cite:`zimanRhoCalculation1960`:

.. math::
    \rho(T) = \frac{4\pi m_e}{n_c e^2 k_B T} \int_0^{\infty} d\omega \, \hbar \omega \, 
    \alpha^2_{\text{tr}}F(\omega) \, n(\omega, T)\left[1 + n(\omega, T)\right],

or altenatively, Allen's resistivity formula :cite:`allenRhoCalculation1978` :

.. math::
    \rho(T) &= \frac{2\pi V_{uc} k_B T}{e^2 \hbar N_F \langle v_z^2 \rangle} \int_0^{\infty} d\omega \, 
    \omega \, \frac{({\frac{\hbar \omega}{2k_B T}})^2}{\sinh^2 ({\frac{\hbar \omega}{2k_B T}})} \, \alpha^2_{\text{tr}}F(\omega)\\
    &=\frac{2\pi V_{uc}}{e^2 N_F \langle v_z^2 \rangle k_B T} \int_0^{\infty} d\omega \, \hbar \omega \, 
    \alpha^2_{\text{tr}}F(\omega) \, n(\omega, T)\left[1 + n(\omega, T)\right],

where :math:`\langle v_z^2 \rangle` is the the average square of the Fermi velocity along the transport z direction.

.. note::
    The above equations are derived under the assumption of isotropy and are therefore might only suitable for isotropic metals.

2. Using the electrical conductivity tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conductivity can also be directly obtained from the electrical conductivity tensor calculated using the BTE:

.. math::
    \sigma_{\alpha\beta} = - \frac{e}{V_{uc}}\sum_{\nu}\int \frac{d^3k}{\Omega_{\text{BZ}}}
     v_{\nu\mathbf{k}\alpha}\partial_{E_\beta} f_{\nu\mathbf{k}}.

And thus we have:

.. math::
    \rho(T)_{\alpha\beta} = (\sigma^{-1}(T))_{\alpha\beta}.

.. note::
    This method is not limited to isotropic metals and is applicable to all metallic systems.