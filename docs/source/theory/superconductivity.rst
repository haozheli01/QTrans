.. _theory_superconductivity:

==================
Superconductivity
==================

Allen-Dynes critical temperature
----------------------------------

The superconducting critical temperature :math:`T_\text{c}` can be estimated using the semi-empirical Allen-Dynes formula.
We start from the phonon linewidth, which can be expressed as:

.. math::
    \gamma_{\mathbf{q}b} = 2\pi \omega_{\mathbf{q}b}\sum_{\mu\nu\mathbf{k}} {\left|g^{\mathbf{q}b}_{\mu\mathbf{k+q},\nu\mathbf{k}}\right|}^{2}
    \delta(\varepsilon_{\nu\mathbf{k}} - \varepsilon_\text{F})\delta(\varepsilon_{\mu\mathbf{k+q}} - \varepsilon_\text{F}).

Then, the Eliashberg spectral function can be written as:

.. math::
    \alpha^{2}F(\omega) = \frac{1}{2\pi N(\varepsilon_\text{F})}\sum_{\mathbf{q}b}
    \frac{\gamma_{\mathbf{q}b}}{\omega_{\mathbf{q}b}}\delta(\omega - \omega_{\mathbf{q}b}).

So, the electron phonon coupling constant can be estimated by:

.. math::
    \lambda = 2\int\frac{\alpha^2 F(\omega)}{\omega}d\omega.

The logarithmic average of phonon frequencies :math:`\omega_\text{log}` is defined as:

.. math::
    \omega_\text{log} = \text{exp}\left[ \frac{2}{\lambda} \int \frac{\alpha^2 F(\omega)}{\omega}\text{log}(\omega)d\omega \right],

Finally, we can use Allen-Dynes formula to compute the :math:`T_\text{c}` as:

.. math::
    T_\text{c} = \frac{\omega_\text{log}}{1.2}\text{exp}(-\frac{1.04(1+\lambda)}{\lambda - \mu^{*}(1+0.62\lambda)}),

where :math:`\mu^{*}` is the semi-empirical Coulomb pseudopotential. 


Anisotropic Superconductivity
----------------------------------

In progress

