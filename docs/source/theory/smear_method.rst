.. _tutorial-smearing:
.. _theory_smearing:

================
Smearing Methods
================

Brillouin-zone integrals contain Dirac delta functions that select electronic
states or enforce energy conservation. On a finite reciprocal-space mesh,
**QTrans** replaces these delta functions with finite-width kernels. The
choice of kernel and width controls the balance between mesh-sampling noise
and numerical broadening.

Broadened Delta Functions
-------------------------

For an energy difference :math:`\Delta E` and a positive energy width
:math:`\eta`, the broadened delta function is

.. math::

    \delta_\eta(\Delta E)
    = \frac{1}{\eta}K\left(\frac{\Delta E}{\eta}\right),
    \qquad
    \int_{-\infty}^{\infty}K(x)\,dx=1.

The prefactor :math:`1/\eta` preserves the normalization and gives
:math:`\delta_\eta` units of inverse energy. For example, the broadened
electronic density of states is

.. math::

    D_\eta(E)
    = \sum_\nu\int_{\mathrm{BZ}}\frac{d^3k}{\Omega_{\mathrm{BZ}}}
    \delta_\eta(\varepsilon_{\nu\mathbf{k}}-E).

Here, :math:`\nu` is the band index and :math:`\Omega_{\mathrm{BZ}}` is the
Brillouin-zone volume. Spin degeneracy is included separately through the
state counting or integration weights; smearing does not introduce a spin
factor.

The numerical helper ``w0gauss(x, ngauss)`` evaluates the dimensionless kernel
:math:`K(x)`. The following expressions describe its analytic kernels; the
implementation limits exponential arguments for numerical stability.

Gaussian Smearing
-----------------

For ``ngauss = 0``, the kernel is a simple Gaussian:

.. math::

    K_{\mathrm{G}}(x)=\frac{1}{\sqrt{\pi}}\exp(-x^2),
    \qquad
    \delta_\eta^{\mathrm{G}}(\Delta E)
    = \frac{1}{\eta\sqrt{\pi}}
    \exp\left[-\left(\frac{\Delta E}{\eta}\right)^2\right].

This positive, symmetric kernel is the default. Its standard deviation is
:math:`\eta/\sqrt{2}`, and its full width at half maximum is
:math:`2\sqrt{\ln2}\,\eta`. Thus the input width :math:`\eta` is neither the
standard deviation nor the full width at half maximum.

Fermi-Dirac Smearing
--------------------

For ``ngauss = -99``, the kernel is the derivative of a Fermi-Dirac occupation:

.. math::

    K_{\mathrm{FD}}(x)
    = \frac{\exp(x)}{[1+\exp(x)]^2}
    = \frac{1}{4\cosh^2(x/2)}.

This kernel is positive and symmetric. With :math:`\eta=k_{\mathrm{B}}T`, it
coincides with the physical thermal window

.. math::

    -\frac{\partial f^0(\varepsilon)}{\partial\varepsilon}
    = \frac{1}{4k_{\mathrm{B}}T}
    \operatorname{sech}^2\left(
    \frac{\varepsilon-\mu}{2k_{\mathrm{B}}T}\right),

where :math:`\mu` is the chemical potential. However, selecting this numerical
kernel does not automatically set its width equal to the physical thermal
energy. The ``temperature`` input controls the physical Fermi-Dirac and
Bose-Einstein occupations independently of the numerical smearing widths.

Cold Smearing
-------------

For ``ngauss = -1``, the Marzari–Vanderbilt cold-smearing kernel is

.. math::

    K_{\mathrm{cold}}(x)
    = \frac{1}{\sqrt{\pi}}
    \exp\left[-\left(x-\frac{1}{\sqrt{2}}\right)^2\right]
    (2-\sqrt{2}\,x).

This kernel is asymmetric and can take negative values. Its argument must
therefore follow the code's sign convention: in DOS calculations,
:math:`x=(\varepsilon_{\nu\mathbf{k}}-E)/\eta`. Reversing that argument changes
the broadened spectrum.

Methfessel-Paxton Smearing
--------------------------

For integer orders :math:`1\leq p\leq10`, selected by ``ngauss = p``, the
Methfessel-Paxton kernel is

.. math::

    K_p(x)=\frac{\exp(-x^2)}{\sqrt{\pi}}
    \sum_{m=0}^{p}\frac{(-1)^m}{4^m m!}H_{2m}(x),

where :math:`H_n(x)` denotes the physicists' Hermite polynomial. Order zero
reduces to the simple Gaussian. For example,

.. math::

    K_1(x)=\frac{\exp(-x^2)}{\sqrt{\pi}}\left(\frac{3}{2}-x^2\right).

The higher-order terms cancel low-order moment errors, which can improve
integration of sufficiently smooth quantities. These kernels have negative
lobes and can produce oscillations or negative contributions to a broadened
DOS. They should not be interpreted as probability densities.

Smearing Parameters in QTrans
-----------------------------

All widths in the following table are entered in meV and converted internally
to Hartree. They are energy widths, including ``smearq``; none is a width in
reciprocal-space coordinates.

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Parameter
      - Default
      - Role
    * - :ref:`dos_smear <cal_dos_smear>`
      - 10.0 meV
      - Width of the delta functions in broadened electronic or phonon DOS.
    * - :ref:`dos_gauss_type <cal_dos_gauss_type>`
      - 0
      - Kernel selector for broadened DOS calculations.
    * - :ref:`smeark <cal_smeark>`
      - 25.0 meV
      - Electronic energy width used in metallic Fermi-level calculations,
        Fermi-surface delta functions, and energy-resolved averages.
    * - :ref:`smearq <cal_smearq>`
      - 25.0 meV
      - Width of energy-conservation delta functions in scattering rates and
        of phonon-energy delta functions in Eliashberg spectra.
    * - :ref:`gauss_type <cal_gauss_type>`
      - 0
      - Kernel selector for metallic Fermi-level calculations and selected
        electronic Fermi-surface integrations.

The delta-kernel helper implements ``0``, ``-99``, ``-1``, and positive orders
from ``1`` to ``10``. Other selectors are not implemented as usable kernels.
Availability in this helper does not imply support throughout every workflow.

.. note::

    The cumulative occupation helper ``wgauss`` currently implements only
    ``0`` and ``-99``. Metallic Fermi-level searches use this helper, including
    the reference-energy search in the electronic DOS workflow. Use these two
    selectors for such workflows. Cold and higher-order Methfessel-Paxton
    delta kernels can be evaluated by ``w0gauss``, but their cumulative
    occupations are not implemented in the current code.

For the supported cumulative occupations, the metallic electron count uses

.. math::

    N_e=\sum_{\nu\mathbf{k}}w_{\mathbf{k}}
    W\left(\frac{\varepsilon_F-\varepsilon_{\nu\mathbf{k}}}{\eta}\right),

with the appropriate state-counting weights :math:`w_{\mathbf{k}}` and

.. math::

    W_{\mathrm{G}}(x)=\frac{1}{2}\operatorname{erfc}(-x),
    \qquad
    W_{\mathrm{FD}}(x)=\frac{1}{1+\exp(-x)}.

Both satisfy :math:`dW/dx=K(x)`. The electronic DOS workflow uses
``dos_smear`` and ``dos_gauss_type`` for this search; transport-related
metallic searches use ``smeark`` and ``gauss_type``.

.. note::

    ``gauss_type`` is not a global switch for all delta functions. In the
    current implementation, electron-phonon scattering rates, breakdown
    energy averages, and the scalar transport spectra used for resistivity
    use Gaussian kernels. Superconducting pairing spectra apply ``gauss_type``
    to the electronic Fermi-surface delta functions, while their
    phonon-energy broadening remains Gaussian. Changing ``gauss_type`` to
    ``-99`` therefore does not turn all transport integrations into
    Fermi-Dirac smearing.

When ``tetra_integral`` is enabled for DOS calculations, tetrahedron
integration replaces the broadened DOS delta functions. The DOS integration
then does not use ``dos_smear`` or ``dos_gauss_type``; the electronic
reference-energy search still uses those parameters in the current workflow.

Convergence with Mesh and Width
-------------------------------

A finite width compensates for discrete reciprocal-space sampling, but also
smooths genuine spectral structure. At fixed mesh density, an excessively
small width can leave too few states near the energy shell and produce noisy
or underestimated integrals. A large width can merge spectral features and
admit transitions far from exact energy conservation.

Converge the target observable by refining the electronic and phonon meshes
and reducing the widths together. For scattering calculations, check the
energy-conservation width ``smearq`` separately from the electronic width
``smeark``. For spectra, also converge the output energy-grid spacing; a fine
output grid alone does not improve the underlying Brillouin-zone sampling.
The default widths are starting values rather than universal convergence
settings.

The transition preselection parameter :ref:`e_thr <cal_e_thr>` is an energy
cutoff, not a smearing width. Its default is 75 meV, three times the default
``smearq``. It is configured independently and is not automatically rescaled
when ``smearq`` changes. Check that it retains the relevant kernel tails,
particularly when increasing the width.
