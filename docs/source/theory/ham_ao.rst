=====================================
Hamiltonian in Atomic Orbitals Basis
=====================================

Hamiltonian
------------

Using atomic orbital basis, we could describe the Bloch state :math:`\left | \nu\mathbf{k}  \right \rangle` as :cite:`zhongAcceleratingCalculationElectron2024`:

.. math::
    \left | \nu\mathbf{k}  \right \rangle = \frac{1}{\sqrt{N}}\sum_{n'}^{N} \mathrm{e}^{\mathrm{i}{\mathbf{R}}_{n'}\cdot\mathbf{k}}
    \sum_{j\beta}C^{\mathbf{k},\nu}_{j\beta} \left | \phi_{n'j\beta}  \right \rangle,
    \label{eq:blochstate}

where :math:`\nu` is the band index, and :math:`\mathbf{k}` is the wave vector of the Bloch state. :math:`N` is the number of unit cells
in Born-von Karman supercell, and :math:`\mathbf{R}_{n'}` is the lattice vector of the :math:`n\text{th}` unit cell.
The :math:`C^{\mathbf{k},\nu}_{j\beta}` is the expansion coefficient of :math:`\left | \phi_{n'j\beta}  \right \rangle`.
And the atomic orbits satisfy:

.. math::
    \left \langle \mathbf{r} | \phi_{n'j\beta}  \right \rangle = \phi_{j\beta}(\mathbf{r} - \mathbf{R}_{n'} - \boldsymbol{\tau}_{j}),

where :math:`\boldsymbol{\tau}_{j}` is the relative position of the :math:`j\text{th}` atom in its unit cell, and :math:`\beta` is the index
of orbit.

The Hamiltonian in atomic orbital representation can be expressed by:

.. math::
    \hat{\mathcal{H}}=\sum_{m'n'i'j'\alpha'\beta'} \widetilde{H}_{m'i'\alpha',n'j'\beta'} \left| \phi_{m'i'\alpha'}\right \rangle \left \langle \phi_{n'j'\beta'}\right|,

where :math:`\widetilde{H} = S^{-1}HS^{-1}` and :math:`S_{mi\alpha,nj\beta} = \left\langle \phi_{mi\alpha}| \phi_{nj\beta} \right\rangle`.
The hamiltoninian matrix elements satisfy :math:`H_{mi\alpha,nj\beta} = \left \langle \phi_{mi\alpha}| \hat{\mathcal{H}} | \phi_{nj\beta}\right \rangle`
   

Electron Eigen Problem
------------------------

Given the inherent non-orthogonality of atomic orbital basis sets, the determination of electronic energies and wave functions
necessitates the solution of a generalized eigenvalue problem :cite:`zhongTransferableEquivariantGraph2023`. Using Fourier transformation,
we write down the Hamiltonian matrix element in the reciprocal space as:

.. math::
    H^{\mathbf{k}}_{i\alpha,j\beta} = \sum_{n}\mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_{n}}H_{0i\alpha,nj\beta}

and also the overlap matrix element as:

.. math::
    S^{\mathbf{k}}_{i\alpha,j\beta} = \sum_{n}\mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}_{n}}S_{0i\alpha,nj\beta}

and the generalized eigenvalue problem can be expressed as:

.. math::
    H^{\mathbf{k}}\psi_{\nu\mathbf{k}} = \varepsilon_{\nu\mathbf{k}}S^{\mathbf{k}}\psi_{\nu\mathbf{k}},

where :math:`\varepsilon_{\nu\mathbf{k}}` is the electronic energy of Bloch state :math:`\left|\nu\mathbf{k}\right\rangle`.
