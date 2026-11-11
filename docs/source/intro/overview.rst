==============
Overview
==============

Introduction
-------------------

**QTrans** is a software package designed for calculating transport phenomena arising from quasiparticle scattering.
It employs a finite-difference method to compute electron-quasiparticle coupling matrix elements, which are then used to
determine properties such as electrical conductivity, resistivity, intrinsic breakdown strength, and
superconducting critical temperature. **QTrans** supports input from both first-principles calculations and predicted
Hamiltonians from machine learning methods such as
`HamGNN <https://github.com/QuantumLab-ZY/HamGNN>`_. These predictions can
reduce the cost of obtaining electronic data for displaced supercells.


**QTrans** components
---------------------

**QTrans** consists of two main components: **preparation** and **calculation**.
Together, they provide a workflow from electronic-structure and phonon data
to electron-phonon coupling and transport properties.

* **Preparation** (``QTrans_prep``) converts electronic-structure outputs into
  the data format used by **QTrans** and constructs the real-space gradient
  matrices required for electron-phonon coupling calculations. It can use
  Hamiltonians obtained from first-principles calculations or machine
  learning predictions, and also provides tools for preparing phonon
  force-set data.

* **Calculation** (``QTrans_run``) reads the prepared electronic, phonon, and
  coupling data to evaluate band structures, densities of states,
  electron-phonon matrix elements, and derived properties, including
  conductivity, carrier mobility, resistivity, intrinsic dielectric
  breakdown strength, and superconducting critical temperature.

The preparation step produces reusable datasets, allowing multiple
calculations with different sampling meshes, temperatures, and other
settings without repeating the underlying data preparation.

Software interfaces
-------------------

The following software packages provide electronic data, Hamiltonian
predictions, or phonon data used by **QTrans**. Available preparation
interfaces depend on the workflow; consult the corresponding tutorial.

* `OpenMX`_ (Open source package for Material eXplorer) is a software package for 
  nanoscale material simulations based on density functional theory (DFT),
  norm-conserving pseudopotentials, and pseudo-atomic localized basis functions.

* `ABACUS`_ (Atomic-orbital Based Ab-initio Computation at USTC) is an open-source 
  computer code package based on density functional theory (DFT). 

* `SIESTA`_ is both a method and its computer program implementation, to perform efficient 
  electronic structure calculations and ab initio molecular dynamics simulations of 
  molecules and solids.

* `HONPAS`_ (Hefei Order-N Packages for Ab initio Simulations) is an ab initio 
  electronic structure calculation software package for linear scaling 
  first-principles density functional theory (DFT) calculations of large-scale systems.

* `HamGNN`_ is an E(3) equivariant graph neural network designed to train and predict 
  ab initio tight-binding (TB) Hamiltonians for molecules and solids.

* `Phonopy`_ is an open source package for phonon calculations at harmonic and quasi-harmonic levels.


.. _OpenMX: https://www.openmx-square.org/
.. _ABACUS: https://abacus.deepmodeling.com/en/latest/
.. _HONPAS: https://honpas.ustc.edu.cn/
.. _SIESTA: https://siesta-project.org/siesta/index.html
.. _HamGNN: https://github.com/QuantumLab-ZY/HamGNN
.. _Phonopy: https://phonopy.github.io/phonopy/
