# QTrans

**Project website:** [QTrans on GitHub Pages](https://www.qtrans.com.cn/)

## Introduction

**QTrans** provides an efficient framework for evaluating electron-phonon coupling (EPC) matrix elements using a finite-difference approach.
It enables the prediction of metallic conductivity, resistivity, carrier mobility, intrinsic breakdown field strength, and superconducting transition temperature.

An installed **QTrans** package provides two command-line tools:

- `QTrans_prep` prepares the input data required for EPC calculations.
- `QTrans_run` calculates EPC matrix elements and related physical properties.

### Capabilities

- MPI-parallel preparation and calculation workflows.
- Finite-difference Hamiltonian gradients in a localized atomic-orbital basis.
- EPC matrix elements at arbitrary electron and phonon wave vectors.
- Input data obtained from first-principles packages, including
  [OpenMX](https://www.openmx-square.org/) and
  [ABACUS](https://abacus.deepmodeling.com/).
- Hamiltonians predicted by machine-learning models such as
  [HamGNN](https://github.com/QuantumLab-ZY/HamGNN), which can substantially
  reduce the cost of EPC calculations for large systems.
- Compatibility with LDA and GGA calculations, hybrid exchange-correlation
  functionals, and DFT+U calculations.
- Nonmagnetic, collinear magnetic, and noncollinear magnetic systems,
  including systems with spin-orbit coupling.
- Dense and sparse-matrix workflows. The optional C extension combines sparse
  matrices with OpenMP parallelism to accelerate transport calculations.

Detailed documents and examples are available on the
[QTrans project website](https://www.qtrans.com.cn/).

## References

If **QTrans** contributes to published work, please cite the references relevant
to the methods used in the calculation:

1. Y. Zhong, S. Liu, B. Zhang, Z. Tao, Y. Sun, W. Chu, X.-G. Gong, J.-H. Yang,
    and H. Xiang, "Accelerating the Calculation of Electron-Phonon Coupling
    Strength with Machine Learning", *Nature Computational Science* **4**,
    615-625 (2024). [doi:10.1038/s43588-024-00668-7](https://doi.org/10.1038/s43588-024-00668-7)

2. X. Li, Z. Cai, S. Liu, H. Li, Y. Sun, X.-G. Gong, and J.-H. Yang,
    "Electron-phonon couplings within density functional theory and beyond: Methods and application to FeSe",
    *Physical Review B* **113**, 184521 (2026). [doi:10.1103/f3l8-rktd](https://doi.org/10.1103/f3l8-rktd)


3. S. Liu, H. Xiang, X.-G. Gong, and J.-H. Yang, "Intrinsic Breakdown
    Strength: Theoretical Derivation and First-Principles Calculations",
    *Physical Review Letters* **134**, 086301 (2025).
    [doi:10.1103/PhysRevLett.134.086301](https://doi.org/10.1103/PhysRevLett.134.086301)


## Contributors

QTrans is developed and maintained by:

- Dr. Shixu Liu
- Haozhe Li
- Dr. Yang Zhong
- Jiarui Bu
- Xingding Li

Scientific advisors:

- Prof. Xin-Gao Gong
- Prof. Hongjun Xiang
- Prof. Ji-Hui Yang

