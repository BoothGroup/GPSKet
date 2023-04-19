# Scripts for *ab initio* benchmarks with GPS
This folder contains scripts to generate the data presented in our upcoming manuscript *A framework for efficient ab initio electronic structure with Gaussian Process States* by Yannic Rath and George H. Booth.

The following scripts are included:
- [H10_amplitudes.py](H10_amplitudes.py): This script generates the exact ground state wavefunction amplitudes for a 1D chain of 10 Hydrogen atoms (minimal basis) in different molecular orbital basis representations (canonical, split-local, local) with [PySCF](https://pyscf.org/)
- [H_chain_timing_analysis.py](H_chain_timing_analysis.py): This script prints the mean timing to evaluate a single local energy for a one-dimensional hydrogen chain with a uniform state. This either utilizes a pruning of vanishing terms (if a pruning threshold greater than 0 is used), or an efficient full contraction over all terms (if the pruning threshold is set to 0).
- [H2O.py](H2O.py): Ground state approximation of an H2O molecule in a 6-31G basis with the GPS (together with an optional augmentation by different reference states)
- [H50_1D.py](H50_1D.py): Ground state approximation for 1D chain of 50 Hydrogen atoms (minimal basis set) with GPS in local basis
- [H4x4x4.py](H4x4x4.py): Ground state approximation for cubic crystal of 4x4x4 Hydrogen atoms (minimal basis set) with GPS augmented by a single Slater determinant, script also includes the approximation of the one-body reduced density matrix
