# GPSKet
This is a plugin for NetKet (www.netket.org), extending it by additional functionality (mostly not yet available in the core code).
We mostly understand this codebase a playground to test out new ideas and approaches, especially around the Gaussian Process State (GPS) ansatz for many-body systems.
As such, the code is not tested to the high standards of released software. Different parts of the code might be merged into the NetKet code in the future.

Functionality implemented in this plugin includes:
- GPS ansatz (including an autoregressive extension)
- Fermionic mean-field ansatzes (Slater determinants, Pfaffians)
- Backflow wavefunctions
- On-the-fly evaluations of local energies for different Hamiltonians
- Fast model updating (some models) for scaling improvements in the local energy evaluations (and sampling via the Metropolis-Hastings algorithm)
- Efficient ab-initio Hamiltonians
- Supervised learning of GPS within Bayesian frameworks

The code is, at this stage, largely undocumented, but different example applications can be found in the tutorials folder.
Further documentation and additional examples will be provided in the near future.
