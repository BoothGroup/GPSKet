import netket as nk
import qGPSKet.models as qGPS

from qGPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from qGPSKet.sampler.fermionic_hopping import MetropolisHopping, MetropolisFastHopping
from qGPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonian

from qGPSKet.models import qGPS

from pyscf import scf, gto, ao2mo, fci, lo
from pyscf.tools import ring

import jax.numpy as jnp


"""
This first bit just sets up the Hamiltonian with PySCF.
In particular it gives us the 1 and 2 electron integrals (h1 and h2) which are required to set up the ab-initio
Hamiltonian with qGPSKet/NetKet.
The Hamiltonian is either represented in a canonical or a "local" orbital basis.
"""
local_basis = True

mol = gto.Mole()

# as an example we set up a ring of 8 Hydrogen atoms with dist 1 a0
mol.build(
    atom = [('H', x) for x in ring.make(8, 1)],
    basis = 'sto-3g',
    symmetry = True,
    unit="Bohr"
)

nelec = mol.nelectron
print('Number of electrons: ', nelec)

myhf = scf.RHF(mol)
ehf = myhf.scf()
norb = myhf.mo_coeff.shape[1]
print('Number of molecular orbitals: ', norb)

# Get hamiltonian elements
# 1-electron 'core' hamiltonian terms, transformed into MO basis
h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

# Get 2-electron electron repulsion integrals, transformed into MO basis
eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)

# Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
# Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
h2 = ao2mo.restore(1, eri, norb)

# Transform to a local orbital basis if wanted
if local_basis:
    loc_coeff = lo.orth_ao(mol, 'meta_lowdin')
    ovlp = myhf.get_ovlp()
    # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
    assert(np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)),np.eye(norb)))
    # Find the hamiltonian in the local basis
    hij_local = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
    hijkl_local = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
    h1 = hij_local
    h2 = hijkl_local

# For testing purposes we can get the GS energy with PySCF, for larger systems this should
# be more efficient than the solver in NetKet.
energy_mo, fcivec_mo = fci.direct_spin1.FCI().kernel(h1, h2, norb, mol.nelectron)

# Note: energy_mo does NOT include the nuclear repulsion energy, the total GS energy is given by
nuc_en = mol.energy_nuc()
gs_energy = energy_mo + nuc_en


"""
This is the qGPSKet/NetKet bit of the calculation.
The key elements different from standard netket calculations are:
1.) Setting up a fermionic discrete Hilbert space:
    In this Hilbert space configurations are represented as a list of L 8-bit integers
    of which only two bits are used. The first (least significant) bit of each int
    encodes whether the alpha/spin-up channel is occupied at the particular site and the
    second bit encodes whether the beta/spin-down channel is occupied. We might change
    this representation at one point but currently this is the best trade-off between
    memory efficiency and convenience.
2.) Setting up the ab-initio Hamiltonian:
    This requires the defined Hilbert space as well as the one and two electron integrals
    as generated above. For a detailed description of the Hamiltonian definition see
    [Neuscamman (2013), https://doi.org/10.1063/1.4829835].
3.) A sampler to generate configurations:
    Non-autoregressive ansatzes require custom transition rules for the Metropolis-Hastings
    algorithm to generate proposals. Currently only a hopping transition is implemented
    for which a randomly selected electron hops from one site to another (thus always
    conserving the total magnetization and electron number of the initial config).
    Two different versions of this hopping sampler are currently available,
    the MetropolisHopping class which follows the default NetKet design, as well as the
    MetropolisFastHopping class which includes the fast update mechanism which can be
    used with the qGPS ansatz.
    More samplers should probably be implemented in the future.
    For the autoregressive ansatz, the direct sampler can be used for the
    Fermionic systems but needs to be amended to take electron number and magnetization
    conservation into account (if this is wanted).
    TODO: this needs to be implemented and checked.
"""

# Set up Hilbert space
hi = FermionicDiscreteHilbert(norb, n_elec=(nelec//2,nelec//2))

# Set up ab-initio Hamiltonian
ha = AbInitioHamiltonian(hi, h1, h2)


# If we want, we can compare the exact energies given by the PySCF and the NetKet solver
# e_mo_nk = nk.exact.lanczos_ed(ha)[0]
# assert(np.allclose(e_mo_nk, energy_mo))


# Use Metropolis-Hastings sampler with hopping rule (including fast updates for qGPS)
sa = MetropolisFastHopping(hi, n_chains_per_rank=1)

# Define the model and the variational state
model = qGPS(hi, 10, dtype=jnp.complex128)
vs = nk.vqs.MCState(sa, model, n_samples=1000)


# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.02)
qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
sr = nk.optimizer.SR(qgt=qgt)

# Variational Monte Carlo driver
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Run optimization
for it in gs.iter(1000,1):
    en = gs.energy.mean + nuc_en
    print("Iteration: {}, Energy: {}, Rel. energy_error: {}".format(it, en, abs((gs_energy - en)/gs_energy)), flush=True)

