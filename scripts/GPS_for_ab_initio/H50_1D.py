import sys

import pickle

from os.path import exists

import numpy as np

import jax.numpy as jnp

from pyscf import scf, gto, ao2mo, lo

import netket as nk
from netket.utils.mpi import (
    MPI_py_comm as _MPI_comm,
    node_number as _rank,
)

import GPSKet.models as qGPS
from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.sampler.fermionic_hopping import MetropolisHopping
from GPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonianOnTheFly
from GPSKet.models import qGPS
from GPSKet.nn.initializers import normal

# Input arguments
M = int(sys.argv[1]) # GPS support dimension
dist = float(sys.argv[2]) # inter-atomic distance (in units of a_0)

n_samples = 10000 # total number of samples (approximate if run in parallel)

L = 50 # number of H atoms

# Construct basis + one- and two-electron integrals with PySCF
mol = gto.Mole()

mol.build(
    atom = [('H', (x, 0., 0.)) for x in dist*np.arange(L)],
    basis = 'sto-6g',
    symmetry = True,
    unit="Bohr"
)

nelec = mol.nelectron
print('Number of electrons: ', nelec)

myhf = scf.RHF(mol)
ehf = myhf.scf()
norb = myhf.mo_coeff.shape[1]
print('Number of molecular orbitals: ', norb)

h1 = np.zeros((norb, norb))
h2 = np.zeros((norb, norb, norb, norb))

if _rank == 0:
    if exists("./basis.npy"):
        loc_coeff = np.load("./basis.npy")
    else:
        loc_coeff = lo.orth_ao(mol, 'meta_lowdin')
        localizer = lo.Boys(mol, loc_coeff)
        localizer.verbose = 4
        localizer.init_guess = None
        loc_coeff = localizer.kernel()
        np.save("basis.npy", loc_coeff)
    ovlp = myhf.get_ovlp()
    # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
    assert(np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)),np.eye(norb)))
    # Find the hamiltonian in the local basis
    hij_local = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
    hijkl_local = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
    h1 = hij_local
    h2 = hijkl_local

_MPI_comm.Bcast(h1)
_MPI_comm.barrier()

h2_slice = np.empty((h2.shape[2],h2.shape[3]))

for i in range(h2.shape[0]):
    for j in range(h2.shape[1]):
        np.copyto(h2_slice, h2[i,j,:,:])
        _MPI_comm.Bcast(h2_slice)
        _MPI_comm.barrier()
        np.copyto(h2[i,j,:,:], h2_slice)

nuc_en = mol.energy_nuc()

# Set up Hilbert space
hi = FermionicDiscreteHilbert(norb, n_elec=(nelec//2,nelec//2))

# Set up ab-initio Hamiltonian
ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)


# Use Metropolis-Hastings sampler with hopping rule
sa = MetropolisHopping(hi, n_sweeps=200, n_chains_per_rank=1)

# Model definition
model = qGPS(hi, M, dtype=jnp.complex128, init_fun=normal(1.e-1), apply_fast_update=True)


# Variational state
vs = nk.vqs.MCState(sa, model, n_samples=n_samples, n_discard_per_chain=100, chunk_size=1)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)
qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
sr = nk.optimizer.SR(qgt=qgt, diag_shift=0.01)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Set up checkpointing
if _rank == 0:
    if exists("./out.txt"):
        vs.parameters = pickle.load(open("parameters.pickle", "rb"))
        out_prev = np.genfromtxt("out.txt", usecols=(0,1,2,3))
        if out_prev.shape[0] > 0:
            best_var_arg = np.argmin(out_prev[:,3])
            best_var = out_prev[best_var_arg, 3]
            count = out_prev.shape[0] - best_var_arg
        else:
            best_var = None
            count = 0
        print("continuing calculation")
    else:
        best_var = None
        pickle.dump(vs.parameters, open("best_pars.pickle", "wb"))
        with open("out.txt", "w") as fl:
            fl.write("")
        count = 0
        print("starting new calculation")
else:
    best_var = None
    count = 0

best_var = _MPI_comm.bcast(best_var, root=0)
count = _MPI_comm.bcast(count, root=0)
vs.parameters = _MPI_comm.bcast(vs.parameters, root=0)

max_count = 100

# Optimization loop
while count < max_count:
    if _rank == 0:
        pickle.dump(vs.parameters, open("parameters.pickle", "wb"))
    dp = gs._forward_and_backward()
    en = gs.energy.mean + nuc_en
    if best_var is None:
        best_var = gs.energy.variance
        count = 0
    else:
        if gs.energy.variance < best_var:
            best_var = gs.energy.variance
            if _rank == 0:
                pickle.dump(vs.parameters, open("best_pars.pickle", "wb"))
            count = 0
        else:
            count += 1
    sampler_acceptance = vs.sampler_state.acceptance
    if count < max_count:
        gs.update_parameters(dp)
    if _rank == 0:
        print(en, gs.energy.variance, sampler_acceptance, gs.energy.R_hat, gs.energy.tau_corr)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}  {}  {}  {}  {}  {}\n".format(np.real(en), np.imag(en), gs.energy.error_of_mean, gs.energy.variance, sampler_acceptance, gs.energy.R_hat, gs.energy.tau_corr, vs.n_samples))
