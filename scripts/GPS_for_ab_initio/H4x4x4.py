import sys

import pickle

from os.path import exists

import numpy as np

import jax.numpy as jnp

from numba import jit

from flax import linen as nn

from pyscf import scf, gto, ao2mo, lo

import netket as nk
from netket.utils.mpi import (
    MPI_py_comm as _MPI_comm,
    node_number as _rank,
    mpi_sum as _mpi_sum,
)
from netket.utils.types import Array

import GPSKet
import GPSKet.models as qGPS
from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.sampler.fermionic_hopping import MetropolisHopping
from GPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonianOnTheFly
from GPSKet.nn.initializers import normal
from GPSKet.models import qGPS
from GPSKet.operator.fermion import apply_hopping

# Input arguments
N = int(sys.argv[1])  # GPS support dimension
dist = float(sys.argv[2])  # inter-atomic distance (in units of Å)

n_samples = 10000  # total number of samples (approximate if run in parallel)

L = 4  # linear dimension of the cubic crystal of H atoms

# Construct basis + one- and two-electron integrals with PySCF
mol = gto.Mole()

atoms = []

for x in range(L):
    for y in range(L):
        for z in range(L):
            atoms.append(("H", (x * dist, y * dist, z * dist)))

mol.build(atom=atoms, basis="sto-6g", symmetry=True, unit="A")

nelec = mol.nelectron
print("Number of electrons: ", nelec)

myhf = scf.RHF(mol)
ehf = myhf.scf()
norb = myhf.mo_coeff.shape[1]
print("Number of molecular orbitals: ", norb)

mol.max_memory = 1.0e9

h1 = np.zeros((norb, norb))
h2 = np.zeros((norb, norb, norb, norb))

if _rank == 0:
    if exists("./basis.npy"):
        loc_coeff = np.load("./basis.npy")
    else:
        loc_coeff = myhf.mo_coeff
        loc_coeff = lo.orth_ao(mol, "meta_lowdin")
        localizer = lo.Boys(mol, loc_coeff)
        localizer.verbose = 4
        localizer.init_guess = None
        loc_coeff = localizer.kernel()
        np.save("basis.npy", loc_coeff)
    if exists("./h1.npy") and exists("./h2.npy"):
        h1 = np.load("./h1.npy")
        h2 = np.load("./h2.npy")
    else:
        ovlp = myhf.get_ovlp()
        # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
        assert np.allclose(
            np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)), np.eye(norb)
        )
        # Find the hamiltonian in the local basis
        h1 = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
        h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
        np.save("h1.npy", h1)
        np.save("h2.npy", h2)

_MPI_comm.Bcast(h1)
_MPI_comm.barrier()

h2_slice = np.empty((h2.shape[2], h2.shape[3]))

for i in range(h2.shape[0]):
    for j in range(h2.shape[1]):
        np.copyto(h2_slice, h2[i, j, :, :])
        _MPI_comm.Bcast(h2_slice)
        _MPI_comm.barrier()
        np.copyto(h2[i, j, :, :], h2_slice)

nuc_en = mol.energy_nuc()

# Set up Hilbert space
hi = FermionicDiscreteHilbert(norb, n_elec=(nelec // 2, nelec // 2))

# Set up ab-initio Hamiltonian
ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)


# Use Metropolis-Hastings sampler with hopping rule
sa = MetropolisHopping(hi, n_sweeps=200, n_chains_per_rank=50)

# Define qGPS x SD model


class SlaterqGPS(nn.Module):
    SD: GPSKet.models.Slater
    qGPS: GPSKet.models.qGPS
    apply_fast_update: bool = True

    @nn.compact
    def __call__(self, x, cache_intermediates=False, update_sites=None) -> Array:
        return self.SD(
            x, cache_intermediates=cache_intermediates, update_sites=update_sites
        ) + self.qGPS(
            x, cache_intermediates=cache_intermediates, update_sites=update_sites
        )


# Run mean-field calcs for initialization of reference state
eigs, vecs = np.linalg.eigh(h1)

mf = scf.RHF(mol)

mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(hi.size)
mf._eri = ao2mo.restore(8, h2, hi.size)

# Assumes RHF
assert hi._n_elec[0] == hi._n_elec[1]

init_dens = np.dot(vecs[:, : mol.nelectron // 2], vecs[:, : mol.nelectron // 2].T)
mf.kernel(dm0=init_dens)


if not mf.converged:
    mf = scf.newton(mf)
    mf.kernel(mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
assert mf.converged

# store the canonical orbitals in phi
phi = mf.mo_coeff[:, : mol.nelectron // 2]

_MPI_comm.Bcast(phi, root=0)


def slater_init(key, shape, dtype=jnp.complex128):
    return jnp.array(phi).astype(dtype).reshape((1, norb, nelec // 2))


inner_SD = GPSKet.models.Slater(hi, init_fun=slater_init, dtype=jnp.complex128)

qGPS_part = qGPS(
    hi,
    N,
    dtype=jnp.complex128,
    init_fun=normal(sigma=1.0e-3, dtype=jnp.complex128),
    apply_fast_update=True,
)

model = SlaterqGPS(inner_SD, qGPS_part)

# Variational state
vs = nk.vqs.MCState(sa, model, n_samples=n_samples, n_discard_per_chain=100)


# Set up computation of 1RDM expectation values
@jit(nopython=True)
def get_conn_1RDM(x):
    x_prime = np.empty((x.shape[0], 2, norb, norb, x.shape[1]), dtype=np.uint8)
    mels = np.empty((x.shape[0], 2, norb, norb), dtype=np.complex128)
    for batch_id in range(x.shape[0]):
        is_occ_up = (x[batch_id] & 1).astype(np.bool8)
        is_occ_down = (x[batch_id] & 2).astype(np.bool8)
        up_count = np.cumsum(is_occ_up)
        down_count = np.cumsum(is_occ_down)
        for i in range(norb):
            for j in range(norb):
                x_prime[batch_id, 0, i, j, :] = x[batch_id, :]
                x_prime[batch_id, 1, i, j, :] = x[batch_id, :]
                mels[batch_id, 0, i, j] = apply_hopping(
                    i, j, x_prime[batch_id, 0, i, j, :], 1, cummulative_count=up_count
                )
                mels[batch_id, 1, i, j] = apply_hopping(
                    i, j, x_prime[batch_id, 1, i, j, :], 2, cummulative_count=down_count
                )
    return x_prime, mels


def get_1RDM(state):
    x = state.samples.reshape((-1, norb))
    log_vals = jnp.expand_dims(state.log_value(x), (1, 2, 3))
    x_primes, mels = get_conn_1RDM(np.asarray(x, dtype=np.uint8))
    all_log_vals_conn = []
    for i in range(x_primes.shape[0]):
        conf = x_primes[i, :, :, :]
        log_vals_conn = state.log_value(jnp.array(conf.reshape((-1, norb)))).reshape(
            (2, norb, norb)
        )
        all_log_vals_conn.append(log_vals_conn)
    all_log_vals_conn = jnp.array(all_log_vals_conn)
    total_samples = _mpi_sum(log_vals.shape[0])
    return (
        _mpi_sum(
            np.array(jnp.sum(jnp.exp(all_log_vals_conn - log_vals) * mels, axis=0))
        )
        / total_samples
    )


# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.075)
qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
sr = nk.optimizer.SR(qgt=qgt, diag_shift=0.01)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

min_global_count = 1000

# Set up checkpointing
if _rank == 0:
    if exists("./out.txt"):
        vs.parameters = pickle.load(open("parameters.pickle", "rb"))
        out_prev = np.genfromtxt("out.txt", usecols=(0, 1, 2, 3))
        if out_prev.shape[0] > 0:
            best_var_arg = np.argmin(out_prev[:, 3])
            best_var = out_prev[best_var_arg, 3]
            if best_var_arg < min_global_count:
                count = out_prev.shape[0] - best_var_arg
            else:
                best_var = None
                count = 0
            global_count = out_prev.shape[0]
        else:
            best_var = None
            count = 0
            global_count = 0
        print("continuing calculation")
    else:
        best_var = None
        pickle.dump(vs.parameters, open("best_pars.pickle", "wb"))
        with open("out.txt", "w") as fl:
            fl.write("")
        count = 0
        global_count = 0
        print("starting new calculation")
else:
    best_var = None
    count = 0
    global_count = 0

best_var = _MPI_comm.bcast(best_var, root=0)
count = _MPI_comm.bcast(count, root=0)
global_count = _MPI_comm.bcast(global_count, root=0)
vs.parameters = _MPI_comm.bcast(vs.parameters, root=0)

max_count = 250

# Optimization loop
while count < max_count:
    if _rank == 0:
        pickle.dump(vs.parameters, open("parameters.pickle", "wb"))
    dp = gs._forward_and_backward()
    oneRDM = np.array(get_1RDM(vs))
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
    if global_count < min_global_count:
        best_var = None
    sampler_acceptance = vs.sampler_state.acceptance
    if count < max_count:
        gs.update_parameters(dp)
    if _rank == 0:
        print(
            en,
            gs.energy.variance,
            sampler_acceptance,
            gs.energy.R_hat,
            gs.energy.tau_corr,
        )
        np.save("oneBRDM_{}.npy".format(global_count), oneRDM)
        with open("out.txt", "a") as fl:
            fl.write(
                "{}  {}  {}  {}  {}  {}  {}  {}\n".format(
                    np.real(en),
                    np.imag(en),
                    gs.energy.error_of_mean,
                    gs.energy.variance,
                    sampler_acceptance,
                    gs.energy.R_hat,
                    gs.energy.tau_corr,
                    vs.n_samples,
                )
            )
    global_count += 1
