from os.path import exists

import numpy as np

import jax
import jax.numpy as jnp

import netket as nk
from netket.utils.types import Array

import GPSKet
import GPSKet.models as qGPS
from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.sampler.fermionic_hopping import MetropolisHopping
from GPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonianOnTheFly
from GPSKet.models import qGPS
from GPSKet.nn.initializers import normal
from GPSKet.models.pfaffian import ZeroMagnetizationPfaffian

from pyscf import scf, gto, ao2mo, lo

from netket.utils.mpi import (
    MPI_py_comm as _MPI_comm,
    node_number as _rank,
)

from flax import linen as nn

import sys

import pickle

# Input arguments
M = int(sys.argv[1])  # support dimension of the GPS
ref_state = int(
    sys.argv[2]
)  # reference state -> 0: no reference state, 1: SD, 2: Spin projected SD, 3: Pfaffian, 4: Spin projected Pfaffian, 5: magnetization breaking SD
basis_type = int(sys.argv[3])  # basis choice -> 0: local, 1: canonical, 2: split
n_samples = int(sys.argv[4])  # total number of samples (approximate if run in parallel)


# Construct basis + one- and two-electron integrals with PySCF
mol = gto.Mole()

mol.build(
    atom=[
        ["H", (0.0, 0.795, -0.454)],
        ["H", (0.0, -0.795, -0.454)],
        ["O", (0.0, 0.0, 0.113)],
    ],
    basis="6-31G",
    unit="Angstrom",
)

nelec = mol.nelectron
print("Number of electrons: ", nelec)

myhf = scf.RHF(mol)
ehf = myhf.scf()
norb = myhf.mo_coeff.shape[1]
print("Number of molecular orbitals: ", norb)

h1 = np.zeros((norb, norb))
h2 = np.zeros((norb, norb, norb, norb))

if _rank == 0:
    if exists("./basis.npy"):
        loc_coeff = np.load("./basis.npy")
    else:
        loc_coeff = myhf.mo_coeff
        if basis_type != 1:
            loc_coeff = lo.orth_ao(
                mol, "meta_lowdin"
            )  # Using "lowdin" might improve the starting guess for a subsequent Boys localization
            if basis_type == 0:
                localizer = lo.Boys(mol, loc_coeff)
                localizer.verbose = 4
                localizer.init_guess = None
                loc_coeff = localizer.kernel()
            if basis_type == 2:
                localizer = lo.Boys(mol, myhf.mo_coeff[:, : nelec // 2])
                localizer.verbose = 4
                loc_coeff_occ = localizer.kernel()
                localizer = lo.Boys(mol, myhf.mo_coeff[:, nelec // 2 :])
                localizer.verbose = 4
                loc_coeff_vrt = localizer.kernel()
                loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)
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

_MPI_comm.Bcast(h1, root=0)

h2_slice = np.empty((h2.shape[2], h2.shape[3]))

for i in range(h2.shape[0]):
    for j in range(h2.shape[1]):
        np.copyto(h2_slice, h2[i, j, :, :])
        _MPI_comm.Bcast(h2_slice, root=0)
        np.copyto(h2[i, j, :, :], h2_slice)

nuc_en = mol.energy_nuc()

# Set up Hilbert space
hi = FermionicDiscreteHilbert(norb, n_elec=(nelec // 2, nelec // 2))

# Set up ab-initio Hamiltonian
ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)

# Use Metropolis-Hastings sampler with hopping rule
sa = MetropolisHopping(hi, n_sweeps=200, n_chains_per_rank=1)

# Model definitions


class PfaffianqGPS(nn.Module):
    Pfaffian: nn.module
    qGPS: GPSKet.models.qGPS
    apply_fast_update: bool = True

    @nn.compact
    def __call__(self, x, cache_intermediates=False, update_sites=None) -> Array:
        if cache_intermediates or (update_sites is not None):
            indices_save = self.variable(
                "intermediates_cache", "samples", lambda: jnp.zeros(0, dtype=x.dtype)
            )
        if update_sites is not None:

            def update_fun(saved_config, update_sites, occs):
                def scan_fun(carry, count):
                    return (carry.at[update_sites[count]].set(occs[count]), None)

                return jax.lax.scan(
                    scan_fun,
                    saved_config,
                    jnp.arange(update_sites.shape[0]),
                    reverse=True,
                )[0]

            full_x = jax.vmap(update_fun, in_axes=(0, 0, 0), out_axes=0)(
                indices_save.value, update_sites, x
            )
        else:
            full_x = x
        if cache_intermediates:
            indices_save.value = full_x
        y = (
            GPSKet.models.slater.occupancies_to_electrons(full_x, hi._n_elec)
            .at[:, nelec // 2 :]
            .add(norb)
        )
        if M == 0:
            return self.Pfaffian(y)
        else:
            return self.Pfaffian(y) + self.qGPS(
                x, cache_intermediates=cache_intermediates, update_sites=update_sites
            )


class SlaterqGPS(nn.Module):
    SD: nn.module
    qGPS: GPSKet.models.qGPS
    apply_fast_update: bool = True

    @nn.compact
    def __call__(self, x, cache_intermediates=False, update_sites=None) -> Array:
        if M == 0:
            return self.SD(
                x, cache_intermediates=cache_intermediates, update_sites=update_sites
            )
        else:
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


def pfaffian_init(key, shape, dtype=jnp.complex128):
    out = jnp.array(np.einsum("in,jn->ij", phi, phi)).astype(dtype)
    # out += jax.nn.initializers.normal(dtype=out.dtype)(key, shape=out.shape, dtype=dtype)
    return out


def slater_init(key, shape, dtype=jnp.complex128):
    out = jnp.array(phi).astype(dtype).reshape((1, norb, nelec // 2))
    # out += jax.nn.initializers.normal(dtype=out.dtype)(key, shape=out.shape, dtype=dtype)
    return out


def full_slater_init(key, shape, dtype=jnp.complex128):
    out = jnp.block(
        [
            [jnp.array(phi).astype(dtype), jnp.zeros(phi.shape, dtype=dtype)],
            [jnp.zeros(phi.shape, dtype=dtype), jnp.array(phi).astype(dtype)],
        ]
    )
    # out += jax.nn.initializers.normal(dtype=out.dtype)(key, shape=out.shape, dtype=dtype)
    return out.reshape((1, 2 * norb, nelec))


qGPS_part = qGPS(
    hi,
    M,
    dtype=jnp.complex128,
    init_fun=normal(sigma=1.0e-1, dtype=jnp.complex128),
    apply_fast_update=True,
)

# 0: no ref_state, 1: SD, 2: Spin projected SD, 3: Pfaffian, 4: Spin projected Pfaffian, 5: magnetization breaking SD
if ref_state == 1:
    inner_SD = GPSKet.models.slater.Slater(
        hi, init_fun=slater_init, dtype=jnp.complex128, apply_fast_update=True
    )
    model = SlaterqGPS(inner_SD, qGPS_part)
elif ref_state == 2:
    inner_SD = GPSKet.models.slater.Slater(
        hi,
        init_fun=slater_init,
        dtype=jnp.complex128,
        S2_projection=GPSKet.models.pfaffian.get_gauss_leg_elements_Sy(3),
        apply_fast_update=True,
    )
    model = SlaterqGPS(inner_SD, qGPS_part)
elif ref_state == 3:
    inner_SD = ZeroMagnetizationPfaffian(
        norb, init_fun=pfaffian_init, dtype=jnp.complex128
    )
    model = PfaffianqGPS(inner_SD, qGPS_part)
elif ref_state == 4:
    inner_SD = ZeroMagnetizationPfaffian(
        norb,
        init_fun=pfaffian_init,
        S2_projection=GPSKet.models.pfaffian.get_gauss_leg_elements_Sy(3),
        dtype=jnp.complex128,
    )
    model = PfaffianqGPS(inner_SD, qGPS_part)
elif ref_state == 5:
    inner_SD = GPSKet.models.slater.Slater(
        hi,
        init_fun=full_slater_init,
        dtype=jnp.complex128,
        fixed_magnetization=False,
        apply_fast_update=True,
    )
    model = SlaterqGPS(inner_SD, qGPS_part)
else:
    model = qGPS_part

# Variational state
vs = nk.vqs.MCState(
    sa, model, n_samples=n_samples, n_discard_per_chain=100, chunk_size=1
)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)
qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
sr = nk.optimizer.SR(qgt=qgt, diag_shift=0.01)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Set up checkpointing
if _rank == 0:
    if exists("./out.txt"):
        vs.parameters = pickle.load(open("parameters.pickle", "rb"))
        out_prev = np.genfromtxt("out.txt", usecols=(0, 1, 2, 3))
        if out_prev.shape[0] > 0:
            best_var_arg = np.argmin(out_prev[:, 3])
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

max_count = 200

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
        print(
            en,
            gs.energy.variance,
            sampler_acceptance,
            gs.energy.R_hat,
            gs.energy.tau_corr,
        )
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
