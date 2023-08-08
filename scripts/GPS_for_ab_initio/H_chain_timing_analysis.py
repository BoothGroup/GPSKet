import sys

import numpy as np

import jax
import jax.numpy as jnp

from pyscf import scf, gto, ao2mo, lo

import netket as nk

from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.sampler.fermionic_hopping import MetropolisHopping
from GPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonianOnTheFly
from GPSKet.operator.hamiltonian.ab_initio_sparse import AbInitioHamiltonianSparse
import time

from flax import linen as nn

# Input arguments
L = int(sys.argv[1])  # linear dimension
dist = float(sys.argv[2])  # inter-atomic distance (in units of a_0)
pruning_threshold = float(sys.argv[3])  # set to 0 for non-sparse implementation

repeats = 10

# Construct basis + one- and two-electron integrals with PySCF
mol = gto.Mole()

mol.build(
    atom=[("H", (x, 0.0, 0.0)) for x in dist * np.arange(L)],
    basis="sto-6g",
    symmetry=True,
    unit="Bohr",
)

nelec = mol.nelectron
print("Number of electrons: ", nelec)

norb = nelec
print("Number of molecular orbitals: ", norb)

loc_coeff = lo.orth_ao(mol, "meta_lowdin")

localizer = lo.Boys(mol, loc_coeff)
localizer.verbose = 4
localizer.init_guess = None
loc_coeff = localizer.kernel()

h1 = np.linalg.multi_dot((loc_coeff.T, scf.hf.get_hcore(mol), loc_coeff))
h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)


# Set up Hilbert space
hi = FermionicDiscreteHilbert(norb, n_elec=(nelec // 2, nelec // 2))


if pruning_threshold != 0.0:
    h1[abs(h1) < pruning_threshold] = 0.0
    h2[abs(h2) < pruning_threshold] = 0.0
    ha = AbInitioHamiltonianSparse(hi, h1, h2)
else:
    ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)

# Use Metropolis-Hastings sampler with hopping rule
sa = MetropolisHopping(hi)


class uniform_state(nn.Module):
    apply_fast_update: bool = True

    @nn.compact
    def __call__(self, x, cache_intermediates=False, update_sites=None):
        self.variable("intermediates_cache", "dummy_var", lambda: None)
        self.param("dummy_param", lambda x: None)
        return jnp.zeros(x.shape[:-1])


# Model definition
model = uniform_state()


# Variational state
vs = nk.vqs.MCState(sa, model, n_samples=100, n_discard_per_chain=100, chunk_size=1)

key = jax.random.PRNGKey(1)

# Compute exp val once to disregard compilation time
samps = hi.random_state(key=key, size=100, dtype=jnp.uint8)
key = jax.random.split(key)[0]
vs._samples = samps
en = vs.expect(ha)

timings = []

for i in range(repeats):
    samps = hi.random_state(key=key, size=100, dtype=jnp.uint8)
    key = jax.random.split(key)[0]
    vs._samples = samps
    time_start = time.time()
    en = vs.expect(ha)
    time_end = time.time()
    timings.append((time_end - time_start) / 100)


with open("evaluation_timing.txt", "w") as fl:
    fl.write("L  Mean time Std time\n")
    fl.write(
        "{}  {}  {}\n".format(L, np.mean(np.array(timings)), np.std(np.array(timings)))
    )
