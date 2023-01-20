import os
import optax
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import GPSKet as qk
from functools import partial
from GPSKet.datasets.h2o import BasisType


def count_spins_fermionic(spins):
    zeros = jnp.zeros(spins.shape[0])
    up_spins = spins&1
    down_spins = (spins&2)/2
    return jnp.stack([zeros, up_spins, down_spins, zeros], axis=-1).astype(jnp.int32)

@partial(jax.vmap, in_axes=(0, None, None))
def renormalize_log_psi_fermionic(n_spins, hilbert, index):
    log_psi = jnp.zeros(hilbert.local_size)
    diff = jnp.array(hilbert._n_elec, jnp.int32)-n_spins[1:3]
    log_psi = jax.lax.cond(
        diff[0] == 0,
        lambda log_psi: log_psi.at[1].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        diff[1] == 0,
        lambda log_psi: log_psi.at[2].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        (diff == 0).any(),
        lambda log_psi: log_psi.at[3].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        (diff >= (hilbert.size-index)).any(),
        lambda log_psi: log_psi.at[0].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        (diff[0] >= (hilbert.size-index)).any(),
        lambda log_psi: log_psi.at[2].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        (diff[1] >= (hilbert.size-index)).any(),
        lambda log_psi: log_psi.at[1].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    return log_psi

# Get dataset
dataset = qk.datasets.get_h2o_dataset(basis_type=BasisType.LOCAL, select_largest=500)

# Setup Hilbert space and Hamiltonian
_DATA = "/tmp/GPSKet_data/"
h1 = np.load(os.path.join(_DATA, f"h1_{BasisType.LOCAL}.npy"))
h2 = np.load(os.path.join(_DATA, f"h2_{BasisType.LOCAL}.npy"))
norb = dataset[0].shape[1]
nelec = 10
hi = qk.hilbert.FermionicDiscreteHilbert(N=norb, n_elec=(nelec//2,nelec//2))
ha = qk.operator.hamiltonian.AbInitioHamiltonianOnTheFly(hi, h1 , h2)

# Setup model, sampler, and variational state
to_indices = lambda x: x.astype(jnp.uint8)
model = qk.models.ARqGPS(hi, 2, dtype=jnp.complex128, to_indices=to_indices, count_spins=count_spins_fermionic, renormalize_log_psi=renormalize_log_psi_fermionic)
sa = qk.sampler.ARDirectSampler(hi)
vs = nk.vqs.MCState(sa, model, n_samples=1000)

# Initialize optimizer
op = optax.experimental.split_real_and_imaginary(optax.adam(learning_rate=0.01))

# Setup state fitting driver
driver = qk.driver.ARStateFitting(dataset, ha, op, variational_state=vs, mini_batch_size=32)

# Run fitting
n_iters = 1000
for it in driver.iter(n_iters, 1):
    print(it, driver.loss, flush=True)