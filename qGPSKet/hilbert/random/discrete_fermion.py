import numpy as np
import netket as nk
from qGPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
import jax

import jax.numpy as jnp

@nk.hilbert.random.random_state.dispatch
def random_state(hilb: FermionicDiscreteHilbert, key, batches: int, *, out=None, dtype=jnp.uint8):
    shape = (batches, hilb._size)

    if hilb._n_elec is None:
        out = jax.random.choice(key, jnp.array(hilb.local_states), shape=shape)
    else:
        def scan_fun(key, val):
            key, subkey = jax.random.split(key)
            up_pos = jax.random.choice(key, hilb.size, shape=(hilb._n_elec[0],), replace=False)
            down_pos = jax.random.choice(subkey, hilb.size, shape=(hilb._n_elec[1],), replace=False)
            out = jnp.zeros(hilb.size, dtype=dtype)
            out = out.at[up_pos].add(1)
            out = out.at[down_pos].add(2)
            return key, out
        out = jax.lax.scan(scan_fun, key, None, length=batches)[1]
    return out