import netket as nk
from GPSKet.hilbert import ASEPDiscreteHilbert
import jax

import jax.numpy as jnp

@nk.hilbert.random.random_state.dispatch
def random_state(hilb: ASEPDiscreteHilbert, key, batches: int, *, dtype=jnp.uint8):
    shape = (batches, hilb.size)
    out = jax.random.choice(key, jnp.array(hilb.local_states, dtype), shape=shape)
    return out