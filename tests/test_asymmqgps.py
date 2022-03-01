import jax
import jax.numpy as jnp
import numpy as np
from qGPSKet.models import ASymmqGPS
from qGPSKet.hilbert import FermionicDiscreteHilbert


key_in, key_ma = jax.random.split(jax.random.PRNGKey(np.random.randint(0, 100)))
B = 16
N = 10
n_elec = [5, 5]

hi = FermionicDiscreteHilbert(N, n_elec=n_elec)
n_dets = 1
ma = ASymmqGPS(hi, n_dets)

x = hi.random_state(key_in, B)
variables = ma.init(key_ma, x)
log_psi = ma.apply(variables, x)
assert log_psi.shape == (B,)