import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from qGPSKet.models import ASymmqGPS, ASymmqGPSProd
from qGPSKet.hilbert import FermionicDiscreteHilbert


key_in, key_ma = jax.random.split(jax.random.PRNGKey(np.random.randint(0, 100)))
B = 16
N = 10
n_elec = [5, 5]

# Test #1: evaluate single-determinant model without symmetries
hi = FermionicDiscreteHilbert(N, n_elec=n_elec)
n_dets = 1
ma = ASymmqGPS(hi, n_dets)
x = hi.random_state(key_in, B)
variables = ma.init(key_ma, x)
log_psi = ma.apply(variables, x)
assert log_psi.shape == (B,)

# Test #2: evaluate single-determinant model with symmetries
g = nk.graph.Chain(N, pbc=True)
symmetries = g.automorphisms().to_array().T
def apply_symmetries(y):
    return jax.vmap(lambda tau: jnp.take(tau, y), in_axes=-1, out_axes=-1)(symmetries)
ma = ASymmqGPS(
    hi, n_dets,
    apply_symmetries=apply_symmetries
)
log_psi = ma.apply(variables, x)
assert log_psi.shape == (B,)

# Test #3: evaluate multi-determinant model without symmetries
n_dets = 4
ma = ASymmqGPS(hi, n_dets)
variables = ma.init(key_ma, x)
log_psi = ma.apply(variables, x)
assert log_psi.shape == (B,)

# Test #4: evaluate multi-determinant model with symmetries
ma = ASymmqGPS(
    hi, n_dets,
    apply_symmetries=apply_symmetries
)
log_psi = ma.apply(variables, x)
assert log_psi.shape == (B,)

# Test #5: evaluate multi-determinant model with projective symmetrization
ma = ASymmqGPS(
    hi, n_dets,
    apply_symmetries=apply_symmetries,
    symmetrization='projective'
)
assert log_psi.shape == (B,)

# Test #6: evaluate multi-determinant model with product over determinants
n_dets = 3
ma = ASymmqGPSProd(
    hi, n_dets,
    apply_symmetries=apply_symmetries,
)
variables = ma.init(key_ma, x)
log_psi = ma.apply(variables, x)
assert log_psi.shape == (B,)

# Test #7: multi-determinant model with product over determinants should only
# work for odd numbers of determinants
n_dets = 2
ma = ASymmqGPSProd(
    hi, n_dets,
    apply_symmetries=apply_symmetries,
)
try:
    variables = ma.init(key_ma, x)
except Exception as error:
    assert isinstance(error, AssertionError)