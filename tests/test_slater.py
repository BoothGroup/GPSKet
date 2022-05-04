import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from qGPSKet.models import Slater, occupancies_to_electrons
from qGPSKet.hilbert import FermionicDiscreteHilbert
from tqdm import tqdm


key_in, key_ma = jax.random.split(jax.random.PRNGKey(np.random.randint(0, 100)))
B = 16
L = 10
n_elec = [5, 5]
dtype = jnp.complex128
n_dets = 3
g = nk.graph.Chain(L, pbc=True)
symmetries = g.automorphisms().to_array().T
n_syms = symmetries.shape[-1]
def apply_symmetries(y):
    return jax.vmap(lambda tau: jnp.take(tau, y), in_axes=-1, out_axes=-1)(symmetries)
hi = FermionicDiscreteHilbert(L, n_elec=n_elec)
x = hi.random_state(key_in, B)

# Test #1: evaluate Slater
# Amplitudes should be equal to:
# Ψ(x) = ∑_τ∑_k det(ɸ_k^↑(τx))det(ɸ_k^↓(τx))
ma = Slater(hi, n_dets, dtype=dtype, apply_symmetries=apply_symmetries)
variables = ma.init(key_ma, x)
log_psi_test = ma.apply(variables, x)
log_psi_true = np.zeros(B, dtype)
params = variables.unfreeze()['params']
orbitals_up = params['orbitals_up']
orbitals_down = params['orbitals_down']
y = occupancies_to_electrons(x.astype(jnp.int32), n_elec)
y_t = apply_symmetries(y)
for i in tqdm(range(B), desc="Test #1"):
    sum_over_syms = 0.0+0j
    for j in range(n_syms):
        y_up = y_t[i,:n_elec[0],j]
        y_down = y_t[i,n_elec[0]:,j]
        sum_over_dets = 0.0+0j
        for k in range(n_dets):
            phi_up = orbitals_up[k, y_up, :]
            phi_down = orbitals_down[k, y_down, :]
            (s_up, log_det_up) = jnp.linalg.slogdet(phi_up)
            (s_down, log_det_down) = jnp.linalg.slogdet(phi_down)
            log_sd = log_det_up + log_det_down + jnp.log(s_up*s_down+0j)
            sum_over_dets += np.exp(log_sd)
        sum_over_syms += sum_over_dets
    if np.issubdtype(dtype, np.complexfloating):
        log_psi_true[i] = np.log(sum_over_syms)
    else:
        log_psi_true[i] = np.log(sum_over_syms).real
np.testing.assert_allclose(log_psi_test, log_psi_true)