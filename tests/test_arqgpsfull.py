import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from tqdm import tqdm
from jax.scipy.special import logsumexp
from GPSKet.models import ARqGPSFull

key_in, key_ma = jax.random.split(jax.random.PRNGKey(np.random.randint(0, 100)))
L = 20
M = 2
dtype = jnp.complex128
batch_size = 16

g = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(1/2, N=g.n_nodes)

arqgps = ARqGPSFull(
    hi, M,
    dtype=dtype
)

# Test #1
# Log-amplitude for a configuration x should be equal to:
# \log(\psi(x)) = \sum_{i=1}^L(\sum_{n=1}^M\epsilon_{x_i,n,i,i}(\prod_{j=1}^{i-1}\epsilon_{x_j,n,j,i}))-\log(\sum_x'|\exp(\sum_{n=1}^M\epsilon_{x',n,i,i}(\prod_{j=1}^{i-1}\epsilon_{x_j,n,j,i}))|^2)/2
inputs = hi.random_state(key_in, batch_size)
variables = arqgps.init(key_ma, inputs)
log_psi_test = arqgps.apply(variables, inputs)
log_psi_true = np.zeros(batch_size, dtype=dtype)
epsilon = np.asarray(variables.unfreeze()['params']['epsilon'], dtype=dtype)
for k in tqdm(range(batch_size), desc="Test #1"):
    log_psi = 0+0*1j
    x_k = hi.states_to_local_indices(inputs[k])
    for i in range(L):
        log_psi_cond = np.zeros(hi.local_size, dtype=dtype)
        for n in range(M):
            var_prod = 1+0*1j
            for j in range(i):
                var_prod *= epsilon[x_k[j], n, j, i]
            log_psi_cond += epsilon[:, n, i, i]*var_prod
        bond_sum = log_psi_cond[x_k[i]]
        normalization = 0.5*logsumexp(2*log_psi_cond.real)
        log_psi += bond_sum-normalization
    log_psi_true[k] = log_psi

np.testing.assert_allclose(log_psi_test, log_psi_true)

symmetries = g.automorphisms().to_array().T
apply_symmetries = lambda x: jnp.take(x, symmetries, axis=-1)
arqgps_symm = ARqGPSFull(
    hi, M,
    dtype=dtype,
    apply_symmetries=apply_symmetries
)

# Test #2
# Symmetrized amplitudes should be equal to average of
# amplitudes from non-symmetric model over
# symmetry transformed input configurations
log_psi_symm = arqgps_symm.apply(variables, inputs)
n_symm = symmetries.shape[-1]
log_psi = jnp.zeros((batch_size, n_symm), dtype=jnp.complex128)
for t in tqdm(range(n_symm), desc="Test #2"):
    inputs_t = jnp.take_along_axis(inputs, jnp.tile(symmetries[:, t], (batch_size, 1)), 1)
    y = arqgps.apply(variables, inputs_t)
    log_psi = log_psi.at[:, t].set(y)
log_psi_real = 0.5*logsumexp(2*log_psi.real, axis=-1, b=1/n_symm)
log_psi_imag = logsumexp(1j*log_psi.imag, axis=-1).imag
log_psi = log_psi_real+1j*log_psi_imag

np.testing.assert_allclose(log_psi_symm.real, log_psi.real)

# Test #3
# Probabilities from .conditionals should match those over
# L sites from ._conditional for constrained and unconstrained Hilbert space
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
symmetries = g.automorphisms().to_array().T
apply_symmetries = lambda x: jnp.take(x, symmetries, axis=-1)
arqgps_symm = ARqGPSFull(
    hi, M,
    dtype=dtype,
    apply_symmetries=apply_symmetries
)
inputs = hi.random_state(key_in, batch_size)
variables = arqgps_symm.init(key_ma, inputs)
psi_cond_test = jnp.zeros((batch_size, L, 2), dtype=jnp.float64)
for l in tqdm(range(L), desc="Test #3.1, unconstrained"):
    p, variables = arqgps_symm.apply(variables, inputs, l, method=ARqGPSFull._conditional, mutable=True)
    psi_cond_test = psi_cond_test.at[:, l, :].set(p)
psi_cond = arqgps_symm.apply(variables, inputs, method=ARqGPSFull.conditionals)

np.testing.assert_allclose(psi_cond_test, psi_cond)

hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes, total_sz=0)
symmetries = g.automorphisms().to_array().T
apply_symmetries = lambda x: jnp.take(x, symmetries, axis=-1)
arqgps_symm = ARqGPSFull(
    hi, M,
    dtype=dtype,
    apply_symmetries=apply_symmetries
)
inputs = hi.random_state(key_in, batch_size)
variables = arqgps_symm.init(key_ma, inputs)
psi_cond_test = jnp.zeros((batch_size, L, 2), dtype=jnp.float64)
for l in tqdm(range(L), desc="Test #3.2, constrained"):
    p, variables = arqgps_symm.apply(variables, inputs, l, method=ARqGPSFull._conditional, mutable=True)
    psi_cond_test = psi_cond_test.at[:, l, :].set(p)
psi_cond = arqgps_symm.apply(variables, inputs, method=ARqGPSFull.conditionals)

np.testing.assert_allclose(psi_cond_test, psi_cond)