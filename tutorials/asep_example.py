import jax
import jax.numpy as jnp
import netket as nk
from scipy.sparse.linalg import eigsh
from qGPSKet.hilbert import ASEPDiscreteHilbert
from qGPSKet.operator.hamiltonian import AsymmetricSimpleExclusionProcess
from qGPSKet.models import qGPS, ARqGPS
from qGPSKet.sampler import ARDirectSampler
from qGPSKet.nn import normal


# Set up Hilbert space
L = 10
hi = ASEPDiscreteHilbert(L)

# Set up lattice
g = nk.graph.Chain(L, pbc=True)

# Set up ASEP model
lambd = 0.0
alpha = beta = gamma = delta = 0.5
p = q = 0.5
ha = AsymmetricSimpleExclusionProcess(hi, L, lambd, alpha, beta, gamma, delta, p, q)

# Use Metropolis-Hastings sampler with hopping rule
# sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=1, dtype=jnp.uint8)
sa = ARDirectSampler(hi, n_chains_per_rank=300, dtype=jnp.uint8)

# Define the model and the variational state
M = 2
dtype = jnp.float64
init_fun = normal(sigma=0.01, dtype=dtype)
# model = qGPS(hi, M, dtype=dtype)
model = ARqGPS(hi, M, dtype=dtype)
vs = nk.vqs.MCState(sa, model, n_samples=300)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=-0.01)
qgt = nk.optimizer.qgt.QGTJacobianDense(mode="real")
sr = nk.optimizer.SR(qgt=qgt, diag_shift=0.01)

# Variational Monte Carlo driver
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Compute exact energy
gs_energy = eigsh(ha.to_linear_operator(), which="LA", k=1, return_eigenvectors=False)

# Run optimization
for it in gs.iter(300,1):
    en = gs.energy.mean
    print("Iteration: {}, Energy: {}, Abs. energy_error: {}".format(it, en.real, abs(gs_energy - en)), flush=True)

