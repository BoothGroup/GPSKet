import jax
import jax.numpy as jnp
import netket as nk
from scipy.sparse.linalg import eigs
from netket.hilbert import Qubit
from GPSKet.operator.hamiltonian import AsymmetricSimpleExclusionProcess
from GPSKet.models import qGPS, ARqGPS
from GPSKet.sampler import ARDirectSampler
from GPSKet.nn import normal


# Set up Hilbert space
L = 10
hi = Qubit(L)

# Set up ASEP model
lambd = 0.2
alpha = beta = gamma = delta = 0.5
p = q = 0.5
ha = AsymmetricSimpleExclusionProcess(hi, lambd, alpha, beta, gamma, delta, p, q)

# Use Metropolis-Hastings sampler with hopping rule
# sa = nk.sampler.MetropolisLocal(hi, n_chains_per_rank=1)
sa = ARDirectSampler(hi, n_chains_per_rank=300)

# Define the model and the variational state
M = 10
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
gs_energy = eigs(ha.to_linear_operator(), which="LR", k=1, return_eigenvectors=False)

# Run optimization
for it in gs.iter(300, 1):
    en = gs.energy.mean
    print(
        "Iteration: {}, Energy: {}, Abs. energy_error: {}".format(
            it, en.real, abs(gs_energy - en)
        ),
        flush=True,
    )
