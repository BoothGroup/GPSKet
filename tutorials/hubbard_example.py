import jax
import jax.numpy as jnp
import netket as nk
from qGPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from qGPSKet.sampler.fermionic_hopping import MetropolisHopping
from qGPSKet.operator.hamiltonian import FermiHubbard
from qGPSKet.models import ASymmqGPS


# Set up Hilbert space
L = 6
n_elec = (3, 3)
hi = FermionicDiscreteHilbert(L, n_elec=n_elec)

# Set up lattice
g = nk.graph.Chain(L, pbc=True)

# Set up Fermi-Hubbard model
U = 8
ha = FermiHubbard(hi, g.edges(), U=U)

# Use Metropolis-Hastings sampler with hopping rule
sa = MetropolisHopping(hi, n_chains_per_rank=1)

# Define the model and the variational state
n_dets = 2
symmetries = g.automorphisms().to_array().T
def apply_symmetries(y):
    return jax.vmap(lambda tau: jnp.take(tau, y), in_axes=-1, out_axes=-1)(symmetries)
model = ASymmqGPS(hi, n_dets, apply_symmetries=apply_symmetries)
vs = nk.vqs.MCState(sa, model, n_samples=300)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.02)
qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
sr = nk.optimizer.SR(qgt=qgt)

# Variational Monte Carlo driver
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Compute exact energy
gs_energy = nk.exact.lanczos_ed(ha)[0]

# Run optimization
for it in gs.iter(300,1):
    en = gs.energy.mean
    print("Iteration: {}, Energy: {}, Rel. energy_error: {}".format(it, en.real, abs((gs_energy - en)/gs_energy)), flush=True)

