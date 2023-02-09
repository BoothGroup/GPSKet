import jax.numpy as jnp
import netket as nk
from mpi4py import MPI
from GPSKet.operator.hamiltonian import get_J1_J2_Hamiltonian
from GPSKet.models import qGPS, ARqGPS, get_sym_transformation_spin
from GPSKet.sampler import ARDirectSampler
from GPSKet.sampler.metropolis_fast import MetropolisFastExchange

# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Parameters
L = 20
M = 2
ansatz = 'qgps'
dtype = jnp.complex128
sampler = 'metropolis-exchange'
batch_size = 100
n_discard = 100
learning_rate = 0.01
diag_shift = 0.01
n_iters = 100

# Compute samples per rank
if batch_size % n_nodes != 0:
    raise ValueError("Define a batch size that is a multiple of the number of MPI ranks")
samples_per_rank = batch_size // n_nodes

# Get Hamiltonian, Hilbert space and graph
# The on the fly calculation of the local energy is only faster for the
# qGPS model where fast updating can be performed.
ha = get_J1_J2_Hamiltonian(L, on_the_fly_en=(ansatz == "qgps"))
hi = ha.hilbert
g = ha.graph

# Ansatz model
if ansatz == 'qgps':
    model = qGPS(hi, M, dtype=dtype, syms=get_sym_transformation_spin(g))
elif ansatz == 'arqgps':
    apply_symmetries, _ = get_sym_transformation_spin(g, spin_flip=False)
    model = ARqGPS(hi, M, dtype=dtype, apply_symmetries=apply_symmetries)

# Sampler
if sampler == 'metropolis-exchange' and ansatz == "qgps":
    sa = MetropolisFastExchange(hi, graph=g, n_chains=1)
elif sampler == 'metropolis-exchange':
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=1)
elif sampler == 'ar-direct':
    sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)

# Variational quantum state
if sa.is_exact:
    vs = nk.vqs.MCState(sa, model, n_samples=batch_size)
else:
    vs = nk.vqs.MCState(sa, model, n_samples=batch_size, n_discard_per_chain=n_discard)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=learning_rate)
qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=(ansatz == 'qgps'))
sr = nk.optimizer.SR(qgt=qgt, diag_shift=diag_shift)

# Variational Monte Carlo driver
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Run optimization
for it in gs.iter(n_iters,1):
    print(it,gs.energy, flush=True)
