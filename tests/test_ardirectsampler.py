import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from mpi4py import MPI
from qGPSKet.models import ARqGPS
from qGPSKet.sampler import ARDirectSampler


# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Model variables
key = jax.random.PRNGKey(np.random.randint(0, 100))
L = 8
M = 2
dtype = jnp.complex128
batch_size = 20

# Compute samples per rank
if batch_size % n_nodes != 0:
    raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
samples_per_rank = batch_size // n_nodes

# Setup
g = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
ha = nk.operator.Heisenberg(hilbert=hi, graph=g, sign_rule=False)
to_indices = lambda x: jnp.asarray((x+hi.local_size-1)/hi.local_size, jnp.int8)
arqgps = ARqGPS(
    hi, M,
    dtype=dtype,
    to_indices=to_indices
)
symmetries = g.automorphisms().to_array().T
apply_symmetries = lambda x: jnp.take(x, symmetries, axis=-1)
arqgps_symm = ARqGPS(
    hi, M,
    dtype=dtype,
    to_indices=to_indices,
    apply_symmetries=apply_symmetries
)

# Test #1
# Shape of sample should be (1, samples_per_rank, L)
sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
vs = nk.vqs.MCState(sa, arqgps, n_samples=batch_size)
samples = vs.sample()
if rank == 0:
    print("Without symmetries:")
    print(f"- sampler.n_chains_per_rank = {sa.n_chains_per_rank}")
    print(f"- vqs.n_samples = {vs.n_samples}")
    print(f"- vqs.chain_length = {vs.chain_length}")
    print(f"- samples.shape = {samples.shape}")
np.testing.assert_equal(samples.shape, (1, samples_per_rank, L))

sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
vs = nk.vqs.MCState(sa, arqgps_symm, n_samples=batch_size)
samples = vs.sample()
if rank == 0:
    print("With symmetries:")
    print(f"- sampler.n_chains_per_rank = {sa.n_chains_per_rank}")
    print(f"- vqs.n_samples = {vs.n_samples}")
    print(f"- vqs.chain_length = {vs.chain_length}")
    print(f"- samples.shape = {samples.shape}")
np.testing.assert_equal(samples.shape, (1, samples_per_rank, L))

# Test #2
# When sampling from a constrained Hilbert space,
# autoregressive models should generate samples with 
# same total magnetization
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes, total_sz=0)
arqgps = ARqGPS(
    hi, M,
    dtype=dtype,
    to_indices=to_indices
)
arqgps_symm = ARqGPS(
    hi, M,
    dtype=dtype,
    to_indices=to_indices,
    apply_symmetries=apply_symmetries
)
sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
vs = nk.vqs.MCState(sa, arqgps, n_samples=batch_size)
samples = vs.sample()
print("Without symmetries:")
print(f"- samples:\n{samples}")
np.testing.assert_equal(np.sum(np.squeeze(samples), axis=-1), np.zeros(batch_size))

sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
vs = nk.vqs.MCState(sa, arqgps_symm, n_samples=batch_size)
samples = vs.sample()
print("With symmetries:")
print(f"- samples:\n{samples}")
np.testing.assert_equal(np.sum(np.squeeze(samples), axis=-1), np.zeros(batch_size))

# Test #3
# Changing sample batch size shouldn't be a problem
samples = vs.sample(n_samples=2*batch_size)
print(samples.shape)
np.testing.assert_equal(samples.shape, (2, batch_size, L))
