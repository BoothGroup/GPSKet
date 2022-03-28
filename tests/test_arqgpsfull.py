import jax
import jax.numpy
import netket as nk
from qGPSKet.operator.hamiltonian import get_J1_J2_Hamiltonian
from qGPSKet.models import ARqGPSFull
from qGPSKet.sampler import ARDirectSampler


key = jax.random.PRNGKey(2)
ha = get_J1_J2_Hamiltonian(20)
hi = ha.hilbert
model = ARqGPSFull(hi, 2)
batch_size = 16
sa = ARDirectSampler(hi, n_chains_per_rank=batch_size)
vs = nk.vqs.MCState(sa, model, n_samples=batch_size)
energy = vs.expect(ha)
print(energy)