import jax
import jax.numpy
import netket as nk
from qGPSKet.operator.hamiltonian import get_J1_J2_Hamiltonian
from qGPSKet.models import AutoregressiveqGPS


key = jax.random.PRNGKey(2)
ha = get_J1_J2_Hamiltonian(20)
hi = ha.hilbert
ma = AutoregressiveqGPS(hi, 2)
x = hi.random_state(key, 4)
variables = ma.init(key, x)
log_psi = ma.apply(variables, x)