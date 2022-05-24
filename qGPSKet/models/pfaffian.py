import numpy as np
import jax.numpy as jnp
import jax
from netket.utils.types import Array, DType, Callable
from typing import Tuple
from netket.utils import HashableArray
from jax.nn.initializers import normal
from flax import linen as nn


def get_gauss_leg_elements_Sy(n_grid):
    x, w = np.polynomial.legendre.leggauss(n_grid)
    return (HashableArray(np.arccos(x)), HashableArray(w))

"""
This implements a Pfaffian of a matrix as exemplified on WikiPedia, there are certainly better ways which we should adapt in the future,
the derivation of this on WikiPedia does not explain why the trace identity can be carried over to non-positive matrices but the code seems to work.
This approach is also not the numerically most stable one.
See arxiv: 1102.3440 and the corresponding codebase (pfapack) for better implementations of the Pfaffian.
TODO: Improve!
"""
@jax.custom_jvp
def log_pfaffian(mat):
    n = mat.shape[0]//2
    pauli_y = jnp.array([[0, -1.j], [1.j, 0.]])
    vals = jnp.linalg.eigvals(jnp.dot(jnp.kron(pauli_y, jnp.eye(n)).T, mat))
    return (0.5 * jnp.sum(jnp.log(vals)) + jnp.log(1.j) * (n**2))


@log_pfaffian.defjvp
def log_pfaffian_jvp(primals, tangents):
    derivative = 0.5 * jnp.linalg.inv(primals[0]).T
    return (log_pfaffian(primals[0]), derivative.flatten().dot(tangents[0].flatten()))


"""
This implements a general Pfaffian wavefunction.
The states which are fed into this model are assumed to be in first quantization, i.e. denote the
positions of Ne electrons, where the positions correspond to site and spin (i.e. take values from 1 ... 2 L).
TODO: We should devise (and stick to) a general framework how the states acting on first quantized inputs should be set up.
"""
class PfaffianState(nn.Module):
    n_sites : int
    init_fun: Callable = normal()
    dtype: DType =jnp.complex128
    @nn.compact
    def __call__(self, y) -> Array:
        F = self.param("F", self.init_fun, (2 * self.n_sites, 2 * self.n_sites), self.dtype)
        F_occ = jnp.take(F, y, axis=0)
        take_fun = lambda x0, x1: jnp.take(x0, x1, axis=1)
        F_occ = jax.vmap(take_fun)(F_occ, y)
        F_skew = F_occ - jnp.swapaxes(F_occ, 1, 2)
        return jax.vmap(log_pfaffian)(F_skew)


"""
This implements a symmetrised Pfaffian wavefunction which explicitly builds in Sz conservation
with zero magnetization (see e.g. mVMC paper/doc).
This implementation does not check if the Hilbert used actually space satisfies this constraint
so a little bit of care is required if this state makes sense.
The states which are fed into this model are assumed to be in the same representation as for the general pfaffian state
(i.e. positions take values between 1 and 2 L).
This model can also easily be projected onto an S^2 eigenstate via a Gauss-Legendre quadrature of the integral in the projector.
See e.g. the mVMC paper [https://doi.org/10.1016/j.cpc.2018.08.014] where this is explained in detail
(comment YR: I am pretty convinced that there is a (-) sign missing in Eq. (53) of that manuscript, implementation below should be correct)
TODO: Test the symmetrization more extensively.
TODO: Implement sanity checks for this model.
TODO: Add documentation for symmetrization interface (+maybe find a more general way of defining it)
"""
class ZeroMagnetizationPfaffian(PfaffianState):
    # S2_projection is a tuple where the first element gives the rotation angles for the Sy rotation
    # and the second element are the corresponding characters which should be used
    S2_projection: Tuple[HashableArray, HashableArray] = (HashableArray(np.array([0.])), HashableArray(np.array([1.])))
    out_transformation: Callable = lambda x: jax.scipy.special.logsumexp(x, axis=(-1))
    @nn.compact
    def __call__(self, y) -> Array:
        n_e_half = y.shape[-1]//2
        f = self.param("f", self.init_fun, (self.n_sites, self.n_sites), self.dtype)
        def evaluate_pfaff_rotations(angle):
            F = jnp.block([[-f * jnp.cos(angle/2) * jnp.sin(angle/2), f * jnp.cos(angle/2) * jnp.cos(angle/2)],
                           [-f * jnp.sin(angle/2) * jnp.sin(angle/2), f * jnp.cos(angle/2) * jnp.sin(angle/2)]])
            F_occ = jnp.take(F, y, axis=0)
            take_fun = lambda x0, x1: jnp.take(x0, x1, axis=1)
            F_occ = jax.vmap(take_fun)(F_occ, y)
            F_skew = F_occ - jnp.swapaxes(F_occ, 1, 2)
            return jax.vmap(log_pfaffian)(F_skew)

        vals = jax.vmap(evaluate_pfaff_rotations, out_axes=-1)(jnp.array(self.S2_projection[0]))
        vals += jnp.log(jnp.asarray(self.S2_projection[1]))

        return self.out_transformation(vals)

