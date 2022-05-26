import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Tuple
from netket.utils.types import Array, Callable, DType
from qGPSKet.nn import normal
import jax


class SlaterDeterminant(nn.Module):
    n_sites: int
    n_elec: Tuple[int, int]
    dtype: DType = jnp.complex128
    init_fun: Callable = normal()
    symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    out_transformation: Callable = lambda x: jax.scipy.special.logsumexp(x, axis=-1)
    """ The following flag determines whether the S^2 symmetry (with S=0) should be enforced
    by using the same orbitals for up and down spin.
    """
    spin_symmetry_by_structure: bool = False

    @nn.compact
    def __call__(self, x) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
            assert x.shape[-1] == np.sum(self.n_elec)

        # Initialize U matrix, which has block structure in terms of
        # up and down electrons
        U_up = self.param("U_up", self.init_fun, (self.n_sites, self.n_elec[0]), self.dtype)
        if self.spin_symmetry_by_structure:
            U_down = U_up
        else:
            U_down = self.param("U_down", self.init_fun, (self.n_sites, self.n_elec[1]), self.dtype)

        y = self.symmetries(x)

        def evaluate_SD(y_sym):
            # Compute Slater determinant as product of the determinants of the
            # matrices Ũ_up and Ũ_down, which have rows from U_up and U_down
            # corresponding to electron positions in y:
            # SD = det(Ũ_up)det(Ũ_down)
            Ũ_up = jnp.take(U_up, y_sym[:, :self.n_elec[0]], axis=0)
            Ũ_down = jnp.take(U_down, y_sym[:, self.n_elec[0]:], axis=0)
            (s_up, log_det_up) = jnp.linalg.slogdet(Ũ_up)
            (s_down, log_det_down) = jnp.linalg.slogdet(Ũ_down)
            return log_det_up + log_det_down + jnp.log(s_up*s_down+0j)
        value = jax.vmap(evaluate_SD, in_axes=-1, out_axes=-1)(y)

        return self.out_transformation(value)