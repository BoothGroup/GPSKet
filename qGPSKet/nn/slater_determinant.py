import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Tuple
from netket.utils.types import Array, Callable
from qGPSKet.nn import normal


class SlaterDeterminant(nn.Module):
    n_sites: int
    n_elec: Tuple[int, int]
    init_fun: Callable=normal(sigma=0.1, dtype=jnp.float64)

    @nn.compact
    def __call__(self, x) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
            assert x.shape[-1] == np.sum(self.n_elec)

        # Initialize U matrix, which has block structure in terms of
        # up and down electrons
        U_up = self.param("U_up", self.init_fun, (self.n_sites, self.n_elec[0]))
        U_down = self.param("U_down", self.init_fun, (self.n_sites, self.n_elec[1]))

        # Compute Slater determinant as product of the determinants of the
        # matrices Ũ_up and Ũ_down, which have rows from U_up and U_down 
        # corresponding to electron positions in y:
        # SD = det(Ũ_up)det(Ũ_down)
        Ũ_up = jnp.take(U_up, x[:, :self.n_elec[0]], axis=0)
        Ũ_down = jnp.take(U_down, x[:, self.n_elec[0]:], axis=0)
        (s_up, log_det_up) = jnp.linalg.slogdet(Ũ_up)
        (s_down, log_det_down) = jnp.linalg.slogdet(Ũ_down)
        log_sd = log_det_up + log_det_down + jnp.log(s_up*s_down+0j)

        return log_sd