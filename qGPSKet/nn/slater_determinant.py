import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Tuple, Optional
from netket.utils.types import Array, Callable, DType
from netket.utils import HashableArray
from qGPSKet.nn import normal
import jax

"""
This defines a single Slater determinant with a fixed total magnetization.
"""
class SlaterDeterminant(nn.Module):
    n_sites: int
    n_elec: Tuple[int, int]
    dtype: DType = jnp.complex128
    init_fun: Callable = normal()
    symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    out_transformation: Callable = lambda x: jax.scipy.special.logsumexp(x, axis=(-1, -2))
    """ The following flag determines whether the S^2 symmetry (with S=0) should be enforced
    by using the same orbitals for up and down spin.
    """
    spin_symmetry_by_structure: bool = False
    """ This (optional) tuple specifies the angles and characters for spin
    rotations which should be used for the S^2 projection. Only sensible if above flag
    is false.
    """
    S2_projection: Optional[Tuple[HashableArray, HashableArray]] = None

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

        if self.S2_projection is not None:
            full_U = jnp.block([[U_up, jnp.zeros(U_up.shape, dtype=U_up.dtype)],
                                [jnp.zeros(U_down.shape, dtype=U_down.dtype), U_down]])

        y = self.symmetries(x)

        def evaluate_SD(y_sym):
            # Compute Slater determinant as product of the determinants of the
            # matrices Ũ_up and Ũ_down, which have rows from U_up and U_down
            # corresponding to electron positions in y:
            # SD = det(Ũ_up)det(Ũ_down) which only works if no spin rotation is applied
            if self.S2_projection is None:
                Ũ_up = jnp.take(U_up, y_sym[:, :self.n_elec[0]], axis=0)
                Ũ_down = jnp.take(U_down, y_sym[:, self.n_elec[0]:], axis=0)
                (s_up, log_det_up) = jnp.linalg.slogdet(Ũ_up)
                (s_down, log_det_down) = jnp.linalg.slogdet(Ũ_down)
                return jnp.expand_dims(log_det_up + log_det_down + jnp.log(s_up*s_down+0j), -1)
            else:
                def evaluate_spin_rotation(angle):
                    U_rotated = jnp.block([[U_up * jnp.cos(angle/2), -U_down * jnp.sin(angle/2)],
                                           [U_up * jnp.sin(angle/2), U_down * jnp.cos(angle/2)]])
                    take_indices = y_sym.at[:, self.n_elec[0]:].add(self.n_sites)
                    U_submat = jnp.take(U_rotated, take_indices, axis=0)
                    (s_det, log_det) = jnp.linalg.slogdet(U_submat)
                    return log_det + jnp.log(s_det+0j)
                return jax.vmap(evaluate_spin_rotation, in_axes=0, out_axes=-1)(jnp.array(self.S2_projection[0])) + jnp.log(jnp.asarray(self.S2_projection[1]))

        value = jax.vmap(evaluate_SD, in_axes=-1, out_axes=-1)(y)

        return self.out_transformation(value)