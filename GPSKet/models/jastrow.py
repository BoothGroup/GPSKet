import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from jax.nn.initializers import normal
from netket.utils.types import Array, DType, NNInitFunc, Callable
from ..hilbert import FermionicDiscreteHilbert


def up_down_occupancies(x):
    """
    Returns spin-up and -down occupancies at each site

    Args:
        x : an array of input configurations in 2nd quantization of shape (B, L)

    Returns:
        x_up, x_dn : spin-up and -down occupancies (B, L)
    """
    x = jnp.asarray(x, dtype=jnp.int32)
    x_up = jnp.asarray(x & 1, jnp.int32)
    x_dn = jnp.asarray((x & 2) / 2, jnp.int32)
    return x_up, x_dn


class Jastrow(nn.Module):
    """
    Implements a Jastrow wavefunction
    """

    hilbert: FermionicDiscreteHilbert
    """The Hilbert space of the wavefunction model"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    init_fun: NNInitFunc = normal()
    """Initializer for the variational parameters"""
    apply_symmetries: Callable = lambda inputs: jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""

    @nn.compact
    def __call__(self, x) -> Array:
        nsites = x.shape[-1]
        kernel = self.param("kernel", self.init_fun, (nsites, nsites), self.dtype)
        kernel = kernel + kernel.T
        x = self.apply_symmetries(x)  # (B, T)
        x_up, x_dn = up_down_occupancies(x)
        kernel, x_up, x_dn = promote_dtype(kernel, x_up, x_dn, dtype=None)
        y = jax.vmap(
            lambda u, d: jnp.einsum("...i,ij,...j", (1 - u), kernel, (1 - d)),
            in_axes=(-1, -1),
            out_axes=-1,
        )(x_up, x_dn)
        return jnp.sum(y, axis=-1)  # (B,)
