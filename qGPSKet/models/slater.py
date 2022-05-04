import jax
import jax.numpy as jnp
from flax import linen as nn
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils.types import Array, Callable, DType, NNInitFunc
from .asymm_qGPS import occupancies_to_electrons, _evaluate_determinants


# Dimensions:
# - B = batch size
# - L = number of sites
# - N = total number of electrons
# - N_up = number of spin-up electrons
# - N_down = number of spin-down electrons
# - M = number of determinants
# - T = number of symmetries

class Slater(nn.Module):
    """
    Implements a linear combination of Slater determinants, with support for symmetries
    """

    hilbert: HomogeneousHilbert
    """Hilbert space"""
    n_determinants: int = 1
    """Number of determinants"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    init_fun : NNInitFunc = jax.nn.initializers.orthogonal()
    """Initializer for the variational parameters"""
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""


    def setup(self):
        L = self.hilbert.size
        self.n_elec = self.hilbert._n_elec
        self.orbitals_up = self.param("orbitals_up", self.init_fun, (self.n_determinants, L, self.n_elec[0]), self.dtype)
        self.orbitals_down = self.param("orbitals_down", self.init_fun, (self.n_determinants, L, self.n_elec[1]), self.dtype)

    def __call__(self, x) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        x = jnp.asarray(x, jnp.int32) # (B, L)

        # Convert input config representation
        y = occupancies_to_electrons(x, self.hilbert._n_elec) # (B, N)

        # Apply symmetry transformations
        y = self.apply_symmetries(y) # (B, N, T)

        # Evaluate Slater determinants
        log_sd_t = jax.vmap(_evaluate_determinants, in_axes=(None,-1), out_axes=-1)(self, y) # (B, M, T)
        sd_t = jnp.exp(log_sd_t) # (B, M, T)

        # Compute log amplitudes
        log_psi = jnp.log(jnp.sum(sd_t, axis=(-2,-1))) # (B,)
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            return log_psi
        else:
            return log_psi.real