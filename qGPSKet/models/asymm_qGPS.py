import jax
import jax.numpy as jnp
from typing import Tuple
from flax import linen as nn
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils.types import Array, Callable, DType, NNInitFunc
from ..nn.initializers import orthogonal


# Dimensions:
# - B = batch size
# - L = number of sites
# - N = total number of electrons
# - N_up = number of spin-up electrons
# - N_down = number of spin-down electrons
# - M = number of determinants
# - T = number of symmetries

def occupancies_to_electrons(x : Array, n_elec : Tuple[int, int]) -> Array:
    """
    Converts input configs from 2nd quantized representation x to 1st quantized representation y:
        x=[x_1, x_2, ..., x_L]
                    |
                    v
        y=(y_1, y_2, ..., y_{N_up}, y_{N_up+1}, y_{N_up+2}, ..., y_{N_up+N_down})

    Args:
        x : an array of input configurations in 2nd quantization of shape (B, L)
        n_elec : a tuple of ints N_up and N_down specifying the number of spin-up and spin-down electrons

    Returns:
        y : input configurations transformed into 1st quantization representation (B, N_up+N_down)
    """
    batch_size = x.shape[0]
    _, y_up = jnp.nonzero(x&1, size=batch_size*n_elec[0])
    _, y_down = jnp.nonzero((x&2)/2, size=batch_size*n_elec[1])
    y_up = jnp.reshape(y_up, (batch_size, -1))
    y_down = jnp.reshape(y_down, (batch_size, -1))
    y = jnp.column_stack([y_up, y_down])
    return y

class ASymmqGPS(nn.Module):
    """
    Implements the antisymmetric qGPS Ansatz with support for multiple determinants,
    symmetries and different symmetrization methods
    """

    hilbert: HomogeneousHilbert
    """Hilbert space"""
    n_determinants: int
    """Number of determinants"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    init_fun : NNInitFunc = orthogonal()
    """Initializer for the variational parameters"""
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""
    symmetrization: str = 'kernel'
    """Symmetrization method"""


    def setup(self):
        L = self.hilbert.size
        self.n_elec = self.hilbert._n_elec
        self.orbitals_up = self.param("orbitals_up", self.init_fun, (self.n_determinants, L, self.n_elec[0]), self.dtype)
        self.orbitals_down = self.param("orbitals_down", self.init_fun, (self.n_determinants, L, self.n_elec[1]), self.dtype)

    def __call__(self, x : Array) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        x = jnp.asarray(x, jnp.int32) # (B, L)

        # Convert input config representation
        y = occupancies_to_electrons(x, self.hilbert._n_elec) # (B, N)

        # Apply symmetry transformations
        y = self.apply_symmetries(y) # (B, N, T)

        # Evaluate Slater determinants
        log_sd_t = jax.vmap(_evaluate_determinants, in_axes=(None,-1), out_axes=-1)(self, y) # (B, M, T)
        sd_t = jnp.exp(log_sd_t)

        # Compute log amplitudes
        if self.symmetrization == 'kernel':
            psi_t = jnp.sum(sd_t, axis=-2) # (B, T)
            psi = jnp.sinh(jnp.sum(psi_t, axis=-1)) # (B,)
        elif self.symmetrization == 'projective':
            psi_t = jnp.sinh(jnp.sum(sd_t, axis=-2)) # (B, T)
            psi = jnp.sum(psi_t, axis=-1) # (B,)
        log_psi = jnp.log(psi)
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            return log_psi
        else:
            return log_psi.real

class ASymmqGPSProd(nn.Module):
    """
    Implements the antisymmetric qGPS Ansatz as an odd product of antisymmetric wavefunctions
    """

    hilbert: HomogeneousHilbert
    """Hilbert space"""
    n_determinants: int
    """Number of determinants"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    init_fun : NNInitFunc = orthogonal()
    """Initializer for the variational parameters"""
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""


    def setup(self):
        assert self.n_determinants % 2 != 0
        L = self.hilbert.size
        self.n_elec = self.hilbert._n_elec
        self.orbitals_up = self.param("orbitals_up", self.init_fun, (self.n_determinants, L, self.n_elec[0]), self.dtype)
        self.orbitals_down = self.param("orbitals_down", self.init_fun, (self.n_determinants, L, self.n_elec[1]), self.dtype)

    def __call__(self, x : Array) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        x = jnp.asarray(x, jnp.int32) # (B, L)

        # Convert input config representation
        y = occupancies_to_electrons(x, self.hilbert._n_elec) # (B, N)

        # Apply symmetry transformations
        y = self.apply_symmetries(y) # (B, N, T)

        # Evaluate Slater determinants
        log_sd_t = jax.vmap(_evaluate_determinants, in_axes=(None,-1), out_axes=-1)(self, y) # (B, M, T)
        sd_t = jnp.exp(log_sd_t)

        # Compute log amplitudes
        psi_t = jnp.prod(jnp.sinh(sd_t), axis=-2) # (B, T)
        psi = jnp.sum(psi_t, axis=-1) # (B,)
        log_psi = jnp.log(psi)
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            return log_psi
        else:
            return log_psi.real

def _evaluate_determinants(model : nn.Module, y_t : Array) -> Array:
    Ũ_up = jnp.take(model.orbitals_up, y_t[:, :model.n_elec[0]], axis=1) # (M, B, N_up, N_up)
    Ũ_down = jnp.take(model.orbitals_down, y_t[:, model.n_elec[0]:], axis=1) # (M, B, N_down, N_down)
    # Compute log of det with sign
    # N.B.: always returns a complex number, even for real parameters
    (s_up, log_det_up) = jnp.linalg.slogdet(Ũ_up)
    (s_down, log_det_down) = jnp.linalg.slogdet(Ũ_down)
    log_sd = log_det_up + log_det_down + jnp.log(s_up*s_down+0j)
    return jnp.transpose(log_sd) # (B, M)
