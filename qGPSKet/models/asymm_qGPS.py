import jax.numpy as jnp
from flax import linen as nn
from netket.utils.types import Array, Callable, DType, NNInitFunc
from .slater import Slater
from ..hilbert.discrete_fermion import FermionicDiscreteHilbert
from ..nn.initializers import orthogonal


# Dimensions:
# - B = batch size
# - L = number of sites
# - N = total number of electrons
# - N_up = number of spin-up electrons
# - N_down = number of spin-down electrons
# - M = number of determinants
# - T = number of symmetries


class ASymmqGPS(nn.Module):
    """
    Implements the antisymmetric qGPS Ansatz with support for multiple determinants,
    symmetries and different symmetrization methods
    """

    hilbert: FermionicDiscreteHilbert
    """Hilbert space"""
    n_determinants: int = 1
    """Number of determinants"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    init_fun : NNInitFunc = orthogonal()
    """Initializer for the variational parameters"""
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""
    symmetrization: str = 'kernel'
    """Symmetrization method"""
    spin_symmetry_by_structure: bool = False
     """ Flag determines whether the S^2 symmetry (with S=0) should be enforced
    by using the same orbitals for up and down spin.
    """
    apply_fast_update: bool = True
    """Whether fast update is used in the computation of the Slater determinants"""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        x = jnp.asarray(x, jnp.int32) # (B, L)

        if self.symmetrization == 'kernel':
            def out_transformation(y):
                y = jnp.exp(y) # (B, M, S, T)
                y = jnp.sum(y, axis=1) # (B, S, T)
                y = jnp.sinh(jnp.sum(y, axis=(-2, -1))) # (B,)
                y = jnp.log(y)
                return y
        elif self.symmetrization == 'projective':
            def out_transformation(y):
                y = jnp.exp(y) # (B, M, S, T)
                y = jnp.sinh(jnp.sum(y, axis=1)) # (B, S, T)
                y = jnp.sum(y, axis=(-2, -1)) # (B,)
                y = jnp.log(y)
                return y

        log_psi = Slater(
            self.hilbert,
            self.n_determinants,
            dtype=self.dtype,
            init_fun=self.init_fun,
            symmetries=self.apply_symmetries,
            out_transformation=out_transformation,
            apply_fast_update=self.apply_fast_update,
            spin_symmetry_by_structure=self.spin_symmetry_by_structure
        )(x)
        return log_psi

class ASymmqGPSProd(nn.Module):
    """
    Implements the antisymmetric qGPS Ansatz as an odd product of antisymmetric wavefunctions
    """

    hilbert: FermionicDiscreteHilbert
    """Hilbert space"""
    n_determinants: int = 1
    """Number of determinants"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    init_fun : NNInitFunc = orthogonal()
    """Initializer for the variational parameters"""
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""
    spin_symmetry_by_structure: bool = False
     """ Flag determines whether the S^2 symmetry (with S=0) should be enforced
    by using the same orbitals for up and down spin.
    """
    apply_fast_update: bool = True
    """Whether fast update is used in the computation of the Slater determinants"""

    def setup(self):
        assert self.n_determinants % 2 != 0

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        x = jnp.asarray(x, jnp.int32) # (B, L)

        def out_transformation(y):
            y = jnp.exp(y) # (B, M, S, T)
            y = jnp.prod(jnp.sinh(y), axis=1) # (B, S, T)
            y = jnp.sum(y, axis=(-2, -1)) # (B,)
            y = jnp.log(y)
            return y

        log_psi = Slater(
            self.hilbert,
            self.n_determinants,
            dtype=self.dtype,
            init_fun=self.init_fun,
            symmetries=self.apply_symmetries,
            out_transformation=out_transformation,
            apply_fast_update=self.apply_fast_update,
            spin_symmetry_by_structure=self.spin_symmetry_by_structure
        )(x)
        return log_psi
