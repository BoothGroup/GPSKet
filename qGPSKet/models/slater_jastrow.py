import jax.numpy as jnp
from jax.nn.initializers import normal
from flax import linen as nn
from netket.models import Jastrow
from netket.utils.types import Array, DType, NNInitFunc
from .slater import Slater
from .jastrow import Jastrow
from ..hilbert.discrete_fermion import FermionicDiscreteHilbert
from ..nn.initializers import orthogonal


class SlaterJastrow(nn.Module):
    """
    Implements a Slater-Jastrow wavefunction
    """

    hilbert: FermionicDiscreteHilbert
    """The Hilbert space of the wavefunction model"""
    dtype: DType=jnp.complex128
    """Type of the variational parameters"""
    n_determinants: int=1
    """Number of determinants"""
    slater_init_fun: NNInitFunc=orthogonal()
    """Initializer for the variational parameters of the Slater determinant"""
    jastrow_init_fun: NNInitFunc=normal()
    """Initializer for the variational parameters of the Jastrow coefficient"""
    apply_fast_update: bool=True
    """Whether fast update is used in the computation of the Slater determinants"""
    spin_symmetry_by_structure: bool=True
    """Whether the α and β orbitals are the same or not"""
    fixed_magnetization: bool=True
    """Whether magnetization should be conserved or not"""

    @nn.compact
    def __call__(self, x) -> Array:
        slater = Slater(
            self.hilbert,
            n_determinants=self.n_determinants,
            dtype=self.dtype,
            init_fun=self.slater_init_fun,
            spin_symmetry_by_structure=self.spin_symmetry_by_structure,
            fixed_magnetization=self.fixed_magnetization,
            apply_fast_update=self.apply_fast_update
        )(x)
        jastrow = Jastrow(
            self.hilbert,
            dtype=self.dtype,
            init_fun=self.jastrow_init_fun
        )(x)
        return slater+jastrow