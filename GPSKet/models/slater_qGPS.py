import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
from flax import linen as nn
from typing import Union, Tuple
from netket.utils.types import Array, DType, NNInitFunc, Callable
from .slater import Slater
from .qGPS import qGPS, no_syms
from ..hilbert.discrete_fermion import FermionicDiscreteHilbert
from ..nn.initializers import orthogonal, normal


class SlaterqGPS(nn.Module):
    """
    Implements a Slater-Jastrow wavefunction where the Jastrow factor is a qGPS
    """

    hilbert: FermionicDiscreteHilbert
    """The Hilbert space of the wavefunction model"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    n_determinants: int = 1
    """Number of determinants"""
    M: int = 1
    """Number of qGPS support dimensions"""
    slater_init_fun: Union[NNInitFunc, Tuple[NNInitFunc, NNInitFunc]] = orthogonal()
    """Initializer for the variational parameters of the Slater determinant"""
    qGPS_init_fun: NNInitFunc = normal()
    """Initializer for the variational parameters of the qGPS"""
    slater_apply_symmetries: Callable = lambda inputs: jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations in the Slater determinant"""
    qGPS_apply_symmetries: Callable = no_syms()
    """Function to apply symmetries to configurations in the qGPS"""
    out_transformation: Callable = lambda x: jax.scipy.special.logsumexp(
        x, axis=(1, -1, -2)
    )
    """Final output transformation. Its input has shape (B, M, S, T)."""
    apply_fast_update: bool = True
    """Whether fast update is used in the computation of the Slater determinants"""
    spin_symmetry_by_structure: bool = True
    """Whether the α and β orbitals are the same or not"""
    fixed_magnetization: bool = True
    """Whether magnetization should be conserved or not"""

    @nn.compact
    def __call__(self, x, cache_intermediates=False, update_sites=None) -> Array:
        slater = Slater(
            self.hilbert,
            n_determinants=self.n_determinants,
            dtype=self.dtype,
            init_fun=self.slater_init_fun,
            symmetries=self.slater_apply_symmetries,
            spin_symmetry_by_structure=self.spin_symmetry_by_structure,
            fixed_magnetization=self.fixed_magnetization,
            out_transformation=self.out_transformation,
            apply_fast_update=self.apply_fast_update
        )(x, cache_intermediates=cache_intermediates, update_sites=update_sites)
        jastrow = qGPS(
            self.hilbert,
            self.M,
            dtype=self.dtype,
            init_fun=self.qGPS_init_fun,
            syms=self.qGPS_apply_symmetries,
            apply_fast_update=self.apply_fast_update
        )(x, cache_intermediates=cache_intermediates, update_sites=update_sites)
        return slater + jastrow
