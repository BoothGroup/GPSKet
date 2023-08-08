import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
from netket.utils.types import Array, NNInitFunc, Callable
from netket.utils import HashableArray
from .backflow import Backflow
from .jastrow import Jastrow
from ..hilbert.discrete_fermion import FermionicDiscreteHilbert


class BackflowJastrow(nn.Module):
    """
    Implements a linear combination of Slater determinants with Backflow orbitals, multiplied by a Jastrow correlation factor
    """

    hilbert: FermionicDiscreteHilbert
    """The Hilbert space of the wavefunction model"""
    orbitals: Tuple[HashableArray]
    """Tuple of initial orbitals for the Backflow determinants"""
    correction_fun: Tuple[nn.Module]
    """Tuple of modules that compute the correction to the initial Backflow orbitals"""
    jastrow_init_fun: NNInitFunc = jax.nn.initializers.normal()
    """Initializer for the variational parameters of the Jastrow coefficient"""
    backflow_apply_symmetries: Callable = lambda inputs: jnp.expand_dims(
        inputs, axis=-1
    )
    """Function to apply symmetries to configurations in the Slater determinant"""
    jastrow_apply_symmetries: Callable = lambda inputs: jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations in the Jastrow factor"""
    apply_fast_update: bool = True
    """Whether fast update is used in the computation of the Slater determinants"""
    spin_symmetry_by_structure: bool = True
    """Whether the α and β orbitals are the same or not"""
    fixed_magnetization: bool = True
    """Whether magnetization should be conserved or not"""

    @nn.compact
    def __call__(self, x) -> Array:
        assert len(self.orbitals) == len(self.correction_fun)
        n_determinants = len(self.orbitals)
        batch_size = x.shape[0]
        log_det = jnp.zeros((batch_size, n_determinants), jnp.complex128)
        for d in range(n_determinants):
            y = Backflow(
                self.hilbert,
                self.orbitals[d],
                self.correction_fun[d],
                apply_symmetries=self.backflow_apply_symmetries,
                spin_symmetry_by_structure=self.spin_symmetry_by_structure,
                fixed_magnetization=self.fixed_magnetization,
                apply_fast_update=self.apply_fast_update,
            )(x)
            log_det = log_det.at[:, d].set(y)
        backflow = jax.scipy.special.logsumexp(log_det, axis=-1)
        jastrow = Jastrow(
            self.hilbert,
            init_fun=self.jastrow_init_fun,
            apply_symmetries=self.jastrow_apply_symmetries,
        )(x)
        return jastrow + backflow
