import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple
from netket.hilbert import HomogeneousHilbert
from netket.hilbert.constraint import DiscreteHilbertConstraint
from netket.utils import struct, StaticRange
from netket.utils.types import Scalar, Array
import jax


class FermionicSumConstraint(DiscreteHilbertConstraint):
    size: int = struct.field(pytree_node=False)
    n_elec: Tuple[Scalar, Scalar] = struct.field(pytree_node=False)

    def __init__(self, size, n_elec):
        self.size = size
        self.n_elec = n_elec

    @jax.jit
    def __call__(self, x: Array) -> Array:
        x_up = jnp.where(x == 3, 1, 0)
        x_up = jnp.where(x == 1, 1, x_up)
        x_dn = jnp.where(x == 3, 1, 0)
        x_dn = jnp.where(x == 2, 1, x_dn)

        return jnp.logical_and(
            jnp.sum(x_up, axis=-1) == self.n_elec[0],
            jnp.sum(x_dn, axis=-1) == self.n_elec[1],
        )

    def __hash__(self):
        return hash(("FermionicSumConstraint", self.n_elec))

    def __eq__(self, other):
        if isinstance(other, FermionicSumConstraint):
            return self.size == other.size and self.n_elec == other.n_elec
        return False


class FermionicDiscreteHilbert(HomogeneousHilbert):
    def __init__(self, N: int = 1, n_elec: Optional[Tuple[int, int]] = None):
        local_states = StaticRange(0, 1, 4, dtype=np.uint8)

        if n_elec is not None:
            constraint = FermionicSumConstraint(N, n_elec)
        else:
            constraint = None

        self._n_elec = n_elec if n_elec is None else n_elec
        self._local_size = 4

        super().__init__(local_states, N, constraint=constraint)

        from .random import discrete_fermion

    def __pow__(self, n):
        if self._n_elec is None:
            n_elec = None
        else:
            n_elec = (n_elec[0] * n, n_elec[1] * n)

        return FermionicDiscreteHilbert(self.size * n, n_elec=n_elec)

    def __repr__(self):
        n_elec = (
            ", n_up={}, n_down={}".format(self._n_elec[0], self._n_elec[1])
            if self._n_elec is not None
            else ""
        )
        return "Fermion(N={} {}))".format(self.size, n_elec)

    def states_to_local_indices(self, x):
        return x.astype(jnp.uint8)
