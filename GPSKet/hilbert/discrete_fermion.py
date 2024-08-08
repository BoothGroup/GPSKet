import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple
from netket.hilbert.custom_hilbert import HomogeneousHilbert
from netket.utils import struct, StaticRange
from netket.utils.types import Scalar, Array


@struct.dataclass
class SumConstraint:

    n_elec: Tuple[Scalar, Scalar] = struct.field(pytree_node=False)

    @jax.jit
    def __call__(self, x: Array) -> Array:
        n_up = jnp.where(x == 3, True, False)
        n_dn = jnp.array(n_up)
        n_up = jnp.where(x == 1, True, n_up)
        n_dn = jnp.where(x == 2, True, n_dn)
        result = jnp.logical_and(n_up.sum(axis=1) == self.n_elec[0], n_dn.sum(axis=1) == self.n_elec[1])
        return result

class FermionicDiscreteHilbert(HomogeneousHilbert):
    def __init__(self, N: int = 1, n_elec: Optional[Tuple[int, int]] = None):
        local_states = StaticRange(0, 1, 4, dtype=np.uint8)

        if n_elec is not None:
            constraints = SumConstraint(n_elec)
        else:
            constraints = None

        self._n_elec = n_elec if n_elec is None else n_elec
        self._local_size = 4

        super().__init__(local_states, N, constraints)

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
