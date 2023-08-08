from typing import Optional, Tuple

from numba import jit

import numpy as np

import jax

import jax.numpy as jnp

import netket as nk

from netket.hilbert.custom_hilbert import HomogeneousHilbert


class FermionicDiscreteHilbert(HomogeneousHilbert):
    def __init__(self, N: int = 1, n_elec: Optional[Tuple[int, int]] = None):
        local_states = np.arange(4, dtype=np.uint8)
        local_states = local_states.tolist()

        if n_elec is not None:

            def constraints(x):
                return self._sum_constraint(x, n_elec)

        else:
            constraints = None

        self._n_elec = n_elec if n_elec is None else n_elec
        self._local_size = 4

        super().__init__(local_states, N, constraints)

        from .random import discrete_fermion

    @staticmethod
    @jit(nopython=True)
    def _sum_constraint(x, n_elec):
        result = np.ones(x.shape[0])
        for i in range(x.shape[0]):
            n_up = 0
            n_down = 0
            for j in range(x.shape[1]):
                if x[i, j] == 1.0 or x[i, j] == 3.0:
                    n_up += 1
                if x[i, j] == 2.0 or x[i, j] == 3.0:
                    n_down += 1
            if n_up == n_elec[0] and n_down == n_elec[1]:
                result[i] = 1
            else:
                result[i] = 0
        return result == 1.0

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
