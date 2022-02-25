import numpy as np
from numba import jit
from netket.graph import AbstractGraph
from typing import List, Tuple, Union
from netket.utils.types import DType
from qGPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from qGPSKet.operator.fermion import FermionicDiscreteOperator, apply_hopping

class FermiHubbard(FermionicDiscreteOperator):
    def __init__(self, hilbert: FermionicDiscreteHilbert, edges: List[Tuple[int, int]], U: float=0.0, t: Union[float, List[float]]=1.):
        super().__init__(hilbert)
        self.U = U
        self.edges = np.array(edges)
        if isinstance(t, List):
            self.t = np.array(t)
        else:
            self.t = np.ones(self.edges.shape[0]) * t

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self) -> DType:
        return float

    def get_conn_flattened(self, x, sections, pad=True):
        x_primes, mels = self._get_conn_flattened_kernel(np.asarray(x, dtype = np.uint8),
                                                         sections, self.U, self.edges, self.t)
        if not pad:
            valid = (mels != 0.)
            x_primes = x_primes[valid, :]
            mels = mels[valid]
            sections[-1] = len(mels)
        return x_primes, mels

    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(x, sections, U, edges, t):
        n_conn = x.shape[0] * (1 + edges.shape[0]*4)
        x_prime = np.empty((n_conn, x.shape[1]), dtype=np.uint8)
        mels = np.empty(n_conn, dtype=np.float64)

        count = 0
        for batch_id in range(x.shape[0]):
            # diagonal element
            x_prime[count, :] = x[batch_id, :]
            mels[count] = U * np.sum(x[batch_id, :] == 3)
            if mels[count] != 0.:
                count += 1

            is_occ_up = (x[batch_id] & 1).astype(np.bool8)
            is_occ_down = (x[batch_id] & 2).astype(np.bool8)

            up_count = np.cumsum(is_occ_up)
            down_count = np.cumsum(is_occ_down)

            # hopping
            for edge_count in range(edges.shape[0]):
                edge = edges[edge_count]

                # spin up
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[0], edge[1], x_prime[count], 1, cummulative_count=up_count)
                if mels[count] != 0.:
                    count += 1
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[1], edge[0], x_prime[count], 1, cummulative_count=up_count)
                if mels[count] != 0.:
                    count += 1

                # spin down
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[0], edge[1], x_prime[count], 2, cummulative_count=down_count)
                if mels[count] != 0.:
                    count += 1
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[1], edge[0], x_prime[count], 2, cummulative_count=down_count)
                if mels[count] != 0.:
                    count += 1
            sections[batch_id] = count

        for l in range(count, n_conn):
            x_prime[l, :] = x[-1, :]
            mels[l] = 0.
        return x_prime, mels

