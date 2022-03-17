from typing import Optional
import numpy as np
from netket.hilbert.custom_hilbert import HomogeneousHilbert
from netket.hilbert._deprecations import graph_to_N_depwarn
from netket.graph import AbstractGraph

class ASEPDiscreteHilbert(HomogeneousHilbert):
    def __init__(
        self,
        N: int = 1,
        graph: Optional[AbstractGraph] = None
    ):
        N = graph_to_N_depwarn(N=N, graph=graph)
        local_states = np.arange(2, dtype=np.uint8)
        local_states = local_states.tolist()

        super().__init__(local_states, N)

        from .random import discrete_asep


    def __pow__(self, n):
        return ASEPDiscreteHilbert(self.size * n)

    def __repr__(self):
        return "ASEP(N={}))".format(self._size)