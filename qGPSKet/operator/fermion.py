import numpy as np

from numba import jit


from netket.operator import DiscreteOperator
from qGPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert

@jit(nopython=True)
def apply_hopping(annihilate_site, create_site, x_prime, spin_int):
    start_occ = x_prime[annihilate_site]
    final_occ = x_prime[create_site]

    left_limit = min(annihilate_site, create_site) + 1
    right_limit = max(annihilate_site, create_site)

    multiplicator = 1.

    if left_limit < right_limit:
        multiplicator *= (-1)**np.sum(((x_prime[left_limit:right_limit])&(spin_int))==(spin_int))

    if create_site != annihilate_site:
        if (not (start_occ & spin_int)) or (final_occ & spin_int):
            multiplicator = 0.0
        else:
            x_prime[annihilate_site] -= spin_int
            x_prime[create_site] += spin_int
    else:
        if not bool(start_occ & spin_int):
            multiplicator = 0.0
    return multiplicator

class FermionicDiscreteOperator(DiscreteOperator):
    """
    Base class for discrete Fermionic operators.
    Maybe we want to add some common logic to this class,
    at the moment this class is an empty super class for concrete fermionic implementations.
    """
    def __init__(self, hilbert: FermionicDiscreteHilbert):
        super().__init__(hilbert)
