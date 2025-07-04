import numpy as np

from numba import jit


from netket.operator import DiscreteOperator
from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert


@jit(nopython=True)
def apply_hopping(
    annihilate_site, create_site, x_prime, spin_int, cummulative_count=None
):
    start_occ = x_prime[annihilate_site]
    final_occ = x_prime[create_site]

    if not (start_occ & spin_int):
        multiplicator = 0
    else:
        if create_site == annihilate_site:
            multiplicator = 1
        else:
            if not (final_occ & spin_int):
                left_limit = min(annihilate_site, create_site)
                right_limit = max(annihilate_site, create_site) - 1

                if cummulative_count is None:
                    parity_count = np.sum(
                        (x_prime[left_limit:right_limit] & spin_int).astype(np.bool)
                    )
                else:
                    parity_count = (
                        cummulative_count[right_limit] - cummulative_count[left_limit]
                    )

                multiplicator = 1

                if parity_count & 1:
                    multiplicator *= -1
            else:
                multiplicator = 0

    if multiplicator != 0:
        x_prime[annihilate_site] -= spin_int
        x_prime[create_site] += spin_int

    return multiplicator


class FermionicDiscreteOperator(DiscreteOperator):
    """
    Base class for discrete Fermionic operators.
    Maybe we want to add some common logic to this class,
    at the moment this class is an empty super class for concrete fermionic implementations.
    """

    def __init__(self, hilbert: FermionicDiscreteHilbert):
        super().__init__(hilbert)
