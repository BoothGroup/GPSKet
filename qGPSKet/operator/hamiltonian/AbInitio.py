import numpy as np

from numba import jit

from netket.utils.types import DType

from qGPSKet.operator.fermion import FermionicDiscreteOperator, apply_hopping
from qGPSKet.operator.fermion import FermionicDiscreteOperator

class AbInitioHamiltonian(FermionicDiscreteOperator):
    def __init__(self, hilbert, h_mat, eri_mat):
        super().__init__(hilbert)
        self.h_mat = h_mat
        self.eri_mat = eri_mat

        # see Neuscamman (2012) for the definition of t
        self.t_mat = self.h_mat - 0.5 * np.einsum("prrq->pq", eri_mat)

    @property
    def dtype(self) -> DType:
        return float

    def get_conn_flattened(self, x, sections):
        x = x.astype(np.uint8)

        x_primes, mels = self._get_conn_flattened_kernel(
            x,
            sections,
            self.t_mat,
            self.eri_mat,
            np.nonzero(self.t_mat),
            np.nonzero(self.eri_mat)
        )

        # this pruning will no longer be required once the method is implemented better
        pruning = np.nonzero(mels == 0.)[0]

        if len(pruning) > 0:
            pruning = np.sort(pruning)
            self.prune_elements_sections(pruning, sections)
            x_primes = np.delete(x_primes, pruning, axis=0)
            mels = np.delete(mels, pruning, axis=0)

        return x_primes, mels

    @staticmethod
    @jit(nopython=True)
    def prune_elements_sections(pruning, sections):
        count = 0
        for b in range(sections.shape[0]):
            while count < len(pruning):
                if pruning[count] >= sections[b]:
                    break
                count += 1
            sections[b] -= count


    # this is done in a *very* naive way
    # TODO: improve!
    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(x, sections, t, eri, nonzero_t, nonzero_eri):
        total_connected_conn = (2*len(nonzero_t[0]) + 3*len(nonzero_eri[0]))*x.shape[0]

        x_prime = np.empty((total_connected_conn, x.shape[1]), dtype=np.uint8)
        mels = np.empty(total_connected_conn, dtype=np.complex128)

        c = 0
        for b in range(x.shape[0]):
            innercount = 0
            # one-body part
            for j in range(len(nonzero_t[0])):
                annihilate_site = nonzero_t[1][j]
                create_site = nonzero_t[0][j]
                x_prime[c + innercount,:] = x[b,:]
                x_prime[c + innercount+1,:] = x[b,:]
                multiplicator = apply_hopping(annihilate_site, create_site, x_prime[c+innercount], 1)
                mels[c+innercount] = multiplicator * t[nonzero_t[0][j], nonzero_t[1][j]]
                multiplicator = apply_hopping(annihilate_site, create_site, x_prime[c+innercount+1], 2)
                mels[c+innercount+1] = multiplicator * t[nonzero_t[0][j], nonzero_t[1][j]]
                innercount += 2

            # two body part
            for j in range(len(nonzero_eri[0])):
                annihilate_site_A = nonzero_eri[3][j]
                create_site_A = nonzero_eri[2][j]
                annihilate_site_B = nonzero_eri[1][j]
                create_site_B = nonzero_eri[0][j]
                x_prime[c + innercount,:] = x[b,:]
                x_prime[c + innercount+1,:] = x[b,:]
                x_prime[c + innercount+2,:] = x[b,:]

                multiplicator = apply_hopping(annihilate_site_A, create_site_A, x_prime[c+innercount], 1)
                if multiplicator != 0:
                    multiplicator *= apply_hopping(annihilate_site_B, create_site_B, x_prime[c+innercount], 1)
                mels[c+innercount] = 0.5 * multiplicator * eri[nonzero_eri[0][j], nonzero_eri[1][j], nonzero_eri[2][j], nonzero_eri[3][j]]

                multiplicator = apply_hopping(annihilate_site_A, create_site_A, x_prime[c+innercount+1], 2)
                if multiplicator != 0:
                    multiplicator *= apply_hopping(annihilate_site_B, create_site_B, x_prime[c+innercount+1], 2)
                mels[c+innercount+1] = 0.5 * multiplicator * eri[nonzero_eri[0][j], nonzero_eri[1][j], nonzero_eri[2][j], nonzero_eri[3][j]]

                multiplicator = apply_hopping(annihilate_site_A, create_site_A, x_prime[c+innercount+2], 2)
                if multiplicator != 0:
                    multiplicator *= apply_hopping(annihilate_site_B, create_site_B, x_prime[c+innercount+2], 1)
                mels[c+innercount+2] = multiplicator * eri[nonzero_eri[0][j], nonzero_eri[1][j], nonzero_eri[2][j], nonzero_eri[3][j]]

                innercount += 3

            c += innercount
            sections[b] = c
        return x_prime, mels

