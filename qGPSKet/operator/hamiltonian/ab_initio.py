import numpy as np

from numba import jit

from netket.utils.types import DType

from qGPSKet.operator.fermion import FermionicDiscreteOperator, apply_hopping

class AbInitioHamiltonian(FermionicDiscreteOperator):
    def __init__(self, hilbert, h_mat, eri_mat):
        super().__init__(hilbert)
        self.h_mat = h_mat
        self.eri_mat = eri_mat

        # see Neuscamman (2013) for the definition of t
        self.t_mat = self.h_mat - 0.5 * np.einsum("prrq->pq", eri_mat)

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self) -> DType:
        return float

    # pad argument is just a dummy at the moment,
    # TODO: include padding for unconstrained Hilbert spaces
    def get_conn_flattened(self, x, sections, pad=True):
        assert(not pad or self.hilbert._has_constraint)

        x_primes, mels = self._get_conn_flattened_kernel(np.asarray(x, dtype = np.uint8),
                                                         sections, self.t_mat, self.eri_mat)

        return x_primes, mels

    # this implementation follows the approach outlined in Neuscamman (2013)
    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(x, sections, t, eri):
        range_indices = np.arange(x.shape[-1])

        x_prime = np.empty((0, x.shape[1]), dtype=np.uint8)
        mels = np.empty(0, dtype=np.complex128)

        c = 0
        for batch_id in range(x.shape[0]):
            is_occ_up = (x[batch_id] & 1).astype(np.bool8)
            is_occ_down = (x[batch_id] & 2).astype(np.bool8)

            up_count = np.cumsum(is_occ_up)
            down_count = np.cumsum(is_occ_down)

            is_empty_up = ~is_occ_up
            is_empty_down = ~is_occ_down

            up_occ_inds = range_indices[is_occ_up]
            down_occ_inds = range_indices[is_occ_down]
            up_unocc_inds = range_indices[is_empty_up]
            down_unocc_inds = range_indices[is_empty_down]

            connected_con = 1
            connected_con += len(up_occ_inds) * len(up_unocc_inds)
            connected_con += len(down_occ_inds) * len(down_unocc_inds)
            connected_con += len(down_occ_inds) * len(down_unocc_inds) * (len(down_occ_inds) - 1) * (len(down_unocc_inds) - 1)
            connected_con += len(up_occ_inds) * len(up_unocc_inds) * (len(up_occ_inds) - 1) * (len(up_unocc_inds) - 1)
            connected_con += len(up_occ_inds) * len(up_unocc_inds) * len(down_occ_inds) * len(down_unocc_inds)

            x_prime = np.append(x_prime, np.empty((connected_con, x.shape[1]), dtype=np.uint8), axis=0)
            mels = np.append(mels, np.empty(connected_con))

            diag_element = 0.0
            for i in up_occ_inds:
                diag_element += t[i, i]
            for i in down_occ_inds:
                diag_element += t[i, i]
            for i in up_occ_inds:
                for j in down_occ_inds:
                    diag_element += eri[i, i, j, j]
            for i in up_occ_inds:
                for j in up_occ_inds:
                    diag_element += 0.5 * eri[i, i, j, j]
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    diag_element += 0.5 * eri[i, a, a, i]
            for i in down_occ_inds:
                for j in down_occ_inds:
                    diag_element += 0.5 * eri[i, i, j, j]
            for i in down_occ_inds:
                for a in down_unocc_inds:
                    diag_element += 0.5 * eri[i, a, a, i]

            x_prime[c, :] = x[batch_id, :]
            mels[c] = diag_element
            c += 1

            # one-body parts
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    x_prime[c, :] = x[batch_id, :]
                    multiplicator = apply_hopping(i, a, x_prime[c], 1,
                                                  cummulative_count=up_count)
                    value = t[i, a]
                    for k in up_occ_inds:
                        value += eri[i, a, k, k]
                    for k in down_occ_inds:
                        value += eri[i, a, k, k]
                    for k in up_unocc_inds:
                        value += 0.5 * eri[i, k, k, a]
                    for k in up_occ_inds:
                        value -= 0.5 * eri[k, a, i, k]
                    mels[c] = multiplicator * value
                    c += 1
            for i in down_occ_inds:
                for a in down_unocc_inds:
                    x_prime[c, :] = x[batch_id, :]
                    multiplicator = apply_hopping(i, a, x_prime[c], 2,
                                                  cummulative_count=down_count)
                    value = t[i, a]
                    for k in down_occ_inds:
                        value += eri[i, a, k, k]
                    for k in up_occ_inds:
                        value += eri[i, a, k, k]
                    for k in down_unocc_inds:
                        value += 0.5 * eri[i, k, k, a]
                    for k in down_occ_inds:
                        value -= 0.5 * eri[k, a, i, k]
                    mels[c] = multiplicator * value
                    c += 1

            # two body parts
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    for j in up_occ_inds:
                        for b in up_unocc_inds:
                            if i != j and a != b:
                                x_prime[c, :] = x[batch_id, :]
                                multiplicator = apply_hopping(i, a, x_prime[c], 1,
                                                              cummulative_count=up_count)
                                multiplicator *= apply_hopping(j, b, x_prime[c], 1,
                                                              cummulative_count=up_count)
                                # take first hop into account
                                left_limit = min(j, b)
                                right_limit = max(j, b) - 1
                                if i <= right_limit and i > left_limit:
                                    multiplicator *= -1
                                if a <= right_limit and a > left_limit:
                                    multiplicator *= -1

                                mels[c] = 0.5 * multiplicator * eri[i,a,j,b]
                                c += 1
            for i in down_occ_inds:
                for a in down_unocc_inds:
                    for j in down_occ_inds:
                        for b in down_unocc_inds:
                            if i != j and a != b:
                                x_prime[c, :] = x[batch_id, :]
                                multiplicator = apply_hopping(i, a, x_prime[c], 2,
                                                                cummulative_count=down_count)
                                multiplicator *= apply_hopping(j, b, x_prime[c], 2,
                                                                cummulative_count=down_count)
                                # take first hop into account
                                left_limit = min(j, b)
                                right_limit = max(j, b) - 1
                                if i <= right_limit and i > left_limit:
                                    multiplicator *= -1
                                if a <= right_limit and a > left_limit:
                                    multiplicator *= -1
                                mels[c] = 0.5 * multiplicator * eri[i,a,j,b]
                                c += 1
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    for j in down_occ_inds:
                        for b in down_unocc_inds:
                            x_prime[c, :] = x[batch_id, :]
                            multiplicator = apply_hopping(i, a, x_prime[c], 1,
                                                          cummulative_count=up_count)
                            multiplicator *= apply_hopping(j, b, x_prime[c], 2,
                                                           cummulative_count=down_count)
                            mels[c] = multiplicator * eri[i,a,j,b]
                            c += 1
            sections[batch_id] = c
        return x_prime, mels
