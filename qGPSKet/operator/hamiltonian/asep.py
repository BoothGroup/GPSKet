import numpy as np
from numba import jit
from netket.utils.types import DType
from netket.operator import DiscreteOperator
from qGPSKet.hilbert import ASEPDiscreteHilbert
from qGPSKet.operator.asep import apply_creation, apply_annihilation, apply_hopping, apply_particle_hole

class AsymmetricSimpleExclusionProcess(DiscreteOperator):
    def __init__(self, hilbert: ASEPDiscreteHilbert, L: int, lambd: float = 0.0, alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.5, delta: float = 0.5, p: float = 0.5, q: float = 0.5):
        super().__init__(hilbert)
        self.L = L
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.p = p
        self.q = q

    @property
    def is_hermitian(self) -> bool:
        flag = False
        params_equal = np.sum([self.alpha, self.beta, self.gamma, self.delta]) == 2.0
        if self.lambd == 0.0 and  params_equal and self.p == self.q:
            flag = True
        return flag

    @property
    def dtype(self) -> DType:
        return float

    # pad argument is just a dummy atm -> TODO: improve this!
    def get_conn_flattened(self, x, sections, pad=True):
        x_primes, mels = self._get_conn_flattened_kernel(np.asarray(x, dtype = np.uint8),
                                                         sections, self.L, self.lambd, self.alpha, self.beta, self.gamma, self.delta, self.p, self.q)
        return x_primes, mels

    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(x, sections, L, lambd, alpha, beta, gamma, delta, p, q):
        exp_lambd = np.exp(lambd)
        exp_minus_lambd = np.exp(-lambd)

        n_conn = x.shape[0] * 4*(1+L)
        x_prime = np.empty((n_conn, x.shape[1]), dtype=np.uint8)
        mels = np.empty(n_conn, dtype=np.float64)

        count = 0
        for batch_id in range(x.shape[0]):
            # alpha term
            x_prime[count, :] = x[batch_id, :]
            mels[count] = alpha * exp_lambd * apply_creation(0, x_prime[count])
            count += 1
            x_prime[count, :] = x[batch_id, :]
            mels[count] = -alpha * (1-x_prime[count, 0])
            count += 1

            # gamma term
            x_prime[count, :] = x[batch_id, :]
            mels[count] = gamma * exp_minus_lambd * apply_annihilation(0, x_prime[count])
            count += 1
            x_prime[count, :] = x[batch_id, :]
            mels[count] = -gamma * x_prime[count, 0]
            count += 1

            # beta term
            x_prime[count, :] = x[batch_id, :]
            mels[count] = beta * exp_minus_lambd * apply_creation(L-1, x_prime[count])
            count += 1
            x_prime[count, :] = x[batch_id, :]
            mels[count] = -beta * (1-x_prime[count, L-1])
            count += 1

            # delta term
            x_prime[count, :] = x[batch_id, :]
            mels[count] = delta * exp_lambd * apply_annihilation(L-1, x_prime[count])
            count += 1
            x_prime[count, :] = x[batch_id, :]
            mels[count] = -delta * x_prime[count, L-1]
            count += 1

            for i in range(L-1):
                # p term
                x_prime[count, :] = x[batch_id, :]
                mels[count] = p * exp_lambd * apply_hopping(i, i+1, x_prime[count])
                count += 1
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -p * apply_particle_hole(i, i+1, x_prime[count])
                count += 1

                # q term
                x_prime[count, :] = x[batch_id, :]
                mels[count] = q * exp_minus_lambd * apply_hopping(i+1, i, x_prime[count])
                count += 1
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -q * apply_particle_hole(i+1, i, x_prime[count])
                count += 1

            sections[batch_id] = count
        return x_prime, mels

