import jax
import jax.numpy as jnp
from flax import linen as nn
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils.types import Array, Callable
from ..nn import SlaterDeterminant


def occupancies_to_electrons(x, n_elec):
    batch_size = x.shape[0]
    _, y_up = jnp.nonzero(x&1, size=batch_size*n_elec[0])
    _, y_down = jnp.nonzero((x&2)/2, size=batch_size*n_elec[1])
    y_up = jnp.reshape(y_up, (batch_size, -1))
    y_down = jnp.reshape(y_down, (batch_size, -1))
    y = jnp.column_stack([y_up, y_down])
    return y

class ASymmqGPS(nn.Module):
    hilbert: HomogeneousHilbert
    n_determinants: int
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    symmetrization: str = 'kernel'


    # Dimensions:
    # - B = batch size
    # - L = number of sites
    # - N = total number of electrons
    # - N_up = number of spin-up electrons
    # - N_down = number of spin-down electrons
    # - M = number of determinants
    # - T = number of symmetries

    def setup(self):
        self._determinants = [
            SlaterDeterminant(self.hilbert.size, self.hilbert._n_elec) for _ in range(self.n_determinants)
        ]

    def __call__(self, x) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        x = jnp.asarray(x, jnp.int32) # (B, L)

        # Convert input configs from 2nd quantized representation x
        # to 1st quantized representation y:
        # x=[x_1, x_2, ..., x_L]
        #                |
        #                v
        # y=(y_1, y_2, ..., y_{N_up}, y_{N_up+1}, y_{N_up+2}, ..., y_{N_up+N_down})

        y = occupancies_to_electrons(x, self.hilbert._n_elec) # (B, N)

        # Apply symmetry transformations
        y = self.apply_symmetries(y) # (B, N, T)

        if self.symmetrization == 'kernel':
            # Evaluate Slater determinants
            sd = jnp.zeros(x.shape[0])
            for i in range(self.n_determinants):
                log_sd_i = jax.vmap(self._determinants[i], in_axes=-1, out_axes=-1)(y) # (B, T)
                sd = sd + jnp.sum(jnp.exp(log_sd_i), axis=-1)

            # Compute log amplitudes
            log_psi = jnp.log(jnp.sinh(sd)) #(B,)
        elif self.symmetrization == 'projective':
            # Evaluate Slater determinants
            def evaluate_determinants(y_t):
                sd_t = jnp.zeros(x.shape[0])
                for i in range(self.n_determinants):
                    log_sd_i = self._determinants[i](y_t)
                    sd_t = sd_t + jnp.exp(log_sd_i)
                return jnp.sinh(sd_t) # (B,)
            sd = jax.vmap(evaluate_determinants, in_axes=-1, out_axes=-1)(y) # (B, T)

            # Compute log amplitudes
            log_psi = jnp.log(jnp.sum(sd, axis=-1)) # (B,)

        return log_psi

class ASymmqGPSProd(nn.Module):
    hilbert: HomogeneousHilbert
    n_determinants: int
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)


    # Dimensions:
    # - B = batch size
    # - L = number of sites
    # - N = total number of electrons
    # - N_up = number of spin-up electrons
    # - N_down = number of spin-down electrons
    # - M = number of determinants
    # - T = number of symmetries

    def setup(self):
        assert self.n_determinants % 2 != 0
        self._determinants = [
            SlaterDeterminant(self.hilbert.size, self.hilbert._n_elec) for _ in range(self.n_determinants)
        ]

    def __call__(self, x) -> Array:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        x = jnp.asarray(x, jnp.int32) # (B, L)

        # Convert input configs from 2nd quantized representation x
        # to 1st quantized representation y:
        # x=[x_1, x_2, ..., x_L]
        #                |
        #                v
        # y=(y_1, y_2, ..., y_{N_up}, y_{N_up+1}, y_{N_up+2}, ..., y_{N_up+N_down})

        y = occupancies_to_electrons(x, self.hilbert._n_elec) # (B, N)

        # Apply symmetry transformations
        y = self.apply_symmetries(y) # (B, N, T)

        # Evaluate Slater determinants
        def evaluate_determinants(y_t):
            sd_t = jnp.ones(x.shape[0])
            for i in range(self.n_determinants):
                log_sd_i = self._determinants[i](y_t)
                sd_t = sd_t * jnp.sinh(jnp.exp(log_sd_i))
            return sd_t # (B,)
        sd = jax.vmap(evaluate_determinants, in_axes=-1, out_axes=-1)(y) # (B, T)

        # Compute log amplitudes
        log_psi = jnp.log(jnp.sum(sd, axis=-1)) # (B,)

        return log_psi