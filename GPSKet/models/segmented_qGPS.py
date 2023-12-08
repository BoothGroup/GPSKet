import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
from netket.utils.types import NNInitFunc, Array, DType
from netket.hilbert import Spin
from netket.hilbert.homogeneous import HomogeneousHilbert
from GPSKet.nn.initializers import normal
from GPSKet.hilbert import FermionicDiscreteHilbert


@jax.jit
def count_spins(indices):
    """
    Counts the number of up- and down-spins up to each site
    in a batch of configurations, where:
        - 0: down-spin
        - 1: up-spin

    Args:
        - configs : batch of configurations (batch_size, L)

    Returns:
        - counts : number of up- and down-spins in configs (batch_size, 2, L)
    """
    n_up = jnp.cumsum(((indices[:, :-1] + 1) & 2) // 2, axis=-1)
    n_dn = jnp.cumsum((indices[:, :-1] + 1) & 1, axis=-1)
    counts = jnp.stack([n_up, n_dn], axis=1)
    counts = jnp.pad(counts, ((0, 0), (0, 0), (1, 0)), mode='constant')
    return counts

@jax.jit
def count_fermions(configs):
    """
    Counts the number of spin-up and spin-down electrons up to each orbital
    in a batch of configurations in 2nd quantization representation where
    occupation of the orbitals is described as:
        - 0: unoccupied
        - 1: singly-occupied with a spin-up electron
        - 2: singly-occupied with a spin-down electron
        - 3: doubly-occupied

    Args:
        - configs : batch of configurations (batch_size, L)

    Returns:
        - counts : number of spin-up and spin-down electrons in configs (batch_size, 2, L)
    """
    n_up = jnp.cumsum(configs[:, :-1] & 1, axis=-1)
    n_dn = jnp.cumsum((configs[:, :-1] & 2) // 2, axis=-1)
    counts = jnp.stack([n_up, n_dn], axis=1)
    counts = jnp.pad(counts, ((0, 0), (0, 0), (1, 0)), mode='constant')
    return counts

class SegGPS(nn.Module):
    """
    Implements a GPS Ansatz inspired by segmented basis sets in Quantum Chemistry, where the amplitudes for
    local states at the i-th site depends on the number of particles seen before in a predefined ordering
    """
    hilbert: HomogeneousHilbert
    M: int
    dtype: DType = jnp.complex128
    init_fun: Optional[
        NNInitFunc
    ] = None


    def setup(self):
        self.L = self.hilbert.size
        self.local_dim = self.hilbert.local_size
        if isinstance(self.hilbert, FermionicDiscreteHilbert):
            self.max_up, self.max_dn = self.hilbert._n_elec
            self._count_fn = count_fermions
        elif isinstance(self.hilbert, Spin) and int(2*self.hilbert._s) == 1:
            if self.L % 2 == 0:
                l = int(self.hilbert._total_sz)
                L_half = int(self.L) // 2
                self.max_up = L_half + l
                self.max_dn = L_half - l
            else:
                m = int(2 * self.hilbert._total_sz)
                L_half = int(self.L - abs(m)) // 2
                if m > 0:
                    self.max_up = L_half + m
                    self.max_dn = L_half
                else:
                    self.max_up = L_half
                    self.max_dn = L_half - m
            self._count_fn = count_spins
        else:
            raise ValueError(f"Hilbert spaces of type {type(self.hilbert)} are not supported yet.")

    @nn.compact
    def __call__(self, inputs) -> Array:
        indices = self.hilbert.states_to_local_indices(inputs)

        if self.init_fun is None:
            init = normal(dtype=self.dtype)
        else:
            init = self.init_fun

        # NOTE: might be implementable at some point as a ragged tensor
        # (see: https://github.com/google/jax/issues/17863)
        epsilon = self.param(
            "epsilon", init, (self.local_dim, self.M, self.L, self.max_up+1, self.max_dn+1), self.dtype
        )

        # Count spin-up and spin-down particles up to each site
        counts = self._count_fn(indices)

        # Evaluate site products
        def evaluate_site_product(sample, count):
            values = jnp.take_along_axis(epsilon, jnp.expand_dims(sample, (0, 1, 3, 4)), axis=0)
            values = jnp.take_along_axis(values, jnp.expand_dims(count[0], (0, 1, 3, 4)), axis=3)
            values = jnp.take_along_axis(values, jnp.expand_dims(count[1], (0, 1, 3, 4)), axis=4)
            site_product = jnp.prod(values, axis=2)
            return jnp.reshape(site_product, -1)
        site_products = jax.vmap(evaluate_site_product)(indices, counts)

        # Compute log-amplitudes
        log_psi = jnp.sum(site_products, axis=-1)
        return log_psi