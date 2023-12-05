import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
from netket.utils.types import NNInitFunc, Array, DType
from netket.hilbert.homogeneous import HomogeneousHilbert
from GPSKet.nn.initializers import normal
from GPSKet.hilbert import FermionicDiscreteHilbert


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
        # TODO: extend this to work with spin models
        assert isinstance(self.hilbert, FermionicDiscreteHilbert)
        self.L = self.hilbert.size
        self.local_dim = self.hilbert.local_size
        self.max_up, self.max_dn = self.hilbert._n_elec

    @nn.compact
    def __call__(self, inputs) -> Array:
        # TODO: add support for fast updating
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

        # Count up- and down-spin electrons up to each site (batch_size, local_dim, L)
        n_up = jnp.cumsum(indices[:, :-1] & 1, axis=-1)
        n_dn = jnp.cumsum((indices[:, :-1] & 2) // 2, axis=-1)
        counts = jnp.stack([n_up, n_dn], axis=1)
        counts = jnp.pad(counts, ((0, 0), (0, 0), (1, 0)), mode='constant')

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