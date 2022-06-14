import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from netket.utils import HashableArray
from netket.utils.types import NNInitFunc, Array, DType, Callable
from typing import Tuple, Union
from netket.hilbert.homogeneous import HomogeneousHilbert
from qGPSKet.nn.initializers import normal

import warnings

# helper function to get the symmetry transformation functions for spin systems
def get_sym_transformation_spin(graph, automorphisms=True, spin_flip=True):
    if automorphisms:
        syms = graph.automorphisms().to_array().T
        inv_syms = np.zeros(syms.shape, dtype=syms.dtype)
        for i in range(syms.shape[0]):
            for j in range(syms.shape[1]):
                inv_syms[syms[i,j], j] = i
        syms = jnp.array(syms)
        inv_syms = jnp.array(inv_syms)
        if spin_flip:
            def symmetries(samples):
                out = jnp.take(samples, syms, axis=-1)
                out = jnp.concatenate((out, 1-out), axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.concatenate((inv_syms[indices], inv_syms[indices]), axis=-1)
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1])
                inv_sym_occs = jnp.concatenate((inv_sym_occs, 1-inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites

        else:
            def symmetries(samples):
                out = jnp.take(samples, syms, axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = inv_syms[indices]
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1])
                return inv_sym_occs, inv_sym_sites
    else:
        if spin_flip:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                out = jnp.concatenate((out, 1-out), axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_sites = jnp.concatenate((inv_sym_sites, inv_sym_sites), axis=-1)
                inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
                inv_sym_occs = jnp.concatenate((inv_sym_occs, 1-inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites
        else:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
                return inv_sym_occs, inv_sym_sites
    return (symmetries, inv_symmetries)

# default syms function (no symmetries)
def no_syms():
    symmetries = lambda samples : jnp.expand_dims(samples, axis=-1)
    inv_sym = lambda sample_at_indices, indices : (jnp.expand_dims(sample_at_indices, axis=-1), jnp.expand_dims(indices, axis=-1))
    return (symmetries, inv_sym)

# TODO: the framework for the symmetrisation could (and should) definitely be improved at one point
class qGPS(nn.Module):
    # TODO: add documentation
    hilbert: HomogeneousHilbert
    M: int
    dtype: DType = jnp.complex128
    init_fun: NNInitFunc = normal()
    to_indices: Callable = lambda samples : samples.astype(jnp.uint8)
    """
    syms is a tuple of two function representing the symmetry operations.
    the first function creates all symmetrically equivalent copies of the test configuration
    (represented by the indices into the epsilon tensor).
    The second function is optional but is required if fast updating is performed.
    It takes two arguments, a single configuration (i.e. a list of indices into the epsilon tensor),
    as well as a list of site indices indicating which sites have changed the occupancy.
    It returns a tuple of two arrays.
    The first returned array should represent the occupancies of all symmetrically equivalent configurations at the updated positions.
    The second array returns the transformed site indices for all symmetry operations.
    For all returned arrays of the syms function, the last dimension corresponds to the total number of symmetry operations.
    """
    syms: Union[Callable, Tuple[Callable, Callable]] = no_syms()
    out_transformation: Callable = lambda argument : jnp.sum(argument, axis=(-2,-1))
    apply_fast_update: bool = True

    def setup(self):
        if type(self.syms) == tuple:
            self.symmetries = self.syms[0]
            self.symmetries_inverse = self.syms[1]
        else:
            self.symmetries = self.syms
            assert (not self.apply_fast_update)
        self.L = self.hilbert.size
        self.local_dim = self.hilbert.local_size

    """
    Note: It might be cleaner to use the `intermediates` interface provided by flax
    to cache intermediate values. However, that stacks intermediates from multiple
    calls by default so these could not be fed back into the model without overwritting
    this behaviour. In order to avoid this pitfall in the model definition, we thus
    use our own interface to store intermediate values which can be fed back into the
    model as variables (as required for fast wavefunction updates).
    """
    @nn.compact
    def __call__(self, inputs, cache_intermediates=False, update_sites=None):

        indices = self.to_indices(inputs)

        epsilon = self.param("epsilon", self.init_fun, (self.local_dim, self.M, self.L), self.dtype)

        # Register the cache variables
        if update_sites is not None or cache_intermediates:
            saved_configs = self.variable("intermediates_cache", "samples", lambda : None)
            saved_site_product = self.variable("intermediates_cache", "site_prod", lambda : None)

        if update_sites is not None:
            indices_save = saved_configs.value
            old_samples = jax.vmap(jnp.take, in_axes=(0, 0), out_axes=0)(indices_save, update_sites)

            def inner_site_product_update(site_prod_old, new_occs, old_occs, sites):
                site_prod_new = site_prod_old / (epsilon[old_occs,:,sites].prod(axis=0))
                site_prod_new = site_prod_new * (epsilon[new_occs,:,sites].prod(axis=0))
                return site_prod_new

            def outer_site_product_update(site_prod_old, sample_new, sample_old, update_sites):
                inv_sym_new, inv_sym_sites = self.symmetries_inverse(sample_new, update_sites)
                inv_sym_old, inv_sym_sites = self.symmetries_inverse(sample_old, update_sites)
                return jax.vmap(inner_site_product_update, in_axes=(-1, -1, -1, -1), out_axes=-1)(site_prod_old, inv_sym_new, inv_sym_old, inv_sym_sites)

            site_product_old = saved_site_product.value
            site_product = jax.vmap(outer_site_product_update, in_axes=(0, 0, 0, 0), out_axes=0)(site_product_old, indices, old_samples, update_sites)
        else:
            def evaluate_site_product(sample):
                return jnp.take_along_axis(epsilon, sample, axis=0).prod(axis=-1).reshape(-1)

            def get_site_prod(sample):
                return jax.vmap(evaluate_site_product, in_axes=-1, out_axes=-1)(self.symmetries(sample))

            transformed_samples = jnp.expand_dims(indices, (1, 2)) # required for the inner take_along_axis

            site_product = jax.vmap(get_site_prod)(transformed_samples)

        if cache_intermediates:
            saved_site_product.value = site_product
            if update_sites is not None:
                def update_fun(saved_config, update_sites, occs):
                    def scan_fun(carry, count):
                        return (carry.at[update_sites[count]].set(occs[count]), None)
                    return jax.lax.scan(scan_fun, saved_config, jnp.arange(update_sites.shape[0]), reverse=True)[0]
                full_samples = jax.vmap(update_fun, in_axes=(0, 0, 0), out_axes=0)(indices_save.value, update_sites, indices)
            else:
                full_samples = indices

            saved_configs.value = full_samples

        return self.out_transformation(site_product)
