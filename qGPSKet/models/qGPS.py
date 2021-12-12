import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from netket.utils import HashableArray
from netket.utils.types import NNInitFunc, Array, DType, Callable
from typing import Tuple, Union
from qGPSKet.nn.initializers import normal


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
            def inv_symmetries(sample, indices):
                inv_sym_sites = jnp.concatenate((inv_syms[indices], inv_syms[indices]), axis=-1)
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample[indices], axis=-1), syms.shape[1])
                inv_sym_occs = jnp.concatenate((inv_sym_occs, 1-inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites

        else:
            def symmetries(samples):
                out = jnp.take(samples, syms, axis=-1)
                return out
            def inv_symmetries(sample, indices):
                inv_sym_sites = inv_syms[indices]
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample[indices], axis=-1), syms.shape[1])
                return inv_sym_occs, inv_sym_sites
    else:
        if spin_flip:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                out = jnp.concatenate((out, 1-out), axis=-1)
                return out
            def inv_symmetries(sample, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_sites = jnp.concatenate((inv_sym_sites, inv_sym_sites), axis=-1)
                inv_sym_occs = jnp.expand_dims(sample[indices], axis=-1)
                inv_sym_occs = jnp.concatenate((inv_sym_occs, 1-inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites
        else:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                return out
            def inv_symmetries(sample, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_occs = jnp.expand_dims(sample[indices], axis=-1)
                return inv_sym_occs, inv_sym_sites
    return (symmetries, inv_symmetries)

# default syms function (no symmetries)
def no_syms():
    symmetries = lambda samples : jnp.expand_dims(samples, axis=-1)
    inv_sym = lambda sample, indices : (jnp.expand_dims(jnp.take(sample, indices), axis=-1), jnp.expand_dims(indices, axis=-1))
    return (symmetries, inv_sym)

# TODO: the framework for the symmetrisation could (and should) definitely be improved at one point
class qGPS(nn.Module):
    # TODO: add documentation
    M: int
    local_dim: int = 2
    dtype: DType = jnp.complex128
    init_fun: NNInitFunc = normal(dtype=dtype)
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
    For all returned arrays of the syms function, the last dimension corresponds to the totl number of symmetry operations.  
    """
    syms: Union[Callable, Tuple[Callable, Callable]] = no_syms()
    before_sym_op: Callable = lambda argument : argument
    final_op: Callable = lambda argument : argument

    def setup(self):
        if type(self.syms) == tuple:
            self.symmetries = self.syms[0]
            self.symmetries_inverse = self.syms[1]
            self.fast_update = True
        else:
            self.symmetries = self.syms
            self.fast_update = False

    @nn.compact
    def __call__(self, inputs, save_site_prod=False, update_sites=None):
        if len(inputs.shape) == 1:
            inputs = jnp.expand_dims(inputs, 0)

        epsilon = self.param("epsilon", self.init_fun, (self.local_dim, self.M, inputs.shape[-1]), self.dtype)

        indices = self.to_indices(inputs)

        if save_site_prod or update_sites is not None:
            site_product_save = self.variable("workspace", "site_prod", lambda : jnp.zeros(0, dtype=self.dtype))
            indices_save = self.variable("workspace", "samples", lambda : jnp.zeros(0, dtype=indices.dtype))

        if update_sites is None or not self.fast_update:
            transformed_samples = self.symmetries(indices)

            # TODO: maybe this can be improved
            def take_site_product(indices, epsilon):
                return jnp.take_along_axis(epsilon, indices, axis=0).prod(axis=-1).reshape(-1)

            batched = jax.vmap(take_site_product, (0, None), 0)
            batched = jax.vmap(batched, (-1, None), -1)

            transformed_samples = jnp.expand_dims(transformed_samples, (-4, -3))
            site_prod = batched(transformed_samples, epsilon)
        else:
            if len(update_sites.shape) == 1:
                update_sites = jnp.expand_dims(update_sites, 0)
            site_prod = site_product_save.value
            new_samples = indices
            old_samples = indices_save.value

            def inner_site_product_update(site_prod, new_occs, old_occs, sites):
                site_prod_new = site_prod / (epsilon[old_occs,:,sites].prod(axis=0))
                site_prod_new = site_prod_new * (epsilon[new_occs,:,sites].prod(axis=0))
                return site_prod_new

            def outer_site_product_update(site_product, sample_new, sample_old, update_sites):
                inv_sym_new, inv_sym_sites = self.symmetries_inverse(sample_new, update_sites)
                inv_sym_old, inv_sym_sites = self.symmetries_inverse(sample_old, update_sites)

                return jax.vmap(inner_site_product_update, in_axes=(-1, -1, -1, -1), out_axes=-1)(site_product, inv_sym_new, inv_sym_old, inv_sym_sites)
            site_prod = jax.vmap(outer_site_product_update, in_axes=(0,0,0,0), out_axes=0)(site_prod, new_samples, old_samples, update_sites)

        if save_site_prod:
            site_product_save.value = site_prod
            indices_save.value = indices

        qGPS_out = self.before_sym_op(site_prod.sum(axis=-2))
        qGPS_out = self.final_op(qGPS_out.sum(axis=-1))
        return qGPS_out
