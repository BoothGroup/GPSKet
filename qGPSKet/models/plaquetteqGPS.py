import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from netket.utils import HashableArray
from netket.utils.types import NNInitFunc, Array, DType, Callable
from typing import Tuple, Union
from netket.hilbert.homogeneous import HomogeneousHilbert
from qGPSKet.nn.initializers import normal

from qGPSKet.models.qGPS import no_syms


import warnings


# TODO: Improve symmetrisation framework (same as for qGPS)
class PlaquetteqGPS(nn.Module):
    # TODO: add documentation
    hilbert: HomogeneousHilbert
    M: int
    plaquettes: HashableArray
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
    out_transformation: Callable = lambda argument : jnp.sum(argument, axis=(-3,-2,-1))
    apply_fast_update: bool = False # Careful with zero-valued parameters here, TODO: implement fallback...

    def setup(self):
        if type(self.syms) == tuple:
            self.symmetries = self.syms[0]
            self.symmetries_inverse = self.syms[1]
        else:
            self.symmetries = self.syms
            if self.apply_fast_update:
                warnings.warn("Attention! Fast updating is not applied in qGPS as the inverse symmetry operations are not supplied.")
            self.apply_fast_update = False
        self.L = self.hilbert.size
        self.local_dim = self.hilbert.local_size
        plaquettes = np.array(self.plaquettes)
        inv_plaquette_ids = -1 * np.ones((plaquettes.shape[0], self.L), dtype=int)
        for i in range(inv_plaquette_ids.shape[0]):
            for j in range(plaquettes.shape[1]):
                inv_plaquette_ids[i, plaquettes[i,j]] = j
        self.inv_plaquette_ids = HashableArray(inv_plaquette_ids)


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

        epsilon = self.param("epsilon", self.init_fun, (self.local_dim, self.plaquettes.shape[0], self.M, self.plaquettes.shape[1]), self.dtype)

        plaquettes = np.asarray(self.plaquettes)
        inv_plaquette_ids = np.asarray(self.inv_plaquette_ids)

        if cache_intermediates or (update_sites is not None):
            site_product_save = self.variable("intermediates_cache", "site_prod", lambda : jnp.zeros(0, dtype=self.dtype))
            indices_save = self.variable("intermediates_cache", "samples", lambda : jnp.zeros(0, dtype=indices.dtype))

        if update_sites is None:
            def evaluate_site_product(sample):
                def single_plaquette_eval(plaquette_ids, epsi):
                    result = jnp.take_along_axis(epsi, sample[:,:,plaquette_ids], axis=0).prod(axis=-1)
                    return result.reshape(-1)
                return jax.vmap(single_plaquette_eval, in_axes=(0, 1), out_axes=0)(plaquettes, epsilon)

            def get_site_prod(sample):
                return jax.vmap(evaluate_site_product, in_axes=-1, out_axes=-1)(self.symmetries(sample))

            transformed_samples = jnp.expand_dims(indices, (1, 2)) # required for the inner take_along_axis
            site_product = jax.vmap(get_site_prod)(transformed_samples)
        else:
            site_product_old = site_product_save.value
            new_samples = indices
            old_samples = jax.vmap(jnp.take, in_axes=(0, 0), out_axes=0)(indices_save.value, update_sites)

            def inner_site_product_update(site_prod_old, new_occs, old_occs, sites):
                def single_plaquette_update(sp_old, epsi, inv_plaquette):
                    transformed_sites = inv_plaquette[sites]
                    valid_elements = transformed_sites != -1
                    update = jnp.where(jnp.expand_dims(valid_elements,1), x=epsi[new_occs, :, transformed_sites], y=1.).prod(axis=0)
                    update /= jnp.where(jnp.expand_dims(valid_elements,1), x=epsi[old_occs, :, transformed_sites], y=1.).prod(axis=0)
                    return sp_old * update
                return jax.vmap(single_plaquette_update, in_axes=(0, 1, 0), out_axes=0)(site_prod_old, epsilon, inv_plaquette_ids)

            def outer_site_product_update(site_prod_old, sample_new, sample_old, update_sites):
                inv_sym_new, inv_sym_sites = self.symmetries_inverse(sample_new, update_sites)
                inv_sym_old, inv_sym_sites = self.symmetries_inverse(sample_old, update_sites)

                return jax.vmap(inner_site_product_update, in_axes=(-1, -1, -1, -1), out_axes=-1)(site_prod_old, inv_sym_new, inv_sym_old, inv_sym_sites)

            site_product = jax.vmap(outer_site_product_update, in_axes=(0, 0, 0, 0), out_axes=0)(site_product_old, new_samples, old_samples, update_sites)

        if cache_intermediates:
            site_product_save.value = site_product
            if update_sites is not None:
                def update_fun(saved_config, update_sites, occs):
                    return saved_config.at[update_sites].set(occs)
                indices_save.value = jax.vmap(update_fun, in_axes=(0, 0, 0), out_axes=0)(indices_save.value, update_sites, indices)
            else:
                indices_save.value = indices

        # site_product has dim N_batch x number of plaquettes x M x Number of syms
        return self.out_transformation(site_product)