import numpy as np
import jax
from jax.nn.initializers import zeros
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Union, Optional, Tuple, List
from netket.utils import HashableArray
from netket.utils.types import DType, NNInitFunc, Callable, Array
from flax import linen as nn
from GPSKet.nn.initializers import normal
from GPSKet.models import qGPS
from GPSKet.models.qGPS import no_syms
from .autoreg_qGPS import _normalize, gpu_cond, AbstractARqGPS


class ARqGPSFull(AbstractARqGPS):
    """
    Implements the fully variational autoregressive formulation of the QGPS Ansatz,
    with support for symmetries and Hilbert spaces constrained to the
    zero magnetization sector.
    """

    M: Union[int, HashableArray] # If M is a list, it defines a per-site support dimension -> this should be faster to evaluate but gives a significant compilation overhead
    """Bond dimension"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    machine_pow: int = 2
    """Exponent required to normalize the output"""
    init_fun: Optional[NNInitFunc] = None # Defaults to qGPS-normal with the parameter dtype
    """Initializer for the variational parameters"""
    normalize: bool=True
    """Whether the Ansatz should be normalized"""
    apply_symmetries: Union[Callable, Tuple[Callable, Callable]] = no_syms()
    """
    Function to apply symmetries to configurations (see qGPS model definition
    for an explanation of the tuple also specifying the inverse symmetry operation
    for fast updating)
    """
    # TODO: extend to cases beyond D=2
    count_spins: Callable = lambda spins : jnp.stack([(spins+1)&1, ((spins+1)&2)/2], axis=-1).astype(jnp.int32)
    """Function to count down and up spins"""
    # TODO: extend to cases where total_sz != 0
    renormalize_log_psi: Callable = lambda n_spins, hilbert, index: jnp.log(jnp.heaviside(hilbert.size//2-n_spins, 0))
    """Function to renormalize conditional log probabilities"""
    out_transformation: Callable=lambda argument: jnp.sum(argument, axis=-1)
    """Function of the output layer, by default sums over bond dimension"""
    apply_fast_update: bool = True
    """Whether or not to apply the fast updating in the model"""

    # Dimensions:
    # - B = batch size
    # - D = local dimension
    # - L = number of sites
    # - M = bond dimension
    # - T = number of symmetries

    def _conditional(self, inputs: Array, index: int) -> Array:
        # Convert input configurations into indices
        inputs = self.hilbert.states_to_local_indices(inputs) # (B, L)

        # Compute conditional probability for site at index
        log_psi = _conditional(self, inputs, index) # (B, D)
        if self.normalize:
            log_psi = _normalize(log_psi, self.machine_pow)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p

    def conditionals(self, inputs: Array) -> Array:
        # Convert input configurations into indices
        inputs = self.hilbert.states_to_local_indices(inputs) # (B, L)

        # Compute conditional probabilities for all sites
        log_psi, _ = _conditionals(self, inputs) # (B, L, D)
        if self.normalize:
            log_psi = _normalize(log_psi, self.machine_pow)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p

    def setup(self):
        if self.init_fun is None:
            init = normal(dtype=self.dtype)
        else:
            init = self.init_fun
        if isinstance(self.M, HashableArray):
            self._epsilon  = tuple([self.param("epsilon_{}".format(i), init, (self.hilbert.local_size, np.asarray(self.M)[i], i+1), self.dtype) for i in range(self.M.shape[0])])
        else:
            self._epsilon  = self.param("epsilon", init, (self.hilbert.local_size, self.M, int(self.hilbert.size * (self.hilbert.size + 1)/2)), self.dtype)
        if self.apply_fast_update:
            self._saved_configs = self.variable("intermediates_cache", "samples", lambda : None)
            if isinstance(self.M, HashableArray):
                self._saved_context_product  = tuple([self.variable("intermediates_cache", "context_prod_{}".format(i), lambda : None) for i in range(self.M.shape[0])])
            else:
                self._saved_context_product = self.variable("intermediates_cache", "context_prod", lambda : None)
        if self.hilbert.constrained:
            self._n_spins = self.variable("cache", "spins", zeros, None, (1, self.hilbert.local_size))
        if self.apply_fast_update:
            # We can only apply the fast-updating of we have the inverse symmetry operation function as well
            assert (type(self.apply_symmetries) == tuple)

    def __call__(self, inputs: Array, cache_intermediates=False, update_sites=None) -> Array:
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0) # (B, L)

        # Generate the full configurations from the partial ones and get inverse symmetries if a fast update is performed
        if update_sites is not None:
            # The old occupancies
            saved_input = self._saved_configs.value

            # Old occupancies at the updated sites
            prev_occupancies = jax.vmap(jnp.take, in_axes=(0, 0), out_axes=0)(saved_input, update_sites)

            # Compute a tuple containing the transformed occupancy and the site to index the epsilon tensor for each symmetry operation

            # Old occupancy
            old_occupancy, site_indices = self.apply_symmetries[1](prev_occupancies, update_sites) # (B, #updates, T), (B, #updates, T)

            # Updated occupancy
            new_occupancy, site_indices = self.apply_symmetries[1](inputs, update_sites) # (B, #updates, T), (B, #updates, T)

            update_args = (self.hilbert.states_to_local_indices(old_occupancy), self.hilbert.states_to_local_indices(new_occupancy), site_indices)

            def update_fun(saved_config, update_sites, occs):
                def scan_fun(carry, count):
                    return (carry.at[update_sites[count]].set(occs[count]), None)
                return jax.lax.scan(scan_fun, saved_config, jnp.arange(update_sites.shape[0]), reverse=True)[0]
            full_samples = jax.vmap(update_fun, in_axes=(0, 0, 0), out_axes=0)(self._saved_configs.value, update_sites, inputs)
        else:
            update_args = None
            full_samples = inputs

        # Transform inputs according to symmetries
        if type(self.apply_symmetries) == tuple:
            inputs = self.apply_symmetries[0](inputs) # (B, L, T)
            full_samples_sym = self.apply_symmetries[0](full_samples) # (B, L, T)
        else:
            inputs = self.apply_symmetries(inputs) # (B, L, T)
            full_samples_sym = self.apply_symmetries(full_samples) # (B, L, T)
        n_symm = inputs.shape[-1]

        # Convert input configurations into indices
        inputs = self.hilbert.states_to_local_indices(inputs) # (B, L, T)
        full_samples_sym = self.hilbert.states_to_local_indices(full_samples_sym) # (B, L, T)
        batch_size = inputs.shape[0]

        # Compute conditional log-probabilities
        if update_sites is not None:
            if isinstance(self._epsilon, tuple):
                old_contexts = [cont_prod.value for cont_prod in self._saved_context_product]
            else:
                old_contexts = self._saved_context_product.value

            log_psi, context_products = jax.vmap(_conditionals, in_axes=(None, -1, -1, -1), out_axes=(-1, -1))(self, full_samples_sym, update_args, old_contexts) # (B, L, D, T), (L, B, M, T)
        else:
            log_psi, context_products = jax.vmap(_conditionals, in_axes=(None, -1, None, None), out_axes=(-1, -1))(self, full_samples_sym, None, None) # (B, L, D, T), (L, B, M, T)
        if self.normalize:
            log_psi = _normalize(log_psi, self.machine_pow, axis=-2)

        # Take conditionals along sites-axis according to input indices
        log_psi = jnp.take_along_axis(log_psi, jnp.expand_dims(full_samples_sym, axis=2), axis=2) # (B, L, 1, T)
        log_psi = jnp.sum(log_psi, axis=1) # (B, 1, T)
        log_psi = jnp.reshape(log_psi, (batch_size, n_symm)) # (B, T)

        # Compute symmetrized log-amplitudes
        log_psi_symm_re = (1/self.machine_pow)*logsumexp(self.machine_pow*log_psi.real, axis=-1, b=1/n_symm)
        log_psi_symm_im = logsumexp(1j*log_psi.imag, axis=-1).imag
        log_psi_symm = log_psi_symm_re+1j*log_psi_symm_im

        if cache_intermediates:
            if isinstance(self._epsilon, tuple):
                for i in range(len(context_products)):
                    self._saved_context_product[i].value = context_products[i]
            else:
                self._saved_context_product.value = context_products
            self._saved_configs.value = full_samples

        return log_psi_symm # (B,)

def _compute_conditional(model: ARqGPSFull, n_spins: Array, inputs: Array, index: int,
                         update_args: Optional[Tuple[Array, Array, Array]]=None,
                         saved_context_prod: Optional[Array]=None) -> Union[Array, Array, Array]:
    if isinstance(model._epsilon, tuple):
        # Currently, We want this function to be callable with index < 0 for _init_cache function
        if index < 0:
            proper_index = 0
        else:
            proper_index = index
        input_param = model._epsilon[proper_index][:,:,-1]
    else:
        # Get the epsilon sub-tensor for the current index
        lower_index = (index * (index+1))//2

        # Retrieve input parameters
        input_param = model._epsilon[:,:,lower_index+index]

    input_param = jnp.expand_dims(input_param, axis=0) # (1, D, M)

    if update_args is None:
        if isinstance(model._epsilon, tuple):
            inputs = inputs[:,:proper_index+1]
            local_epsilon = model._epsilon[proper_index]
        else:
            local_epsilon = jax.lax.dynamic_slice_in_dim(model._epsilon, lower_index, model.hilbert.size, axis=-1)
        # Compute product of parameters over j<index
        context_param = jnp.expand_dims(local_epsilon, axis=0) # (1, D, M, L)
        inputs_expanded = jnp.expand_dims(inputs, axis=(1,2)) # (B, 1, 1, L)
        context_val = jnp.take_along_axis(context_param, inputs_expanded, axis=1).reshape((-1, *local_epsilon.shape[-2:])) # (B, M, L)

        # Apply masking for sites > index
        context_val = jnp.where(jnp.arange(local_epsilon.shape[-1]) >= index, 1., context_val)
        context_prod = jnp.prod(context_val, axis=-1) # (B, M)
    else:
        old_occupancy, new_occupancy, site_indices = update_args
        if isinstance(model._epsilon, tuple):
            update = (model._epsilon[index][new_occupancy,:,site_indices])
            update /= (model._epsilon[index][old_occupancy,:,site_indices])
        else:
            update = (model._epsilon[new_occupancy,:,site_indices+lower_index])
            update /= (model._epsilon[old_occupancy,:,site_indices+lower_index])
        # Apply masking for sites > index
        update = jnp.where(jnp.expand_dims(site_indices >= index, axis=-1), 1., update)
        context_prod = update.prod(axis=1) * saved_context_prod

    site_prod = input_param * jnp.expand_dims(context_prod, axis=1) # (B, D, M)

    # Compute log conditional probabilities
    log_psi = model.out_transformation(site_prod) # (B, D)

    # Slice inputs at index-1 to count previous spins
    inputs_i = inputs[:, index-1] # (B,)
    # Update spins count if index is larger than 0, otherwise leave as is
    n_spins = gpu_cond(
        index > 0,
        lambda n_spins: n_spins + model.count_spins(inputs_i),
        lambda n_spins: n_spins,
        n_spins
    )

    # If Hilbert space associated with the model is constrained, i.e.
    # model has "n_spins" in "cache" collection, then impose total magnetization.
    # This is done by counting number of up/down spins until index, then if
    # n_spins is >= L/2 the probability of up/down spin at index should be 0,
    # i.e. the log probability becomes -inf
    log_psi = gpu_cond(
        index >= 0,
        lambda log_psi: log_psi+model.renormalize_log_psi(n_spins, model.hilbert, index),
        lambda log_psi: log_psi,
        log_psi
    )
    return n_spins, log_psi, context_prod

def _conditional(model: ARqGPSFull, inputs: Array, index: int) -> Array:
    # Retrieve spins count
    batch_size = inputs.shape[0]
    if model.has_variable("cache", "spins"):
        n_spins = model._n_spins.value
        n_spins = jnp.asarray(n_spins, jnp.int32)
        n_spins = jnp.resize(n_spins, (batch_size, model.hilbert.local_size)) # (B, D)
    else:
        n_spins = jnp.zeros((batch_size, model.hilbert.local_size), jnp.int32)

    # Compute log conditional probabilities
    n_spins, log_psi, _ = _compute_conditional(model, n_spins, inputs, index)

    # Update model cache
    if model.has_variable("cache", "spins"):
        model._n_spins.value = n_spins
    return log_psi # (B, D)

def _conditionals(model: ARqGPSFull, inputs: Array, update_args: Optional[Tuple[Tuple[Array, Array],
                  Tuple[Array, Array]]]=None, saved_context_product: Optional[Array]=None) -> Array:
    # Loop over sites while computing log conditional probabilities
    def _scan_fun(n_spins, index):
        if saved_context_product is not None:
            if isinstance(model._epsilon, tuple):
                n_spins, log_psi, context_product = _compute_conditional(model, n_spins, inputs, index, update_args, saved_context_product[index][:, :])
            else:
                n_spins, log_psi, context_product = _compute_conditional(model, n_spins, inputs, index, update_args, saved_context_product[index, :, :])
        else:
            n_spins, log_psi, context_product = _compute_conditional(model, n_spins, inputs, index)
        n_spins = gpu_cond(
            model.hilbert.constrained,
            lambda n_spins: n_spins,
            lambda n_spins: jnp.zeros_like(n_spins),
            n_spins
        )
        return n_spins, (log_psi, context_product)

    batch_size = inputs.shape[0]
    n_spins = jnp.zeros((batch_size, model.hilbert.local_size), jnp.int32)
    indices = jnp.arange(model.hilbert.size)
    if isinstance(model._epsilon, tuple):
        log_psi = None
        context_product = None
        for i in range(len(indices)):
            n_spins, value = _scan_fun(n_spins, i)
            if log_psi is None:
                log_psi = jnp.expand_dims(value[0], axis=0)
            else:
                log_psi = jnp.append(log_psi, jnp.expand_dims(value[0], axis=0), axis=0)
            if context_product is None:
                context_product = (value[1],)
            else:
                context_product = (*context_product, value[1])
    else:
        _, value = jax.lax.scan(
            _scan_fun,
            n_spins,
            indices
        )
        log_psi, context_product = value
    log_psi = jnp.transpose(log_psi, [1, 0, 2])
    return log_psi, context_product # (B, L, D), (L, B, M)

class ARqGPSModPhaseFull(ARqGPSFull):
    """
    Implements an Ansatz composed of an autoregressive qGPS for the modulus of the amplitude and a qGPS for the phase.
    """

    def setup(self):
        assert jnp.issubdtype(self.dtype, jnp.floating)
        super().setup()
        self._qgps = qGPS(
            self.hilbert, self.hilbert.size,
            dtype=jnp.float64,
            init_fun=self.init_fun)

    def __call__(self, inputs: Array) -> Array:
        log_psi_mod = super().__call__(inputs)
        log_psi_phase = self._qgps(inputs)
        return log_psi_mod + log_psi_phase*1j