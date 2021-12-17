import abc
from typing import Tuple, Union
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import linen as nn
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils.types import NNInitFunc, Array, DType, Callable
from jax.nn.initializers import zeros, ones
import numpy as np
from qGPSKet.nn.initializers import normal


class AbstractARqGPS(nn.Module):
    """
    Base class for autoregressive qGPS.

    Subclasses must implement the methods `__call__` and `conditionals`.
    They can also override `_conditional` to implement the caching for fast autoregressive sampling.

    They must also implement the field `machine_pow`,
    which specifies the exponent to normalize the outputs of `__call__`.
    """

    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous Hilbert spaces are supported."""

    # machine_pow: int = 2 Must be defined on subclasses

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

    def _conditional(self, inputs: Array, index: int) -> Array:
        """
        Computes the conditional probabilities for a site to take a given value.

        It should only be called successively with indices 0, 1, 2, ...,
        as in the autoregressive sampling procedure.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          index: index of the site.

        Returns:
          The probabilities with dimensions (batch, Hilbert.local_size).
        """
        return self.conditionals(inputs)[:, index, :]

    @abc.abstractmethod
    def conditionals(self, inputs: Array) -> Array:
        """
        Computes the conditional probabilities for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).

        Examples:

          >>> import pytest; pytest.skip("skip automated test of this docstring")
          >>>
          >>> p = model.apply(variables, σ, method=model.conditionals)
          >>> print(p[2, 3, :])
          [0.3 0.7]
          # For the 3rd spin of the 2nd sample in the batch,
          # it takes probability 0.3 to be spin down (local state index 0),
          # and probability 0.7 to be spin up (local state index 1).
        """

class ARqGPS(AbstractARqGPS):
    """
    Implements the autoregressive formulation of the QGPS Ansatz,
    with support for symmetries and Hilbert spaces constrained to the
    zero magnetization sector.
    """

    M: int
    """Bond dimension"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    machine_pow: int = 2
    """Exponent required to normalize the output"""
    init_fun: NNInitFunc = normal(dtype=dtype)
    """Initializer for the variational parameters"""
    to_indices: Callable = lambda inputs : inputs.astype(jnp.uint8)
    """Function to convert configurations into indices, e.g. a mapping from {-local_dim/2, local_dim/2}"""
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""

    # Dimensions:
    # - B = batch size
    # - D = local dimension
    # - L = number of sites
    # - M = bond dimension
    # - T = number of symmetries

    def _conditional(self, inputs: Array, index: int) -> Array:
        # Convert input configurations into indices
        inputs = self.to_indices(inputs) # (B, L)
        
        # Compute conditional probability for site at index
        log_psi = _conditional(self, inputs, index) # (B, D)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p

    def conditionals(self, inputs: Array) -> Array:
        # Convert input configurations into indices
        inputs = self.to_indices(inputs) # (B, L)
        
        # Compute conditional probabilities for all sites
        log_psi = _conditionals(self, inputs) # (B, L, D)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p

    def setup(self):
        self._epsilon = self.param("epsilon", self.init_fun, (self.hilbert.local_size, self.M, self.hilbert.size), self.dtype)
        self._cache = self.variable("cache", "inputs", ones, None, (1, self.hilbert.local_size, self.M), self.dtype)
        if self.hilbert.constrained:
            self._n_spins = self.variable("cache", "spins", zeros, None, (1, self.hilbert.local_size))

    def __call__(self, inputs: Array) -> Array:
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0) # (B, L)

        # Convert input configurations into indices
        inputs = self.to_indices(inputs) # (B, L)
        batch_size = inputs.shape[0]

        # Transform inputs according to symmetries
        inputs = self.apply_symmetries(inputs) # (B, L, T)
        n_symm = inputs.shape[-1]

        # Compute conditional log-probabilities
        log_psi = jax.vmap(_conditionals, in_axes=(None, -1), out_axes=-1)(self, inputs) # (B, L, D, T)

        # Take conditionals along sites-axis according to input indices
        log_psi = jnp.take_along_axis(log_psi, jnp.expand_dims(inputs, axis=2), axis=2) # (B, L, 1, T)
        log_psi = jnp.sum(log_psi, axis=1) # (B, 1, T)
        log_psi = jnp.reshape(log_psi, (batch_size, n_symm)) # (B, T)

        # Compute symmetrized log-amplitudes
        log_psi_symm_re = (1/self.machine_pow)*logsumexp(self.machine_pow*log_psi.real, axis=-1, b=1/n_symm)
        log_psi_symm_im = logsumexp(1j*log_psi.imag, axis=-1).imag
        log_psi_symm = log_psi_symm_re+1j*log_psi_symm_im
        return log_psi_symm # (B,)


def _normalize(log_psi: Array, machine_pow: int) -> Array:
    return log_psi - (1/machine_pow)*logsumexp(machine_pow*log_psi.real, axis=-1, keepdims=True)

def _compute_conditional(hilbert: HomogeneousHilbert, cache: Array, n_spins: Array, epsilon: Array, inputs: Array, index: int) -> Union[Tuple, Array]:
    # Slice inputs at index-1 to get cached products
    # (Note: when index=0, it doesn't matter what slice of the cache we take,
    # because it is initialized with ones)
    inputs_i = inputs[:, index-1] # (B,)

    # Compute product of parameters and cache at index along bond dimension
    params_i = jnp.asarray(epsilon, epsilon.dtype)[:, :, index] # (D, M)
    prods = jax.vmap(lambda c, s: params_i*c[s], in_axes=(0, 0))(cache, inputs_i)
    prods = jnp.asarray(prods, epsilon.dtype) # (B, D, M)

    # Update cache if index is positive, otherwise leave as is
    cache = jax.lax.cond(
        index >= 0,
        lambda _: prods,
        lambda _: cache,
        None
    )

    # Compute log conditional probabilities
    log_psi = jnp.sum(prods, axis=-1)

    # Update spins count if index is larger than 0, otherwise leave as is
    # TODO: generalize beyond D=2
    n_spins = jax.lax.cond(
        index > 0,
        lambda n_spins: n_spins + jnp.stack([jnp.abs(inputs_i-1), inputs_i], axis=-1),
        lambda n_spins: n_spins,
        n_spins
    )

    # If Hilbert space associated with the model is constrained, i.e.
    # model has "n_spins" in "cache" collection, then impose total magnetization.  
    # This is done by counting number of up/down spins until index, then if
    # n_spins is >= L/2 the probability of up/down spin at index should be 0,
    # i.e. the log probability becomes -inf
    # TODO: extend to cases where total_sz != 0
    log_psi = jax.lax.cond(
        index >= 0,
        lambda log_psi: log_psi+jnp.log(jnp.heaviside(hilbert.size//2-n_spins, 0)),
        lambda log_psi: log_psi,
        log_psi
    )
    return (cache, n_spins), log_psi

def _conditional(model: ARqGPS, inputs: Array, index: int) -> Array:
    # Retrieve cache
    batch_size = inputs.shape[0]
    cache = model._cache.value
    cache = jnp.asarray(cache, model.dtype)
    cache = jnp.resize(cache, (batch_size, model.hilbert.local_size, model.M)) # (B, D, M)

    # Retrieve spins count
    if model.has_variable("cache", "spins"):
        n_spins = model._n_spins.value
        n_spins = jnp.asarray(n_spins, jnp.int32)
        n_spins = jnp.resize(n_spins, (batch_size, model.hilbert.local_size)) # (B, D)
    else:
        n_spins = jnp.zeros((batch_size, model.hilbert.local_size), jnp.int32)
    
    # Compute log conditional probabilities
    (cache, n_spins), log_psi = _compute_conditional(model.hilbert, cache, n_spins, model._epsilon, inputs, index)
    log_psi = _normalize(log_psi, model.machine_pow)
    
    # Update model cache
    if model.has_variable("cache", "inputs"):
        model._cache.value = cache
    if model.has_variable("cache", "spins"):
        model._n_spins.value = n_spins
    return log_psi # (B, D)

def _conditionals(model: ARqGPS, inputs: Array) -> Array:
    # Convert input configurations into indices
    inputs = model.to_indices(inputs) # (B, L)
    batch_size = inputs.shape[0]

    # Loop over sites while computing log conditional probabilities
    def _scan_fun(carry, index):
        cache, n_spins = carry
        (cache, n_spins), log_psi = _compute_conditional(model.hilbert, cache, n_spins, model._epsilon, inputs, index)
        n_spins = jax.lax.cond(
            model.hilbert.constrained,
            lambda n_spins: n_spins,
            lambda n_spins: jnp.zeros_like(n_spins),
            n_spins
        )
        return (cache, n_spins), log_psi

    cache = jnp.ones((batch_size, model.hilbert.local_size, model.M), model.dtype)
    n_spins = jnp.zeros((batch_size, model.hilbert.local_size))
    indices = jnp.arange(model.hilbert.size)
    _, log_psi = jax.lax.scan(
        _scan_fun,
        (cache, n_spins),
        indices
    )
    log_psi = jnp.transpose(log_psi, [1, 0, 2])
    log_psi = _normalize(log_psi, model.machine_pow)
    return log_psi # (B, L, D)