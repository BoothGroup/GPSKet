import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from typing import Tuple, Union, Optional
from netket.utils import HashableArray
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils.types import NNInitFunc, Array, DType, Callable
from jax.nn.initializers import zeros
from GPSKet.nn.initializers import normal
from .autoreg_qGPS import AbstractARqGPS, gpu_cond, _normalize


class ARPlaquetteqGPS(AbstractARqGPS):
    """
    Implements the autoregressive formulation of the plaquette qGPS Ansatz with weight sharing,
    support for symmetries and Hilbert spaces constrained to the zero magnetization sector.
    """

    M: int
    """Bond dimension"""
    plaquettes: HashableArray
    """Plaquette for each site"""
    masks: HashableArray
    """Autoregressive mask for each site"""
    dtype: DType = jnp.complex128
    """Type of the variational parameters"""
    machine_pow: int = 2
    """Exponent required to normalize the output"""
    init_fun: Optional[
        NNInitFunc
    ] = None  # Defaults to qGPS-normal (sigma = 0.01) with the parameter dtype
    """Initializer for the variational parameters"""
    normalize: bool = True
    """Whether the Ansatz should be normalized"""
    apply_symmetries: Callable = lambda inputs: jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""
    # TODO: extend to cases beyond D=2
    count_spins: Callable = lambda spins: jnp.stack(
        [(spins + 1) & 1, ((spins + 1) & 2) / 2], axis=-1
    ).astype(jnp.int32)
    """Function to count down and up spins"""
    # TODO: extend to cases where total_sz != 0
    renormalize_log_psi: Callable = lambda n_spins, hilbert, index: jnp.log(
        jnp.heaviside(hilbert.size // 2 - n_spins, 0)
    )
    """Function to renormalize conditional log probabilities"""
    out_transformation: Callable = lambda argument: jnp.sum(argument, axis=-1)
    """Function of the output layer, by default sums over bond dimension"""

    # Dimensions:
    # - B = batch size
    # - D = local dimension
    # - L = number of sites
    # - M = bond dimension
    # - T = number of symmetries

    def _conditional(self, inputs: Array, args: Tuple) -> Array:
        # Convert input configurations into indices
        inputs = self.hilbert.states_to_local_indices(inputs)  # (B, L)

        # Compute conditional probability for site at index
        log_psi = _conditional(self, inputs, args)  # (B, D)
        if self.normalize:
            log_psi = _normalize(log_psi, self.machine_pow)
        p = jnp.exp(self.machine_pow * log_psi.real)
        return p

    def conditionals(self, inputs: Array) -> Array:
        # Convert input configurations into indices
        inputs = self.hilbert.states_to_local_indices(inputs)  # (B, L)

        # Compute conditional probabilities for all sites
        log_psi = _conditionals(self, inputs)  # (B, L, D)
        if self.normalize:
            log_psi = _normalize(log_psi, self.machine_pow)
        p = jnp.exp(self.machine_pow * log_psi.real)
        return p

    def setup(self):
        if self.init_fun is None:
            init = normal(sigma=0.01, dtype=self.dtype)
        else:
            init = self.init_fun
        self._epsilon = self.param(
            "epsilon",
            init,
            (self.hilbert.local_size, self.M, self.hilbert.size),
            self.dtype,
        )
        if self.hilbert.constrained:
            self._n_spins = self.variable(
                "cache", "spins", zeros, None, (1, self.hilbert.local_size)
            )

    def __call__(self, inputs: Array) -> Array:
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0)  # (B, L)

        # Transform inputs according to symmetries
        inputs = self.apply_symmetries(inputs)  # (B, L, T)
        n_symm = inputs.shape[-1]

        # Convert input configurations into indices
        inputs = self.hilbert.states_to_local_indices(inputs)  # (B, L, T)
        batch_size = inputs.shape[0]

        # Compute conditional log-probabilities
        log_psi = jax.vmap(_conditionals, in_axes=(None, -1), out_axes=-1)(
            self, inputs
        )  # (B, L, D, T)
        if self.normalize:
            log_psi = _normalize(log_psi, self.machine_pow, axis=-2)

        # Take conditionals along sites-axis according to input indices
        log_psi = jnp.take_along_axis(
            log_psi, jnp.expand_dims(inputs, axis=2), axis=2
        )  # (B, L, 1, T)
        log_psi = jnp.sum(log_psi, axis=1)  # (B, 1, T)
        log_psi = jnp.reshape(log_psi, (batch_size, n_symm))  # (B, T)

        # Compute symmetrized log-amplitudes
        log_psi_symm = (1 / self.machine_pow) * logsumexp(
            self.machine_pow * log_psi.real, axis=-1, b=1 / n_symm
        )
        log_psi_symm_im = logsumexp(1j * log_psi.imag, axis=-1).imag
        log_psi_symm = log_psi_symm + 1j * log_psi_symm_im
        return log_psi_symm  # (B,)


def _compute_conditional(
    hilbert: HomogeneousHilbert,
    n_spins: Array,
    epsilon: Array,
    mask: Array,
    plaquette: Array,
    inputs: Array,
    index: int,
    count_spins: Callable,
    renormalize_log_psi: Callable,
    out_transformation: Callable,
) -> Union[Array, Array]:
    # Slice inputs at index-1 to count previous spins
    inputs_i = inputs[:, index - 1]  # (B,)

    # Retrieve input parameters at j=0
    input_param = jnp.asarray(epsilon, epsilon.dtype)[:, :, 0]  # (D, M)
    input_param = jnp.expand_dims(input_param, axis=0)  # (1, D, M)

    # Compute product of parameters at 0<j<=index
    context_param = jnp.where(mask, epsilon, 1.0)  # (B, M, L)
    context_param = jnp.expand_dims(context_param, axis=0)  # (1, D, M, L)
    inputs = jnp.expand_dims(inputs[:, plaquette], axis=(1, 2))  # (B, 1, 1, L)
    context_val = jnp.take_along_axis(context_param, inputs, axis=1)  # (B, 1, M, L)
    context_prod = jnp.prod(context_val, axis=-1)  # (B, 1, M)
    site_prod = input_param * context_prod  # (B, D, M)

    # Compute log conditional probabilities
    log_psi = out_transformation(site_prod)  # (B, D)

    # Update spins count if index is larger than 0, otherwise leave as is
    n_spins = gpu_cond(
        index > 0,
        lambda n_spins: n_spins + count_spins(inputs_i),
        lambda n_spins: n_spins,
        n_spins,
    )

    # If Hilbert space associated with the model is constrained, i.e.
    # model has "n_spins" in "cache" collection, then impose total magnetization.
    # This is done by counting number of up/down spins until index, then if
    # n_spins is >= L/2 the probability of up/down spin at index should be 0,
    # i.e. the log probability becomes -inf
    log_psi = gpu_cond(
        index >= 0,
        lambda log_psi: log_psi + renormalize_log_psi(n_spins, hilbert, index),
        lambda log_psi: log_psi,
        log_psi,
    )
    return n_spins, log_psi


def _conditional(model: ARPlaquetteqGPS, inputs: Array, args: Tuple) -> Array:
    # Retrieve spins count
    batch_size = inputs.shape[0]
    if model.has_variable("cache", "spins"):
        n_spins = model._n_spins.value
        n_spins = jnp.asarray(n_spins, jnp.int32)
        n_spins = jnp.resize(n_spins, (batch_size, model.hilbert.local_size))  # (B, D)
    else:
        n_spins = jnp.zeros((batch_size, model.hilbert.local_size), jnp.int32)

    # Compute log conditional probabilities
    index, mask, plaquette = args
    n_spins, log_psi = _compute_conditional(
        model.hilbert,
        n_spins,
        model._epsilon,
        mask,
        plaquette,
        inputs,
        index,
        model.count_spins,
        model.renormalize_log_psi,
        model.out_transformation,
    )

    # Update model cache
    if model.has_variable("cache", "spins"):
        model._n_spins.value = n_spins
    return log_psi  # (B, D)


def _conditionals(model: ARPlaquetteqGPS, inputs: Array) -> Array:
    # Loop over sites while computing log conditional probabilities
    def _scan_fun(n_spins, args):
        index, mask, plaquette = args
        n_spins, log_psi_cond = _compute_conditional(
            model.hilbert,
            n_spins,
            model._epsilon,
            mask,
            plaquette,
            inputs,
            index,
            model.count_spins,
            model.renormalize_log_psi,
            model.out_transformation,
        )
        n_spins = gpu_cond(
            model.hilbert.constrained,
            lambda n_spins: n_spins,
            lambda n_spins: jnp.zeros_like(n_spins),
            n_spins,
        )
        return n_spins, log_psi_cond

    batch_size = inputs.shape[0]
    n_spins = jnp.zeros((batch_size, model.hilbert.local_size), jnp.int32)
    indices = jnp.arange(model.hilbert.size)
    masks = np.asarray(model.masks, np.int32)
    plaquettes = np.asarray(model.plaquettes, np.int32)
    _, log_psi = jax.lax.scan(_scan_fun, n_spins, (indices, masks, plaquettes))
    log_psi = jnp.transpose(log_psi, [1, 0, 2])
    return log_psi  # (B, L, D)
