import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Union, List
from netket.utils.types import DType, NNInitFunc, Callable, Array
from flax import linen as nn
from netket.hilbert.homogeneous import HomogeneousHilbert
from qGPSKet.nn.initializers import normal


class ConditionalqGPS(nn.Module):
    size: int
    local_size: int
    M: int
    dtype: DType = jnp.complex128
    init_fun_context: NNInitFunc = normal(dtype=dtype)
    init_fun_inputs: NNInitFunc = normal(dtype=dtype)
    machine_pow: int=2

    @nn.compact
    def __call__(self, context: Array) -> Array:
        # Initialize variational parameters
        context_size = context.shape[-1]
        context_param = self.param("context", self.init_fun_context, (self.local_size, self.M, context_size), self.dtype)
        inputs_param = self.param("inputs", self.init_fun_inputs, (self.local_size, self.M), self.dtype)

        # Compute log conditional amplitudes
        def take_context_val(c):
            indices = jnp.expand_dims(c, axis=(0, 1))
            val = jnp.take_along_axis(context_param, indices, axis=0)
            val = jnp.reshape(val, (self.M, -1))
            return val
        context_val = jax.vmap(take_context_val, in_axes=0)(context) # (B, M, L)
        context_val = jnp.prod(context_val, axis=-1) # (B, M)
        site_prod = jax.vmap(lambda p: inputs_param*p, in_axes=0)(context_val) # (B, D, M)
        log_psi = jnp.sum(site_prod, axis=-1) # (B, D)
        return log_psi # (B, D)

class ARqGPSFull(nn.Module):
    hilbert: HomogeneousHilbert
    M: Union[int, List[int]]
    dtype: DType = jnp.complex128
    init_fun: NNInitFunc = normal(dtype=dtype)
    to_indices: Callable = lambda inputs : inputs.astype(jnp.uint8)
    apply_symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)
    machine_pow: int=2

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

    def _conditional(self, inputs: Array, index: int) -> Array:
        # Compute conditional probability for site at index
        # log_psi = self._conditional_wavefunctions[index](inputs) # (B, D)
        # FIXME: this is inefficient, but has similar scaling as the ideal implementation
        log_psi = _conditionals(self, inputs)[:, index, :]
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p

    def conditionals(self, inputs: Array) -> Array:
        # Compute conditional probabilities for all sites
        log_psi = _conditionals(self, inputs) # (B, L, D)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p

    def setup(self):
        self.local_dim = self.hilbert.local_size
        self.n_sites = self.hilbert.size
        if isinstance(self.M, int):
            M = [self.M]*self.n_sites
        else:
            assert len(self.M) == self.n_sites
            M = self.M
        self._conditional_wavefunctions = [
            ConditionalqGPS(
                i+1,
                self.hilbert.local_size,
                M[i],
                dtype=self.dtype,
                init_fun_context=self.init_fun if i>0 else jax.nn.initializers.ones,
                init_fun_inputs=self.init_fun
            )
            for i in range(self.n_sites)
        ]

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0) # (B, L)

        # Transform inputs according to symmetries
        inputs = self.apply_symmetries(inputs) # (B, L, T)
        n_symm = inputs.shape[-1]

        # Convert input configurations into indices
        inputs = self.to_indices(inputs) # (B, L, T)
        batch_size = inputs.shape[0]

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


def _conditionals(model: ARqGPSFull, inputs: Array) -> Array:
    batch_size = inputs.shape[0]
    log_psi = jnp.zeros((batch_size, model.n_sites, model.local_dim), model.dtype)
    context = jnp.expand_dims(inputs[:, 0], axis=1)
    log_psi_cond = model._conditional_wavefunctions[0](context)
    log_psi.at[:, 0, :].set(log_psi_cond)
    for index in range(1, model.n_sites):
        context = inputs[:, :index]
        log_psi_cond = model._conditional_wavefunctions[index](context)
        log_psi.at[:, index, :].set(log_psi_cond)
    return log_psi
