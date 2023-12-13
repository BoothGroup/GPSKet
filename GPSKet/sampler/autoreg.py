import jax
import numpy as np
from jax import numpy as jnp
from functools import partial
from netket.sampler import Sampler, SamplerState
from netket.utils import struct, HashableArray
from netket.utils.types import PRNGKeyT


def batch_choice(key, a, p):
    """
    Batched version of `jax.random.choice`.

    Attributes:
      key: a PRNGKey used as the random key.
      a: 1D array. Random samples are generated from its elements.
      p: 2D array of shape `(batch_size, a.size)`. Each slice `p[i, :]` is
        the probabilities associated with entries in `a` to generate a sample
        at the index `i` of the output. Can be unnormalized.

    Returns:
      The generated samples as an 1D array of shape `(batch_size,)`.
    """
    p_cumsum = p.cumsum(axis=1)
    r = p_cumsum[:, -1:] * jax.random.uniform(key, shape=(p.shape[0], 1))
    indices = (r > p_cumsum).sum(axis=1)
    out = a[indices]
    return out


@struct.dataclass
class ARDirectSamplerState(SamplerState):
    key: PRNGKeyT
    """state of the random number generator."""

    def __repr__(self):
        return f"{type(self).__name__}(rng state={self.key})"


@struct.dataclass
class ARDirectSampler(Sampler):
    """Direct sampler for autoregressive QGPS"""

    @property
    def is_exact(sampler):
        return True

    def _init_cache(sampler, model, σ, key):
        # FIXME: hacky solution to make sure cache of FastARQGPS._conditional
        # is not updated during init
        if hasattr(model, "plaquettes"):
            L = sampler.hilbert.size
            scan_init = (-1, np.zeros(L), np.arange(L))
        else:
            scan_init = -1
        variables = model.init(key, σ, scan_init, method=model._conditional)
        if "cache" in variables:
            cache = variables["cache"]
        else:
            cache = None
        return cache

    def _init_state(sampler, model, variables, key):
        return ARDirectSamplerState(key=key)

    def _reset(sampler, model, variables, state):
        return state

    def _sample_chain(sampler, model, variables, state, chain_length):
        σ, new_state = _sample_chain(sampler, model, variables, state, chain_length)
        return σ, new_state

    def _sample_next(sampler, model, variables, state):
        σ, new_state = sampler._sample_chain(model, variables, state, 1)
        σ = σ.squeeze(axis=0)
        return new_state, σ


@partial(jax.jit, static_argnums=(1, 4))
def _sample_chain(sampler, model, variables, state, chain_length):
    if "cache" in variables:
        variables.pop("cache")

    def scan_fun(carry, args):
        σ, cache, key = carry
        if cache:
            _variables = {**variables, "cache": cache}
        else:
            _variables = variables
        new_key, key = jax.random.split(key)

        p, mutables = model.apply(
            _variables, σ, args, method=model._conditional, mutable=["cache"]
        )
        if "cache" in mutables:
            cache = mutables["cache"]
        else:
            cache = None

        local_states = jnp.asarray(sampler.hilbert.local_states, dtype=sampler.dtype)
        new_σ = batch_choice(key, local_states, p)
        if hasattr(model, "plaquettes"):
            index = args[0]
        else:
            index = args
        σ = σ.at[:, index].set(new_σ)

        return (σ, cache, new_key), None

    new_key, key_init, key_scan, key_symm = jax.random.split(state.key, 4)

    # We just need a buffer for `σ` before generating each sample
    # The result does not depend on the initial contents in it
    batch_size = chain_length * sampler.n_chains_per_rank
    σ = jnp.zeros(
        (batch_size, sampler.hilbert.size),
        dtype=sampler.dtype,
    )

    # Init `cache` before generating each sample,
    # even if `variables` is not changed and `reset` is not called
    cache = sampler._init_cache(model, σ, key_init)

    indices = jnp.arange(sampler.hilbert.size)
    if hasattr(model, "plaquettes"):
        masks = np.asarray(model.masks, np.int32)
        plaquettes = np.asarray(model.plaquettes, np.int32)
        scan_init = (indices, masks, plaquettes)
    else:
        scan_init = indices

    use_scan = True
    if hasattr(model, "M"):
        if isinstance(model.M, HashableArray):
            use_scan = False

    if use_scan:
        (σ, _, _), _ = jax.lax.scan(
            scan_fun,
            (σ, cache, key_scan),
            scan_init,
        )
    else:
        for i in range(sampler.hilbert.size):
            if hasattr(model, "plaquettes"):
                masks = np.asarray(model.masks, np.int32)
                plaquettes = np.asarray(model.plaquettes, np.int32)
                scan_init = (indices, masks, plaquettes)
                (σ, cache, key_scan), _ = scan_fun(
                    (σ, cache, key_scan),
                    (
                        i,
                        np.asarray(model.masks, np.int32)[i],
                        np.asarray(model.plaquettes, np.int32)[i],
                    ),
                )
            else:
                (σ, cache, key_scan), _ = scan_fun((σ, cache, key_scan), i)

    # Apply symmetries
    if type(model.apply_symmetries) == tuple:
        syms = model.apply_symmetries[0]
    else:
        syms = model.apply_symmetries

    σ = syms(σ)  # (B, L, T)

    # Sample transformations uniformly
    r = jax.random.randint(key_symm, shape=(batch_size,), minval=0, maxval=σ.shape[-1])
    σ = jnp.take_along_axis(σ, jnp.expand_dims(r, axis=(-2, -1)), axis=-1).reshape(
        σ.shape[:-1]
    )  # (B, L)

    σ = σ.reshape((sampler.n_chains_per_rank, chain_length, sampler.hilbert.size))

    new_state = state.replace(key=new_key)
    return σ, new_state
