import jax
import jax.numpy as jnp


def normal(sigma=1.e-1, dtype=complex):
    if dtype is jnp.complex_:
        def init_fun(key, shape, dtype=dtype):
            phases = sigma * jax.random.normal(key, shape, float) *1.j
            return jnp.exp(phases).astype(dtype)
    else:
        def init_fun(key, shape, dtype=dtype):
            var = 1. + sigma * jax.random.normal(key, shape, dtype)
            return var
    return init_fun