import jax
import jax.numpy as jnp
from jax import dtypes


def normal(sigma=1.e-1, dtype=complex):
    def init_fun(key, shape, dtype=dtype):
        if dtype is jnp.complex_:
            phases = sigma * jax.random.normal(key, shape, float) *1.j
            return jnp.exp(phases).astype(dtype)
        else:
            var = 1. + sigma * jax.random.normal(key, shape, dtype)
            return var
    return init_fun

def orthogonal(scale=1.0, column_axis=-1, dtype=jnp.float_):
    """
    Constructs an initializer for a linear combination of matrices with orthogonal columns.

    The shape must be 3D.
    """
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        if len(shape) != 3:
            raise ValueError("Orthogonal initializer requires a 3D shape.")
        ortho_init = jax.nn.initializers.orthogonal(scale=scale, column_axis=column_axis, dtype=dtype)
        W = jnp.zeros(shape, dtype=dtype)
        keys = jax.random.split(key, shape[0])
        for i in range(shape[0]):
            ortho_matrix = ortho_init(keys[i], shape[-2:])
            W = W.at[i].set(ortho_matrix)
        return W
    return init