import jax
import jax.numpy as jnp
from jax import dtypes


def normal(sigma=0.1, dtype=jnp.float_):
    """
    Constructs an initializer for a qGPS model.
    Real parameters are normally distributed around 1.0, while complex parameters have unit length
    and have normally distributed phases around 0.

    Args:
        sigma : width of the normal distribution
        dtype : default dtype of the weights

    Returns:
        init function with signature `(key, shape, dtype) -> Array`
    """
    if jnp.iscomplexobj(dtype):

        def init_fun(key, shape, dtype=dtype):
            phase = jax.random.normal(key, shape, jnp.float32) * sigma
            eps = jnp.exp(1j * phase).astype(dtype)
            return eps

    else:

        def init_fun(key, shape, dtype=dtype):
            eps = jnp.ones(shape, dtype)
            eps += jax.random.normal(key, shape, dtype) * sigma
            return eps

    return init_fun


def orthogonal(scale=1.0, column_axis=-1, dtype=jnp.float_):
    """
    Constructs an initializer for a linear combination of matrices with orthogonal columns.

    Args:
        scale : width of the normal distribution
        column_axis : the axis that contains the columns that should be orthogonal
        dtype : default dtype of the weights

    Returns:
        init function with signature `(key, shape, dtype) -> Array`
        Importantly, the shape must be 3D.
    """

    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        if len(shape) != 3:
            raise ValueError("Orthogonal initializer requires a 3D shape.")
        ortho_init = jax.nn.initializers.orthogonal(
            scale=scale, column_axis=column_axis, dtype=dtype
        )
        W = jnp.zeros(shape, dtype=dtype)
        keys = jax.random.split(key, shape[0])
        for i in range(shape[0]):
            ortho_matrix = ortho_init(keys[i], shape[-2:])
            W = W.at[i].set(ortho_matrix)
        return W

    return init
