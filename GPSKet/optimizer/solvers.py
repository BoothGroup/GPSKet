import jax
import jax.numpy as jnp


@jax.jit
def pinv(A, b):
    A_inv = jnp.linalg.pinv(A, hermitian=True)
    x = jnp.dot(A_inv, b)
    return x