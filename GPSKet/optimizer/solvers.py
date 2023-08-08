import jax.numpy as jnp
from netket.jax import tree_ravel


def pinv(A, b, rcond=1e-12, x0=None):
    del x0
    A = A.to_dense()
    b, unravel = tree_ravel(b)
    A_inv = jnp.linalg.pinv(A, rcond=rcond, hermitian=True)
    x = jnp.dot(A_inv, b)
    return unravel(x), None
