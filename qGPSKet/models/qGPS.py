import jax
import jax.numpy as jnp
import flax.linen as nn
from netket.utils import HashableArray
from netket.utils.types import NNInitFunc, Array, DType, Callable


# default initializer
def qGPS_init(sigma=1.e-1, dtype=complex):
    if dtype is complex:
        def init_fun(key, shape, dtype=dtype):
            phases = sigma * jax.random.normal(key, shape, float) *1.j
            return jnp.exp(phases).astype(dtype)
    else:
        def init_fun(key, shape, dtype=dtype):
            var = 1. + sigma * jax.random.normal(key, shape, dtype)
            return var
    return init_fun

# helper function to get the symmetry transformation functions
def get_sym_transformation(graph, automorphisms=True, spin_flip=True):
    if automorphisms:
        syms = graph.automorphisms().to_array().T
        if spin_flip:
            def symmetries(samples):
                out = jnp.take(samples, syms, axis=-1)
                out = jnp.concatenate((out, 1-out), axis=-1)
                return out
        else:
            def symmetries(samples):
                out = jnp.take(samples, syms, axis=-1)
                return out
    else:
        if spin_flip:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                out = jnp.concatenate((out, 1-out), axis=-1)
                return out
        else:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                return out
    return symmetries

class qGPS(nn.Module):
    # TODO: add documentation
    M: int
    local_dim: int = 2
    dtype: DType = complex
    init_fun: NNInitFunc = qGPS_init(dtype=dtype)
    to_indices: Callable = lambda samples : samples.astype(jnp.uint8)
    apply_symmetries: Callable = lambda samples : jnp.expand_dims(samples, axis=-1)
    before_sym_op: Callable = lambda argument : argument
    final_op: Callable = lambda argument : argument

    @nn.compact
    def __call__(self, inputs):
        if len(inputs.shape) == 1:
            inputs = jnp.expand_dims(inputs, 0)

        epsilon = self.param("epsilon", self.init_fun, (self.local_dim, self.M, inputs.shape[-1]), self.dtype)

        transformed_samples = self.apply_symmetries(self.to_indices(inputs))

        range_ids = jnp.arange(transformed_samples.shape[-2])

        # this is a strange way of computing the site product (but we do want some memory efficiency here)
        def site_prod(indices, epsilon):
            return epsilon[indices,:,range_ids].prod(axis=0)
        batched = jax.vmap(site_prod, (0, None), 0)
        if len(inputs.shape) == 3:
            batched = jax.vmap(batched, (0, None), 0)
        batched = jax.vmap(batched, (-1, None), -1)

        support_dim_prod = batched(transformed_samples, epsilon)

        qGPS_out = self.before_sym_op(support_dim_prod.sum(axis=-2))
        qGPS_out = self.final_op(qGPS_out.sum(axis=-1))
        return qGPS_out
