import netket as nk
from netket.optimizer.qgt.qgt_jacobian_common import choose_jacobian_mode
from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT

import netket.jax as nkjax


from typing import Tuple, Optional, Callable, Any
from netket.utils.types import PyTree

from netket.stats.mpi_stats import (
    sum as _sum
)

from netket.optimizer.qgt.qgt_jacobian_dense_logic import (
    dense_jacobian_real_holo,
    dense_jacobian_cplx,
    tree_to_reim
)

import jax
import jax.numpy as jnp

from functools import partial

"""
This is essentially a copy of the constructor of the QGTJacobianDense quantum geometric tensor from netket
but it is adjusted so that it can be used with the unique samples variational state,
this is still very all very hacky. TODO: improve!
"""

def QGTJacobianDenseUniqueSamples(vstate=None, *, mode: str = None, holomorphic: bool = None, **kwargs) -> "QGTJacobianDenseT":
    assert("rescale_shift" not in kwargs)
    if vstate is None:
        return partial(QGTJacobianDenseUniqueSamples, mode=mode, holomorphic=holomorphic)

    if mode is None:
        mode = choose_jacobian_mode(vstate._apply_fun, vstate.parameters, vstate.model_state, vstate.samples[0], mode=mode, holomorphic=holomorphic)
    else:
        assert(holomorphic is None)

    chunk_size = None

    try:
        chunk_size = vstate.chunk_size
    except:
        pass

    O = prepare_centered_oks(vstate._apply_fun, vstate.parameters, vstate.samples_with_counts, vstate.model_state, mode, chunk_size)
    return QGTJacobianDenseT(O=O, mode=mode, **kwargs)

@partial(jax.jit, static_argnames=("apply_fun", "mode", "chunk_size"))
def prepare_centered_oks(apply_fun: Callable, params: PyTree, samples_and_counts: Tuple[jnp.ndarray, jnp.ndarray], model_state: Optional[PyTree], mode: str, chunk_size: Optional[int]=None) -> PyTree:
    samples = samples_and_counts[0]
    counts = samples_and_counts[1]

    def forward_fn(w, samps):
        return apply_fun({"params": w, **model_state}, samps)

    if mode == "real":
        split_complex_params = True
        jacobian_fun = dense_jacobian_real_holo
    elif mode =="complex":
        split_complex_params = True
        jacobian_fun = dense_jacobian_cplx
    elif mode == "holomorphic":
        split_complex_params = False
        jacobian_fun = dense_jacobian_real_holo
    else:
        assert(False)

    if split_complex_params:
        params, reassemble = tree_to_reim(params)

        def f(w, samps):
            return forward_fn(reassemble(w), samps)
    else:
        f = forward_fn

    def gradf_fun(params, samps):
        return jacobian_fun(f, params, samps)

    jacobians = nkjax.vmap_chunked(gradf_fun, in_axes=(None, 0), chunk_size=chunk_size)(params, samples)

    reshaped_counts = counts.reshape((-1, 1))

    jacobians_mean = _sum(reshaped_counts * jacobians, axis=0, keepdims=True)

    centered_oks =  jnp.sqrt(reshaped_counts) * (jacobians - jacobians_mean)

    return centered_oks

