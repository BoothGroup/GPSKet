import netket as nk
from netket.optimizer.qgt.qgt_jacobian_common import (choose_jacobian_mode, sanitize_diag_shift, to_shift_offset, rescale)
from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT

import netket.jax as nkjax


from typing import Tuple, Optional, Callable, Any
from netket.utils.types import PyTree

from netket.stats.mpi_stats import (
    sum as _sum
)

import jax
import jax.numpy as jnp

from functools import partial

"""
This is essentially a copy of the constructor of the QGTJacobianDense quantum geometric tensor from netket
but it is adjusted so that it can be used with the unique samples variational state,
this is still very all very hacky. TODO: improve!
"""

def QGTJacobianDenseUniqueSamples(vstate=None, *, mode: str = None, holomorphic: bool = None, diag_shift=None, diag_scale=None, **kwargs) -> "QGTJacobianDenseT":
    assert("rescale_shift" not in kwargs)
    assert(diag_scale is None) # Not yet implemented -> TODO: implement support!
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

    samples, counts = vstate.samples_with_counts

    centered_oks = nkjax.jacobian(vstate._apply_fun, vstate.parameters, samples, vstate.model_state, mode=mode, chunk_size=chunk_size, pdf = counts, dense=True, center=True)

    pars_struct = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vstate.parameters)

    return QGTJacobianDenseT(O=centered_oks, mode=mode, _params_structure=pars_struct, diag_shift=diag_shift, **kwargs)
