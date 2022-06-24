import netket as nk
from netket.optimizer.qgt.qgt_jacobian_common import choose_jacobian_mode
from netket.optimizer.qgt.qgt_jacobian_dense_logic import prepare_centered_oks
from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT

from netket.stats import subtract_mean

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
        assert(holomorphic is not None)

    chunk_size = None

    try:
        chunk_size = vstate.chunk_size
    except:
        pass

    samples, counts = vstate.samples
    samples = samples.reshape((-1, samples.shape[-1]))
    counts = counts.reshape(-1)

    O, _ = prepare_centered_oks(vstate._apply_fun, vstate.parameters, samples, vstate.model_state, mode, False, chunk_size)

    O *= jnp.sqrt(counts.reshape(-1, 1))
    O = subtract_mean(O, axis=0)

    return QGTJacobianDenseT(O=O, mode=mode)