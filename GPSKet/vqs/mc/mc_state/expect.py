from functools import partial
from typing import Callable, Optional
import jax
import jax.numpy as jnp
import netket as nk
from netket.utils.types import PyTree
from netket.stats import Stats
from netket import jax as nkjax
from netket.vqs.mc.mc_state.state import MCState

from netket.vqs.mc.mc_state.expect import get_local_kernel, get_local_kernel_arguments

"""
This simply overrides the NetKet default implementation in order to be able to pass
additional arguments to the model apply function (e.g. required for fast updates).
Ultimately this should probably at one point be merged into NetKet.
"""

@nk.vqs.expect.dispatch
def expect_chunked(vstate: MCState, op: nk.operator.AbstractOperator, chunk_size: int) -> Stats:  # noqa: F811
    samples, args = get_local_kernel_arguments(vstate, op)
    local_estimator_fun = get_local_kernel(vstate, op, chunk_size)
    return _expect(
        chunk_size,
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        samples,
        args,
    )


@nk.vqs.expect.dispatch
def expect(vstate: MCState, op: nk.operator.AbstractOperator) -> Stats:  # noqa: F811
    samples, args = get_local_kernel_arguments(vstate, op)
    local_estimator_fun = get_local_kernel(vstate, op)
    return _expect(
        None,
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        samples,
        args,
    )


@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect(
    chunk_size: Optional[int],
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    samples: jnp.ndarray,
    local_value_args: PyTree,
) -> Stats:
    samples_shape = samples.shape

    if jnp.ndim(samples) != 2:
        samples = samples.reshape((-1, samples_shape[-1]))

    def log_pdf(w, samples):
        return machine_pow * model_apply_fun({"params": w, **model_state}, samples).real

    if chunk_size is not None:
        _, op_stats = nkjax.expect(
            log_pdf,
            partial(local_value_kernel, model_apply_fun, chunk_size=chunk_size),
            {"params": parameters, **model_state},
            samples,
            local_value_args,
            n_chains=samples_shape[0],
        )
    else:
        _, op_stats = nkjax.expect(
            log_pdf,
            partial(local_value_kernel, model_apply_fun),
            {"params": parameters, **model_state},
            samples,
            local_value_args,
            n_chains=samples_shape[0],
        )
    return op_stats
