import netket as nk
import netket.jax as nkjax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Literal, Any
from collections import defaultdict
import numpy as np

from netket.utils.types import PyTree

from netket.utils.mpi import (
    node_number as _rank,
    mpi_sum as _mpi_sum,
    n_nodes as _n_nodes,
    mpi_sum_jax as _mpi_sum_jax,
    mpi_max_jax as _mpi_max_jax,
    mpi_allgather_jax as _mpi_allgather_jax,
)

from netket.stats import Stats

from netket.stats.mpi_stats import sum as _sum

from functools import partial

from netket.vqs import get_local_kernel_arguments, get_local_kernel

import jax

""" Very hacky implementation of an MC state which samples until it has accumulated n_samples
unique samples. Expectation values then also include the number of times each of the unique samples
was sampled.

TODO: This needs to be cleaned up, we should think of the best approach to do this, currently only the means
of expectation values can be trusted and sampler properties and variances are probably wrong because the
approach is just hacked into the code with the least possible overhead.
"""


class MCStateUniqueSamples(nk.vqs.MCState):
    def __init__(self, *args, max_sampling_steps=None, batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sampling_steps = max_sampling_steps
        if batch_size is None:
            self.batch_size = self.n_samples
        else:
            self.batch_size = batch_size

    def reset(self):
        self._samples = None
        self._unique_samples = None
        self._relative_counts = None

    @property
    def samples(self) -> jnp.ndarray:
        return self.samples_with_counts[0]

    @property
    def samples_with_counts(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self._unique_samples is None:
            unique_samps = defaultdict(lambda: 0)
            count = 0
            continue_sampling = True

            if self.max_sampling_steps is not None:
                if self.max_sampling_steps <= count:
                    continue_sampling = False

            # Generate a batch of samples
            samps = self.sample(n_samples=self.batch_size)

            while continue_sampling:
                # Merge the samples from all mpi processes
                all_samples = _mpi_allgather_jax(samps)[0].reshape(
                    (-1, samps.shape[-1])
                )

                # Add to the previously sampled configurations, there is probably a much more efficient way of doing this but it's good enough for now
                for samp in np.array(all_samples):
                    unique_samps[tuple(samp)] += 1
                    if len(unique_samps) >= self.n_samples:
                        break

                count += 1

                if self.max_sampling_steps is not None:
                    if self.max_sampling_steps <= count:
                        continue_sampling = False
                if len(unique_samps) >= self.n_samples:
                    continue_sampling = False
                if continue_sampling:
                    samps = self.sample(
                        n_samples=self.batch_size, n_discard_per_chain=0
                    )

            unique_samples = np.tile(np.array(all_samples[0]), (self.n_samples, 1))
            relative_counts = np.zeros(self.n_samples, dtype=float)

            max_id = min(len(unique_samps), self.n_samples)
            np.copyto(
                unique_samples[:max_id, :],
                np.array(list(unique_samps.keys()), dtype=unique_samples.dtype)[
                    :max_id, :
                ],
            )
            np.copyto(
                relative_counts[:max_id],
                np.array(list(unique_samps.values()), dtype=relative_counts.dtype)[
                    :max_id
                ],
            )

            # Split samples and counts across mpi processes
            lower = _rank * self.n_samples_per_rank
            upper = lower + self.n_samples_per_rank
            self._unique_samples = jnp.array(unique_samples[lower:upper, :])
            self._relative_counts = jnp.array(relative_counts[lower:upper])

            self._relative_counts /= _sum(self._relative_counts)

        return (self._unique_samples, self._relative_counts)


""" The following functions just override the NetKet implementation to inject the sample counts into the expectation value evaluation."""


@nk.vqs.expect_and_grad.dispatch(precedence=10)
def expect_and_grad(
    vstate: MCStateUniqueSamples,
    op: nk.operator.AbstractOperator,
    use_covariance: Literal[True],
    chunk_size: Optional[int],
    *,
    mutable: Any
):
    _, args = get_local_kernel_arguments(vstate, op)
    samples_and_counts = vstate.samples_with_counts
    if chunk_size is not None:
        local_estimator = get_local_kernel(vstate, op, chunk_size)
    else:
        local_estimator = get_local_kernel(vstate, op)
    assert mutable is False

    exp, grad = grad_expect_hermitian_chunked(
        chunk_size,
        local_estimator,
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples_and_counts,
        args,
        compute_grad=True,
    )

    return exp, grad


@nk.vqs.expect.dispatch(precedence=10)
def expect(
    vstate: MCStateUniqueSamples,
    op: nk.operator.AbstractOperator,
    chunk_size: Optional[int],
):
    _, args = get_local_kernel_arguments(vstate, op)
    samples_and_counts = vstate.samples_with_counts
    if chunk_size is not None:
        local_estimator = get_local_kernel(vstate, op, chunk_size)
    else:
        local_estimator = get_local_kernel(vstate, op)

    exp = grad_expect_hermitian_chunked(
        chunk_size,
        local_estimator,
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples_and_counts,
        args,
        compute_grad=False,
    )

    return exp


@partial(jax.jit, static_argnums=(0, 1, 2, 7))
def grad_expect_hermitian_chunked(
    chunk_size: Optional[int],
    estimator_fun: Callable,
    model_apply_fun: Callable,
    parameters: PyTree,
    model_state: PyTree,
    samples_and_counts: Tuple[jnp.ndarray, jnp.ndarray],
    estimator_args: PyTree,
    compute_grad=False,
):
    samples = samples_and_counts[0]
    counts = samples_and_counts[1]

    if chunk_size is not None:
        loc_vals = estimator_fun(
            model_apply_fun,
            {"params": parameters, **model_state},
            samples,
            estimator_args,
            chunk_size=chunk_size,
        )
    else:
        loc_vals = estimator_fun(
            model_apply_fun,
            {"params": parameters, **model_state},
            samples,
            estimator_args,
        )

    mean = _sum(counts * loc_vals)

    variance = _sum(counts * (jnp.abs(loc_vals - mean) ** 2))

    loc_val_stats = Stats(mean=mean, variance=variance)

    if compute_grad:
        loc_vals_centered = counts * (loc_vals - loc_val_stats.mean)

        if chunk_size is not None:
            vjp_fun = nkjax.vjp_chunked(
                lambda w, samps: model_apply_fun({"params": w, **model_state}, samps),
                parameters,
                samples,
                conjugate=True,
                chunk_size=chunk_size,
                chunk_argnums=1,
                nondiff_argnums=1,
            )
        else:
            vjp_fun = nkjax.vjp(
                lambda w, samps: model_apply_fun({"params": w, **model_state}, samps),
                parameters,
                samples,
                conjugate=True,
            )[1]

        val_grad = vjp_fun((jnp.conjugate(loc_vals_centered)))[0]

        val_grad = jax.tree_util.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
                target.dtype
            ),
            val_grad,
            parameters,
        )

        return loc_val_stats, jax.tree_util.tree_map(lambda x: _mpi_sum_jax(x)[0], val_grad)
    else:
        return loc_val_stats
