import netket as nk
import netket.jax as nkjax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Any
from collections import defaultdict
import numpy as np

from netket.utils.types import PyTree

from netket.utils.dispatch import TrueT

from netket.utils.mpi import (
    node_number as _rank,
    mpi_sum as _mpi_sum,
    n_nodes as _n_nodes,
    mpi_sum_jax as _mpi_sum_jax,
    mpi_max_jax as _mpi_max_jax
)

from netket.stats import Stats

from netket.stats.mpi_stats import (
    sum as _sum
)

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
class MCStateUniqeSamples(nk.vqs.MCState):
    def __init__(self, *args, max_sampling_steps=None, fill_with_random=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sampling_steps = max_sampling_steps
        self.fill_with_random = fill_with_random

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
            """
            We constructs and synchronize arrays of all samples,
            including the dictionary used for the counting across the mpi ranks.
            This is a little bit inelegant and should maybe be improved.
            """

            # Relevant range of samples for this rank
            start_id = _rank * self.n_samples_per_rank
            end_id = start_id + self.n_samples_per_rank

            unique_samps = defaultdict(lambda: 0)
            count = 0
            continue_sampling = True

            if self.max_sampling_steps is not None:
                if self.max_sampling_steps <= count:
                    continue_sampling = False


            samps = self.sample()
            while(continue_sampling):
                samps_np = np.asarray(samps.reshape((-1, samps.shape[-1])))

                """
                Merge the samples from all mpi processes. This can certainly be
                done better (e.g. by only synchronizing the hashes and not the full samples).
                TODO: improve
                """
                all_samples = np.zeros((self.n_samples, samps_np.shape[-1]), dtype=samps_np.dtype)
                np.copyto(all_samples[start_id:end_id], samps_np)

                all_samples = _mpi_sum(all_samples)

                """
                Now count the number of unique samples,
                TODO: make this more efficient, e.g. using np.unique.
                """
                for samp in all_samples:
                    unique_samps[tuple(samp)] += 1
                    if len(unique_samps) == self.n_samples:
                        break
                count += 1

                if self.max_sampling_steps is not None:
                    if self.max_sampling_steps <= count:
                        continue_sampling = False
                if len(unique_samps) >= self.n_samples:
                    continue_sampling = False
                if continue_sampling:
                    samps = self.sample(n_discard_per_chain=0)

            unique_samples = np.zeros((self.n_samples, samps.shape[-1]), dtype=np.array(samps).dtype)
            relative_counts = np.zeros(unique_samples.shape[0])

            if len(unique_samps) > 0:
                np.copyto(unique_samples, all_samples)
                np.copyto(unique_samples[:len(unique_samps), :], np.array(list(unique_samps.keys()), dtype=np.array(samps).dtype))
                np.copyto(relative_counts[:len(unique_samps)], np.array(list(unique_samps.values()), dtype=float))
            else:
                assert(self.fill_with_random)

            if self.fill_with_random:
                synced_key = _mpi_sum_jax(self.sampler_state.rng)[0]
                added_samples = self.sampler.hilbert.random_state(synced_key, size=(self.n_samples - len(unique_samps)), dtype=samps.dtype)
                np.copyto(unique_samples[len(unique_samps):, :], np.array(added_samples))
                log_probs = 2*self.log_value(added_samples).real
                log_probs = log_probs-_mpi_max_jax(jnp.max(log_probs))[0]
                probs_added = jnp.exp(log_probs)
                if len(unique_samps) > 0:
                    probs_added *= jnp.sum(relative_counts[:len(unique_samps)])/jnp.sum(probs_added) # 50:50 split
                np.copyto(relative_counts[len(unique_samps):], np.array(probs_added))

            # Split samples and counts across mpi processes
            self._unique_samples = jnp.array(unique_samples[start_id:end_id, :])
            self._relative_counts = jnp.array(relative_counts[start_id:end_id])

            self._relative_counts /= _sum(self._relative_counts)

        return (self._unique_samples, self._relative_counts)


""" The following functions just override the NetKet implementation to inject the sample counts into the expectation value evaluation."""
@nk.vqs.expect_and_grad.dispatch(precedence=10)
def expect_and_grad(vstate: MCStateUniqeSamples, op: nk.operator.AbstractOperator, use_covariance: TrueT, chunk_size: Optional[int], *, mutable:Any):
    _, args = get_local_kernel_arguments(vstate, op)
    samples_and_counts = vstate.samples_with_counts
    if chunk_size is not None:
        local_estimator = get_local_kernel(vstate, op, chunk_size)
    else:
        local_estimator = get_local_kernel(vstate, op)
    assert(mutable is False)

    exp, grad = grad_expect_hermitian_chunked(chunk_size, local_estimator, vstate._apply_fun, vstate.parameters, vstate.model_state, samples_and_counts, args, compute_grad=True)

    return exp, grad

@nk.vqs.expect.dispatch(precedence=10)
def expect(vstate: MCStateUniqeSamples, op: nk.operator.AbstractOperator, chunk_size: Optional[int]):
    _, args = get_local_kernel_arguments(vstate, op)
    samples_and_counts = vstate.samples_with_counts
    if chunk_size is not None:
        local_estimator = get_local_kernel(vstate, op, chunk_size)
    else:
        local_estimator = get_local_kernel(vstate, op)

    exp = grad_expect_hermitian_chunked(chunk_size, local_estimator, vstate._apply_fun, vstate.parameters, vstate.model_state, samples_and_counts, args, compute_grad=False)

    return exp

@partial(jax.jit, static_argnums=(0,1,2,7))
def grad_expect_hermitian_chunked(chunk_size: Optional[int], estimator_fun: Callable, model_apply_fun: Callable, parameters: PyTree, model_state: PyTree, samples_and_counts: Tuple[jnp.ndarray, jnp.ndarray], estimator_args: PyTree, compute_grad=False):
    samples = samples_and_counts[0]
    counts = samples_and_counts[1]

    if chunk_size is not None:
        loc_vals = estimator_fun(model_apply_fun, {"params": parameters, **model_state}, samples, estimator_args, chunk_size=chunk_size)
    else:
        loc_vals = estimator_fun(model_apply_fun, {"params": parameters, **model_state}, samples, estimator_args)

    mean = _sum(counts * loc_vals)

    variance = _sum(counts * (jnp.abs(loc_vals - mean)**2))

    loc_val_stats = Stats(mean=mean, variance=variance)

    if compute_grad:
        loc_vals_centered = counts * (loc_vals - loc_val_stats.mean)

        if chunk_size is not None:
            vjp_fun = nkjax.vjp_chunked(lambda w, samps: model_apply_fun({"params": w, **model_state}, samps), parameters, samples, conjugate=True, chunk_size=chunk_size, chunk_argnums=1, nondiff_argnums=1)
        else:
            vjp_fun = nkjax.vjp(lambda w, samps: model_apply_fun({"params": w, **model_state}, samps), parameters, samples, conjugate=True)[1]

        val_grad = vjp_fun((jnp.conjugate(loc_vals_centered)))[0]

        val_grad = jax.tree_map(lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(target.dtype), val_grad, parameters)

        return loc_val_stats, jax.tree_map(lambda x: _mpi_sum_jax(x)[0], val_grad)
    else:
        return loc_val_stats
