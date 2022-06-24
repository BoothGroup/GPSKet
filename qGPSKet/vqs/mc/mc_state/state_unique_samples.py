import netket as nk
import netket.jax as nkjax
import jax.numpy as jnp
from typing import Tuple, Optional
from collections import defaultdict
import numpy as np

from netket.utils.mpi import (
    node_number as _rank,
    mpi_sum as _mpi_sum
)

""" Very hacky implementation of an MC state which samples until it has accumulated n_samples
unique samples. Expectation values then also include the number of times each of the unique samples
was sampled.

TODO: This needs to be cleaned up, we should think of the best approach to do this, currently only the means
of expectation values can be trusted and sampler properties and variances are probably wrong because the
approach is just hacked into the code with the least possible overhead.
"""
class MCStateUniqeSamples(nk.vqs.MCState):
    def reset(self):
        self._samples = None
        self._unique_samples = None
        self._relative_counts = None

    @property
    def samples(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
            while(len(unique_samps) < self.n_samples):
                if count == 0:
                    samps = self.sample()
                else:
                    samps = self.sample(n_discard_per_chain=0)
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
            unique_samples = np.array(list(unique_samps.keys()), dtype=samps.dtype)
            relative_counts = np.array(list(unique_samps.values()), dtype=float)

            """
            Required so that we can reuse all the standard code to evaluate expectation values.
            """
            relative_counts *= self.n_samples/np.sum(relative_counts)

            # Split samples and counts across mpi processes
            self._unique_samples = jnp.array(unique_samples[start_id:end_id, :])
            self._relative_counts = jnp.array(relative_counts[start_id:end_id])

        return (self._unique_samples, self._relative_counts)


""" Below code is a very hacky way to inject the relative counts into the standard
evaluation of expectation values """
def count_wrapped(inner_fun, logpsi, pars, samples, args, *add_args, **kwargs):
    return args[-1].reshape(-1) * inner_fun(logpsi, pars, samples, args[:-1], *add_args, **kwargs)

@nk.vqs.get_local_kernel.dispatch(precedence=10)
def get_local_kernel(vstate: MCStateUniqeSamples, op: nk.operator.AbstractOperator, chunk_size: Optional[int] = None):
    vstate.__class__ = nk.vqs.MCState
    inner_fun = get_local_kernel(vstate, op, chunk_size=chunk_size)
    vstate.__class__ = MCStateUniqeSamples
    return nkjax.HashablePartial(count_wrapped, inner_fun)


@nk.vqs.get_local_kernel_arguments.dispatch(precedence=10)
def get_local_kernel_arguments(vstate: MCStateUniqeSamples, op: nk.operator.AbstractOperator):
    samples = vstate.samples[0]
    vstate.__class__ = nk.vqs.MCState
    args = get_local_kernel_arguments(vstate, op)[1]
    vstate.__class__ = MCStateUniqeSamples
    args_with_counts = (*args, vstate.samples[1])
    return (samples, args_with_counts)
