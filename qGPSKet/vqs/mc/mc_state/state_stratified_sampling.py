import numpy as np

from qGPSKet.vqs.mc.mc_state.state_unique_samples import MCStateUniqeSamples

import jax
import jax.numpy as jnp

from netket.stats.mpi_stats import (
    sum as _sum
)

import netket.jax as nkjax

from netket.utils.mpi import (
    node_number as _rank,
    mpi_max_jax as _mpi_max_jax,
    n_nodes as _n_nodes
)

from typing import Tuple

from dataclasses import replace

""" Implements a state with stratified sampling, splitting the evaluation of expectation values
into a determinisitic evaluation over a fixed set, and a sampled estimate over the complement.
At the moment this is only an implementation for quick testing which is very slow. """
class MCStateStratifiedSampling(MCStateUniqeSamples):
    def __init__(self, deterministic_samples, N_total, *args, rand_norm=True, number_random_samples=None, **kwargs):
        super().__init__(*args, **kwargs)

        assert(self.sampler.n_chains_per_rank == 1)

        self.n_sweeps = self.sampler.n_sweeps

        self.sampler = replace(self.sampler, n_sweeps=1)

        self.deterministic_samples = jnp.array_split(deterministic_samples, _n_nodes)[_rank]

        self.N_complement = N_total - deterministic_samples.shape[0] # Total size of the complement

        self.rand_norm = rand_norm

        self.lookup_dict = {tuple(conf): i for i, conf in enumerate(np.array(deterministic_samples))}

        # Find a valid initial sample (one from the complement)
        key = jax.random.split(self.sampler_state.rng)[0]
        self.current_sample = self.sampler.hilbert.random_state(key, dtype=self.deterministic_samples.dtype).reshape(-1)
        while(tuple(np.array(self.current_sample)) in self.lookup_dict):
            key = jax.random.split(key)[0]
            self.current_sample = self.sampler.hilbert.random_state(key, dtype=self.deterministic_samples.dtype).reshape(-1)
        self.sampler_state = replace(self.sampler_state, σ = self.current_sample.reshape((1,-1)))

        if not self.rand_norm:
            assert(number_random_samples is None)

        if number_random_samples is None:
            self.number_random_samples = self.n_samples_per_rank - self.deterministic_samples.shape[0]
        else:
            self.number_random_samples = len(np.array_split(np.arange(number_random_samples), _n_nodes)[_rank])


    def sample_step(self):
        old_sample = self.current_sample
        self.current_sample = self.sample(chain_length=1, n_discard_per_chain=0).reshape(-1)

        # Reject the sample if it is in the determinisitc set
        if tuple(np.array(self.current_sample).reshape(-1)) in self.lookup_dict:
            self.current_sample = old_sample

        self.sampler_state = replace(self.sampler_state, σ = self.current_sample.reshape((1,-1)))


    @property
    def samples_with_counts(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self._unique_samples is None:
            # Sampling Warm-up
            for i in range(self.n_discard_per_chain):
                for j in range(self.n_sweeps):
                    self.sample_step()

            # Sample from the complement
            remaining_samples = self.n_samples_per_rank - self.deterministic_samples.shape[0]
            sampled_configs = []
            for i in range(remaining_samples):
                for j in range(self.n_sweeps):
                    self.sample_step()
                sampled_configs.append(self.current_sample)

            samples_from_complement = jnp.array(np.array(sampled_configs))

            all_samples = jnp.concatenate((self.deterministic_samples, samples_from_complement))

            log_prob_amps_deterministic =  2 * self.log_value(self.deterministic_samples).real
            log_prob_amps_complement =  2 * self.log_value(samples_from_complement).real

            # Renormalise the probability amplitudes for numerical stability
            rescale_shift = _mpi_max_jax(jnp.max(jnp.concatenate((log_prob_amps_deterministic, log_prob_amps_complement))))[0]
            log_prob_amps_deterministic -= rescale_shift
            log_prob_amps_complement -= rescale_shift

            # Contribution of the determinisitc set to the norm
            norm_deterministic = _sum(jnp.exp(log_prob_amps_deterministic))

            if self.rand_norm:
                # Approximation to the norm correction from a uniformly sampled set
                key = jax.random.split(self.sampler_state.rng)[0]
                random_samples = []
                for i in range(self.number_random_samples):
                    key = jax.random.split(key)[0]
                    samp = self.sampler.hilbert.random_state(key, dtype=self.deterministic_samples.dtype).reshape(-1)
                    while(tuple(np.array(samp)) in self.lookup_dict):
                        key = jax.random.split(key)[0]
                        samp = self.sampler.hilbert.random_state(key, dtype=self.deterministic_samples.dtype).reshape(-1)
                    random_samples.append(np.array(samp))


                random_samps = jnp.array(np.array(random_samples))

                def log_prob(samp):
                    return 2 * self.log_value(samp.reshape((1,-1))).real

                log_probs_sampled = nkjax.vmap_chunked(log_prob, chunk_size=self.chunk_size)(random_samps)

                norm_sampled = self.N_complement * _sum(jnp.exp(log_probs_sampled - rescale_shift))/_sum(random_samps.shape[0])

            else:
                # Approximation to the norm correction from the sampled set (evaluated with self-normalizing importance sampling)
                norm_sampled = self.N_complement * _sum(jnp.exp(2 * log_prob_amps_complement))/_sum(jnp.exp(log_prob_amps_complement))

            norm_estimate = norm_deterministic + norm_sampled

            prefactors_det = jnp.exp(log_prob_amps_deterministic)/norm_estimate
            prefactors_sampled = jnp.ones(log_prob_amps_complement.shape) * (1 - norm_deterministic/norm_estimate) / _sum(len(log_prob_amps_complement))

            self._unique_samples = all_samples
            self._relative_counts = jnp.concatenate((prefactors_det, prefactors_sampled))
        return (self._unique_samples, self._relative_counts)
