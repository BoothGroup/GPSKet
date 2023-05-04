import numpy as np

from GPSKet.vqs.mc.mc_state.state_unique_samples import MCStateUniqueSamples

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
At the moment this is only an implementation for quick testing which is very slow.
The number of samples passed is the number of samples taken in addition to the deterministic set."""
class MCStateStratifiedSampling(MCStateUniqueSamples):
    def __init__(self, deterministic_samples, N_total, *args, rand_norm=True, number_random_samples=None, renormalize=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_sweeps = self.sampler.n_sweeps

        self.sampler = replace(self.sampler, n_sweeps=1)

        self.deterministic_samples = jnp.array_split(deterministic_samples, _n_nodes)[_rank]

        self.N_complement = N_total - deterministic_samples.shape[0] # Total size of the complement

        self.rand_norm = rand_norm

        self.lookup_dict = {tuple(conf): i for i, conf in enumerate(np.array(deterministic_samples))}

        # Find a valid initial sample (one from the complement), very inelegant
        key = jax.random.split(self.sampler_state.rng)[0]

        self.current_sample = []

        for j in range(self.sampler.n_chains_per_rank):
            samp = self.sampler.hilbert.random_state(key, dtype=self.deterministic_samples.dtype).reshape(-1)
            while(tuple(np.array(samp)) in self.lookup_dict):
                key = jax.random.split(key)[0]
                samp = self.sampler.hilbert.random_state(key, dtype=self.deterministic_samples.dtype).reshape(-1)
            self.current_sample.append(samp)
            key = jax.random.split(key)[0]
        self.current_sample = jnp.array(np.array(self.current_sample))
        self.sampler_state = replace(self.sampler_state, σ = self.current_sample)

        if not self.rand_norm:
            assert(number_random_samples is None)

        if number_random_samples is None:
            self.number_random_samples = self.n_samples_per_rank
        else:
            self.number_random_samples = len(np.array_split(np.arange(number_random_samples), _n_nodes)[_rank])

        self.renormalize = renormalize


    def sample_step(self):
        old_sample = self.current_sample
        self.current_sample = self.sample(chain_length=1, n_discard_per_chain=0).reshape(self.current_sample.shape)

        # Reject the samples in the deterministic set, this is quite hacky (and slow!)
        valid = np.ones(self.current_sample.shape[0], dtype=bool)
        for i, samp in enumerate(self.current_sample):
            if tuple(np.array(samp)) in self.lookup_dict:
                valid[i] = False

        self.current_sample = jnp.where(np.expand_dims(valid,-1), self.current_sample, old_sample)

        self.sampler_state = replace(self.sampler_state, σ = self.current_sample)


    @property
    def samples_with_counts(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self._unique_samples is None:
            # Sampling Warm-up
            for i in range(self.n_discard_per_chain):
                for j in range(self.n_sweeps):
                    self.sample_step()

            # Sample from the complement
            sampled_configs = []
            for i in range(self.chain_length):
                for j in range(self.n_sweeps):
                    self.sample_step()
                sampled_configs.append(self.current_sample)

            samples_from_complement = jnp.array(np.array(sampled_configs)).reshape((-1, self.current_sample.shape[-1]))

            all_samples = jnp.concatenate((self.deterministic_samples, samples_from_complement))

            def log_prob(samp):
                return jnp.squeeze(2 * self.log_value(samp.reshape((1,-1))).real)

            log_prob_amps_deterministic =  nkjax.vmap_chunked(log_prob, chunk_size=self.chunk_size)(self.deterministic_samples)
            log_prob_amps_complement =  nkjax.vmap_chunked(log_prob, chunk_size=self.chunk_size)(samples_from_complement)

            total_samples = _sum(len(log_prob_amps_complement))
            if self.renormalize: # Switch off for normalized models (e.g. autoregressive models)
                # Renormalise the probability amplitudes for numerical stability
                rescale_shift = _mpi_max_jax(jnp.max(jnp.concatenate((log_prob_amps_deterministic, log_prob_amps_complement))))[0]
                log_prob_amps_deterministic -= rescale_shift
                log_prob_amps_complement -= rescale_shift
                # Contribution of the determinisitc set to the norm
                norm_deterministic = _sum(jnp.exp(log_prob_amps_deterministic))


                if self.rand_norm:
                    # Approximation to the norm correction from a uniformly sampled set
                    key = jax.random.split(self.sampler_state.rng)[0]
                    random_samples = np.empty((self.number_random_samples, self.deterministic_samples.shape[-1]), dtype=self.deterministic_samples.dtype)
                    found_samples = 0
                    while found_samples < self.number_random_samples:
                        key = jax.random.split(key)[0]
                        proposed_samples = np.array(self.sampler.hilbert.random_state(key, size=self.number_random_samples, dtype=self.deterministic_samples.dtype))
                        for samp in proposed_samples:
                            if tuple(samp) not in self.lookup_dict:
                                np.copyto(random_samples[found_samples], samp)
                                found_samples += 1
                                if found_samples == self.number_random_samples:
                                    break

                    random_samps = jnp.array(random_samples)

                    log_probs_sampled = nkjax.vmap_chunked(log_prob, chunk_size=self.chunk_size)(random_samps)

                    norm_sampled = self.N_complement * _sum(jnp.exp(log_probs_sampled - rescale_shift))/_sum(random_samps.shape[0])

                else:
                    # Approximation to the norm correction from the sampled set (evaluated with self-normalizing importance sampling)
                    norm_sampled = self.N_complement * total_samples /_sum(jnp.exp(-log_prob_amps_complement))

                norm_estimate = norm_deterministic + norm_sampled
            else:
                norm_deterministic = _sum(jnp.exp(log_prob_amps_deterministic))
                norm_estimate = 1.

            prefactors_det = jnp.exp(log_prob_amps_deterministic)/norm_estimate
            prefactors_sampled = jnp.ones(log_prob_amps_complement.shape) * (1 - norm_deterministic/norm_estimate) / total_samples

            self._unique_samples = all_samples
            self._relative_counts = jnp.concatenate((prefactors_det, prefactors_sampled))
        return (self._unique_samples, self._relative_counts)
