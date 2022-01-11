import jax
import jax.numpy as jnp
from jax.experimental import loops
from netket.utils import struct
from netket.sampler.metropolis import MetropolisSampler, MetropolisRule
from netket.sampler.rules.exchange import compute_clusters

class MetropolisRuleWithUpdate(MetropolisRule):
    pass

@struct.dataclass
class MetropolisFastSampler(MetropolisSampler):
    """
    TODO: here we require some checking if the transition rule also returns the updates.
    """
    def _sample_next(sampler, machine, parameters, state):
        """
        Fast implementation of the _sample_next function for qGPS models (allowing for fast updates),
        implementation is based on the original netket implementation for the metropolis sampler.
        Note that the updating is still not strictly constant in the system size as full configurations
        (together with intermediate values) are copied at each sampling step. However there is less
        overhead as the amplitude computation is performed by fast updating.
        """
        new_rng, rng = jax.random.split(state.rng)

        with loops.Scope() as s:
            s.key = rng
            s.σ = state.σ
            value, s.workspace = machine.apply(parameters, state.σ, mutable="workspace", save_site_prod=True)
            s.log_prob = sampler.machine_pow * value.real

            # for logging
            s.accepted = state.n_accepted_proc

            for i in s.range(sampler.n_sweeps):
                # 1 to propagate for next iteration, 1 for uniform rng and n_chains for transition kernel
                s.key, key1, key2 = jax.random.split(s.key, 3)

                σp, log_prob_correction, update_sites = sampler.rule.transition(
                    sampler, machine, parameters, state, key1, s.σ
                )

                params = {**parameters, **s.workspace}
                updated_occupancy = jax.vmap(jnp.take, in_axes=(0, 0), out_axes=0)(σp, update_sites)
                value, new_workspace = machine.apply(params, updated_occupancy, mutable="workspace", save_site_prod=True, update_sites=update_sites)
                proposal_log_prob = (
                    sampler.machine_pow * value.real
                )

                uniform = jax.random.uniform(key2, shape=(sampler.n_chains_per_rank,))
                if log_prob_correction is not None:
                    do_accept = uniform < jnp.exp(
                        proposal_log_prob - s.log_prob + log_prob_correction
                    )
                else:
                    do_accept = uniform < jnp.exp(proposal_log_prob - s.log_prob)

                # do_accept must match ndim of proposal and state (which is 2)
                s.σ = jnp.where(do_accept.reshape(-1, 1), σp, s.σ)

                site_product_old = s.workspace["workspace"]["site_prod"]
                site_product_new = new_workspace["workspace"]["site_prod"]
                samples_old = s.workspace["workspace"]["samples"]
                samples_new = new_workspace["workspace"]["samples"]

                site_product_updated = jnp.where(do_accept.reshape(-1,1,1), site_product_new, site_product_old)
                samples_updated = jnp.where(do_accept.reshape(-1,1), samples_new, samples_old)

                s.workspace = {"workspace": {"site_prod": site_product_updated, "samples": samples_updated}}

                s.accepted += do_accept.sum()

                s.log_prob = jax.numpy.where(
                    do_accept.reshape(-1), proposal_log_prob, s.log_prob
                )

            new_state = state.replace(
                rng=new_rng,
                σ=s.σ,
                n_accepted_proc=s.accepted,
                n_steps_proc=state.n_steps_proc
                + sampler.n_sweeps * sampler.n_chains_per_rank,
            )

        return new_state, new_state.σ
    def dummy(self):
        return self

def MetropolisFastExchange(hilbert, *args, clusters=None, graph=None, d_max=1, **kwargs) -> MetropolisFastSampler:
    from .rules.exchange_with_update import ExchangeRuleWithUpdate

    #TODO: clean this up and follow the standard netket design
    if clusters is None:
        assert(graph is not None)
        clusters = compute_clusters(graph, d_max)
    exchange_rule_with_updates = ExchangeRuleWithUpdate(jnp.array(clusters))

    return MetropolisFastSampler(hilbert, exchange_rule_with_updates, *args, **kwargs)