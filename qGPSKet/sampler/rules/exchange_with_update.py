import jax
import jax.numpy as jnp
from flax import struct
from netket.sampler.rules.exchange import ExchangeRule_

@struct.dataclass
class ExchangeRuleWithUpdate(ExchangeRule_):
    """
    Exchange Update rule which also returns the list of affected sites which is required for the fast metropolis sampler
    """
    returns_updates: bool = True

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]

        # pick a random cluster
        cluster_id = jax.random.randint(
            key, shape=(n_chains,), minval=0, maxval=rule.clusters.shape[0]
        )

        def scalar_update_fun(σ, cluster):
            # sites to be exchanged,
            si = rule.clusters[cluster, 0]
            sj = rule.clusters[cluster, 1]

            σp = σ.at[si].set(σ[sj])
            return (σp.at[sj].set(σ[si]), si, sj)

        out = jax.vmap(scalar_update_fun, in_axes=(0, 0), out_axes=0)(σ, cluster_id)
        update_sites = jnp.stack((out[1], out[2]), axis=-1)
        return (out[0], None, update_sites)