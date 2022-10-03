import jax
import jax.numpy as jnp
from flax import struct
from netket.sampler.metropolis import MetropolisRule
from typing import Optional
from netket.utils.types import Array

def transition_function(key, sample, hop_probability, transition_probs=None, return_updates=False):
    def apply_electron_hop(samp, key):
        keyA, keyB, keyC, keyD = jax.random.split(key, num=4)
        is_occ_up = (samp & 1).astype(bool)
        is_occ_down = (samp & 2).astype(bool)
        occ = is_occ_up.astype(jnp.uint8) + is_occ_down.astype(jnp.uint8)

        hopping_or_exchange = jax.random.choice(keyC, jnp.array([0,1]), p=jnp.array([hop_probability, 1-hop_probability]))
        occ_prob = jnp.where(hopping_or_exchange==0, occ, jnp.logical_and(occ!=2, occ!=0))
        start_site = jax.random.choice(keyA, samp.shape[-1], p=occ_prob)
        spin_probs = jnp.array([is_occ_up[start_site], is_occ_down[start_site]])
        spin = jax.random.choice(keyB, 2, p=spin_probs)+1
        target_site_probs = jnp.where(hopping_or_exchange==0, ~((samp & spin).astype(bool)), jnp.logical_and(jnp.logical_and(occ!=2, occ!=0), ~((samp & spin).astype(bool))))
        if transition_probs is not None:
            target_site = jax.random.choice(keyD, samp.shape[-1], p=transition_probs[start_site, :])
        else:
            target_site = jax.random.choice(keyD, samp.shape[-1], p = target_site_probs)
        # Make sure no unallowed move is applied
        target_site = jnp.where(target_site_probs[target_site]==False, start_site, target_site)
        updated_sample = samp.at[start_site].add(-spin)
        updated_sample = updated_sample.at[target_site].add(spin)
        def get_exchange(_):
            updated_sample_exchanged = updated_sample.at[start_site].add(3-spin)
            return updated_sample_exchanged.at[target_site].add(-(3-spin))
        return (jax.lax.cond(hopping_or_exchange==0, lambda _: updated_sample, get_exchange, None), start_site, target_site)

    keys = jax.random.split(key, num=sample.shape[0])

    dtype = sample.dtype
    sample = jnp.asarray(sample, jnp.uint8)
    updated_sample, start_sites, target_sites = jax.vmap(apply_electron_hop, in_axes=(0, 0), out_axes=(0, 0, 0))(sample, keys)
    updated_sample = jnp.array(updated_sample, dtype)

    if return_updates:
        update_sites = jnp.stack((start_sites, target_sites), axis=-1)
        return (updated_sample, None, update_sites)
    else:
        return (updated_sample, None)

transition_fun_with_update = lambda key, sample, hop_probability, transition_probs : transition_function(key, sample, hop_probability, transition_probs, return_updates=True)
transition_fun_without_update = lambda key, sample, hop_probability, transition_probs : transition_function(key, sample, hop_probability, transition_probs, return_updates=False)

@struct.dataclass
class FermionicHoppingRule(MetropolisRule):
    """
    Fermionic hopping update rule
    """
    hop_probability: float = 1.
    transition_probs: Optional[Array] = None

    def transition(rule, sampler, machine, parameters, state, key, sample):
        return transition_fun_without_update(key, sample, rule.hop_probability, transition_probs=rule.transition_probs)


@struct.dataclass
class FermionicHoppingRuleWithUpdates(MetropolisRule):
    """
    Fermionic hopping update rule which also returns the list of affected sites
    which is required for the fast metropolis sampler
    """
    hop_probability: float = 1.
    transition_probs: Optional[Array] = None

    def transition(rule, sampler, machine, parameters, state, key, sample):
        return transition_fun_with_update(key, sample, rule.hop_probability, transition_probs=rule.transition_probs)
