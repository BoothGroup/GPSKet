import jax
import jax.numpy as jnp
from flax import struct
from netket.sampler.metropolis import MetropolisRule

def transition_function(key, sample, return_updates=False):
    sample = sample.astype(jnp.uint8)
    n_chains = sample.shape[0]

    # pick one of the occupied sites
    is_occ_up = (sample & 1).astype(bool)
    is_occ_down = (sample & 2).astype(bool)
    is_occ = is_occ_up.astype(jnp.uint8) + is_occ_down.astype(jnp.uint8)

    keys = jax.random.split(key, num=n_chains)

    def find_electron(occ_up, occ_down, occ, key):
        keyA, keyB, keyC = jax.random.split(key, num=3)
        start_site = jax.random.choice(keyA, sample.shape[-1], p=occ)
        spin_probs = jnp.zeros(2, dtype=bool).at[0].set(occ_up[start_site])
        spin_probs = spin_probs.at[1].set(occ_down[start_site])
        spin = jax.random.choice(keyB, 2, p=spin_probs)+1
        return (start_site, spin, keyC)

    start_sites, spins, new_keys = jax.vmap(find_electron, in_axes=(0,0,0,0), out_axes=(0,0,0))(is_occ_up, is_occ_down, is_occ, keys)

    candidates = ~(sample[start_sites] & spins).astype(bool)

    def target_site(is_candidate, key):
        site = jax.random.choice(key, sample.shape[-1], p=is_candidate)
        return site

    target_sites = jax.vmap(target_site, in_axes=(0, 0), out_axes=0)(candidates, new_keys)

    def scalar_update_fun(sample, start_site, target_site, spin):
        samplep = sample.at[start_site].add(-spin)
        samplep = samplep.at[target_site].add(spin)
        return samplep

    updated_sample = jax.vmap(scalar_update_fun, in_axes=(0, 0, 0, 0), out_axes=0)(sample, start_sites, target_sites, spins)
    if return_updates:
        update_sites = jnp.stack((start_sites, target_sites), axis=-1)
        return (updated_sample, None, update_sites)
    else:
        return (updated_sample, None)

transition_fun_with_update =  lambda key, sample : transition_function(key, sample, return_updates=True)

transition_fun_without_update = lambda key, sample : transition_function(key, sample, return_updates=False)

@struct.dataclass
class FermionicHoppingRule(MetropolisRule):
    """
    Fermionic hopping update rule
    """
    def transition(rule, sampler, machine, parameters, state, key, sample):
        return transition_fun_without_update(key, sample)

@struct.dataclass
class FermionicHoppingRuleWithUpdates(MetropolisRule):
    """
    Fermionic hopping update rule which also returns the list of affected sites
    which is required for the fast metropolis sampler
    """
    def transition(rule, sampler, machine, parameters, state, key, sample):
        return transition_fun_with_update(key, sample)

