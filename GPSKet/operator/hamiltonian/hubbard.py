import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import netket.jax as nkjax
from numba import jit
from typing import List, Tuple, Union, Optional
from netket.utils.types import DType
from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.operator.fermion import FermionicDiscreteOperator, apply_hopping
from GPSKet.operator.hamiltonian.ab_initio import get_parity_multiplicator_hop


class FermiHubbard(FermionicDiscreteOperator):
    def __init__(self, hilbert: FermionicDiscreteHilbert, edges: List[Tuple[int, int]], U: float=0.0, t: Union[float, List[float]]=1.):
        super().__init__(hilbert)
        self.U = U
        self.edges = np.array(edges)
        if isinstance(t, List):
            self.t = np.array(t)
        else:
            self.t = np.ones(self.edges.shape[0]) * t

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self) -> DType:
        return float

    # pad argument is just a dummy atm -> TODO: improve this!
    def get_conn_flattened(self, x, sections, pad=True):
        x_primes, mels = self._get_conn_flattened_kernel(np.asarray(x, dtype = np.uint8),
                                                         sections, self.U, self.edges, self.t)
        return x_primes, mels

    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(x, sections, U, edges, t):
        n_conn = x.shape[0] * (1 + edges.shape[0]*4)
        x_prime = np.empty((n_conn, x.shape[1]), dtype=np.uint8)
        mels = np.empty(n_conn, dtype=np.float64)

        count = 0
        for batch_id in range(x.shape[0]):
            # diagonal element
            x_prime[count, :] = x[batch_id, :]
            mels[count] = U * np.sum(x[batch_id, :] == 3)
            count += 1

            is_occ_up = (x[batch_id] & 1).astype(np.bool8)
            is_occ_down = (x[batch_id] & 2).astype(np.bool8)

            up_count = np.cumsum(is_occ_up)
            down_count = np.cumsum(is_occ_down)

            # hopping
            for edge_count in range(edges.shape[0]):
                edge = edges[edge_count]

                # spin up
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[0], edge[1], x_prime[count], 1, cummulative_count=up_count)
                count += 1
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[1], edge[0], x_prime[count], 1, cummulative_count=up_count)
                count += 1

                # spin down
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[0], edge[1], x_prime[count], 2, cummulative_count=down_count)
                count += 1
                x_prime[count, :] = x[batch_id, :]
                mels[count] = -t[edge_count] * apply_hopping(edge[1], edge[0], x_prime[count], 2, cummulative_count=down_count)
                count += 1

                sections[batch_id] = count
        return x_prime, mels

""" Wrapper class which can be used to apply the on-the-fly updating,
also includes another flag specifying if fast updating should be applied or not.
"""
class FermiHubbardOnTheFly(FermiHubbard):
    pass

def local_en_on_the_fly(logpsi, pars, samples, args, use_fast_update=False, chunk_size=None):
    edges, U, t = args
    def vmap_fun(sample):
        sample = jnp.asarray(sample, np.uint8)
        is_occ_up = (sample & 1)
        is_occ_down = (sample & 2) >> 1
        up_count = jnp.cumsum(is_occ_up, dtype=np.uint8)
        down_count = jnp.cumsum(is_occ_down, dtype=np.uint8)

        # Compute log_amp of sample
        if use_fast_update:
            log_amp, intermediates_cache = logpsi(pars, jnp.expand_dims(sample, 0), mutable="intermediates_cache", cache_intermediates=True)
            parameters = {**pars, **intermediates_cache}
        else:
            log_amp = logpsi(pars, jnp.expand_dims(sample, 0))

        """ This function returns the log_amp of the connected configuration which is only specified
        by the occupancy on the updated sites as well as the indices of the sites updated."""
        def get_connected_log_amp(updated_occ_partial, update_sites):
            if use_fast_update:
                log_amp_connected = logpsi(parameters, jnp.expand_dims(updated_occ_partial, 0), update_sites=jnp.expand_dims(update_sites, 0))
            else:
                updated_config = sample.at[update_sites].set(updated_occ_partial)
                log_amp_connected = logpsi(pars, jnp.expand_dims(updated_config, 0))
            return log_amp_connected

        local_en = U * jnp.sum(sample == 3)

        def get_hopping_term(spin_int, cumulative_count):

            def apply_hopping(annihilate_site, create_site):

                def hop(operands):
                    annihilate_site, create_site = operands
                    # Updated config at update sites
                    start_occ = sample[annihilate_site]
                    end_occ = sample[create_site]
                    new_occ = jnp.array([start_occ-spin_int, end_occ+spin_int], dtype=jnp.uint8)
                    update_sites = jnp.array([annihilate_site, create_site])

                    # Get parity
                    parity_multiplicator = get_parity_multiplicator_hop(update_sites, cumulative_count)

                    # Evaluate amplitude ratio
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))

                    return parity_multiplicator*amp_ratio.astype(jnp.complex_)

                def no_hop(operands):
                    return 0.*1j

                start_occ = sample[annihilate_site]
                end_occ = sample[create_site]
                multiplicator = jax.lax.cond(
                    jnp.logical_or(~(start_occ & spin_int).astype(bool), (end_occ & spin_int).astype(bool)),
                    no_hop,
                    hop,
                    (annihilate_site, create_site)
                )
                return multiplicator

            def hopping_loop(index, carry):
                edge = edges[index]
                value = apply_hopping(edge[0], edge[1])
                value += apply_hopping(edge[1], edge[0])
                value *= -t[index]
                return carry+value
            return jax.lax.fori_loop(0, edges.shape[0], hopping_loop, 0.)

        local_en += get_hopping_term(1, up_count)
        local_en += get_hopping_term(2, down_count)

        return local_en
    return nkjax.vmap_chunked(vmap_fun, chunk_size=chunk_size)(samples)

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: FermiHubbardOnTheFly):
    samples = vstate.samples
    edges = op.edges
    U = op.U
    t = op.t
    return (samples, (edges, U, t))

@nk.vqs.get_local_kernel.dispatch(precedence=1)
def get_local_kernel(vstate: nk.vqs.MCState, op: FermiHubbardOnTheFly, chunk_size: Optional[int] = None):
    try:
        use_fast_update = vstate.model.apply_fast_update
    except:
        use_fast_update = False
    return nkjax.HashablePartial(local_en_on_the_fly, use_fast_update=use_fast_update, chunk_size=chunk_size)