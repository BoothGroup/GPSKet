import numpy as np
import netket as nk
import jax.numpy as jnp
import jax
from numba import jit
import netket.jax as nkjax

from typing import Optional

from functools import partial

from netket.utils.types import DType
from qGPSKet.operator.fermion import FermionicDiscreteOperator, apply_hopping
from qGPSKet.models import qGPS

class AbInitioHamiltonian(FermionicDiscreteOperator):
    def __init__(self, hilbert, h_mat, eri_mat):
        """ Though not entirely necessary it makes our life a little bit easier to restrict
        ourselves to fixed electron number/magnetization hilbert spaces. """
        assert(hilbert._n_elec is not None)
        super().__init__(hilbert)
        self.h_mat = h_mat
        self.eri_mat = eri_mat

        # See [Neuscamman (2013), https://doi.org/10.1063/1.4829835] for the definition of t
        self.t_mat = self.h_mat - 0.5 * np.einsum("prrq->pq", eri_mat)

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self) -> DType:
        return float

    # Pad argument is just a dummy at the moment,
    # TODO: include padding for unconstrained Hilbert spaces
    def get_conn_flattened(self, x, sections, pad=True):
        assert(not pad or self.hilbert._has_constraint)

        x_primes, mels = self._get_conn_flattened_kernel(np.asarray(x, dtype = np.uint8),
                                                         sections, self.t_mat, self.eri_mat)

        return x_primes, mels

    # This implementation follows the approach outlined in [Neuscamman (2013), https://doi.org/10.1063/1.4829835].
    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(x, sections, t, eri):
        range_indices = np.arange(x.shape[-1])

        x_prime = np.empty((0, x.shape[1]), dtype=np.uint8)
        mels = np.empty(0, dtype=np.complex128)

        c = 0
        for batch_id in range(x.shape[0]):
            is_occ_up = (x[batch_id] & 1).astype(np.bool8)
            is_occ_down = (x[batch_id] & 2).astype(np.bool8)

            up_count = np.cumsum(is_occ_up)
            down_count = np.cumsum(is_occ_down)

            is_empty_up = ~is_occ_up
            is_empty_down = ~is_occ_down

            up_occ_inds = range_indices[is_occ_up]
            down_occ_inds = range_indices[is_occ_down]
            up_unocc_inds = range_indices[is_empty_up]
            down_unocc_inds = range_indices[is_empty_down]

            connected_con = 1
            connected_con += len(up_occ_inds) * len(up_unocc_inds)
            connected_con += len(down_occ_inds) * len(down_unocc_inds)
            connected_con += len(down_occ_inds) * len(down_unocc_inds) * (len(down_occ_inds) - 1) * (len(down_unocc_inds) - 1)
            connected_con += len(up_occ_inds) * len(up_unocc_inds) * (len(up_occ_inds) - 1) * (len(up_unocc_inds) - 1)
            connected_con += len(up_occ_inds) * len(up_unocc_inds) * len(down_occ_inds) * len(down_unocc_inds)

            x_prime = np.append(x_prime, np.empty((connected_con, x.shape[1]), dtype=np.uint8), axis=0)
            mels = np.append(mels, np.empty(connected_con))

            diag_element = 0.0
            for i in up_occ_inds:
                diag_element += t[i, i]
            for i in down_occ_inds:
                diag_element += t[i, i]
            for i in up_occ_inds:
                for j in down_occ_inds:
                    diag_element += eri[i, i, j, j]
            for i in up_occ_inds:
                for j in up_occ_inds:
                    diag_element += 0.5 * eri[i, i, j, j]
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    diag_element += 0.5 * eri[i, a, a, i]
            for i in down_occ_inds:
                for j in down_occ_inds:
                    diag_element += 0.5 * eri[i, i, j, j]
            for i in down_occ_inds:
                for a in down_unocc_inds:
                    diag_element += 0.5 * eri[i, a, a, i]

            x_prime[c, :] = x[batch_id, :]
            mels[c] = diag_element
            c += 1

            # One-body parts
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    x_prime[c, :] = x[batch_id, :]
                    multiplicator = apply_hopping(i, a, x_prime[c], 1,
                                                  cummulative_count=up_count)
                    value = t[i, a]
                    for k in up_occ_inds:
                        value += eri[i, a, k, k]
                    for k in down_occ_inds:
                        value += eri[i, a, k, k]
                    for k in up_unocc_inds:
                        value += 0.5 * eri[i, k, k, a]
                    for k in up_occ_inds:
                        value -= 0.5 * eri[k, a, i, k]
                    mels[c] = multiplicator * value
                    c += 1
            for i in down_occ_inds:
                for a in down_unocc_inds:
                    x_prime[c, :] = x[batch_id, :]
                    multiplicator = apply_hopping(i, a, x_prime[c], 2,
                                                  cummulative_count=down_count)
                    value = t[i, a]
                    for k in down_occ_inds:
                        value += eri[i, a, k, k]
                    for k in up_occ_inds:
                        value += eri[i, a, k, k]
                    for k in down_unocc_inds:
                        value += 0.5 * eri[i, k, k, a]
                    for k in down_occ_inds:
                        value -= 0.5 * eri[k, a, i, k]
                    mels[c] = multiplicator * value
                    c += 1

            # Two body parts
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    for j in up_occ_inds:
                        for b in up_unocc_inds:
                            if i != j and a != b:
                                x_prime[c, :] = x[batch_id, :]
                                multiplicator = apply_hopping(i, a, x_prime[c], 1,
                                                              cummulative_count=up_count)
                                multiplicator *= apply_hopping(j, b, x_prime[c], 1,
                                                              cummulative_count=up_count)
                                # take first hop into account
                                left_limit = min(j, b)
                                right_limit = max(j, b) - 1
                                if i <= right_limit and i > left_limit:
                                    multiplicator *= -1
                                if a <= right_limit and a > left_limit:
                                    multiplicator *= -1

                                mels[c] = 0.5 * multiplicator * eri[i,a,j,b]
                                c += 1
            for i in down_occ_inds:
                for a in down_unocc_inds:
                    for j in down_occ_inds:
                        for b in down_unocc_inds:
                            if i != j and a != b:
                                x_prime[c, :] = x[batch_id, :]
                                multiplicator = apply_hopping(i, a, x_prime[c], 2,
                                                              cummulative_count=down_count)
                                multiplicator *= apply_hopping(j, b, x_prime[c], 2,
                                                               cummulative_count=down_count)
                                # Take first hop into account
                                left_limit = min(j, b)
                                right_limit = max(j, b) - 1
                                if i <= right_limit and i > left_limit:
                                    multiplicator *= -1
                                if a <= right_limit and a > left_limit:
                                    multiplicator *= -1
                                mels[c] = 0.5 * multiplicator * eri[i,a,j,b]
                                c += 1
            for i in up_occ_inds:
                for a in up_unocc_inds:
                    for j in down_occ_inds:
                        for b in down_unocc_inds:
                            x_prime[c, :] = x[batch_id, :]
                            multiplicator = apply_hopping(i, a, x_prime[c], 1,
                                                          cummulative_count=up_count)
                            multiplicator *= apply_hopping(j, b, x_prime[c], 2,
                                                           cummulative_count=down_count)
                            mels[c] = multiplicator * eri[i,a,j,b]
                            c += 1
            sections[batch_id] = c
        return x_prime, mels

""" Wrapper class which can be used to apply the on-the-fly updating,
also includes another flag specifying if fast updating should be applied or not.
"""
class AbInitioHamiltonianOnTheFly(AbInitioHamiltonian):
    pass

""" Helper function which returns the parity for an electron hop by counting
how many electrons the hopping electron moved past. Careful!, this
is only valid if it is a valid electron move, this function does NOT do any
check if the move is valid (in contrast to the apply_hopping function of the
general fermion operator file)"""
def get_parity_multiplicator_hop(update_sites, cummulative_el_count):
    limits = jnp.sort(update_sites)
    parity_count = (cummulative_el_count[limits[1] - 1] - cummulative_el_count[limits[0]])
    # Type promotion is important, gives incorrect results if not cast to unsigned int
    return (jnp.int32(1) - 2 * (parity_count & 1))

def local_en_on_the_fly(n_elecs, logpsi, pars, samples, args, use_fast_update=False, chunk_size=None):
    t = args[0]
    eri = args[1]

    n_sites = samples.shape[-1]
    def vmap_fun(sample):
        is_occ_up = (sample & 1)
        is_occ_down = (sample & 2) >> 1
        up_count = jnp.cumsum(is_occ_up, dtype=int)
        down_count = jnp.cumsum(is_occ_down, dtype=int)
        is_empty_up = 1 >> is_occ_up
        is_empty_down = 1 >> is_occ_down

        up_occ_inds, = jnp.nonzero(is_occ_up, size=n_elecs[0])
        down_occ_inds, = jnp.nonzero(is_occ_down, size=n_elecs[1])
        up_unocc_inds, = jnp.nonzero(is_empty_up, size=n_sites-n_elecs[0])
        down_unocc_inds, = jnp.nonzero(is_empty_down, size=n_sites-n_elecs[1])

        """ The implementation mostly follows the construction of the connected configurations
        as applied in the get_conn_flattened method which is maybe more readable. This is based
        on the approach as presented in [Neuscamman (2013), https://doi.org/10.1063/1.4829835]."""

        # All the diagonal contributions
        local_en = jnp.sum(t[up_occ_inds, up_occ_inds])
        local_en += jnp.sum(t[down_occ_inds, down_occ_inds])
        local_en += jnp.sum(eri[up_occ_inds, up_occ_inds, :, :][:, down_occ_inds, down_occ_inds])
        local_en += 0.5 * jnp.sum(eri[up_occ_inds, up_occ_inds, :, :][:, up_occ_inds, up_occ_inds])
        local_en += 0.5 * jnp.sum(eri[up_occ_inds, :, :, up_occ_inds][:, up_unocc_inds, up_unocc_inds])
        local_en += 0.5 * jnp.sum(eri[down_occ_inds, down_occ_inds, :, :][:, down_occ_inds, down_occ_inds])
        local_en += 0.5 * jnp.sum(eri[down_occ_inds, :, :, down_occ_inds][:, down_unocc_inds, down_unocc_inds])

        # The following part evaluates the contributions from the connected configurations.

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
                """
                Careful: Go through update_sites in reverse order to ensure the actual updates (which come first in the array)
                are applied and not the dummy updates.
                Due to the non-determinism of updates with .at, we cannot use this and need to scan explicitly.
                """
                def scan_fun(carry, count):
                    return (carry.at[update_sites[count]].set(updated_occ_partial[count]), None)
                updated_config = jax.lax.scan(scan_fun, sample, jnp.arange(len(update_sites)), reverse=True)[0]
                log_amp_connected = logpsi(pars, jnp.expand_dims(updated_config, 0))
            return log_amp_connected

        def get_one_body_term_up(i, a):
            # Updated config at update sites
            new_occ = jnp.array([sample[i]-1, sample[a]+1], dtype=jnp.uint8)
            update_sites = jnp.array([i, a])

            # Get parity
            parity_multiplicator = get_parity_multiplicator_hop(update_sites, up_count)

            # Evaluate amplitude ratio
            log_amp_connected = get_connected_log_amp(new_occ, update_sites)
            amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
            value = t[i, a]
            value += jnp.sum(eri[i, a, up_occ_inds, up_occ_inds])
            value += jnp.sum(eri[i, a, down_occ_inds, down_occ_inds])
            value += 0.5 * jnp.sum(eri[i, up_unocc_inds, up_unocc_inds, a])
            value -= 0.5 * jnp.sum(eri[up_occ_inds, a, i, up_occ_inds])
            return value * amp_ratio * parity_multiplicator

        local_en += jnp.sum(jax.vmap(jax.vmap(get_one_body_term_up, in_axes=(None, 0)), in_axes=(0, None))(up_occ_inds, up_unocc_inds))

        def get_one_body_term_down(i, a):
            # Updated config at update sites
            new_occ = jnp.array([sample[i]-2, sample[a]+2], dtype=jnp.uint8)
            update_sites = jnp.array([i, a])

            # Get parity
            parity_multiplicator = get_parity_multiplicator_hop(update_sites, down_count)

            # Evaluate amplitude ratio
            log_amp_connected = get_connected_log_amp(new_occ, update_sites)
            amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
            value = t[i, a]
            value += jnp.sum(eri[i, a, down_occ_inds, down_occ_inds])
            value += jnp.sum(eri[i, a, up_occ_inds, up_occ_inds])
            value += 0.5 * jnp.sum(eri[i, down_unocc_inds, down_unocc_inds, a])
            value -= 0.5 * jnp.sum(eri[down_occ_inds, a, i, down_occ_inds])
            return value * amp_ratio * parity_multiplicator

        local_en += jnp.sum(jax.vmap(jax.vmap(get_one_body_term_down, in_axes=(None, 0)), in_axes=(0, None))(down_occ_inds, down_unocc_inds))

        def two_body_up_up_occ(index_outer, val_outer):
            i = up_occ_inds[index_outer]
            def two_body_up_up_unocc(index_inner, val_inner):
                a = up_unocc_inds[index_inner]
                occ_inds_outer_removed = up_occ_inds[jnp.nonzero(up_occ_inds != i, size=len(up_occ_inds)-1)]
                unocc_inds_outer_removed = up_unocc_inds[jnp.nonzero(up_unocc_inds != a, size=len(up_unocc_inds)-1)]

                new_occ_outer = jnp.array([sample[i]-1, sample[a]+1], dtype=jnp.uint8)
                update_sites_outer = jnp.array([i, a])

                # Get parity multiplicator for first hop
                parity_multiplicator_outer = get_parity_multiplicator_hop(update_sites_outer, up_count)

                def inner_loop(j, b):
                    new_occ_inner = jnp.array([sample[j]-1, sample[b]+1], dtype=jnp.uint8)
                    update_sites_inner = jnp.array([j, b])

                    # Get parity multiplicator for second hop (this does not take first hop into account)
                    parity_multiplicator_inner = get_parity_multiplicator_hop(update_sites_inner, up_count)

                    parity_multiplicator = parity_multiplicator_outer * parity_multiplicator_inner

                    # Evaluate the modification required to include the first hop
                    limits_inner = jnp.sort(update_sites_inner)
                    left_lim = limits_inner[0]
                    right_lim = (limits_inner[1]-1)
                    parity_multiplicator = jnp.where((i <= right_lim) & (i > left_lim), -parity_multiplicator, parity_multiplicator)
                    parity_multiplicator = jnp.where((a <= right_lim) & (a > left_lim), -parity_multiplicator, parity_multiplicator)

                    # Combined update to the config
                    new_occ = jnp.concatenate((new_occ_outer, new_occ_inner))
                    update_sites = jnp.concatenate((update_sites_outer, update_sites_inner))

                    # Get amplitude ratio
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))

                    return (eri[i,a,j,b] * parity_multiplicator * amp_ratio)
                inner_contraction = jax.vmap(jax.vmap(inner_loop, in_axes=(None, 0)), in_axes=(0, None))(occ_inds_outer_removed, unocc_inds_outer_removed)
                return val_inner + 0.5 * jnp.sum(inner_contraction)
            return jax.lax.fori_loop(0, len(up_unocc_inds), two_body_up_up_unocc, val_outer)
        local_en = jax.lax.fori_loop(0, len(up_occ_inds), two_body_up_up_occ, local_en)


        def two_body_down_down_occ(index_outer, val_outer):
            i = down_occ_inds[index_outer]
            def two_body_down_down_unocc(index_inner, val_inner):
                a = down_unocc_inds[index_inner]
                occ_inds_outer_removed = down_occ_inds[jnp.nonzero(down_occ_inds != i, size=len(down_occ_inds)-1)]
                unocc_inds_outer_removed = down_unocc_inds[jnp.nonzero(down_unocc_inds != a, size=len(down_unocc_inds)-1)]

                new_occ_outer = jnp.array([sample[i]-2, sample[a]+2], dtype=jnp.uint8)
                update_sites_outer = jnp.array([i, a])

                # Get parity multiplicator for first hop
                parity_multiplicator_outer = get_parity_multiplicator_hop(update_sites_outer, down_count)

                def inner_loop(j, b):
                    new_occ_inner = jnp.array([sample[j]-2, sample[b]+2], dtype=jnp.uint8)
                    update_sites_inner = jnp.array([j, b])

                    # Get parity multiplicator for second hop (this does not take first hop into account)
                    parity_multiplicator_inner = get_parity_multiplicator_hop(update_sites_inner, down_count)

                    parity_multiplicator = parity_multiplicator_outer * parity_multiplicator_inner

                    # Evaluate the modification required to include the first hop
                    limits_inner = jnp.sort(update_sites_inner)
                    left_lim = limits_inner[0]
                    right_lim = (limits_inner[1]-1)
                    parity_multiplicator = jnp.where(jnp.logical_and((i <= right_lim), (i > left_lim)), -parity_multiplicator, parity_multiplicator)
                    parity_multiplicator = jnp.where(jnp.logical_and((a <= right_lim), (a > left_lim)), -parity_multiplicator, parity_multiplicator)

                    # Combined update to the config
                    new_occ = jnp.concatenate((new_occ_outer, new_occ_inner))
                    update_sites = jnp.concatenate((update_sites_outer, update_sites_inner))

                    # Get amplitude ratio
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))

                    return (eri[i,a,j,b] * parity_multiplicator * amp_ratio)
                inner_contraction = jax.vmap(jax.vmap(inner_loop, in_axes=(None, 0)), in_axes=(0, None))(occ_inds_outer_removed, unocc_inds_outer_removed)
                return val_inner + 0.5 * jnp.sum(inner_contraction)
            return jax.lax.fori_loop(0, len(down_unocc_inds), two_body_down_down_unocc, val_outer)
        local_en = jax.lax.fori_loop(0, len(down_occ_inds), two_body_down_down_occ, local_en)



        # Two body contribution (up, down)

        """ Helper functions to create the new_occ and update_sites arrays
        based on whether the site is already in the update_sites array or not
        (required since we cannot jit if statements). If the site is already in
        the update sites, we update this occupancy accordingly but still add the
        site index to the list of updated sites and add the original sample occupation
        to the lists of new occupancies, so that effectively the amplitude is unaffected
        by this additional update in the fast updating. If no fast updating is performed it
        is therefore necessary to ensure that the new configuration takes the actual update
        which is at the position of the first occurance the update site.
        This construction keeps the shapes fixed so that everything stays jittable.
        """
        def get_updated_occ_previous_move(first_update_occ, update_sites, site_index, spin_update):
            full_update_sites = jnp.append(update_sites, site_index)
            updated_occ = jnp.append(first_update_occ, sample[site_index])
            first_matching_index = jnp.nonzero(full_update_sites == site_index, size=1)[0][0]
            updated_occ = updated_occ.at[first_matching_index].add(spin_update)
            return (updated_occ, full_update_sites)


        def two_body_up_down_occ(index_outer, val_outer):
            i = up_occ_inds[index_outer]
            def two_body_up_down_unocc(index_inner, val_inner):
                a = up_unocc_inds[index_inner]

                new_occ_outer = jnp.array([sample[i]-1, sample[a]+1], dtype=jnp.uint8)
                update_sites_outer = jnp.array([i, a])

                # Get parity multiplicator for first hop
                parity_multiplicator_up = get_parity_multiplicator_hop(update_sites_outer, up_count)

                def inner_loop(j, b):
                    new_occ_inner = jnp.array([sample[j]-2, sample[b]+2], dtype=jnp.uint8)
                    update_sites_inner = jnp.array([j, b])
                    new_occ_updated, update_sites_updated = get_updated_occ_previous_move(new_occ_outer, update_sites_outer, j, -2)
                    new_occ_final, update_sites_final = get_updated_occ_previous_move(new_occ_updated, update_sites_updated, b, 2)

                    parity_multiplicator_down = get_parity_multiplicator_hop(jnp.array([j, b]), down_count)

                    parity_multiplicator = parity_multiplicator_up * parity_multiplicator_down

                    # Get amplitude ratio
                    log_amp_connected = get_connected_log_amp(new_occ_final, update_sites_final)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))

                    return (eri[i,a,j,b] * parity_multiplicator * amp_ratio)
                inner_contraction = jax.vmap(jax.vmap(inner_loop, in_axes=(None, 0)), in_axes=(0, None))(down_occ_inds, down_unocc_inds)
                return val_inner + jnp.sum(inner_contraction)
            return jax.lax.fori_loop(0, len(up_unocc_inds), two_body_up_down_unocc, val_outer)
        local_en = jax.lax.fori_loop(0, len(up_occ_inds), two_body_up_down_occ, local_en)

        return local_en
    return nkjax.vmap_chunked(vmap_fun, chunk_size=chunk_size)(samples)

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: AbInitioHamiltonianOnTheFly):
    samples = vstate.samples
    t = jnp.array(op.t_mat)
    eri = jnp.array(op.eri_mat)
    return (samples, (t, eri))

@nk.vqs.get_local_kernel.dispatch(precedence=1)
def get_local_kernel(vstate: nk.vqs.MCState, op: AbInitioHamiltonianOnTheFly, chunk_size: Optional[int] = None):
    try:
        use_fast_update = vstate.model.apply_fast_update
    except:
        use_fast_update = False
    return nkjax.HashablePartial(local_en_on_the_fly, vstate.hilbert._n_elec, use_fast_update=use_fast_update, chunk_size=chunk_size)
