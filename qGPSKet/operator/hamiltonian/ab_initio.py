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
how many electrons the hopping electron moves past moved past. Careful!, this
is only valid if it is a valid electron move, this function does NOT do any
check if the move is valid (in contrast to the apply_hopping function of the
general fermion operator file)"""
def get_parity_multiplicator_hop(update_sites, cummulative_el_count):
    limits = jnp.sort(update_sites)
    parity_count = (cummulative_el_count[limits[1] - 1] - cummulative_el_count[limits[0]])
    # Type promotion is important, gives incorrect results if not cast to unsigned int
    return (jnp.int32(1) - 2 * (parity_count & 1))

def local_en_on_the_fly(logpsi, pars, samples, args, use_fast_update=False, chunk_size=None):
    t = args[0]
    eri = args[1]
    def vmap_fun(sample):
        is_occ_up = (sample & 1)
        is_occ_down = (sample & 2) >> 1
        up_count = jnp.cumsum(is_occ_up, dtype=int)
        down_count = jnp.cumsum(is_occ_down, dtype=int)
        is_empty_up = 1 >> is_occ_up
        is_empty_down = 1 >> is_occ_down

        """The following construction ensures that this is jittable.
        When electron/magnetization numbers are conserved (as it is often the case),
        we know the correct size of these nonzero arrays. Then we could use scans
        in the following but we want to keep things general for now and stick to
        the jax while loop constructions and fill up the nonzero arrays with "-1"'s.
        Just to be safe, we increase the maximal expected size by 1 in order to always have
        at least one "-1" in the array."""
        up_occ_inds = jnp.nonzero(is_occ_up, size=len(is_occ_up)+1, fill_value=-1)[0]
        down_occ_inds = jnp.nonzero(is_occ_down, size=len(is_occ_down)+1, fill_value=-1)[0]
        up_unocc_inds = jnp.nonzero(is_empty_up, size=len(is_empty_up)+1, fill_value=-1)[0]
        down_unocc_inds = jnp.nonzero(is_empty_down, size=len(is_empty_down)+1, fill_value=-1)[0]

        """ The code should definitely be made more readable at one point,
        maybe we want to use the jax.experimental.loops interface for this
        but it is not entirely clear how polished that is at the moment and
        if there might be some speed penalties, we definitely want speed here.
        The implementation mostly follows the construction of the connected configurations
        as applied in the get_conn_flattened method which is more readable. This is based
        on the approach as presented in [Neuscamman (2013), https://doi.org/10.1063/1.4829835]."""


        """ This defines the stopping criterion for the loops over the indices
        (we stop looping over the index array when we find a "-1" which indicates
        that no more sites exists with the wanted occupancy)."""
        def loop_stop_cond(index_array, arg):
            return index_array[arg[0]] != -1
        up_occ_cond = partial(loop_stop_cond, up_occ_inds)
        up_unocc_cond = partial(loop_stop_cond, up_unocc_inds)
        down_occ_cond = partial(loop_stop_cond, down_occ_inds)
        down_unocc_cond = partial(loop_stop_cond, down_unocc_inds)

        # Evaluate the contribution from the diagonal, TODO: readability of the code can certainly be improved
        def evaluate_t_elements_sum_diag(loop_indices):
            def while_body(arg):
                diag_element = arg[1] + t[loop_indices[arg[0]], loop_indices[arg[0]]]
                return (arg[0] + 1, diag_element, loop_indices)
            return jax.lax.while_loop(partial(loop_stop_cond, loop_indices), while_body, (0, 0., loop_indices))[1]

        def compute_eri_sum_diag(loop_indices, tensor_ids):
            indices_0 = loop_indices[tensor_ids[0]]
            indices_1 = loop_indices[tensor_ids[1]]
            indices_2 = loop_indices[tensor_ids[2]]
            indices_3 = loop_indices[tensor_ids[3]]
            def outer_while_body(outer_arg):
                outer_count = outer_arg[0]
                def inner_while_body(inner_arg):
                    inner_count = inner_arg[0]
                    count = (outer_count, inner_count)
                    index_0 = indices_0[count[tensor_ids[0]]]
                    index_1 = indices_1[count[tensor_ids[1]]]
                    index_2 = indices_2[count[tensor_ids[2]]]
                    index_3 = indices_3[count[tensor_ids[3]]]
                    diag_element = inner_arg[1] + eri[index_0, index_1, index_2, index_3]
                    return (inner_count+1, diag_element)
                diag_element = jax.lax.while_loop(partial(loop_stop_cond, loop_indices[1]), inner_while_body, (0, outer_arg[1]))[1]
                return (outer_count + 1, diag_element)
            return jax.lax.while_loop(partial(loop_stop_cond, loop_indices[0]), outer_while_body, (0, 0.))[1]

        # All the diagonal contributions
        local_en = evaluate_t_elements_sum_diag(up_occ_inds)
        local_en += evaluate_t_elements_sum_diag(down_occ_inds)
        local_en += compute_eri_sum_diag((up_occ_inds, down_occ_inds), (0, 0, 1, 1))
        local_en += 0.5 * compute_eri_sum_diag((up_occ_inds, up_occ_inds), (0, 0, 1, 1))
        local_en += 0.5 * compute_eri_sum_diag((up_occ_inds, up_unocc_inds), (0, 1, 1, 0))
        local_en += 0.5 * compute_eri_sum_diag((down_occ_inds, down_occ_inds), (0, 0, 1, 1))
        local_en += 0.5 * compute_eri_sum_diag((down_occ_inds, down_unocc_inds), (0, 1, 1, 0))


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
                updated_config = sample.at[update_sites].set(updated_occ_partial)
                log_amp_connected = logpsi(pars, jnp.expand_dims(updated_config, 0))
            return log_amp_connected

        # One body terms
        def get_one_body_terms(spin_int):
            if spin_int == 1:
                occ_inds = up_occ_inds
                unocc_inds = up_unocc_inds
                el_count = up_count
                other_spin_occ_inds = down_occ_inds
                occ_cond = up_occ_cond
                unocc_cond = up_unocc_cond
            elif spin_int == 2:
                occ_inds = down_occ_inds
                unocc_inds = down_unocc_inds
                el_count = down_count
                other_spin_occ_inds = up_occ_inds
                occ_cond = down_occ_cond
                unocc_cond = down_unocc_cond

            def outer_loop_occ(arg):
                i = occ_inds[arg[0]]
                def inner_loop_unocc(arg):
                    a = unocc_inds[arg[0]]

                    # Updated config at update sites
                    new_occ = jnp.array([sample[i]-spin_int, sample[a]+spin_int], dtype=jnp.uint8)
                    update_sites = jnp.array([i, a])

                    # Get parity
                    parity_multiplicator = get_parity_multiplicator_hop(update_sites, el_count)

                    # Evaluate amplitude ratio
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))

                    value = t[i, a]

                    # Additional loops for one body contribution as specified in Neuscamman paper
                    def j_loop(arg):
                        j = occ_inds[arg[0]]
                        return (arg[0] + 1, arg[1] + eri[i,a,j,j] - 0.5 * eri[j,a,i,j])
                    value += jax.lax.while_loop(up_occ_cond, j_loop, (0, 0.))[1]

                    def j_bar_loop(arg):
                        j_bar = other_spin_occ_inds[arg[0]]
                        return (arg[0] + 1, arg[1] + eri[i,a,j_bar,j_bar])
                    value += jax.lax.while_loop(down_occ_cond, j_bar_loop, (0, 0.))[1]

                    def b_loop(arg):
                        b = unocc_inds[arg[0]]
                        return (arg[0] + 1, arg[1] + 0.5 * eri[i,b,b,a])
                    value += jax.lax.while_loop(unocc_cond, b_loop, (0, 0.))[1]
                    value *= amp_ratio * parity_multiplicator

                    return (arg[0] + 1, arg[1] + value)
                value = jax.lax.while_loop(unocc_cond, inner_loop_unocc, (0, 0.))[1]
                return (arg[0] + 1, arg[1] + value)
            return jax.lax.while_loop(occ_cond, outer_loop_occ, (0, 0.))[1]

        # One body term up spin
        local_en += get_one_body_terms(1)
        # One body term down spin
        local_en += get_one_body_terms(2)

        # Two body terms
        def evaluate_two_body_terms_same_spin(spin_int):
            if spin_int == 1:
                occ_inds = up_occ_inds
                unocc_inds = up_unocc_inds
                el_count = up_count
                other_spin_occ_inds = down_occ_inds
                occ_cond = up_occ_cond
                unocc_cond = up_unocc_cond
            elif spin_int == 2:
                occ_inds = down_occ_inds
                unocc_inds = down_unocc_inds
                el_count = down_count
                other_spin_occ_inds = up_occ_inds
                occ_cond = down_occ_cond
                unocc_cond = down_unocc_cond

            site_range_occ = jnp.arange(len(occ_inds))
            site_range_unocc = jnp.arange(len(unocc_inds))
            def outer_loop_occ(arg):
                i = occ_inds[arg[0]]

                # Little bit hacky way of removing index i from the list of indices for j
                occ_inds_outer_removed = occ_inds[jnp.nonzero(site_range_occ!=arg[0], size=len(occ_inds)-1, fill_value=-1)]

                inner_occ_cond = partial(loop_stop_cond, occ_inds_outer_removed)

                def outer_loop_unocc(arg):
                    a = unocc_inds[arg[0]]

                    # Little bit hacky way of removing index a from the list of indices for b
                    unocc_inds_outer_removed = unocc_inds[jnp.nonzero(site_range_unocc!=arg[0], size=len(unocc_inds)-1, fill_value=-1)]

                    inner_unocc_cond = partial(loop_stop_cond, unocc_inds_outer_removed)

                    new_occ_outer = jnp.array([sample[i]-spin_int, sample[a]+spin_int], dtype=jnp.uint8)
                    update_sites_outer = jnp.array([i, a])

                    # Get parity multiplicator for first hop
                    parity_multiplicator_outer = get_parity_multiplicator_hop(update_sites_outer, el_count)

                    def inner_loop_occ(arg):
                        j = occ_inds_outer_removed[arg[0]]

                        def inner_loop_unocc(arg):
                            b = unocc_inds_outer_removed[arg[0]]

                            new_occ_inner = jnp.array([sample[j]-spin_int, sample[b]+spin_int], dtype=jnp.uint8)
                            update_sites_inner = jnp.array([j, b])

                            # Get parity multiplicator for second hop (this does not take first hop into account)
                            parity_multiplicator_inner = get_parity_multiplicator_hop(update_sites_inner, el_count)

                            parity_multiplicator = parity_multiplicator_outer * parity_multiplicator_inner

                            # Evaluate the modification required to include the first hop
                            limits_inner = jnp.sort(update_sites_inner)
                            left_lim = limits_inner[0]
                            right_lim = (limits_inner[1]-1)
                            parity_multiplicator = jax.lax.cond((i <= right_lim) & (i > left_lim), lambda x: -x,
                                                                lambda x: x, parity_multiplicator)
                            parity_multiplicator = jax.lax.cond((a <= right_lim) & (a > left_lim), lambda x: -x,
                                                                lambda x: x, parity_multiplicator)

                            # Combined update to the config
                            new_occ = jnp.concatenate((new_occ_outer, new_occ_inner))
                            update_sites = jnp.concatenate((update_sites_outer, update_sites_inner))

                            # Get amplitude ratio
                            log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                            amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))

                            return (arg[0] + 1, arg[1] + eri[i,a,j,b] * parity_multiplicator * amp_ratio)
                        return (arg[0] + 1, jax.lax.while_loop(inner_unocc_cond, inner_loop_unocc, (0, arg[1]))[1])
                    return (arg[0] + 1, jax.lax.while_loop(inner_occ_cond, inner_loop_occ, (0, arg[1]))[1])
                return (arg[0] + 1, jax.lax.while_loop(unocc_cond, outer_loop_unocc, (0, arg[1]))[1])
            return 0.5 * jax.lax.while_loop(occ_cond, outer_loop_occ, (0, 0.))[1]

        # Two body contribution (up, up)
        local_en += evaluate_two_body_terms_same_spin(1)

        # Two body contribution (down, down)
        local_en += evaluate_two_body_terms_same_spin(2)

        # Two body contribution (up, down)

        """ Helper functions to create the new_occ and update_sites arrays
        based on whether the site is already in the update_sites array or not
        (required since we cannot jit if statements). If the site is already in
        the update sites, we update this occupancy accordingly but still add the
        site index to the list of updated sites. If the fast updating is applied,
        the original sample occupation is added to the lists of new occupancies,
        so that effectively the amplitude is unaffected by this additional update.
        If no fast update is performed, we add the updated occupancy instead of
        the original one to the list of new occupancies, as in this case the full
        connected configuration is generated and this ensures that the correct one
        is created.
        This construction keeps the shapes fixed so that everything stays jittable.
        This is a very hacky approach and should be done better in the future.
        TODO: improve!

        WARNING: This method only works correctly if the site is at most contained
        once in the update_sites array passed to the function.
        """
        def get_updated_occ_previous_move(first_update_occ, update_sites, site_index, spin_update):
            in_arr = jnp.squeeze(jnp.nonzero(update_sites == site_index, size=1, fill_value=-1)[0])
            def update(_):
                updated_occ = first_update_occ.at[in_arr].add(spin_update)
                if use_fast_update:
                    return (jnp.append(updated_occ, sample[site_index]), jnp.append(update_sites, site_index))
                else:
                    return (jnp.append(updated_occ, updated_occ[in_arr]), jnp.append(update_sites, site_index))
            def append(_):
                return (jnp.append(first_update_occ, sample[site_index]+spin_update),
                        jnp.append(update_sites, site_index))
            return jax.lax.cond(in_arr == -1, append, update, None)

        def up_loop_occ(arg):
            i = up_occ_inds[arg[0]]

            def up_loop_unocc(arg):
                a = up_unocc_inds[arg[0]]

                new_occ = jnp.array([sample[i]-1, sample[a]+1], dtype=jnp.uint8)
                update_sites = jnp.array([i, a])

                # Get parity multiplicator first hop
                parity_multiplicator_up = get_parity_multiplicator_hop(update_sites, up_count)

                def down_loop_occ(arg):
                    j = down_occ_inds[arg[0]]

                    new_occ_updated, update_sites_updated = get_updated_occ_previous_move(new_occ, update_sites, j, -2)

                    def down_loop_unocc(arg):
                        b = down_unocc_inds[arg[0]]

                        new_occ_final, update_sites_final = get_updated_occ_previous_move(new_occ_updated, update_sites_updated, b, 2)
                        """Now update_sites_final is the list of sites where the occupancy changes,
                        new_occ_final holds the corresponding updated occupancies at these sites"""

                        # Get parity multiplicator second hop
                        parity_multiplicator_down = get_parity_multiplicator_hop(jnp.array((j, b)), down_count)

                        parity_multiplicator = parity_multiplicator_up * parity_multiplicator_down

                        # Get amplitude ratio
                        log_amp_connected = get_connected_log_amp(new_occ_final, update_sites_final)
                        amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))

                        value = arg[1] + eri[i,a,j,b] * parity_multiplicator * amp_ratio
                        return (arg[0] + 1, value)
                    return (arg[0] + 1, jax.lax.while_loop(down_unocc_cond, down_loop_unocc, (0, arg[1]))[1])
                return (arg[0] + 1, jax.lax.while_loop(down_occ_cond, down_loop_occ, (0, arg[1]))[1])
            return (arg[0] + 1, jax.lax.while_loop(up_unocc_cond, up_loop_unocc, (0, arg[1]))[1])

        local_en += jax.lax.while_loop(up_occ_cond, up_loop_occ, (0, 0.))[1]

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
    except NameError:
        use_fast_update = False
    return nkjax.HashablePartial(local_en_on_the_fly, use_fast_update=use_fast_update, chunk_size=chunk_size)
