import numpy as np
import netket as nk
import jax.numpy as jnp
import jax
from numba import jit
import netket.jax as nkjax

from typing import Optional

from functools import partial

from GPSKet.operator.hamiltonian.ab_initio import (
    AbInitioHamiltonianOnTheFly,
    get_parity_multiplicator_hop,
)

from netket.utils.types import DType
from GPSKet.operator.fermion import FermionicDiscreteOperator, apply_hopping
from GPSKet.models import qGPS


class AbInitioHamiltonianSparse(AbInitioHamiltonianOnTheFly):
    """Implementation of an ab initio Hamiltonian utilizing sparse structure in the
    one- and two-electron integrals. If a localized basis is used, this gives a reduction to O(N^2)
    terms which need to be evaluated for each local energy. Currently, the sparse structure is set
    up in the constructor resulting in a bit of memory overhead.
    TODO: Improve memory footprint.
    """

    def __init__(self, hilbert, h_mat, eri_mat):
        super().__init__(hilbert, h_mat, eri_mat)

        # Set up the sparse structure

        """
        Start/end ids into the flattened arrays holding the nonzero
        orbital ids and values for each first index into the one-electron array
        start_id[i] = self.h1_nonzero_start[i]
        end_id[i] = self.h1_nonzero_start[i+1]
        non_zero_ids(i) = non_zero_ids_flattened[start_id[i]:end_id[i]]
        non_zero_vals(i) = non_zero_vals_flattened[start_id[i]:end_id[i]]
        """
        self.h1_nonzero_range = np.zeros(self.h_mat.shape[0] + 1, dtype=int)
        self.h1_nonzero_ids_flat = np.zeros(0, dtype=int)
        self.h1_nonzero_vals_flat = np.zeros(0, dtype=self.h_mat.dtype)

        # Construct flattened arrays
        for i in range(self.h_mat.shape[0]):
            nonzeros = np.nonzero(self.h_mat[i, :])[0]
            self.h1_nonzero_range[i + 1] = self.h1_nonzero_range[i] + len(nonzeros)
            self.h1_nonzero_ids_flat = np.append(self.h1_nonzero_ids_flat, nonzeros)
            self.h1_nonzero_vals_flat = np.append(
                self.h1_nonzero_vals_flat, self.h_mat[i, nonzeros]
            )

        """
        Start/end ids into the flattened arrays holding the nonzero
        orbital ids and values for each (i,j) index pair into the eri array (indexed as eri[i,a,j,b])
        start_id[i,j] = self.h2_nonzero_start[i, j]
        end_id[i,j] = self.h2_nonzero_start[i, j+1]
        non_zero_ids(i,j) = non_zero_ids_flattened[start_id[i,j]:end_id[i,j]] -> index pair(a,b)
        non_zero_vals(i,j) = non_zero_vals_flattened[start_id[i,j]:end_id[i,j]]
        """
        self.h2_nonzero_range = np.zeros(
            (self.eri_mat.shape[0], self.eri_mat.shape[2] + 1), dtype=int
        )
        self.h2_nonzero_ids_flat = np.zeros((0, 2), dtype=int)
        self.h2_nonzero_vals_flat = np.zeros(0, dtype=self.eri_mat.dtype)

        # Construct flattened arrays
        for i in range(self.eri_mat.shape[0]):
            for j in range(self.eri_mat.shape[2]):
                nonzeros = np.array(np.nonzero(self.eri_mat[i, :, j, :]))
                self.h2_nonzero_range[i, j + 1] = (
                    self.h2_nonzero_range[i, j] + nonzeros.shape[1]
                )
                if j == self.eri_mat.shape[2] - 1 and i != self.eri_mat.shape[0] - 1:
                    self.h2_nonzero_range[i + 1, 0] = (
                        self.h2_nonzero_range[i, j] + nonzeros.shape[1]
                    )
                self.h2_nonzero_ids_flat = np.append(
                    self.h2_nonzero_ids_flat, nonzeros.T, axis=0
                )
                self.h2_nonzero_vals_flat = np.append(
                    self.h2_nonzero_vals_flat,
                    self.eri_mat[i, nonzeros[0, :], j, nonzeros[1, :]],
                )


def local_en_on_the_fly(
    n_elecs, logpsi, pars, samples, args, use_fast_update=False, chunk_size=None
):
    h1_nonzero_range = args[0]
    h1_nonzero_ids_flat = args[1]
    h1_nonzero_vals_flat = args[2]

    h2_nonzero_range = args[3]
    h2_nonzero_ids_flat = args[4]
    h2_nonzero_vals_flat = args[5]

    n_sites = samples.shape[-1]

    def vmap_fun(sample):
        sample = jnp.asarray(sample, jnp.uint8)
        is_occ_up = sample & 1
        is_occ_down = (sample & 2) >> 1
        up_count = jnp.cumsum(is_occ_up, dtype=int)
        down_count = jnp.cumsum(is_occ_down, dtype=int)
        is_empty_up = 1 >> is_occ_up
        is_empty_down = 1 >> is_occ_down

        (up_occ_inds,) = jnp.nonzero(is_occ_up, size=n_elecs[0])
        (down_occ_inds,) = jnp.nonzero(is_occ_down, size=n_elecs[1])
        (up_unocc_inds,) = jnp.nonzero(is_empty_up, size=n_sites - n_elecs[0])
        (down_unocc_inds,) = jnp.nonzero(is_empty_down, size=n_sites - n_elecs[1])

        # Compute log_amp of sample
        if use_fast_update:
            log_amp, intermediates_cache = logpsi(
                pars,
                jnp.expand_dims(sample, 0),
                mutable="intermediates_cache",
                cache_intermediates=True,
            )
            parameters = {**pars, **intermediates_cache}
        else:
            log_amp = logpsi(pars, jnp.expand_dims(sample, 0))

        """ This function returns the log_amp of the connected configuration which is only specified
        by the occupancy on the updated sites as well as the indices of the sites updated."""

        def get_connected_log_amp(updated_occ_partial, update_sites):
            if use_fast_update:
                log_amp_connected = logpsi(
                    parameters,
                    jnp.expand_dims(updated_occ_partial, 0),
                    update_sites=jnp.expand_dims(update_sites, 0),
                )
            else:
                """
                Careful: Go through update_sites in reverse order to ensure the actual updates (which come first in the array)
                are applied and not the dummy updates.
                Due to the non-determinism of updates with .at, we cannot use this and need to scan explicitly.
                """

                def scan_fun(carry, count):
                    return (
                        carry.at[update_sites[count]].set(updated_occ_partial[count]),
                        None,
                    )

                updated_config = jax.lax.scan(
                    scan_fun, sample, jnp.arange(len(update_sites)), reverse=True
                )[0]
                log_amp_connected = logpsi(pars, jnp.expand_dims(updated_config, 0))
            return log_amp_connected

        # Computes term from single electron hop

        # up spin
        def compute_1B_up(i):
            def inner_loop(a_index, val):
                a = h1_nonzero_ids_flat[a_index]

                def valid_hop():
                    # Updated config at update sites
                    new_occ = jnp.array([sample[i] - 1, sample[a] + 1], dtype=jnp.uint8)
                    update_sites = jnp.array([i, a])
                    # Get parity
                    parity_multiplicator = get_parity_multiplicator_hop(
                        update_sites, up_count
                    )
                    # Evaluate amplitude ratio
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
                    return amp_ratio * parity_multiplicator

                def invalid_hop():
                    return jax.lax.select(
                        i == a,
                        jnp.array(1, dtype=log_amp.dtype),
                        jnp.array(0, dtype=log_amp.dtype),
                    )

                return val + h1_nonzero_vals_flat[a_index] * jax.lax.cond(
                    is_empty_up[a], valid_hop, invalid_hop
                )

            return jax.lax.fori_loop(
                h1_nonzero_range[i],
                h1_nonzero_range[i + 1],
                inner_loop,
                jnp.array(0, dtype=log_amp.dtype),
            )

        local_en = jnp.sum(jax.vmap(compute_1B_up)(up_occ_inds))

        def compute_1B_down(i):
            def inner_loop(a_index, val):
                a = h1_nonzero_ids_flat[a_index]

                def valid_hop():
                    # Updated config at update sites
                    new_occ = jnp.array([sample[i] - 2, sample[a] + 2], dtype=jnp.uint8)
                    update_sites = jnp.array([i, a])
                    # Get parity
                    parity_multiplicator = get_parity_multiplicator_hop(
                        update_sites, down_count
                    )
                    # Evaluate amplitude ratio
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
                    return amp_ratio * parity_multiplicator

                def invalid_hop():
                    return jax.lax.select(
                        i == a,
                        jnp.array(1, dtype=log_amp.dtype),
                        jnp.array(0, dtype=log_amp.dtype),
                    )

                return val + h1_nonzero_vals_flat[a_index] * jax.lax.cond(
                    is_empty_down[a], valid_hop, invalid_hop
                )

            return jax.lax.fori_loop(
                h1_nonzero_range[i],
                h1_nonzero_range[i + 1],
                inner_loop,
                jnp.array(0, dtype=log_amp.dtype),
            )

        local_en += jnp.sum(jax.vmap(compute_1B_down)(down_occ_inds))

        # Helper function which updates a config, also taking into account a previous update
        def update_config(site, update_sites, updated_conf, spin_int, create):
            update_sites = jnp.append(update_sites, site)
            updated_conf = jnp.append(updated_conf, sample[site])
            first_matching_index = jnp.nonzero(update_sites == site, size=1)[0][0]
            valid = jax.lax.select(
                create,
                ~(updated_conf[first_matching_index] & spin_int).astype(bool),
                (updated_conf[first_matching_index] & spin_int).astype(bool),
            )
            updated_conf = updated_conf.at[first_matching_index].add(
                jax.lax.select(create, spin_int, -spin_int)
            )
            return updated_conf, valid, update_sites

        def two_body_up_up_occ(inds):
            i = up_occ_inds[inds[0]]
            j = up_occ_inds[inds[1]]
            update_sites_ij = jnp.array([i, j])
            new_occ_ij = jnp.array([sample[i] - 1, sample[j] - 1], dtype=jnp.uint8)
            parity_count_ij = up_count[i] + up_count[j] - 2

            def inner_loop(ab_index, val):
                a = h2_nonzero_ids_flat[ab_index, 0]
                b = h2_nonzero_ids_flat[ab_index, 1]

                new_occ_ijb, valid_b, update_sites_ijb = update_config(
                    b, update_sites_ij, new_occ_ij, 1, True
                )
                new_occ, valid_a, update_sites = update_config(
                    a, update_sites_ijb, new_occ_ijb, 1, True
                )
                valid = valid_a & valid_b

                def get_val():
                    parity_count = parity_count_ij + up_count[a] + up_count[b]
                    parity_count -= (
                        (a >= j).astype(int)
                        + (a >= i).astype(int)
                        + (b >= j).astype(int)
                        + (b >= i).astype(int)
                        - (a >= b).astype(int)
                        + (j > i).astype(int)
                    )
                    parity_multiplicator = -2 * (parity_count & 1) + 1
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
                    return (
                        h2_nonzero_vals_flat[ab_index]
                        * amp_ratio
                        * parity_multiplicator
                    )

                value = jax.lax.cond(
                    valid, get_val, lambda: jnp.array(0.0, dtype=log_amp.dtype)
                )
                return val + value

            return jax.lax.fori_loop(
                h2_nonzero_range[i, j],
                h2_nonzero_range[i, j + 1],
                inner_loop,
                jnp.array(0, dtype=log_amp.dtype),
            )

        local_en += jnp.sum(
            jax.vmap(two_body_up_up_occ)(jnp.triu_indices(up_occ_inds.shape[0], k=1))
        )

        def two_body_down_down_occ(inds):
            i = down_occ_inds[inds[0]]
            j = down_occ_inds[inds[1]]
            update_sites_ij = jnp.array([i, j])
            new_occ_ij = jnp.array([sample[i] - 2, sample[j] - 2], dtype=jnp.uint8)
            parity_count_ij = down_count[i] + down_count[j] - 2

            def inner_loop(ab_index, val):
                a = h2_nonzero_ids_flat[ab_index, 0]
                b = h2_nonzero_ids_flat[ab_index, 1]

                new_occ_ijb, valid_b, update_sites_ijb = update_config(
                    b, update_sites_ij, new_occ_ij, 2, True
                )
                new_occ, valid_a, update_sites = update_config(
                    a, update_sites_ijb, new_occ_ijb, 2, True
                )
                valid = valid_a & valid_b

                def get_val():
                    parity_count = parity_count_ij + down_count[a] + down_count[b]
                    parity_count -= (
                        (a >= j).astype(int)
                        + (a >= i).astype(int)
                        + (b >= j).astype(int)
                        + (b >= i).astype(int)
                        - (a >= b).astype(int)
                        + (j > i).astype(int)
                    )
                    parity_multiplicator = -2 * (parity_count & 1) + 1
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
                    return (
                        h2_nonzero_vals_flat[ab_index]
                        * amp_ratio
                        * parity_multiplicator
                    )

                value = jax.lax.cond(
                    valid, get_val, lambda: jnp.array(0.0, dtype=log_amp.dtype)
                )
                return val + value

            return jax.lax.fori_loop(
                h2_nonzero_range[i, j],
                h2_nonzero_range[i, j + 1],
                inner_loop,
                jnp.array(0, dtype=log_amp.dtype),
            )

        local_en += jnp.sum(
            jax.vmap(two_body_down_down_occ)(
                jnp.triu_indices(down_occ_inds.shape[0], k=1)
            )
        )

        def two_body_up_down_occ(inds):
            i = up_occ_inds[inds[0]]
            j = down_occ_inds[inds[1]]
            update_sites_i = jnp.array([i])
            new_occ_i = jnp.array([sample[i] - 1], dtype=jnp.uint8)
            new_occ_ij, _, update_sites_ij = update_config(
                j, update_sites_i, new_occ_i, 2, False
            )
            parity_count_ij = up_count[i] + down_count[j] - 2

            def inner_loop(ab_index, val):
                a = h2_nonzero_ids_flat[ab_index, 0]
                b = h2_nonzero_ids_flat[ab_index, 1]

                new_occ_ijb, valid_b, update_sites_ijb = update_config(
                    b, update_sites_ij, new_occ_ij, 2, True
                )
                new_occ, valid_a, update_sites = update_config(
                    a, update_sites_ijb, new_occ_ijb, 1, True
                )
                valid = valid_a & valid_b

                def get_val():
                    parity_count = parity_count_ij + up_count[a] + down_count[b]
                    parity_count -= (a >= i).astype(int) + (b >= j).astype(int)
                    parity_multiplicator = -2 * (parity_count & 1) + 1
                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
                    return (
                        h2_nonzero_vals_flat[ab_index]
                        * amp_ratio
                        * parity_multiplicator
                    )

                value = jax.lax.cond(
                    valid, get_val, lambda: jnp.array(0.0, dtype=log_amp.dtype)
                )
                return val + value

            return jax.lax.fori_loop(
                h2_nonzero_range[i, j],
                h2_nonzero_range[i, j + 1],
                inner_loop,
                jnp.array(0, dtype=log_amp.dtype),
            )

        row_inds, col_inds = jnp.indices((up_occ_inds.shape[0], down_occ_inds.shape[0]))
        local_en += jnp.sum(
            jax.vmap(two_body_up_down_occ)((row_inds.flatten(), col_inds.flatten()))
        )

        return local_en

    return nkjax.vmap_chunked(vmap_fun, chunk_size=chunk_size)(samples)


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: AbInitioHamiltonianSparse):
    samples = vstate.samples
    h1_nonzero_range = jnp.array(op.h1_nonzero_range)
    h1_nonzero_ids_flat = jnp.array(op.h1_nonzero_ids_flat)
    h1_nonzero_vals_flat = jnp.array(op.h1_nonzero_vals_flat)

    h2_nonzero_range = jnp.array(op.h2_nonzero_range)
    h2_nonzero_ids_flat = jnp.array(op.h2_nonzero_ids_flat)
    h2_nonzero_vals_flat = jnp.array(op.h2_nonzero_vals_flat)

    return (
        samples,
        (
            h1_nonzero_range,
            h1_nonzero_ids_flat,
            h1_nonzero_vals_flat,
            h2_nonzero_range,
            h2_nonzero_ids_flat,
            h2_nonzero_vals_flat,
        ),
    )


@nk.vqs.get_local_kernel.dispatch(precedence=1)
def get_local_kernel(
    vstate: nk.vqs.MCState,
    op: AbInitioHamiltonianSparse,
    chunk_size: Optional[int] = None,
):
    try:
        use_fast_update = vstate.model.apply_fast_update
    except:
        use_fast_update = False
    return nkjax.HashablePartial(
        local_en_on_the_fly,
        vstate.hilbert._n_elec,
        use_fast_update=use_fast_update,
        chunk_size=chunk_size,
    )
