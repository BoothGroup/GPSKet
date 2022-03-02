import numpy as np
import netket as nk
import jax.numpy as jnp
import jax
from numba import jit
import netket.jax as nkjax

from qGPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonian

from typing import Optional

from functools import partial

from netket.utils.types import DType
from qGPSKet.operator.fermion import FermionicDiscreteOperator, apply_hopping
from qGPSKet.models import qGPS

class AbInitioHamiltonianSparse(AbInitioHamiltonian):
    def __init__(self, hilbert, h_mat, eri_mat):
        super(AbInitioHamiltonian, self).__init__(hilbert)

        if h_mat is not None:
            assert(self.hilbert.size == h_mat.shape[0] == h_mat.shape[1])

            non_zero = np.sum(h_mat != 0.)
            self.h_nonzero_inds = np.zeros(non_zero, dtype=int)
            self.h_nonzero_vals = np.zeros(non_zero, dtype=h_mat.dtype)
            self.h_nonzero_secs = np.zeros(self.hilbert.size+1, dtype=int)

            count = 0
            for j in range(self.hilbert.size):
                nonzeros = h_mat[j,:].nonzero()[0]
                self.h_nonzero_inds[count:(count+len(nonzeros))] = nonzeros
                self.h_nonzero_vals[count:(count+len(nonzeros))] = h_mat[j, nonzeros]
                self.h_nonzero_secs[j+1] = self.h_nonzero_secs[j] + len(nonzeros)
                count += len(nonzeros)

            self.h_nonzero_inds = jnp.array(self.h_nonzero_inds)
            self.h_nonzero_vals = jnp.array(self.h_nonzero_vals)
            self.h_nonzero_secs = jnp.array(self.h_nonzero_secs)
        else:
            self.h_nonzero_inds = None
            self.h_nonzero_vals = None
            self.h_nonzero_secs = None

        if eri_mat is not None:
            assert(self.hilbert.size == eri_mat.shape[0] == eri_mat.shape[1] == eri_mat.shape[2] == eri_mat.shape[3])

            non_zero = np.sum(eri_mat != 0.)
            self.eri_nonzero_inds = np.zeros((non_zero,2), dtype=int)
            self.eri_nonzero_vals = np.zeros(non_zero, dtype=eri_mat.dtype)
            self.eri_nonzero_secs = np.zeros(self.hilbert.size**2+1, dtype=int)

            count = 0
            for j in range(self.hilbert.size):
                for i in range(self.hilbert.size):
                    nonzeros = np.array(eri_mat[:,i,:,j].nonzero()).T
                    self.eri_nonzero_inds[count:(count+nonzeros.shape[0]), :] = nonzeros
                    self.eri_nonzero_vals[count:(count+nonzeros.shape[0])] = eri_mat[nonzeros[:,0],i,nonzeros[:,1],j]
                    self.eri_nonzero_secs[j*self.hilbert.size + i + 1] = self.eri_nonzero_secs[j*self.hilbert.size + i] + nonzeros.shape[0]
                    count += nonzeros.shape[0]

            self.eri_nonzero_inds = jnp.array(self.eri_nonzero_inds)
            self.eri_nonzero_vals = jnp.array(self.eri_nonzero_vals)
            self.eri_nonzero_secs = jnp.array(self.eri_nonzero_secs)
        else:
            self.eri_nonzero_inds = None
            self.eri_nonzero_vals = None
            self.eri_nonzero_secs = None

    """
    This is really only for testing purposes, expectation value automatically
    applies code below.
    """
    def get_conn_flattened(self, x, sections, pad=True):
        assert(not pad or self.hilbert._has_constraint)

        x_primes, mels = self._get_conn_flattened_kernel(np.asarray(x, dtype = np.uint8), sections, np.array(self.h_nonzero_vals),
                                                         np.array(self.h_nonzero_inds), np.array(self.h_nonzero_secs),
                                                         np.array(self.eri_nonzero_vals), np.array(self.eri_nonzero_inds),
                                                         np.array(self.eri_nonzero_secs))

        return x_primes, mels

    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(x, sections, h, h_nz, h_nz_secs, eri, eri_nz, eri_nz_secs):
        range_indices = np.arange(x.shape[-1])

        x_prime = np.empty((0, x.shape[1]), dtype=np.uint8)
        mels = np.empty(0, dtype=np.complex128)

        n_orbs = x.shape[-1]

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

            connected_con = len(up_occ_inds) * x.shape[1]
            connected_con += len(down_occ_inds) * x.shape[1]
            connected_con += len(down_occ_inds) * (len(down_occ_inds)-1) * x.shape[1] * x.shape[1]
            connected_con += len(up_occ_inds) * (len(up_occ_inds)-1) * x.shape[1] * x.shape[1]
            connected_con += len(up_occ_inds) * len(down_occ_inds) * x.shape[1] * x.shape[1]
            end_val = c + connected_con

            x_conn = np.empty((connected_con, x.shape[1]), dtype=np.uint8)
            x_conn[:,:] = x[batch_id,:]
            x_prime = np.append(x_prime, x_conn, axis=0)
            mels = np.append(mels, np.zeros(connected_con))

            # One-body parts
            for i in up_occ_inds:
                for k in range(h_nz_secs[i], h_nz_secs[i+1]):
                    a = h_nz[k]
                    x_prime[c, :] = x[batch_id, :]
                    multiplicator = apply_hopping(i, a, x_prime[c], 1,
                                                  cummulative_count=up_count)
                    mels[c] = multiplicator * h[k]
                    c += 1
            for i in down_occ_inds:
                for k in range(h_nz_secs[i], h_nz_secs[i+1]):
                    a = h_nz[k]
                    x_prime[c, :] = x[batch_id, :]
                    multiplicator = apply_hopping(i, a, x_prime[c], 2,
                                                  cummulative_count=down_count)
                    mels[c] = multiplicator * h[k]
                    c += 1

            # Two body parts
            for i in up_occ_inds:
                parity_count_i = (up_count[i]-1)
                for j in up_occ_inds:
                    if j != i:
                        parity_count_j = (parity_count_i + up_count[j] - 1)
                        if j > i:
                            parity_count_j = parity_count_j - 1
                        for k in range(eri_nz_secs[j*n_orbs+i], eri_nz_secs[j*n_orbs+i+1]):
                            a = eri_nz[k, 1]
                            b = eri_nz[k, 0]
                            parity_count = parity_count_j + up_count[a] + up_count[b]
                            parity_count -= int(a >= j) + int(a >= i) + int(b >= j) + int(b >= i)
                            parity_count += int(b >= a)
                            if parity_count & 1:
                                multiplicator = -1
                            else:
                                multiplicator = 1
                            x_prime[c, :] = x[batch_id, :]
                            x_prime[c, i] -= 1
                            x_prime[c, j] -= 1
                            if x_prime[c, a] & 1:
                                multiplicator *= 0
                            else:
                                x_prime[c, a] += 1
                                if x_prime[c, b] & 1:
                                    multiplicator *= 0
                                else:
                                    x_prime[c, b] += 1
                            mels[c] = 0.5 * multiplicator * eri[k]
                            c += 1

            for i in down_occ_inds:
                parity_count_i = (down_count[i]-1)
                for j in down_occ_inds:
                    if j != i:
                        parity_count_j = (parity_count_i + down_count[j] - 1)
                        if j > i:
                            parity_count_j = parity_count_j - 1
                        for k in range(eri_nz_secs[j*n_orbs+i], eri_nz_secs[j*n_orbs+i+1]):
                            a = eri_nz[k, 1]
                            b = eri_nz[k, 0]
                            parity_count = parity_count_j + down_count[a] + down_count[b]
                            parity_count -= int(a >= j) + int(a >= i) + int(b >= j) + int(b >= i)
                            parity_count += int(b >= a)
                            if parity_count & 1:
                                multiplicator = -1
                            else:
                                multiplicator = 1
                            x_prime[c, :] = x[batch_id, :]
                            x_prime[c, i] -= 2
                            x_prime[c, j] -= 2
                            if x_prime[c, a] & 2:
                                multiplicator *= 0
                            else:
                                x_prime[c, a] += 2
                                if x_prime[c, b] & 2:
                                    multiplicator *= 0
                                else:
                                    x_prime[c, b] += 2
                            mels[c] = 0.5 * multiplicator * eri[k]
                            c += 1

            for i in up_occ_inds:
                parity_count_i = (up_count[i]-1)
                for j in down_occ_inds:
                    parity_count_j = parity_count_i + down_count[j] - 1
                    for k in range(eri_nz_secs[j*n_orbs+i], eri_nz_secs[j*n_orbs+i+1]):
                        a = eri_nz[k, 1]
                        b = eri_nz[k, 0]
                        parity_count = parity_count_j + down_count[a] + up_count[b]
                        parity_count -= int(a >= j) + int(b >= i)
                        if parity_count & 1:
                            multiplicator = -1
                        else:
                            multiplicator = 1

                        x_prime[c, :] = x[batch_id, :]
                        x_prime[c, i] -= 1
                        x_prime[c, j] -= 2
                        if x_prime[c, a] & 2:
                            multiplicator = 0
                        else:
                            x_prime[c, a] += 2
                            if x_prime[c, b] & 1:
                                multiplicator = 0
                            else:
                                x_prime[c, b] += 1
                        mels[c] = multiplicator * eri[k]
                        c += 1
            c = end_val
            sections[batch_id] = c
        return x_prime, mels

def local_en_on_the_fly(logpsi, pars, samples, args, use_fast_update=False, chunk_size=None):
    h = args[0]
    h_nonzero_inds = args[1]
    h_nonzero_secs = args[2]
    eri = args[3]
    eri_nonzero_inds = args[4]
    eri_nonzero_secs = args[5]

    n_orbs = samples.shape[-1]

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

        # Compute log_amp of sample
        if use_fast_update:
            log_amp, intermediates_cache = logpsi(pars, sample, mutable="intermediates_cache", cache_intermediates=True)
            parameters = {**pars, **intermediates_cache}
        else:
            log_amp = logpsi(pars, sample)

        """ This function returns the log_amp of the connected configuration which is only specified
        by the occupancy on the updated sites as well as the indices of the sites updated."""
        def get_connected_log_amp(updated_occ_partial, update_sites):
            if use_fast_update:
                log_amp_connected = logpsi(parameters, updated_occ_partial, update_sites=update_sites)
            else:
                updated_config = sample.at[update_sites].set(updated_occ_partial)
                log_amp_connected = logpsi(pars, updated_config)
            return log_amp_connected

        # Computes term from single electron hop
        def compute_connected_1B(args):
            i = args[0][0]
            a = args[0][1]
            spin_int = args[1]
            el_count = args[2]
            def valid_hop(_):
                # Updated config at update sites
                new_occ = jnp.array([sample[i]-spin_int, sample[a]+spin_int], dtype=jnp.uint8)
                update_sites = jnp.array([i, a])
                # Get parity
                parity_count = el_count[i] + el_count[a] - (a>=i).astype(int) - 1
                parity_multiplicator = -2*(parity_count & 1) + 1
                # Evaluate amplitude ratio
                log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
                return (amp_ratio * parity_multiplicator).astype(complex)
            def invalid_hop(_):
                return 0.j
            return jax.lax.cond((sample[a]&spin_int).astype(bool), invalid_hop, valid_hop, None)

        # One-body up
        def outer_loop_occ(arg):
            i = up_occ_inds[arg[0]]
            def inner_loop(arg):
                a = h_nonzero_inds[arg[0]]
                value = jax.lax.cond(a != i, compute_connected_1B, lambda _: jnp.array(1., dtype=complex), ((i, a), 1, up_count))
                return (arg[0] + 1, arg[1] + h[arg[0]] * value)
            def stop_cond(arg):
                return arg[0] < h_nonzero_secs[i+1]
            value = jax.lax.while_loop(stop_cond, inner_loop, (h_nonzero_secs[i], 0.))[1]
            return (arg[0] + 1, arg[1] + value)
        local_en = jax.lax.while_loop(up_occ_cond, outer_loop_occ, (0, 0.))[1]

        # One-body down
        def outer_loop_occ(arg):
            i = down_occ_inds[arg[0]]
            def inner_loop(arg):
                a = h_nonzero_inds[arg[0]]
                value = jax.lax.cond(a != i, compute_connected_1B, lambda _: jnp.array(1., dtype=complex), ((i, a), 2, down_count))
                return (arg[0] + 1, arg[1] + h[arg[0]] * value)
            def stop_cond(arg):
                return arg[0] < h_nonzero_secs[i+1]
            value = jax.lax.while_loop(stop_cond, inner_loop, (h_nonzero_secs[i], 0.))[1]
            return (arg[0] + 1, arg[1] + value)
        local_en += jax.lax.while_loop(down_occ_cond, outer_loop_occ, (0, 0.))[1]

        occ_inds = (up_occ_inds, down_occ_inds)
        spin_ints = jnp.array([1, 2])
        electron_counts = (up_count, down_count)
        occ_conds = (up_occ_cond, down_occ_cond)

        # Helper function which adds an update by a creation or an annihilation operator
        def update_config(site, update_sites, updated_conf, spin_int, create):
            index_in_sites = jnp.squeeze(jnp.nonzero(update_sites==site, size=1, fill_value=-1)[0])
            update_sign = jax.lax.cond(create, lambda _: 1, lambda _: -1, None)
            def update(index):
                valid = jnp.array((updated_conf[index]&spin_int), dtype=bool) ^ create
                new_occ = jnp.append(updated_conf.at[index].add(update_sign * spin_int), 4)
                update_sites_new = jnp.append(update_sites, -1)
                return new_occ, valid, update_sites_new
            def append(index):
                valid = jnp.array((sample[site]&spin_int), dtype=bool) ^ create
                new_occ = jnp.append(updated_conf, sample[site] + update_sign * spin_int)
                update_sites_new = jnp.append(update_sites, site)
                return new_occ, valid, update_sites_new
            return jax.lax.cond(jnp.squeeze(index_in_sites!=-1), update, append, index_in_sites)


        # Two body terms
        # TODO: Clean this up, make this more readable, add documentation
        def get_two_body_contraction(spin_indices):
            def first_loop_occ(arg):
                i = occ_inds[spin_indices[0]][arg[0]]
                update_sites_i = jnp.array([i])
                new_occ_i = jnp.array([sample[i]-spin_ints[spin_indices[0]]])
                parity_count_i = electron_counts[spin_indices[0]][i] - 1
                valid_i = True
                def second_loop_occ(arg):
                    j = occ_inds[spin_indices[1]][arg[0]]
                    new_occ_j, valid_j, update_sites_j = update_config(j, update_sites_i, new_occ_i, spin_ints[spin_indices[1]], False)
                    parity_count_j = parity_count_i + electron_counts[spin_indices[1]][j] - 1
                    parity_count_j -= jnp.array((j > i)*(spin_indices[0] == spin_indices[1]), dtype=int)
                    def compute_inner(_):
                        def inner_loop_unocc(arg):
                            a = eri_nonzero_inds[arg[0], 1]
                            b = eri_nonzero_inds[arg[0], 0]
                            new_occ_a, valid_a, update_sites_a = update_config(a, update_sites_j, new_occ_j, spin_ints[spin_indices[1]], True)
                            new_occ_b, valid_b, update_sites_b = update_config(b, update_sites_a, new_occ_a, spin_ints[spin_indices[0]], True)
                            valid = jnp.logical_and(valid_a, valid_b)

                            def get_val(_):
                                parity_count = parity_count_j + electron_counts[spin_indices[1]][a] + electron_counts[spin_indices[0]][b]
                                parity_count -= jnp.array(a >= j, dtype=int) + jnp.array((a >= i)*(spin_indices[0] == spin_indices[1]), dtype=int)
                                parity_count -= jnp.array((b >= j)*(spin_indices[0] == spin_indices[1]), dtype=int) + jnp.array(b >= i, dtype=int)
                                parity_count += jnp.array((b >= a)*(spin_indices[0] == spin_indices[1]), dtype=int)
                                parity_multiplicator = -2*(parity_count & 1) + 1
                                true_updates = jnp.logical_and(new_occ_b != 4, new_occ_b != sample[update_sites_b])
                                no_updates = jnp.sum(true_updates)
                                def updates_0(_):
                                    return jnp.array(1., dtype=complex)
                                def update(no_updates, _):
                                    update_inds = jnp.nonzero(true_updates, size=no_updates)[0]
                                    new_occ = new_occ_b[update_inds]
                                    update_sites = update_sites_b[update_inds]
                                    log_amp_connected = get_connected_log_amp(new_occ, update_sites)
                                    amp_ratio = jnp.squeeze(jnp.exp(log_amp_connected - log_amp))
                                    return amp_ratio
                                amp_ratio = jax.lax.switch(no_updates, [updates_0, partial(update, 1),
                                                                        partial(update, 2), partial(update, 3),
                                                                        partial(update, 4)], None)
                                return (eri[arg[0]] * amp_ratio * parity_multiplicator).astype(complex)

                            value = jax.lax.cond(valid, get_val, lambda _: jnp.array(0., dtype=complex), None)
                            return (arg[0] + 1, arg[1] + value)
                        def stop_cond(arg):
                            return arg[0] < eri_nonzero_secs[j*n_orbs+i+1]
                        return jax.lax.while_loop(stop_cond, inner_loop_unocc, (eri_nonzero_secs[j*n_orbs+i], 0.))[1]
                    value = jax.lax.cond(valid_j, compute_inner, lambda _: jnp.array(0., dtype=complex), None)
                    return (arg[0] + 1, arg[1] + value)
                value = jax.lax.while_loop(partial(loop_stop_cond, occ_inds[spin_indices[1]]), second_loop_occ, (0, 0.))[1]
                return (arg[0] + 1, arg[1] + value)
            return jax.lax.while_loop(partial(loop_stop_cond, occ_inds[spin_indices[0]]), first_loop_occ, (0, 0.))[1]

        local_en += 0.5 * get_two_body_contraction((0,0))
        local_en += 0.5 * get_two_body_contraction((1,1))
        local_en += get_two_body_contraction((0,1))

        return local_en
    return nkjax.vmap_chunked(vmap_fun, chunk_size=chunk_size)(samples)

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: AbInitioHamiltonianSparse):
    samples = vstate.samples
    return (samples, (op.h_nonzero_vals, op.h_nonzero_inds, op.h_nonzero_secs, op.eri_nonzero_vals,
                      op.eri_nonzero_inds, op.eri_nonzero_secs))

@nk.vqs.get_local_kernel.dispatch(precedence=1)
def get_local_kernel(vstate: nk.vqs.MCState, op: AbInitioHamiltonianSparse, chunk_size: Optional[int] = None):
    try:
        use_fast_update = vstate.model.apply_fast_update
    except NameError:
        use_fast_update = False
    return nkjax.HashablePartial(local_en_on_the_fly, use_fast_update=use_fast_update, chunk_size=chunk_size)
