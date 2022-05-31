import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional
from qGPSKet.hilbert import FermionicDiscreteHilbert
from netket.utils.types import Array, Callable, DType, NNInitFunc
from netket.utils import HashableArray
from .asymm_qGPS import occupancies_to_electrons, _evaluate_determinants
from functools import partial

# Dimensions:
# - B = batch size
# - L = number of sites
# - N = total number of electrons
# - N_up = number of spin-up electrons
# - N_down = number of spin-down electrons
# - M = number of determinants
# - S = number of spin rotations for S^2 projection
# - T = number of symmetries

class Slater(nn.Module):
    """
    This defines a set of M Slater determinants.
    Per default these are summed together but other ways of combining them are possible.
    """

    hilbert: FermionicDiscreteHilbert
    """Hilbert space"""

    n_determinants: int = 1
    """Number of determinants"""

    dtype: DType = jnp.complex128

    init_fun : NNInitFunc = jax.nn.initializers.orthogonal()

    symmetries: Callable = lambda inputs : jnp.expand_dims(inputs, axis=-1)

    out_transformation: Callable = lambda x: jax.scipy.special.logsumexp(x, axis=(1, -1, -2))
    """Final output transformation. Its input has shape (B, M, S, T)."""

    spin_symmetry_by_structure: bool = False
    """ Flag determines whether the S^2 symmetry (with S=0) should be enforced
    by using the same orbitals for up and down spin.
    """

    S2_projection: Optional[Tuple[HashableArray, HashableArray]] = None
    """ This (optional) tuple specifies the angles and characters for spin
    rotations which should be used for the S^2 projection. Only sensible if above flag
    is false.
    """

    fixed_magnetization: bool = True
    """Whether this is a SD with fixed particle number and magnetization or not."""

    apply_fast_update: bool = True


    def setup(self):
        """ Just do some sanity checks that the chosen parameters actually make sense.
        TODO: Clean this up."""
        assert(self.hilbert._n_elec is not None) # SD currently only implemented for a Hilbert with a fixed magnetization/electron number
        if self.spin_symmetry_by_structure:
            assert(self.S2_projection is None) # While technically possible this does not make sense.
        if self.S2_projection is not None:
            assert(self.fixed_magnetization)

    @nn.compact
    def __call__(self, x, cache_intermediates=False, update_sites=None) -> Array:
        n_sites = self.hilbert.size

        # This code is applied if the value is calculated from scratch
        if update_sites is None:
            # First set up the full matrices of orbitals (with spin-down orbitals indexed by indices site + L)
            if self.fixed_magnetization:
                U_up = self.param("U_up", self.init_fun, (self.n_determinants, n_sites, self.hilbert._n_elec[0]), self.dtype)
                if self.spin_symmetry_by_structure:
                    U_down = U_up
                else:
                    U_down = self.param("U_down", self.init_fun, (self.n_determinants, n_sites, self.hilbert._n_elec[1]), self.dtype)

                def get_full_U(up_part, down_part):
                    return jnp.block([[up_part, jnp.zeros((n_sites, down_part.shape[1]), dtype=up_part.dtype)],
                                      [jnp.zeros((n_sites, up_part.shape[1]), dtype=up_part.dtype), down_part]])

                full_U = jax.vmap(get_full_U)(U_up, U_down) # (M, 2 * L, N)
            else:
                full_U = self.param("U", self.init_fun, (self.n_determinants, 2*n_sites, self.hilbert._n_elec[0]+self.hilbert._n_elec[1]), self.dtype) # (M, 2 * L, N)

            # Now include the rotations, after this full_U will have shape (M, 2*L, N, S)
            if self.S2_projection is None:
                full_U = jnp.expand_dims(full_U, axis=-1) #(M, 2*L, N, S)
            else:
                def apply_rotation(angle):
                    # Apply the rotation to the orbitals
                    U00 = full_U[:, :n_sites, :self.hilbert._n_elec[0]] * jnp.cos(angle/2) + full_U[:, n_sites:, :self.hilbert._n_elec[0]] * jnp.sin(angle/2)
                    U10 = full_U[:, :n_sites, :self.hilbert._n_elec[0]] * jnp.sin(angle/2) + full_U[:, n_sites:, :self.hilbert._n_elec[0]] * jnp.cos(angle/2)
                    U01 = full_U[:, :n_sites, self.hilbert._n_elec[0]:] * jnp.cos(angle/2) - full_U[:, n_sites:, self.hilbert._n_elec[0]:] * jnp.sin(angle/2)
                    U11 = -full_U[:, :n_sites, self.hilbert._n_elec[0]:] * jnp.sin(angle/2) + full_U[:, n_sites:, self.hilbert._n_elec[0]:] * jnp.cos(angle/2)
                    U_rotated = jnp.block([[U00, U01],
                                           [U10, U11]])
                    return U_rotated
                full_U = jax.vmap(apply_rotation, in_axes=0, out_axes=-1)(jnp.array(self.S2_projection[0])) #(M, 2*L, N, S)

            # Convert second quantized representation to first quantized representation
            y = occupancies_to_electrons(x, self.hilbert._n_elec)
            y = self.symmetries(y).at[:, self.hilbert._n_elec[0]:, :].add(n_sites) # From now on a position >= L correspond to the spin-down orbitals

            # Construct the sub-matrices where the rows of unoccupied sites have been removed
            inner_take_over_rotations = jax.vmap(partial(jnp.take, axis=1), in_axes=(-1, None), out_axes=-1) # vmap over rotations
            U_submats_per_sample = jax.vmap(inner_take_over_rotations, in_axes=(None, -1), out_axes=-1) # vmap over symmetries
            U_submats = jax.vmap(U_submats_per_sample, in_axes=(None, 0), out_axes=0)(full_U, y) # vmap over batch

            # Now evaluate the determinants
            def evaluate_SD(U_submat):
                if self.S2_projection is None and self.fixed_magnetization:
                    # Compute Slater determinant as product of the determinants of the
                    # spin-up and spin-down orbital submatrices:
                    # SD = det(Ũ_up)det(Ũ_down) which only works if no spin rotation is applied and the magnetization is conserved
                    (s_up, log_det_up) = jnp.linalg.slogdet(U_submat[:, :, :self.hilbert._n_elec[0], :self.hilbert._n_elec[0]])
                    (s_down, log_det_down) = jnp.linalg.slogdet(U_submat[:, :, self.hilbert._n_elec[0]:, self.hilbert._n_elec[0]:])
                    return log_det_up + log_det_down + jnp.log(s_up*s_down+0j)
                else:
                    (s_det, log_det) = jnp.linalg.slogdet(U_submat)
                    return log_det + jnp.log(s_det+0j)

            evaluate_over_rotations = jax.vmap(evaluate_SD, in_axes=-1, out_axes=-1)
            log_det_values = jax.vmap(evaluate_over_rotations, in_axes=-1, out_axes=-1)(U_submats) # (B, M, S, T)

            # If we store the intermediates for fast updating, we need to invert the sub-matrices
            if cache_intermediates:
                inverse_over_rotations = jax.vmap(jnp.linalg.inv, in_axes=-1, out_axes=-1) # vmap over rotations
                inverted_submats = jax.vmap(inverse_over_rotations, in_axes=-1, out_axes=-1)(U_submats) # vmap over symmetries, output has shape (B, M, N, N, S, T)

        # Apply fast updating of the determinants
        else:
            """ Consecutive updates are not yet supported (but will hopefully be soon)
            -> requires a bit of checking what type of update we want to perform (see below),
            as well as updating the inverse of the submatrix of U """
            assert(not cache_intermediates)

            # First we need to determine which electrons move (and where they move)
            occupancies_save = self.variable("intermediates_cache", "occupancies", lambda : None).value
            old_occupancies = jax.vmap(jnp.take, in_axes=(0, 0), out_axes=0)(occupancies_save, update_sites)
            spin_up_updates = (old_occupancies & 1).astype(int) - (x & 1).astype(int)
            spin_down_updates = ((old_occupancies & 2).astype(int) - (x & 2).astype(int))//2
            updates = jnp.concatenate((spin_up_updates, spin_down_updates), axis=-1) # 0: nothing, 1: remove electron, -1: add electron

            # Determines the sites electrons jump to, the electron ids of the jumping electrons and the parity sign obtained from this move
            @jax.vmap
            def get_add_site_el_id_parity(update_sites_single, updates_single, sites_to_el_single, cum_elec_count):
                update_sites_spin_channel_split = jnp.concatenate((update_sites_single, update_sites_single+n_sites))
                """ We pad the update arrays with (-1)s to indicate dummy updates, this means that the determinant which is taken below
                might be a little bit more expensive than necessary (as we might (and typically) have less updates than indicated
                by the size of the update_sites array). We could also apply a little hack and assume that the length of the update
                sites array is equal to the number of electron updates but the implementation with the padding is a bit more general
                and safer to use. """
                add_ids, = jnp.nonzero(updates_single == -1, size=update_sites_single.shape[0], fill_value=-1)
                add_sites = jnp.where(add_ids != -1, update_sites_spin_channel_split[add_ids], -1)
                remove_ids, = jnp.nonzero(updates_single == 1, size=update_sites_single.shape[0], fill_value=-1)
                remove_sites = jnp.where(remove_ids != -1, update_sites_spin_channel_split[remove_ids], -1)
                el_ids = jnp.where(remove_ids != -1, sites_to_el_single[remove_sites], -1)

                """ Now we need to compute the additional sign we get from pretending these are updates in first quantization but
                we want to have the update w.r.t. configs in second quantization (i.e. with a well-defined ordering of the electrons).
                Maybe one day we want to code up the Hamiltonians + samplers in first quantization, then this would not be required
                but until then we essentially need to revert the computed parity sign we evaluated in the Hamiltonian."""
                def loop_fun(index, val):
                    sign, cum_elec_count_add, cum_elec_count_rm = val

                    """ Note that the arrays add_ids, add_sites, remove_ids, remove_sites, el_ids are padded with (-1)s.
                    This can lead to unwanted effects if the code below is modified. """

                    # Count the number of electrons we move past in this move
                    no_of_electrons_passed = abs(cum_elec_count_rm[index]-cum_elec_count_add[index]).astype(int)
                    # Correction if the remove site is beyond the add site
                    no_of_electrons_passed = jnp.where(remove_sites[index] > add_sites[index], no_of_electrons_passed-1, no_of_electrons_passed)

                    # Sign modification
                    new_sign = sign * ((-1)**(no_of_electrons_passed & 1))

                    # Modify the cummulative electron counts for the following updates
                    cum_elec_count_add_sites_updated = jnp.where(add_sites >= add_sites[index], cum_elec_count_add, cum_elec_count_add+1)
                    cum_elec_count_add_sites_updated = jnp.where(add_sites >= remove_sites[index], cum_elec_count_add_sites_updated, cum_elec_count_add_sites_updated-1)

                    cum_elec_count_rm_sites_updated = jnp.where(remove_sites >= add_sites[index], cum_elec_count_rm, cum_elec_count_rm+1)
                    cum_elec_count_rm_sites_updated = jnp.where(remove_sites >= remove_sites[index], cum_elec_count_rm_sites_updated, cum_elec_count_rm_sites_updated-1)
                    return (new_sign, cum_elec_count_add_sites_updated, cum_elec_count_rm_sites_updated)

                cum_elec_count_add = cum_elec_count[add_sites]
                cum_elec_count_rm = cum_elec_count[remove_sites]

                sign_update, _, _ = jax.lax.fori_loop(0, add_sites.shape[0], loop_fun, (1, cum_elec_count_add, cum_elec_count_rm))

                return add_sites, el_ids, sign_update

            sites_to_electron_ids_save = self.variable("intermediates_cache", "sites_to_electron_ids", lambda : None).value
            cumulative_electron_count_save = self.variable("intermediates_cache", "cumulative_electron_count", lambda : None).value
            add_sites, moving_electron_ids, sign_update = get_add_site_el_id_parity(update_sites, updates, sites_to_electron_ids_save, cumulative_electron_count_save)

            # Apply the symmetries to the add_sites
            add_sites_expanded = jnp.expand_dims(add_sites, axis=-1)
            add_sites_sym = self.symmetries(add_sites%n_sites)
            add_sites_sym += (add_sites_expanded//n_sites) * n_sites # (B, N_updates, T)
            add_sites_sym = jnp.where(add_sites_expanded != -1, add_sites_sym, -1)

            def get_determinant_update(add_sites_single, moving_electron_ids_single, update_matrix):
                """ The update is just the determinant of a matrix which has those rows from the update matrix corresponding to the sites
                where an electron is added, and the columns of the electrons which are moving. """
                up_mat = jnp.eye(add_sites_single.shape[0])
                up_mat = jnp.where(jnp.expand_dims(add_sites_single, axis=-1) != -1, update_matrix[jnp.ix_(add_sites_single, moving_electron_ids_single)], up_mat)
                (s_det_update, log_det_update) = jnp.linalg.slogdet(up_mat)
                return log_det_update + jnp.log(s_det_update+0j)

            log_det_update_per_determinant = jax.vmap(get_determinant_update, in_axes=(None, None, 0), out_axes=0) # vmap over determinants
            log_det_update_per_sample = jax.vmap(log_det_update_per_determinant, in_axes=(0, 0, 0), out_axes=0) # vmap over batch dimension
            log_det_update_per_rotation = jax.vmap(log_det_update_per_sample, in_axes=(None, None, -1), out_axes=-1) # vmap over rotations
            log_det_update_per_symmetry = jax.vmap(log_det_update_per_rotation, in_axes=(-1, None, -1), out_axes=-1) # vmap over symmetries

            update_matrices_save = self.variable("intermediates_cache", "update_matrices", lambda : None).value
            log_dets_save = self.variable("intermediates_cache", "log_dets", lambda : None).value
            log_det_values = log_det_update_per_symmetry(add_sites_sym, moving_electron_ids, update_matrices_save) + log_dets_save
            log_det_values += jnp.expand_dims(jnp.log(sign_update +0.j), axis=(1,2,3))


        # Store everything we need to store for subsequent fast updates
        if cache_intermediates:
            # Store the inverted occupied submatrices (required for double updates)
            self.variable("intermediates_cache", "inverted_submats", lambda : None).value = inverted_submats

            """ Store the update matrices (U.dot(inv(U_occ))); this is O(N^2 * L) but allows for O(1) updates to the determinant values:
            If we make a single set of O(L) different updates at a time (as e.g. in local energy evaluation for lattice models),
            then it would potentially be faster overall not to do evaluate this matrix-matrix product in the setup and instead do O(L) updates
            (though there would still be a benefit from the fast update).
            If we do consecutive O(1) updates (e.g. in the Metropolis sampling), we should definitely not compute the full
            matrix-matrix contraction as this essentially makes the fast updating useless -> TODO: find a good way to determine whether
            or not to evaluate the matrix-matrix product which allows for O(1) updates to the SD values.
            We could only evaluate this matrix-matrix contraction in the update step (rather than the setup step)
            and decide based on the number of updates whether or not to pre-compute it. This however needs a mod of the
            ab-initio Hamiltonian. """
            update_matrices_save = self.variable("intermediates_cache", "update_matrices", lambda : None)
            update_matrices_save.value = jnp.einsum("ijkl,miknlp->mijnlp", full_U, inverted_submats) # (B, M, L, N, S, T)

            # Store the full U matrices
            self.variable("intermediates_cache", "full_U", lambda : None).value = full_U

            # Store a mapping from sites to electron indices (-1 values denote unoccupied sites, spin-down sites are indexed by indices site_no + L)
            electron_positions = occupancies_to_electrons(x, self.hilbert._n_elec).at[:, self.hilbert._n_elec[0]:].add(n_sites)
            def set_occ_per_sample(electron_positions_single):
                def set_occ(i, sites_to_els):
                    return sites_to_els.at[electron_positions_single[i]].set(i)
                return jax.lax.fori_loop(0, electron_positions_single.shape[0], set_occ, -jnp.ones(self.hilbert.size*2, dtype=int))
            sites_to_electron_ids_save = self.variable("intermediates_cache", "sites_to_electron_ids", lambda : None)
            sites_to_electron_ids_save.value = jax.vmap(set_occ_per_sample)(electron_positions)

            # Store the occupancies
            self.variable("intermediates_cache", "occupancies", lambda : None).value = x

            # Store the cummulative elextron counts (required for fast evaluation of the parity update)
            cumulative_electron_count_save = self.variable("intermediates_cache", "cumulative_electron_count", lambda : None)
            cumulative_electron_count_save.value = jnp.cumsum(jnp.concatenate((x & 1, (x & 2)//2), axis=1), axis=1)

            # Store the calculated determinants
            self.variable("intermediates_cache", "log_dets", lambda : None).value = log_det_values

        if self.S2_projection is not None:
            log_det_values += jnp.log(jnp.asarray(self.S2_projection[1])).reshape((-1,1))

        return self.out_transformation(log_det_values)
