import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional
from qGPSKet.hilbert import FermionicDiscreteHilbert
from netket.utils.types import Array, Callable, DType, NNInitFunc
from netket.utils import HashableArray
from .asymm_qGPS import occupancies_to_electrons, _evaluate_determinants

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
    This defines a set of M Slater magnetization conserving determinants.
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


    def setup(self):
        """ Just do some sanity checks that the chosen parameters actually make sense.
        TODO: Clean this up."""
        assert(self.hilbert._n_elec is not None) # SD currently only implemented for a Hilbert
        if self.spin_symmetry_by_structure:
            assert(self.S2_projection is None) # While technically possible this does not make sense.
        if self.S2_projection is not None:
            assert(self.fixed_magnetization)

    @nn.compact
    def __call__(self, x) -> Array:
        n_sites = self.hilbert.size

        if self.fixed_magnetization:
            U_up = self.param("U_up", self.init_fun, (self.n_determinants, n_sites, self.hilbert._n_elec[0]), self.dtype)
            if self.spin_symmetry_by_structure:
                U_down = U_up
            else:
                U_down = self.param("U_down", self.init_fun, (self.n_determinants, n_sites, self.hilbert._n_elec[1]), self.dtype)

            def get_full_U(up_part, down_part):
                return jnp.block([[up_part, jnp.zeros((n_sites, down_part.shape[1]), dtype=up_part.dtype)],
                                  [jnp.zeros((n_sites, up_part.shape[1]), dtype=up_part.dtype), down_part]])

            full_U = jax.vmap(get_full_U)(U_up, U_down)
        else:
            full_U = self.param("U", self.init_fun, (self.n_determinants, 2*n_sites, self.hilbert._n_elec[0]+self.hilbert._n_elec[1]), self.dtype)

        y = occupancies_to_electrons(x, self.hilbert._n_elec)

        y = self.symmetries(y).at[:, self.hilbert._n_elec[0]:, :].add(n_sites) # From now on a position >= L correspond to the spin-down orbitals

        def evaluate_SD(U_submat):
            if self.S2_projection is None and self.fixed_magnetization:
                # Compute Slater determinant as product of the determinants of the
                # spin-up and spin-down orbital submatrices:
                # SD = det(Ũ_up)det(Ũ_down) which only works if no spin rotation is applied and the magnetization is conserved
                (s_up, log_det_up) = jnp.linalg.slogdet(U_submat[:self.hilbert._n_elec[0], :self.hilbert._n_elec[0]])
                (s_down, log_det_down) = jnp.linalg.slogdet(U_submat[self.hilbert._n_elec[0]:, self.hilbert._n_elec[0]:])
                return log_det_up + log_det_down + jnp.log(s_up*s_down+0j)
            else:
                (s_det, log_det) = jnp.linalg.slogdet(U_submat)
                return log_det + jnp.log(s_det+0j)

        def evaluate_SD_batch(y_batch):
            def evaluate_for_different_SD(U):
                def evaluate_SD_sym(y_sym):
                    if self.S2_projection is None:
                        U_submat = jnp.take(U, y_sym, axis=0)
                        log_det = evaluate_SD(U_submat)
                        return jnp.expand_dims(log_det, axis=-1)
                    else:
                        def get_SD_rotation(angle):
                            # Apply the rotation to the orbitals
                            U00 = U[:n_sites, :self.hilbert._n_elec[0]] * jnp.cos(angle/2) + U[n_sites:, :self.hilbert._n_elec[0]] * jnp.sin(angle/2)
                            U10 = U[:n_sites, :self.hilbert._n_elec[0]] * jnp.sin(angle/2) + U[n_sites:, :self.hilbert._n_elec[0]] * jnp.cos(angle/2)
                            U01 = U[:n_sites, self.hilbert._n_elec[0]:] * jnp.cos(angle/2) - U[n_sites:, self.hilbert._n_elec[0]:] * jnp.sin(angle/2)
                            U11 = -U[:n_sites, self.hilbert._n_elec[0]:] * jnp.sin(angle/2) + U[n_sites:, self.hilbert._n_elec[0]:] * jnp.cos(angle/2)
                            U_rotated = jnp.block([[U00, U01],
                                                   [U10, U11]])
                            U_submat = jnp.take(U_rotated, y_sym, axis=0)
                            return evaluate_SD(U_submat)
                        return jax.vmap(get_SD_rotation, in_axes=0, out_axes=-1)(jnp.array(self.S2_projection[0]))
                return jax.vmap(evaluate_SD_sym, in_axes=-1, out_axes=-1)(y_batch)
            return jax.vmap(evaluate_for_different_SD)(full_U)
        value = jax.vmap(evaluate_SD_batch)(y) # (B, M, S, T)

        if self.S2_projection is not None:
            value += jnp.log(jnp.asarray(self.S2_projection[1])).reshape((-1,1))

        return self.out_transformation(value)