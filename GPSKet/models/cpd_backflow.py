import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Optional
from GPSKet.hilbert import FermionicDiscreteHilbert
from GPSKet.models import occupancies_to_electrons
from GPSKet.nn.initializers import normal
from netket.utils.array import HashableArray
from netket.utils.types import Array, DType, NNInitFunc


@jax.jit
def CPD_single_orbital(epsilon_a: Array, n_b: Array) -> Array:
    """
    Computes the CP-decomposition of a single multi-particle orbital
    and returns the site products for caching

    Args:
        epsilon_a: the variational parameter tensor of the α-th orbital, has shape (N, 4, M, K)
        n_b: occupations of K orbitals in the environment of the α-th orbital, has shape (K,)
    
    Returns:
        - an array of shape (N,) with the values of the multi-particle orbital for every electron
    """
    v = jnp.take_along_axis(epsilon_a, jnp.expand_dims(n_b, (0, 1, 2)), axis=1) # (N, 1, M, K)
    p = jnp.prod(v, axis=-1).squeeze(axis=1) # (N, M)
    o = jnp.sum(p, axis=-1) # (N,)
    return  o

def CPD(epsilon, n_b):
    if len(n_b.shape) == 2:
        # Environments matrix is not None
        in_axes = (0, 0)
    else:
        in_axes = (0, None)
    return jax.vmap(CPD_single_orbital, in_axes=in_axes)(epsilon, n_b)

CPD_batched = jax.vmap(CPD, in_axes=(None, 0))

class CPDBackflow(nn.Module):
    """
    Implements a backflow model with configuration-dependent CP-decomposed molecular orbitals,
    with support for restricted, unsrestricted or generalized spin-orbitals.
    
    The interaction between orbitals is all-to-all by default, but it can be restricted to only
    K orbitals by defining an environment matrix of shape (n_orbitals, K) which contains the indices
    of orbitals β for every orbitals α.
    """

    hilbert: FermionicDiscreteHilbert
    """The Hilbert space of the wavefunction model"""
    M: int
    """Support dimension of the CP-decomposition"""
    environments: HashableArray = Optional[None]
    """Matrix of shape (n_orbitals, K), where K <= n_orbitals"""
    dtype: DType = jnp.complex128
    """Parameter dtype"""
    init_fun: NNInitFunc = normal
    """Initialization function for the variational parameters"""
    restricted: bool = True
    """Whether the α and β orbitals are the same or not"""
    fixed_magnetization: bool = True
    """Whether magnetization should be conserved or not"""

    def setup(self):
        # Size of local Hilbert space
        D = self.hilbert._local_size
        # Number of atomic orbitals (also indicated as L in the code below)
        self._norb = self.hilbert.size
        # Number of spin-up and down electrons (n_elec_up, n_elec_down)
        self._n_elec = self.hilbert._n_elec
        # Total number of electrons
        N = np.sum(self._n_elec)
        # Number of other orbitals in the environment of each
        K = self.environments.shape[1] if self.environments else self._norb
        if self.fixed_magnetization:
            if self.restricted:
                assert self._n_elec[0] == self._n_elec[1]
                shape = (self._norb, N//2, D, self.M, K)
            else:
                shape = (self._norb, N, D, self.M, K)
        else:
            shape = (2*self._norb, N, D, self.M, K)
        if self.init_fun is None:
            init_fun = normal(dtype=self.dtype)
        else:
            init_fun = self.init_fun
        if self.environments is not None:
            self._environments = self.variable(
                "orbitals", "environments", lambda: jnp.array(self.environments)
            )
        self._epsilon = self.param("epsilon", init_fun, shape, self.dtype)

    @nn.compact
    def __call__(self, n) -> Array:
        # TODO: Add support for fast-updating
        # Convert 2nd quantized configurations to orbital indices
        R = occupancies_to_electrons(n, self._n_elec) # (B, N)
        R = jnp.expand_dims(R, axis=-1)

        # If an environments matrix is defined, then take only occupations of orbitals in environment of each.
        # This matrix needs to be of shape (L, K), with the α-th row corresponding to the K indices of orbitals
        # β in the environment of orbital α
        if self.environments is not None:
            n = jnp.take_along_axis(
                jnp.expand_dims(n, 2),
                jnp.expand_dims(self._environments.value, 0),
                axis=1
            ) # (B, L, K)

        # Compute orbitals and log-determinant
        orbitals = CPD_batched(self._epsilon, n) # (B, L, N)
        if self.fixed_magnetization:
            r_up, r_dn = jnp.split(R, np.array([self._n_elec[0]]), axis=1) # (B, N_up, 1), (B, N_dn, 1)
            if self.restricted:
                # Restricted spin-orbitals
                U_up = jnp.take_along_axis(orbitals, r_up, axis=1)
                U_dn = jnp.take_along_axis(orbitals, r_dn, axis=1)
            else:
                # Unrestricted spin-orbitals
                U_up = jnp.take_along_axis(orbitals[:, :, : self._n_elec[0]], r_up, axis=1)
                U_dn = jnp.take_along_axis(orbitals[:, :, self._n_elec[0] :], r_dn, axis=1)
            (s_up, log_det_up) = jnp.linalg.slogdet(U_up)
            (s_dn, log_det_dn) = jnp.linalg.slogdet(U_dn)
            log_det = log_det_up + log_det_dn + jnp.log(s_up * s_dn + 0j)  # (B,)
        else:
            # Generalized spin-orbitals
            R = R.at[:, self._n_elec[0] :, :].add(self._norb) # (B, N, 1)
            U = jnp.take_along_axis(orbitals, R, axis=1)
            (s, log_det) = jnp.linalg.slogdet(U)
            log_det = log_det + jnp.log(s + 0j)  # (B,)

        return log_det