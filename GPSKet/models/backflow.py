import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from jax.scipy.special import logsumexp
from netket.utils import HashableArray
from netket.utils.types import Array, Callable
from GPSKet.hilbert import FermionicDiscreteHilbert
from GPSKet.models import occupancies_to_electrons


class Backflow(nn.Module):
    """
    Implements a (un)restricted backflow model which can conserve or break magnetization
    """

    hilbert: FermionicDiscreteHilbert
    """The Hilbert space of the wavefunction model"""
    orbitals: HashableArray
    """Initial orbitals, e.g. Hartree-Fock"""
    correction_fun: nn.Module
    """Module that computes the correction to the initial orbitals"""
    apply_symmetries: Callable = lambda inputs: jnp.expand_dims(inputs, axis=-1)
    """Function to apply symmetries to configurations"""
    apply_fast_update: bool = True
    """Whether to allow fast-updating or not"""
    spin_symmetry_by_structure: bool = True
    """Whether the α and β orbitals are the same or not"""
    fixed_magnetization: bool = True
    """Whether magnetization should be conserved or not"""

    @nn.compact
    def __call__(self, x, cache_intermediates=False, update_sites=None) -> Array:
        norb = self.hilbert.size
        nelec = self.hilbert._n_elec
        # TODO: improve performance by scanning over symmetries
        orbitals = jnp.array(self.orbitals)  # (L, N)
        corrections = self.correction_fun(
            x, cache_intermediates=cache_intermediates, update_sites=update_sites
        )  # (B, L, N, T)
        if cache_intermediates or (update_sites is not None):
            indices_save = self.variable(
                "intermediates_cache", "samples", lambda: jnp.zeros(0, dtype=x.dtype)
            )
        if update_sites is not None:

            def update_fun(saved_config, update_sites, occs):
                def scan_fun(carry, count):
                    return (carry.at[update_sites[count]].set(occs[count]), None)

                return jax.lax.scan(
                    scan_fun,
                    saved_config,
                    jnp.arange(update_sites.shape[0]),
                    reverse=True,
                )[0]

            full_x = jax.vmap(update_fun, in_axes=(0, 0, 0), out_axes=0)(
                indices_save.value, update_sites, x
            )
        else:
            full_x = x
        if cache_intermediates:
            indices_save.value = full_x
        y = occupancies_to_electrons(full_x, nelec)
        y = self.apply_symmetries(y)  # (B, N, T)
        y = jnp.expand_dims(y, axis=-2)  # (B, N, 1, T)
        if self.fixed_magnetization:
            y_up, y_dn = jnp.split(y, np.array([nelec[0]]), axis=1)
            if self.spin_symmetry_by_structure:
                ɸ_up = jnp.take_along_axis(
                    jnp.expand_dims(orbitals, axis=(0, -1)), y_up, axis=1
                )
                ɸ_dn = jnp.take_along_axis(
                    jnp.expand_dims(orbitals, axis=(0, -1)), y_dn, axis=1
                )
                Δ_up = jnp.take_along_axis(corrections, y_up, axis=1)
                Δ_dn = jnp.take_along_axis(corrections, y_dn, axis=1)
            else:
                ɸ_up = jnp.take_along_axis(
                    jnp.expand_dims(orbitals[:, : nelec[0]], axis=(0, -1)), y_up, axis=1
                )
                ɸ_dn = jnp.take_along_axis(
                    jnp.expand_dims(orbitals[:, nelec[0] :], axis=(0, -1)), y_dn, axis=1
                )
                Δ_up = jnp.take_along_axis(
                    corrections[:, :, : nelec[0], :], y_up, axis=1
                )
                Δ_dn = jnp.take_along_axis(
                    corrections[:, :, nelec[0] :, :], y_dn, axis=1
                )
            ɸ_up = jnp.transpose(ɸ_up, (0, 3, 1, 2))  # (B, T, N, N)
            ɸ_dn = jnp.transpose(ɸ_dn, (0, 3, 1, 2))  # (B, T, N, N)
            Δ_up = jnp.transpose(Δ_up, (0, 3, 1, 2))  # (B, T, N, N)
            Δ_dn = jnp.transpose(Δ_dn, (0, 3, 1, 2))  # (B, T, N, N)
            (s_up, log_det_up) = jnp.linalg.slogdet(ɸ_up + Δ_up)
            (s_dn, log_det_dn) = jnp.linalg.slogdet(ɸ_dn + Δ_dn)
            log_det = log_det_up + log_det_dn + jnp.log(s_up * s_dn + 0j)  # (B, T)
        else:
            y = y.at[:, nelec[0] :, :].add(norb)
            ɸ = jnp.take_along_axis(jnp.expand_dims(orbitals, axis=(0, -1)), y, axis=1)
            Δ = jnp.take_along_axis(corrections, y, axis=1)
            ɸ = jnp.transpose(ɸ, (0, 3, 1, 2))  # (B, T, N, N)
            Δ = jnp.transpose(Δ, (0, 3, 1, 2))  # (B, T, N, N)
            (s, log_det) = jnp.linalg.slogdet(ɸ + Δ)
            log_det = log_det + jnp.log(s + 0j)  # (B, T)
        return logsumexp(log_det, axis=-1)
