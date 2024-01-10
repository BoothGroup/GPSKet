import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import GPSKet as qk
from scipy.linalg import circulant
from functools import partial
from flax import linen as nn
from netket.hilbert import HomogeneousHilbert
from netket.graph import AbstractGraph
from netket.utils import HashableArray
from netket.utils.types import Array
from ml_collections import ConfigDict
from typing import Union, Tuple, Callable, Optional


_MODELS = {
    "GPS": qk.models.qGPS,
    "FilterGPS": qk.models.qGPS,
    "ARGPS": qk.models.ARqGPSFull,
    "MaskedGPS": qk.models.ARqGPSFull,
    "ARFilterGPS": qk.models.ARPlaquetteqGPS,
    "MaskedFilterGPS": qk.models.ARPlaquetteqGPS,
}


def get_model(
    config: ConfigDict,
    hilbert: HomogeneousHilbert,
    graph: Optional[AbstractGraph] = None,
) -> nn.Module:
    """
    Return the model for a wavefunction Ansatz

    Args:
        config : experiment configuration file
        hilbert : Hilbert space on which the model should act
        graph : graph associated with the Hilbert space (optional)

    Returns:
        the model for the wavefunction Ansatz
    """
    name = config.model_name
    try:
        ma_cls = _MODELS[name]
    except KeyError:
        raise ValueError(f"Model {name} is not a valid class or is not supported yet.")
    if config.model.dtype == "real":
        dtype = jnp.float64
    elif config.model.dtype == "complex":
        dtype = jnp.complex128
    M = config.model.M
    init_fn = qk.nn.initializers.normal(sigma=config.model.sigma, dtype=dtype)
    if graph:
        groups = config.model.symmetries.split(",")
        translations = (
            "translations" in groups
            or groups[0] == "all"
            or groups[0] == "automorphisms"
        )
        point_symmetries = (
            "point-symmetries" in groups
            or groups[0] == "all"
            or groups[0] == "automorphisms"
        )
        spin_flip = (
            "spin-flip" in groups or groups[0] == "all" or groups[0] == "automorphisms"
        )
        symmetries_fn, inv_symmetries_fn = get_symmetry_transformation_spin(
            name, translations, point_symmetries, spin_flip, graph
        )
    else:
        symmetries_fn, inv_symmetries_fn = qk.models.no_syms()
    if name == "GPS" or name == "FilterGPS":
        axis = (-2, -1)
    else:
        axis = -1
    out_trafo = lambda x: jnp.sum(x, axis=axis)
    if name != "GPS" and name != "FilterGPS":
        if isinstance(hilbert, nk.hilbert.Spin):
            count_spins_fn = count_spins
            renormalize_log_psi_fn = renormalize_log_psi
        elif isinstance(hilbert, qk.hilbert.FermionicDiscreteHilbert):
            count_spins_fn = count_spins_fermionic
            renormalize_log_psi_fn = renormalize_log_psi_fermionic
        args = [hilbert, M]
        if "Filter" in name:
            args.extend(get_plaquettes_and_masks(hilbert, graph))
            apply_symmetries = symmetries_fn
        else:
            apply_symmetries = (symmetries_fn, inv_symmetries_fn)
        normalize = "AR" in name
        ma = ma_cls(
            *args,
            dtype=dtype,
            init_fun=init_fn,
            normalize=normalize,
            apply_symmetries=apply_symmetries,
            count_spins=count_spins_fn,
            renormalize_log_psi=renormalize_log_psi_fn,
            out_transformation=out_trafo,
        )
    else:
        if "Filter" in name:
            args = [hilbert, M]
            if config.system_name == "Hubbard1d":
                graph = nk.graph.Chain(config.system.Lx, pbc=config.system.pbc)
            symmetries_fn, inv_symmetries_fn = get_symmetry_transformation_spin(
                name, True, False, False, graph
            )
        else:
            args = [hilbert, hilbert.size * M]
        ma = ma_cls(
            *args,
            dtype=dtype,
            init_fun=init_fn,
            syms=(symmetries_fn, inv_symmetries_fn),
            out_transformation=out_trafo,
        )
    return ma


def get_symmetry_transformation_spin(
    name: str,
    translations: bool,
    point_symmetries: bool,
    spin_flip: bool,
    graph: AbstractGraph,
) -> Union[Tuple[Callable, Callable], Callable]:
    """
    Return the appropriate spin symmetry transformations

    Args:
        name : name of the Ansatz
        translations : whether to include translations or not
        point_symmetries : whether to include point-group symmetries or not
        spin_flip : whether to include spin_flip symmetry or not
        graph : underlying graph of the system

    Returns:
        spin symmetry transformations and their inverses
    """
    automorphisms = translations or point_symmetries
    if automorphisms:
        if translations and point_symmetries:
            syms = graph.automorphisms().to_array().T
        elif translations:
            syms = graph.translation_group().to_array().T
        elif point_symmetries:
            syms = graph.point_group().to_array().T
        inv_syms = np.zeros(syms.shape, dtype=syms.dtype)
        for i in range(syms.shape[0]):
            for j in range(syms.shape[1]):
                inv_syms[syms[i, j], j] = i
        syms = jnp.array(syms)
        inv_syms = jnp.array(inv_syms)
    if name == "GPS" or name == "FilterGPS":
        inv_centre = 1
    else:
        inv_centre = 0
    if automorphisms and spin_flip:

        def symmetries(samples: Array) -> Array:
            out = jnp.take(samples, syms, axis=-1)
            out = jnp.concatenate((out, inv_centre - out), axis=-1)
            return out

        def inv_symmetries(sample_at_indices, indices):
            inv_sym_sites = jnp.concatenate(
                (inv_syms[indices], inv_syms[indices]), axis=-1
            )
            inv_sym_occs = jnp.tile(
                jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1]
            )
            inv_sym_occs = jnp.concatenate(
                (inv_sym_occs, inv_centre - inv_sym_occs), axis=-1
            )
            return inv_sym_occs, inv_sym_sites

    elif automorphisms:

        def symmetries(samples: Array) -> Array:
            out = jnp.take(samples, syms, axis=-1)
            return out

        def inv_symmetries(sample_at_indices, indices):
            inv_sym_sites = inv_syms[indices]
            inv_sym_occs = jnp.tile(
                jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1]
            )
            return inv_sym_occs, inv_sym_sites

    elif spin_flip:

        def symmetries(samples: Array) -> Array:
            out = jnp.expand_dims(samples, axis=-1)
            out = jnp.concatenate((out, inv_centre - out), axis=-1)
            return out

        def inv_symmetries(sample_at_indices, indices):
            inv_sym_sites = jnp.expand_dims(indices, axis=-1)
            inv_sym_sites = jnp.concatenate((inv_sym_sites, inv_sym_sites), axis=-1)
            inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
            inv_sym_occs = jnp.concatenate(
                (inv_sym_occs, inv_centre - inv_sym_occs), axis=-1
            )
            return inv_sym_occs, inv_sym_sites

    else:

        def symmetries(samples: Array) -> Array:
            out = jnp.expand_dims(samples, axis=-1)
            return out

        def inv_symmetries(sample_at_indices, indices):
            inv_sym_sites = jnp.expand_dims(indices, axis=-1)
            inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
            return inv_sym_occs, inv_sym_sites

    return symmetries, inv_symmetries


def count_spins(spins: Array) -> Array:
    """
    Count the number of up- and down-spins in a batch of local configurations x_i,
    where x_i can be equal to:
        - 0 if it is occupied by an down-spin
        - 1 if it is occupied by a up-spin

    Args:
        spins : array of local configurations (batch,)

    Returns:
        the number of down- and up-spins for each configuration in the batch (batch, 2)
    """
    return jnp.stack([(spins + 1) & 1, ((spins + 1) & 2) / 2], axis=-1).astype(
        jnp.int32
    )


def count_spins_fermionic(spins: Array) -> Array:
    """
    Count the spin-up and down electrons in a batch of local occupations x_i,
    where x_i can be equal to:
        - 0 if it is unoccupied
        - 1 if it is occupied by a single spin-up electron
        - 2 if it is occupied by a single spin-down electron
        - 3 if it is doubly-occupied

    Args:
        spins : array of local configurations (batch,)

    Returns:
        the number of spin-up and down electrons for each configuration in the batch (batch, 4)
    """
    zeros = jnp.zeros(spins.shape[0])
    up_spins = spins & 1
    down_spins = (spins & 2) / 2
    return jnp.stack([zeros, up_spins, down_spins, zeros], axis=-1).astype(jnp.int32)


def renormalize_log_psi(
    n_spins: Array, hilbert: HomogeneousHilbert, index: int
) -> Array:
    """
    Renormalize the log-amplitude to conserve the number of up- and down-spins

    Args:
        n_spins : number of up- and down-spins up to index (batch, 2)
        hilbert : Hilbert space from which configurations are sampled
        index : site index

    Returns:
        renormalized log-amplitude (batch,)
    """
    return jnp.log(jnp.heaviside(hilbert.size // 2 - n_spins, 0))


@partial(jax.vmap, in_axes=(0, None, None))
def renormalize_log_psi_fermionic(
    n_spins: Array, hilbert: HomogeneousHilbert, index: int
) -> Array:
    """
    Renormalize the log-amplitude to conserve the number of spin-up and down electrons

    Args:
        n_spins : number of spin-up and down electrons up to index (batch, 4)
        hilbert : Hilbert space from which configurations are sampled
        index : site index

    Returns:
        renormalized log-amplitude (batch,)
    """
    # Compute difference between spin-up (spin-down) electrons up to index and
    # total number of spin-up (spin-down) electrons
    diff = jnp.array(hilbert._n_elec, jnp.int32) - n_spins[1:3]

    # 1. if the number of spin-up (spin-down) electrons until index
    #    is equal to n_elec_up (n_elec_down), then set to 0 the probability
    #    of sampling a singly occupied orbital with a spin-up (spin-down)
    #    electron, as well as the probability of sampling a doubly occupied orbital
    log_psi = jnp.zeros(hilbert.local_size)
    log_psi = jax.lax.cond(
        diff[0] == 0,
        lambda log_psi: log_psi.at[1].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi,
    )
    log_psi = jax.lax.cond(
        diff[1] == 0,
        lambda log_psi: log_psi.at[2].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi,
    )
    log_psi = jax.lax.cond(
        (diff == 0).any(),
        lambda log_psi: log_psi.at[3].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi,
    )

    # 2. if the number of spin-up (spin-down) electrons that still need to be
    #    distributed, is smaller or equal than the number of sites left, then set the probability
    #    of sampling an empty orbital and one with the opposite spin to 0
    log_psi = jax.lax.cond(
        (diff[0] >= (hilbert.size - index)).any(),
        lambda log_psi: log_psi.at[np.array([0, 2])].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi,
    )
    log_psi = jax.lax.cond(
        (diff[1] >= (hilbert.size - index)).any(),
        lambda log_psi: log_psi.at[np.array([0, 1])].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi,
    )
    return log_psi

def get_plaquettes_and_masks(hilbert: HomogeneousHilbert, graph: AbstractGraph):
    """
    Return the filter plaquettes and masks for a filter-based GPS Ansatz

    Args:
        hilbert : Hilbert space on which the model should act
        graph : graph associated with the Hilbert space

    Returns:
        a tuple containing the filter plaquettes and masks for a filter-based GPS Ansatz
    """
    L = hilbert.size
    if graph and graph.ndim == 2 and graph.pbc.all():
        translations = graph.translation_group().to_array()
        plaquettes = translations[np.argsort(translations[:, 0])]
        plaquettes = HashableArray(plaquettes)
    else:
        plaquettes = HashableArray(circulant(np.arange(L)))
    masks = HashableArray(
        np.where(plaquettes >= np.repeat([np.arange(L)], L, axis=0).T, 0, 1)
    )
    return (plaquettes, masks)
