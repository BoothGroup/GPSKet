import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from netket.utils.mpi import node_number as _MPI_rank
from netket.utils.mpi import MPI_py_comm as _MPI_comm
from netket.hilbert import HomogeneousHilbert
from netket.utils import HashableArray
from netket.utils.types import Array
from ml_collections import ConfigDict
from pyscf import gto, scf, ao2mo
from typing import Union, Tuple, Optional, Callable
from GPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly, FermiHubbardOnTheFly
from GPSKet.models import Backflow, qGPS, CPDBackflow


Hamiltonian = Union[AbInitioHamiltonianOnTheFly, FermiHubbardOnTheFly]

def get_model(config: ConfigDict, hilbert: HomogeneousHilbert, hamiltonian: Hamiltonian, workdir: Optional[str]=None) -> nn.Module:
    """
    Return the model for a wavefunction Ansatz

    Args:
        config : experiment configuration file
        hilbert : Hilbert space on which the model should act
        hamiltonian : Hamiltonian of the system
        workdir : working directory (optional)

    Returns:
        the model for the wavefunction Ansatz
    """
    if config.model.dtype == 'real':
        dtype = jnp.float64
    elif config.model.dtype == 'complex':
        dtype = jnp.complex128
    if isinstance(hamiltonian, AbInitioHamiltonianOnTheFly):
        phi = get_hf_orbitals_from_file(
            config.system,
            hilbert._n_elec,
            workdir,
            restricted=config.model.restricted, 
            fixed_magnetization=config.model.fixed_magnetization
        )
    else:
        phi = get_hf_orbitals(
            hamiltonian,
            restricted=config.model.restricted,      
            fixed_magnetization=config.model.fixed_magnetization
        )
    if config.model.get('exchange_cutoff', None) == None:
        norb = hilbert.size
        nelec = np.sum(hilbert._n_elec)
        out_trafo, total_supp_dim = get_backflow_out_transformation(
            config.model.M,
            norb,
            nelec,
            config.model.restricted,
            config.model.fixed_magnetization
        )
        def init_fun(key, shape, dtype):
            epsilon = jnp.ones(shape, dtype=dtype)
            first_supp_dim = np.prod(phi.shape)
            epsilon = epsilon.at[:, : first_supp_dim, 0].set(
                phi.flatten()
            )
            epsilon = epsilon.at[:, first_supp_dim :, 0].set(0.0)
            epsilon += jax.nn.initializers.normal(config.model.sigma, dtype=epsilon.dtype)(
                key, shape=epsilon.shape, dtype=dtype
            )
            return epsilon
        orbitals = None
        correction_fn = qGPS(
            hilbert,
            total_supp_dim,
            dtype=dtype,
            init_fun=init_fun,
            out_transformation=out_trafo,
            apply_fast_update=True
        )
        ma = Backflow(
            hilbert,
            correction_fn,
            orbitals=orbitals,
            spin_symmetry_by_structure=config.model.restricted,
            fixed_magnetization=config.model.fixed_magnetization,
            apply_fast_update=True
        )
    else:
        if isinstance(hamiltonian, AbInitioHamiltonianOnTheFly):
            environments = get_top_k_orbital_indices(
                config.system,
                config.model.exchange_cutoff,
                workdir
            )
            environments = HashableArray(environments)
        else:
            raise ValueError("Range cutoff is currently only supported for molecular systems.")
        def init_fun(key, shape, dtype):
            epsilon = jnp.ones(shape, dtype=dtype)
            epsilon = epsilon.at[:, :, :, 0, 0].set(
                jnp.expand_dims(phi, axis=-1)
            )
            epsilon = epsilon.at[:, :, :, 1:, 0].set(0.0)
            epsilon += jax.nn.initializers.normal(config.model.sigma, dtype=epsilon.dtype)(
                key, shape=epsilon.shape, dtype=dtype
            )
            return epsilon
        ma = CPDBackflow(
            hilbert,
            config.model.M,
            environments=environments,
            dtype=dtype,
            init_fun=init_fun,
            restricted=config.model.restricted,
            fixed_magnetization=config.model.fixed_magnetization
        )
    return ma
    
def get_hf_orbitals(hamiltonian: FermiHubbardOnTheFly, restricted: bool=True, fixed_magnetization: bool=True) -> Array:
    """
    Return the Hartree-Fock mean-field orbitals for the Fermi-Hubbard system

    Args:
        hamiltonian: FermiHubbard Hamiltionian object
        restricted : whether the α and β orbitals are the same or not
        fixed_magnetization : whether magnetization should be conserved or not

    Returns:
        an array of molecular orbital coefficients for the Hartree-Fock orbitals
    """
    if _MPI_rank == 0:
        # Setup system
        norb = hamiltonian.hilbert.size
        n_elec = hamiltonian.hilbert._n_elec
        nelec = np.sum(n_elec)
        mol = gto.Mole()
        mol.nelectron = nelec
        mol.spin = n_elec[0] - n_elec[1]
        h1 = np.zeros((norb, norb))
        for i, edge in enumerate(hamiltonian.edges):
            h1[edge[0], edge[1]] = -hamiltonian.t[i]
            h1[edge[1], edge[0]] = -hamiltonian.t[i]
        h2 = np.zeros((norb, norb, norb, norb))
        np.fill_diagonal(h2, hamiltonian.U)

        # Calculate the mean-field Hartree-Fock energy and wave function
        if restricted:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = ao2mo.restore(8, h2, norb)
        _, vecs = np.linalg.eigh(h1)

        # Optimize
        if not restricted:
            # Break spin-symmetry
            dm_alpha, dm_beta = mf.get_init_guess()
            dm_beta[:2, :2] = 0
            init_dens = (dm_alpha, dm_beta)
            mf = scf.newton(mf)
            mf.kernel(dm0=init_dens)
            mo1 = mf.stability(external=True)[0]
            mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo1))
            mo1 = mf.stability(external=True)[0]
            assert (mf.converged)
        else:
            # Check that orbitals are restricted
            assert (n_elec[0] == n_elec[1])
            init_dens = np.dot(vecs[:, :n_elec[0]], vecs[:, :n_elec[0]].T)
            mf.kernel(dm0=init_dens)
            if not mf.converged:
                mf = scf.newton(mf)
                mf.kernel(mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
            assert (mf.converged)
        
        # Return orbitals
        nelec = np.sum(n_elec)
        if fixed_magnetization:
            if restricted:
                orbitals = mf.mo_coeff[:, :n_elec[0]]
            else:
                mo_coeff = np.reshape(mf.mo_coeff, (2, norb, norb))
                orbitals = np.concatenate([mo_coeff[0, :, :n_elec[0]], mo_coeff[1, :, :n_elec[1]]], axis=1)
        else:
            orbitals = np.zeros((2*norb, nelec))
            mo_coeff = np.reshape(mf.mo_coeff, (2, norb, norb))
            orbitals[:norb, :n_elec[0]] = mo_coeff[0, :, :n_elec[0]]
            orbitals[norb:, n_elec[0]:] = mo_coeff[1, :, :n_elec[1]]
    else:
        orbitals = None
    orbitals = _MPI_comm.bcast(orbitals, root=0)
    return orbitals

def get_hf_orbitals_from_file(config: ConfigDict, n_elec: Tuple[int, int], workdir: Optional[str]=None, restricted: bool=True, fixed_magnetization: bool=True) -> Array:
    """
    Return the Hartree-Fock mean-field orbitals for a molecular system by loading them from a file

    Args:
        config: configuration file of the system
        n_elec: number of spin-up and spin-down electrons
        workdir: path to the directory where the Hartree-Fock orbitals are stored
        restricted : whether the α and β orbitals are the same or not
        fixed_magnetization : whether magnetization should be conserved or not

    Returns:
        an array of molecular orbital coefficients for the Hartree-Fock orbitals
    """
    if _MPI_rank == 0:
        if workdir == None:
            workdir = os.getcwd()
        hf_orbitals_path = os.path.join(workdir, 'hf_orbitals.npy')
        if os.path.exists(hf_orbitals_path):
            hf_orbitals = np.load(hf_orbitals_path) # (norb, nelec//2) or (norb, nelec)
        else:
            raise FileNotFoundError('No HF orbitals found in workdir')
        nelec = np.sum(n_elec)
        if fixed_magnetization:
            if restricted:
                # RHF
                assert hf_orbitals.shape[1] == nelec//2
                orbitals = hf_orbitals
            else:
                # UHF
                if hf_orbitals.shape[1] == nelec//2:
                    orbitals = np.concatenate((hf_orbitals, hf_orbitals), axis=1)
                else:
                    assert hf_orbitals.shape[1] == nelec
                    orbitals = hf_orbitals
        else:
            # GHF
            norb = hf_orbitals.shape[0]
            if config.get('n_elec', None):
                hf_orbitals_a = hf_orbitals[:, :n_elec[0]]
                hf_orbitals_b = hf_orbitals[:, :n_elec[1]]
                orbitals = np.zeros((2*norb, nelec))
                orbitals[:norb, :n_elec[0]] = hf_orbitals_a
                orbitals[norb:, n_elec[1]:] = hf_orbitals_b
            else:
                nelec = hf_orbitals.shape[1]*2
                orbitals = np.zeros((2*norb, nelec))
                orbitals[:norb, :nelec//2] = hf_orbitals
                orbitals[norb:, nelec//2:] = hf_orbitals
    else:
        orbitals = None
    orbitals = _MPI_comm.bcast(orbitals, root=0)
    return orbitals

def get_backflow_out_transformation(M: int, norb: int, nelec: int, restricted: bool=True, fixed_magnetization: bool=True) -> Tuple[Callable, int]:
    """
    Return the transformation of the ouput layer for a GPS model to work within a backflow Ansatz

    Args:
        M : support dimension of each GPS backflow orbital model
        norb : number of orbitals
        nelec : number of electrons
        restricted : whether the α and β orbitals are the same or not
        fixed_magnetization : whether magnetization should be conserved or not

    Returns:
        a callable function that is applied in the output layer of a GPS model and
        the total support dimension of the GPS model
    """
    if fixed_magnetization:
        if restricted:
            shape = (M, norb, nelec//2)
        else:
            shape = (M, norb, nelec)
    else:
        shape = (M, 2*norb, nelec)
    def out_trafo(x):
        batch_size = x.shape[0]
        n_syms = x.shape[-1]
        # Reshape output into (B, M, L, N, T)
        x = jnp.reshape(x, (batch_size,)+shape+(n_syms,))
        # Sum over support dim M
        out = jnp.sum(x, axis=1)
        return out
    return out_trafo, np.prod(shape)

def get_top_k_orbital_indices(config: ConfigDict, exchange_cutoff: int, workdir: Optional[str]=None) -> Array:
    """
    Return the top-k most coupled orbital indices for each orbital

    Args:
        config: configuration file for the system
        exchange_cutoff: number of orbitals considered
        workdir: path to the directory where the Hartree-Fock exchange matrix is stored

    Returns
        an array of shape (norb, exchange_cutoff) of orbital indices
    """
    if _MPI_rank == 0:
        # Load exchange matrix
        if workdir == None:
            workdir = os.getcwd()
        exchange_path = os.path.join(workdir, 'exchange.npy')
        if os.path.exists(exchange_path):
            em = np.load(exchange_path) # (norb, norb)
        else:
            raise FileNotFoundError('No exchange matrix found in workdir')
        
        # Transform to a local orbital basis, if necessary
        if 'local' in config.basis:
            basis_path = os.path.join(workdir, 'basis.npy')
            if os.path.exists(basis_path):
                basis = np.load(basis_path) # (norb, norb)
            else:
                raise FileNotFoundError('No basis file found in workdir')
            em = np.linalg.multi_dot((basis.T, em, basis))

        # Generate environment matrix of top-K closest coupled orbitals for each orbital
        top_k_orbital_indices = np.flip(np.argsort(np.abs(em), axis=1)[:, -exchange_cutoff:], axis=1)
    else:
        top_k_orbital_indices = None
    top_k_orbital_indices = _MPI_comm.bcast(top_k_orbital_indices, root=0)
    return top_k_orbital_indices
        