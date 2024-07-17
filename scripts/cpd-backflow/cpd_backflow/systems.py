import os
import numpy as np
import netket as nk
import GPSKet as qk
from typing import Union, Optional
from ml_collections import ConfigDict
from netket.utils.mpi import node_number as _MPI_rank
from netket.utils.mpi import MPI_py_comm as _MPI_comm
from netket.operator import AbstractOperator
from GPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly, AbInitioHamiltonianSparse
from GPSKet.operator.hamiltonian import FermiHubbardOnTheFly
from pyscf import scf, gto, ao2mo, lo


def get_system(config: ConfigDict, workdir: Optional[str]=None) -> AbstractOperator:
    """
    Return the Hamiltonian for a system

    Args:
        config : experiment configuration file
        workdir : working directory (optional)

    Returns:
        Hamiltonian for the system
    """
    name = config.system_name
    if name in ['Hchain', 'Hsheet', 'H2O']:
        store_exchange = config.model.get('exchange_cutoff', None) is not None
        return get_molecular_system(config.system, workdir=workdir, store_exchange=store_exchange)
    elif name == 'FermiHubbard':
        return get_FermiHubbard_system(config.system)
    else:
        raise ValueError(f"Could not find system with name {name}")
    
def build_molecule(config: ConfigDict) -> gto.Mole:
    """
    Build a molecular PySCF object from a system configuration file

    Args:
        config : system configuration dictionary

    Returns:
        Mole object
    """
    if config.get('atom', None):
        atom = config.atom
    else:
        atom = config.molecule
    if config.get('n_elec', None):
        spin = config.n_elec[0]-config.n_elec[1]
    else:
        spin = 0
    mol = gto.M(
        atom=atom,
        basis=config.basis_set,
        symmetry=config.symmetry,
        unit=config.unit,
        spin=spin
    )
    return mol

def get_molecular_system(config: ConfigDict, workdir: Optional[str]=None, store_exchange: bool=False) -> Union[AbInitioHamiltonianOnTheFly, AbInitioHamiltonianSparse]:
    """
    Return the Hamiltonian for a molecular system

    Args:
        config : system configuration dictionary
        workdir : working directory
        store_exchange : flag to store the exchange matrix

    Returns:
        Hamiltonian for the molecular system
    """
    # Setup Hilbert space
    if _MPI_rank == 0:
        mol = build_molecule(config)
        spin = mol.spin
        n_elec = mol.nelec
        nelec = np.sum(n_elec)
        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.ROHF(mol)
        mf.scf()
        norb = mf.mo_coeff.shape[1]
        print(f"Number of molecular orbitals: {norb}")
        print(f"Number of α and β electrons: {n_elec}")
    else:
        norb = None
        n_elec = None
        nelec = None
    norb = _MPI_comm.bcast(norb, root=0) # Number of molecular orbitals
    n_elec = _MPI_comm.bcast(n_elec, root=0) # Number of α and β electrons
    nelec = _MPI_comm.bcast(nelec, root=0) # Total number of electrons

    hi = qk.hilbert.FermionicDiscreteHilbert(N=norb, n_elec=n_elec)

    # Get hamiltonian elements
    if _MPI_rank == 0:
        # Transform to a local orbital basis if wanted
        if workdir == None:
            workdir = os.getcwd()
        basis_path = os.path.join(workdir, "basis.npy")
        h1_path = os.path.join(workdir, "h1.npy")
        h2_path = os.path.join(workdir, "h2.npy")
        hf_orbitals_path = os.path.join(workdir, "hf_orbitals.npy")
        if (os.path.exists(basis_path) and
                os.path.exists(h1_path) and
                os.path.exists(h2_path) and
                os.path.exists(hf_orbitals_path)):
            basis = np.load(basis_path)
            h1 = np.load(h1_path)
            h2 = np.load(h2_path)
            hf_orbitals = np.load(hf_orbitals_path)
        else:
            # Transform to a local orbital basis if wanted
            if 'local' in config.basis:
                loc_coeff = lo.orth_ao(mol, 'lowdin')
                if 'boys' in config.basis:
                    localizer = lo.Boys(mol, mo_coeff=loc_coeff)
                    localizer.init_guess = None
                    loc_coeff = localizer.kernel()
                elif 'pipek-mezey' in config.basis:
                    loc_coeff = lo.PipekMezey(mol, mo_coeff=loc_coeff).kernel()
                elif 'edmiston-ruedenberg' in config.basis:
                    loc_coeff = lo.EdmistonRuedenberg(mol, mo_coeff=loc_coeff).kernel()
                elif 'split' in config.basis:
                    localizer = lo.Boys(mol, mf.mo_coeff[:,:nelec//2])
                    loc_coeff_occ = localizer.kernel()
                    localizer = lo.Boys(mol, mf.mo_coeff[:, nelec//2:])
                    loc_coeff_vrt = localizer.kernel()
                    loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)
                basis = loc_coeff
            elif config.basis == 'canonical':
                basis = mf.mo_coeff
            else:
                raise ValueError("Unknown basis, please choose between: 'canonical', 'local-boys', 'local-pipek-mezey', 'local-edmiston-ruedenberg' and 'local-split'.")
            ovlp = mf.get_ovlp()
            # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
            assert(np.allclose(np.linalg.multi_dot((basis.T, ovlp, basis)), np.eye(norb)))
            # Find the hamiltonian in the basis
            canonical_to_local_trafo = basis.T.dot(ovlp.dot(mf.mo_coeff))
            h1 = np.linalg.multi_dot((basis.T, mf.get_hcore(), basis))
            h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), norb)
            if spin == 0:
                hf_orbitals = canonical_to_local_trafo[:, :nelec//2]
            else:
                hf_orbitals = np.concatenate(
                    (canonical_to_local_trafo[:, :n_elec[0]], canonical_to_local_trafo[:, :n_elec[1]]),
                    axis=1
                )
            # Prune Hamiltonian elements below threshold
            if config.get('pruning_threshold', None) is not None and config.pruning_threshold != 0.0:
                h1[abs(h1) < config.pruning_threshold] = 0.0
                h2[abs(h2) < config.pruning_threshold] = 0.0
            np.save(basis_path, basis)
            np.save(h1_path, h1)
            np.save(h2_path, h2)
            np.save(hf_orbitals_path, hf_orbitals)
        # Store exchange matrix, if necessary
        if store_exchange:
            exchange_path = os.path.join(workdir, "exchange.npy")
            if not os.path.exists(exchange_path):
                # Get converged density matrix from the Hartree Fock
                dm = mf.make_rdm1()
                # Get exchange matrix
                _, em = mf.get_jk(mol, dm)
                # Transform to a local orbital basis, if necessary
                if 'local' in config.basis:
                    em = np.linalg.multi_dot((basis.T, em, basis))
                np.save(exchange_path, em)
    else:
        h1 = None
        h2 = None
    h1 = _MPI_comm.bcast(h1, root=0)
    h2 = _MPI_comm.bcast(h2, root=0)

    # Setup Hamiltonian
    if config.get('pruning_threshold', None) is not None and config.pruning_threshold != 0.0:
        ha = AbInitioHamiltonianSparse(hi, h1, h2)
    else:
        ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)
    return ha

def get_FermiHubbard_system(config: ConfigDict) -> FermiHubbardOnTheFly:
    """
    Return the Hamiltonian for a Fermi-Hubbard systemg

    Args:
        config : system configuration dictionary

    Returns:
        Hamiltonian for the Fermi-Hubbard system
    """
    # Setup graph and Hilbert space
    Lx = config.Lx
    Ly = config.get('Ly', 1)
    t = config.t
    U = config.U
    n_elec = config.get('n_elec', None)
    if Ly > 1:
        # config.pbc:
        # - 'PBC-PBC' = periodic boundary conditions in both dimensions
        # - 'PBC-APBC' = periodic boundary conditions in one dimension, and anti-periodic in the other
        #   - implemented by changing the sign on the hopping terms in one direction
        # - 'PBC-OBC' = periodic boundary conditions in one dimension, and open in the other
        if config.pbc in ['PBC', 'PBC-PBC', 'PBC-APBC']:
            pbc = [True, True]
        elif 'OBC' in config.pbc:
            pbc = [True, False] 
        g = nk.graph.Grid([Lx, Ly], pbc=pbc)
    else:
        # config.pbc = True/False
        g = nk.graph.Chain(Lx, pbc=config.pbc)
    if n_elec is None:
        n_elec = (g.n_nodes//2, g.n_nodes//2)
    hi = qk.hilbert.FermionicDiscreteHilbert(g.n_nodes, n_elec=n_elec)

    # Setup Hamiltonian
    edges = np.array(g.edges())
    t = np.ones(edges.shape[0])*config.t
    if Ly > 1:
        if 'APBC' in config.pbc:
            for i, edge in enumerate(edges):
                if np.abs(edge[0]-edge[1]) // config.Ly == (config.Lx-1):
                    t[i] *= -1.0
    else:
        if config.pbc and Lx % 4 == 0:
            t[-1] *= -1.0
    ha = FermiHubbardOnTheFly(hi, edges, U=U, t=t)
    return ha
