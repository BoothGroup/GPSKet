import numpy as np
import netket as nk
import GPSKet as qk
from ml_collections import ConfigDict
from netket.operator import AbstractOperator, Heisenberg
from GPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly
from GPSKet.operator.hamiltonian import FermiHubbardOnTheFly
from pyscf import scf, gto, ao2mo, lo
from VMCutils import MPIVars


def get_system(config: ConfigDict) -> AbstractOperator:
    """
    Return the Hamiltonian for a system

    Args:
        config : experiment configuration file

    Returns:
        Hamiltonian for the system
    """
    name = config.system_name
    if "Heisenberg" in name or "J1J2" in name:
        return get_Heisenberg_system(config.system)
    elif config.system.get("molecule", None):
        return get_molecular_system(config.system)
    elif "Hubbard" in name:
        return get_Hubbard_system(config.system)


def get_Heisenberg_system(config: ConfigDict) -> Heisenberg:
    """
    Return the Hamiltonian for Heisenberg system

    Args:
        config : system configuration dictionary

    Returns:
        Hamiltonian for the Heisenberg system
    """
    # Setup Hilbert space
    Lx = config.Lx
    Ly = config.get("Ly", None)
    J1 = config.J1
    J2 = config.get("J2", 0.0)

    # Setup Hamiltonian
    if J2 != 0.0:
        sign_rule = (config.sign_rule, False)
    else:
        sign_rule = config.sign_rule
    ha = qk.operator.hamiltonian.get_J1_J2_Hamiltonian(
        Lx, Ly=Ly, J1=J1, J2=J2, sign_rule=sign_rule, pbc=config.pbc, on_the_fly_en=True
    )
    return ha


def get_molecular_system(config: ConfigDict) -> AbInitioHamiltonianOnTheFly:
    """
    Return the Hamiltonian for a molecular system

    Args:
        config : system configuration dictionary

    Returns:
        Hamiltonian for the molecular system
    """
    # Setup Hilbert space
    if MPIVars.rank == 0:
        mol = gto.Mole()
        molecule = config.get("molecule")
        mol.build(
            atom=molecule,
            basis=config.basis_set,
            symmetry=config.symmetry,
            unit=config.unit,
        )
        nelec = mol.nelectron
        print("Number of electrons: ", nelec)

        myhf = scf.RHF(mol)
        myhf.scf()
        norb = myhf.mo_coeff.shape[1]
        print("Number of molecular orbitals: ", norb)
    else:
        norb = None
        nelec = None
    norb = MPIVars.comm.bcast(norb, root=0)
    nelec = MPIVars.comm.bcast(nelec, root=0)

    hi = qk.hilbert.FermionicDiscreteHilbert(N=norb, n_elec=(nelec // 2, nelec // 2))

    # Get hamiltonian elements
    if MPIVars.rank == 0:
        # 1-electron 'core' hamiltonian terms, transformed into MO basis
        h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

        # Get 2-electron electron repulsion integrals, transformed into MO basis
        eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,) * 4, compact=False)

        # Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
        # Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
        h2 = ao2mo.restore(1, eri, norb)

        # Transform to a local orbital basis if wanted
        if "local" in config.basis:
            loc_coeff = lo.orth_ao(mol, "meta_lowdin")
            ovlp = myhf.get_ovlp()
            # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
            assert np.allclose(
                np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)), np.eye(norb)
            )
            # Find the hamiltonian in the local basis
            hij_local = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
            hijkl_local = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
            h1 = hij_local
            h2 = hijkl_local
    else:
        h1 = None
        h2 = None
    h1 = MPIVars.comm.bcast(h1, root=0)
    h2 = MPIVars.comm.bcast(h2, root=0)

    # Setup Hamiltonian
    ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)
    return ha


def get_Hubbard_system(config: ConfigDict) -> FermiHubbardOnTheFly:
    """
    Return the Hamiltonian for Hubbard system at half-filling

    Args:
        config : system configuration dictionary

    Returns:
        Hamiltonian for the Hubbard system at half-filling
    """
    # Setup Hilbert space
    Lx = config.Lx
    t = config.t
    U = config.U

    # TODO: add support for 2d system
    g = nk.graph.Chain(Lx, pbc=config.pbc)
    hi = qk.hilbert.FermionicDiscreteHilbert(
        g.n_nodes, n_elec=(g.n_nodes // 2, g.n_nodes // 2)
    )

    # Setup Hamiltonian
    edges = np.array([[i, (i + 1) % Lx] for i in range(Lx)])
    t = t * np.ones(Lx)
    if config.pbc and Lx % 4 == 0:
        t[-1] *= -1
    ha = FermiHubbardOnTheFly(hi, edges, U=U, t=t)
    return ha
