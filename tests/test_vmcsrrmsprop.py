import pytest
import optax
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from netket.utils import mpi
from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.sampler.fermionic_hopping import MetropolisFastHopping
from GPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonianOnTheFly
from GPSKet.models import qGPS
from GPSKet.driver import VMC_SRRMSProp
from pyscf import scf, gto, ao2mo, lo
from pyscf.tools import ring


def _setup_hamiltonian(local_basis=False):
    # Set up a ring of 8 Hydrogen atoms with dist 1 a0
    mol = gto.Mole()
    mol.build(
        atom=[("H", x) for x in ring.make(8, 1)], basis="sto-3g", symmetry=True, unit="Bohr"
    )

    nelec = mol.nelectron
    print("Number of electrons: ", nelec)

    myhf = scf.RHF(mol)
    ehf = myhf.scf()
    norb = myhf.mo_coeff.shape[1]
    print("Number of molecular orbitals: ", norb)

    # Get hamiltonian elements
    # 1-electron 'core' hamiltonian terms, transformed into MO basis
    h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

    # Get 2-electron electron repulsion integrals, transformed into MO basis
    eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,) * 4, compact=False)

    # Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
    # Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
    h2 = ao2mo.restore(1, eri, norb)

    # Transform to a local orbital basis if wanted
    if local_basis:
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

    # Set up Hilbert space
    hi = FermionicDiscreteHilbert(norb, n_elec=(nelec // 2, nelec // 2))

    # Set up ab-initio Hamiltonian
    ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)

    return ha

def _setup_vmc(ha, hi, complex=True, mode='real'):
    # Use Metropolis-Hastings sampler with hopping rule (including fast updates for qGPS)
    sa = MetropolisFastHopping(hi, n_chains_per_rank=1)

    # Define the model and the variational state
    dtype = jnp.complex128 if complex else jnp.float64
    if not complex:
        out_trafo = lambda argument: jnp.sum(argument, axis=(-2, -1)).astype(jnp.complex128)
    else:
        out_trafo = lambda argument: jnp.sum(argument, axis=(-2, -1))
    model = qGPS(hi, 10, dtype=dtype, out_transformation=out_trafo)
    vs = nk.vqs.MCState(sa, model, n_samples=128, seed=42)

    # Optimizer and drivers
    op = nk.optimizer.Sgd(learning_rate=0.02)
    qgt = nk.optimizer.qgt.QGTJacobianDense(mode=mode)

    return op, vs, qgt

def test_SRRMSProp_real_vs_complex():
    """
    VMC_SRRMSProp should yield the same optimization of a positive definite wavefunction
    with `mode=real` and `mode=complex`
    """
    ha = _setup_hamiltonian()
    hi = ha.hilbert
    n_iters = 5
    diag_shift = 0.01
    op, vs_real, _ = _setup_vmc(ha, hi, complex=False, mode='real')
    vmc_real = VMC_SRRMSProp(ha, op, diag_shift=diag_shift, variational_state=vs_real, jacobian_mode='real')
    logger_real = nk.logging.RuntimeLog()
    vmc_real.run(n_iter=n_iters, out=logger_real)

    op, vs_cplx, _ = _setup_vmc(ha, hi, complex=False, mode='complex')
    vmc_cplx = VMC_SRRMSProp(ha, op, diag_shift=diag_shift, variational_state=vs_cplx, jacobian_mode='complex')
    logger_cplx = nk.logging.RuntimeLog()
    vmc_cplx.run(n_iter=n_iters, out=logger_cplx)

    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-4), vmc_real.state.parameters, vmc_cplx.state.parameters
    )

    if mpi.rank == 0 and jax.process_count() == 0:
        energy_real = logger_real.data["Energy"]["Mean"]
        energy_cplx = logger_cplx.data["Energy"]["Mean"]
        np.testing.assert_allclose(energy_real, energy_cplx, atol=1e-10)

def test_SRRMSProp_jacobian_mode():
    """
    VMC_SRRMSProp should raise an error when attempting to set `jacobian_mode` to something else than
    `real` or `complex`
    """
    ha = _setup_hamiltonian()
    hi = ha.hilbert
    op, vs, _ = _setup_vmc(ha, hi)
    vmc = VMC_SRRMSProp(ha, op, diag_shift=0.1, variational_state=vs)
    assert vmc.jacobian_mode == 'complex'
    vmc.run(n_iter=1)

    with pytest.raises(ValueError):
        vmc = VMC_SRRMSProp(ha, op, diag_shift=0.1, variational_state=vs, jacobian_mode='holomorphic')

def test_SRRMSProp_schedules():
    """
    VMC_SRRMSProp should support schedule for `diag_shift`
    """
    ha = _setup_hamiltonian()
    hi = ha.hilbert
    op, vs, _ = _setup_vmc(ha, hi, mode='complex')
    vmc = VMC_SRRMSProp(
        ha,
        op,
        diag_shift=optax.linear_schedule(0.1, 0.001, 100),
        variational_state=vs
    )
    vmc.run(n_iter=5)
