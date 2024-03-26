import numpy as np

from pyscf import scf, gto, ao2mo, fci, lo


# Set up system
dist = 1.8

mol = gto.Mole()

mol.build(
    atom=[("H", (x, 0.0, 0.0)) for x in dist * np.arange(10)],
    basis="sto-6g",
    symmetry=True,
    unit="Bohr",
)

nelec = mol.nelectron
print("Number of electrons: ", nelec)

myhf = scf.RHF(mol)
ehf = myhf.scf()
norb = myhf.mo_coeff.shape[1]
print("Number of molecular orbitals: ", norb)

# Get one- and two-electron integrals for canonical basis

# 1-electron 'core' hamiltonian terms, transformed into MO basis
h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

# Get 2-electron electron repulsion integrals, transformed into MO basis
eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,) * 4, compact=False)

# Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
# Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
h2 = ao2mo.restore(1, eri, norb)

# Get ground state amplitudes
cisolver = fci.direct_spin1.FCISolver(mol)
e, c_canonical = cisolver.kernel(h1, h2, norb, nelec)

np.save("canonical_amplitudes.npy", c_canonical)

# Get one- and two-electron integrals for local basis

loc_coeff = lo.orth_ao(mol, "meta_lowdin")
loc_coeff = lo.Boys(mol, loc_coeff).kernel()

ovlp = myhf.get_ovlp()
# Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
assert np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)), np.eye(norb))
# Find the hamiltonian in the local basis
hij_local = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
hijkl_local = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
h1 = hij_local
h2 = hijkl_local

# Get ground state amplitudes
cisolver = fci.direct_spin1.FCISolver(mol)
e, c_localized = cisolver.kernel(h1, h2, norb, nelec)

np.save("localized_amplitudes.npy", c_localized)


# Get one- and two-electron integrals for split-local basis

loc_coeff_occ = lo.Boys(mol, myhf.mo_coeff[:, : nelec // 2]).kernel()
loc_coeff_vrt = lo.Boys(mol, myhf.mo_coeff[:, nelec // 2 :]).kernel()
loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)

ovlp = myhf.get_ovlp()
# Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
assert np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)), np.eye(norb))
# Find the hamiltonian in the split-local basis
hij_local = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
hijkl_local = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
h1 = hij_local
h2 = hijkl_local

# Get ground state amplitudes
cisolver = fci.direct_spin1.FCISolver(mol)
e, c_split_local = cisolver.kernel(h1, h2, norb, nelec)

np.save("split_localized_amplitudes.npy", c_split_local)
