import os
import numpy as np
from enum import Enum
from typing import Optional, Tuple
from netket.utils.types import Array
from pyscf import scf, gto, ao2mo, fci, lo


_DATA = "/tmp/qGPSKet_data/"

class BasisType(Enum):
    LOCAL = 0 # Meta-Löwdin
    CANONICAL = 1
    SPLIT = 2
    BOYS = 3

def _h2o_raw(basis_type):
    mol = gto.Mole()

    mol.build(
        atom = [['H', (0., 0.795, -0.454)], ['H', (0., -0.795, -0.454)], ['O', (0., 0., 0.113)]],
        basis = '6-31G',
        unit="Angstrom"
    )

    nelec = mol.nelectron
    print('Number of electrons: ', nelec)

    myhf = scf.RHF(mol)
    ehf = myhf.scf()
    norb = myhf.mo_coeff.shape[1]
    print('Number of molecular orbitals: ', norb)

    if not os.path.exists(_DATA):
        os.makedirs(_DATA)

    if not os.path.exists(os.path.join(_DATA, f"h1_{basis_type}.npy")):
        loc_coeff = myhf.mo_coeff
        if basis_type != 1:
            loc_coeff = lo.orth_ao(mol, 'meta_lowdin')
            if basis_type == 3:
                localizer = lo.Boys(mol, loc_coeff)
                localizer.init_guess = None
                loc_coeff = localizer.kernel()
            if basis_type == 2:
                localizer = lo.Boys(mol, myhf.mo_coeff[:,:nelec//2])
                loc_coeff_occ = localizer.kernel()
                localizer = lo.Boys(mol, myhf.mo_coeff[:, nelec//2:])
                loc_coeff_vrt = localizer.kernel()
                loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)
        ovlp = myhf.get_ovlp()
        # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
        assert(np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)),np.eye(norb)))
        # Find the hamiltonian in the local basis
        h1 = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
        h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
        np.save(os.path.join(_DATA, f"h1_{basis_type}.npy"), h1)
        np.save(os.path.join(_DATA, f"h2_{basis_type}.npy"), h2)
    else:
        h1 = np.load(os.path.join(_DATA, f"h1_{basis_type}.npy"))
        h2 = np.load(os.path.join(_DATA, f"h2_{basis_type}.npy"))

    nuc_en = mol.energy_nuc()

    if not os.path.exists(os.path.join(_DATA, f"all_configs_H2O_{basis_type}.npy")):
        # Run FCI
        cisolver = fci.direct_spin1.FCISolver(mol)
        e, c = cisolver.kernel(h1, h2, norb, nelec)
        configs = []
        amps = []
        def to_config(alpha, beta, n):
            string_a = [0] * (n - (len(bin(alpha)) - 2)) + [1 if digit == '1' else 0 for digit in bin(alpha)[2:]]
            string_a.reverse()
            string_b = [0] * (n - (len(bin(beta)) - 2)) + [1 if digit == '1' else 0 for digit in bin(beta)[2:]]
            string_b.reverse()
            return np.array(string_a, dtype=np.uint8) + 2 * np.array(string_b, dtype=np.uint8)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                alpha = fci.cistring.addr2str(norb, nelec//2, i)
                beta = fci.cistring.addr2str(norb, nelec//2, j)
                conf = to_config(alpha, beta, norb)
                configs.append(conf)
                amps.append(c[i,j])
        all_configurations = np.array(configs)
        all_amplitudes = np.array(amps)
        np.save(os.path.join(_DATA, f"all_configs_H2O_{basis_type}.npy"), all_configurations)
        np.save(os.path.join(_DATA, f"all_amplitudes_H2O_{basis_type}.npy"), all_amplitudes)
    else:
        all_configurations = np.load(os.path.join(_DATA, f"all_configs_H2O_{basis_type}.npy"))
        all_amplitudes = np.load(os.path.join(_DATA, f"all_amplitudes_H2O_{basis_type}.npy"))

    return all_configurations, all_amplitudes

def get_h2o_dataset(basis_type: int = BasisType.CANONICAL, select_largest: Optional[int] = None) -> Tuple[Array, Array]:
    configurations, amplitudes = _h2o_raw(basis_type)
    if select_largest:
        largest_amplitudes_ids = np.argsort(np.abs(amplitudes)**2)[-select_largest:]
        amplitudes = amplitudes[largest_amplitudes_ids]
        configurations = configurations[largest_amplitudes_ids]
    return (configurations, amplitudes)
