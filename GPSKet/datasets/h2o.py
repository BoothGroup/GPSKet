import os
import numpy as np
from typing import Optional, Tuple
from netket.utils.types import Array
from pyscf import scf, gto, ao2mo, fci, ci, lo


_DATA = "/tmp/GPSKet_data/"

def _H2O_raw(basis, method, datapath):
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

    if not os.path.exists(datapath):
        os.makedirs(datapath)
    dsid = f"{basis}_{method}"

    if not os.path.exists(os.path.join(datapath, f"h1_{dsid}.npy")):
        loc_coeff = myhf.mo_coeff
        if basis != 'CANONICAL':
            loc_coeff = lo.orth_ao(mol, 'meta_lowdin')
            if basis == 'BOYS':
                localizer = lo.Boys(mol, loc_coeff)
                localizer.init_guess = None
                loc_coeff = localizer.kernel()
            if basis == 'SPLIT':
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
        np.save(os.path.join(datapath, f"h1_{dsid}.npy"), h1)
        np.save(os.path.join(datapath, f"h2_{dsid}.npy"), h2)
    else:
        h1 = np.load(os.path.join(datapath, f"h1_{dsid}.npy"))
        h2 = np.load(os.path.join(datapath, f"h2_{dsid}.npy"))

    nuc_en = mol.energy_nuc()

    if not os.path.exists(os.path.join(datapath, f"all_configs_H2O_{dsid}.npy")):
        if method == 'FCI':
            # Run FCI
            cisolver = fci.direct_spin1.FCISolver(mol)
            e, c = cisolver.kernel(h1, h2, norb, nelec)
        elif method == 'CISD':
            # Run CISD
            cisolver = ci.CISD(myhf)
            e, c = cisolver.kernel()
            c = cisolver.to_fcivec(c)
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
        np.save(os.path.join(datapath, f"all_configs_H2O_{dsid}.npy"), all_configurations)
        np.save(os.path.join(datapath, f"all_amplitudes_H2O_{dsid}.npy"), all_amplitudes)
    else:
        all_configurations = np.load(os.path.join(datapath, f"all_configs_H2O_{dsid}.npy"))
        all_amplitudes = np.load(os.path.join(datapath, f"all_amplitudes_H2O_{dsid}.npy"))

    return all_configurations, all_amplitudes

def get_H2O_dataset(basis: str = 'CANONICAL', method: str = 'FCI', select_largest: Optional[int] = None, datapath: str = _DATA) -> Tuple[Array, Array]:
    """
    Return a dataset of configurations and amplitudes for the ground state of H2O computed with `method` in `basis`.

    Args:
        basis : basis type in which the ground state is computed; currently supported: CANONICAL, LOCAL, BOYS and SPLIT (default: CANONICAL)
        method : method used to compute the ground state, either FCI or CISD (default: FCI)
        select_largest : number of configurations with largest probability returned (default: None)
        datapath : path to where the datasets and intermediate values are stored

    Returns:
        A tuple (configurations, aomplitudes) of all the configurations and amplitudes selected, if `select_largest` is `None` all configurations are returned
    """
    configurations, amplitudes = _H2O_raw(basis.upper(), method.upper(), datapath=datapath)
    if select_largest:
        largest_amplitudes_ids = np.argsort(np.abs(amplitudes)**2)[-select_largest:]
        amplitudes = amplitudes[largest_amplitudes_ids]
        configurations = configurations[largest_amplitudes_ids]
    return (configurations, amplitudes)
