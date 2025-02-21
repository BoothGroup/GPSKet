import os
import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science', 'nature', 'notebook'])


# Spin-spin correlations
def spin_spin_correlation(rdm1, rdm2, P_A, P_B):
    """
    Calculate the spin-spin correlation between atoms A and B using the corrected equation.
    
    Parameters:
    - rdm1: (np.ndarray) the one-body reduced density matrix
    - rdm2: (np.ndarray) the two-body reduced density matrix
    - P_A, P_B: (np.ndarray) projection operators for atoms A and B
    
    Returns:
    - SzASzB: (float) the spin-spin correlation value.

    Note:
    - This function should give the same result as the following snippet
    >>> from vayesta.core.qemb.corrfunc import get_corrfunc
    >>> rdm1 = np.load("rdm_1.npy")
    >>> rdm2 = np.load("rdm_2.npy")
    >>> corr = get_corrfunc(emb, "sz,sz", dm1=rdm1, dm2=rdm2)
    """
    # Transform spin-free 2-RDM into spinful term
    rdm2 = -(rdm2 / 6 + rdm2.transpose(0, 3, 2, 1) / 3)

    # Apply projection operators to the one-body RDMs and sum over indices i and j
    one_body_term = np.einsum('ik,jk,ij->', P_A, P_B, rdm1)

    # Apply projection operators to the two-body RDMs and sum over indices i, j, k, and l
    two_body_term = np.einsum('ij,kl,ijkl->', P_A, P_B, rdm2)

    # Combine terms according to the corrected equation
    SzAszB = 0.25 * one_body_term + 0.5 * two_body_term

    return SzAszB.real

def get_correlations(workdir):
    rdm1 = np.load(os.path.join(workdir, "rdm_1.npy"))
    rdm2 = np.load(os.path.join(workdir, "rdm_2.npy"))
    norb = rdm1.shape[0]
    my_proj = {s: np.zeros((norb, norb)) for s in range(norb)}
    for s in range(norb):
        my_proj[s][s, s] = 1.0
    corr = np.zeros((norb, norb))
    for a in range(norb):
        P_A = my_proj[a]
        for b in range(norb):
            P_B = my_proj[b]
            corr[a, b] = spin_spin_correlation(rdm1, rdm2, P_A, P_B)
    return corr

def radial_correlations(corr):
    bulk = [14, 15, 20, 21]
    C_0 = np.mean([corr[a, a] for a in bulk])
    C_1 = 0.0
    for a in bulk:
        for b in [a-6, a-1, a+6, a+1]:
            C_1 += corr[a, b]
    C_1 = C_1 / 4
    C_sqrt2 = 0.0
    for a in bulk:
        for b in [a-7, a+5, a+7, a-5]:
            C_sqrt2 += corr[a, b]
    C_sqrt2 = C_sqrt2 / 4
    C_2 = 0.0
    for a in bulk:
        for b in [a-12, a-2, a+12, a+2]:
            C_2 += corr[a, b]
    C_2 = C_2 / 4
    return [C_0, C_1, C_sqrt2, C_2]

# Load data
df = pd.read_csv('hsheet_cpd_backflow.csv')
base_path = os.path.dirname(__file__)

# Compute spin-spin correlation functions
Crs = {}
for index, row in df.query('`exchange_cutoff`.isna() and `distance` in [1.2, 3.0]', engine='python').iterrows():
    print(row['uuid'])
    workdir = os.path.join(base_path, "hsheet_rdms", row['uuid'])
    corr = get_correlations(workdir)
    Crs[row['distance']] = radial_correlations(corr)

# Plot
rs = [0, 1, np.sqrt(2), 2]
plt.figure()
plt.minorticks_off()
plt.axhline(0.0, linestyle=":", color="0.4")
for a0, Cr in Crs.items():
    plt.plot(rs, Cr, linestyle="-", marker="o", markersize=12, markeredgecolor="0.2", label=f"$d = {a0}$ Å")
plt.xticks(rs, labels=["0.0", "1.0", r"$\sqrt{2}$", "2.0"])
plt.ylabel(r"$C(r)$")
plt.xlabel(r"$r/d$")
plt.legend();
plt.savefig("hsheet_6x6_corr.pdf", dpi=300, bbox_inches="tight");
