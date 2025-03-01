import pandas as pd
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science', 'nature', 'notebook'])
from pyscf import gto, scf


# Compute full dissociation limit
mol = gto.M(atom='H 0 0 0', basis='sto-6g', spin=1)
mf = scf.RHF(mol)
mf.kernel()
h1_energy = mf.e_tot

# Load reference and CPD backflow data
n_atoms = 36
df = pd.read_csv('hsheet.csv')
rhf_energies = df.query('`method` == "rhf"')
uccsd_energies = df.query('`method` == "uccsd"')
dmrg_energies = df.query('`method` == "dmrg"')
gpsxsd_energies = df.query('`method` == "gpsxsd"')
cpdbf_energies = df.query('`method` == "cpd_backflow"')

# Energy
fig, axs = plt.subplots(2, 1, figsize=(9, 9), height_ratios=[0.7, 0.3], gridspec_kw={'hspace': 0.05}, sharex=True)

## Top panel
axs[0].axhline(h1_energy, linestyle="-", color="0.6", label="Full dissociation limit")
axs[0].plot(rhf_energies['distance'], rhf_energies['energy']/n_atoms, color="C2", label="RHF")
axs[0].plot(uccsd_energies['distance'], uccsd_energies['energy']/n_atoms, color="C4", label="UCCSD")
axs[0].plot(dmrg_energies['distance'], dmrg_energies['energy']/n_atoms, color="C3", marker='o', markersize=6, label="DMRG (M=1024)")
axs[0].plot(gpsxsd_energies['distance'], gpsxsd_energies['energy']/n_atoms, color="0.2", linestyle=":", label=r"$\Psi^{GPS}\times\left|\Phi\right\rangle (M=72)$")
sub_df = cpdbf_energies.query('`exchange_cutoff`.isna() and `M` == 1', engine="python")
label = r"$\Psi^{CPD} (M=1)$"
axs[0].errorbar(sub_df['distance'], sub_df['energy']/n_atoms, yerr=sub_df['errorbar']/n_atoms, color="0.2", linestyle="-", label=label)
sub_df = cpdbf_energies.query('`exchange_cutoff` == 5.0 and `M` == 1')
label = r"$\Psi^{CPD} (M=1, K=5)$"
axs[0].errorbar(sub_df['distance'], sub_df['energy']/n_atoms, yerr=sub_df['errorbar']/n_atoms, color="0.2", linestyle="--", label=label)
axs[0].set_ylabel(r'Energy per atom ($E_h$)')
axs[0].set_ylim((-0.501, -0.459))
axs[0].legend(loc='lower right')

## Bottom panel
dmrg_energies_values = dmrg_energies.query('`distance` != 2.5')['energy'].values
sub_df = cpdbf_energies.query('`distance` != 2.5 and `exchange_cutoff`.isna() and `M` == 1', engine="python")
label = r"$\Psi^{CPD} (M=1)$"
error_wrt_dmrg = (sub_df['energy']-dmrg_energies_values)/n_atoms
axs[1].plot(sub_df['distance'], error_wrt_dmrg, color="0.2", linestyle="-", label=label)
sub_df = cpdbf_energies.query('`distance` != 2.5 and `exchange_cutoff` == 5.0 and `M` == 1')
label = r"$\Psi^{CPD} (M=1, K=5)$"
error_wrt_dmrg = (sub_df['energy']-dmrg_energies_values)/n_atoms
axs[1].plot(sub_df['distance'], error_wrt_dmrg, color="0.2", linestyle="--", label=label)
sub_df = gpsxsd_energies.query('`distance` != 2.5')
error_wrt_dmrg = (sub_df['energy']-dmrg_energies_values)/n_atoms
axs[1].plot(sub_df['distance'], error_wrt_dmrg, color="0.2", linestyle=":", label=r"$\Psi^{GPS}\times\left|\Phi\right\rangle (M=72)$")
axs[1].set_ylabel(r'$(E_{\Psi}-E_{DMRG})/N$')
axs[1].set_xlabel(r'Interatomic distance (Å)')
for label, ax in zip(['a', 'b'], axs):
    ax.annotate(
    label,
    xy=(0, 1), xycoords='axes fraction',
    xytext=(+0.7, -1.0), textcoords='offset fontsize', verticalalignment='top',
    fontsize=18, fontfamily='sans-serif', fontweight='bold',
    bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
plt.savefig('hsheet_6x6_energy.pdf', dpi=300, bbox_inches='tight')
