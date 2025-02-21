import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science', 'nature', 'notebook'])
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from pyscf import gto, scf, cc, fci, ao2mo


loc = LogLocator(10)
fmt = LogFormatterSciNotation(10.0)

# PySCF reference data
mol = gto.Mole()
mol.atom = [('H', [0.0, 0.795, -0.454]), ('H', [0.0, -0.795, -0.454]),	('O', [0.0, 0.0, 0.113])]
mol.basis = '6-31g'
mol.unit = 'angstrom'
mol.symmetry = True
mol.build()

## RHF
mf = scf.RHF(mol)
mf.kernel()
nuc_energy = mf.energy_nuc()
hf_energy = mf.energy_tot()

## CCSD
ccsd = cc.CCSD(mf)
ccsd.kernel()
ccsd_energy = ccsd.e_tot

## Exact energy
ovlp = mf.get_ovlp()
norb = mf.mo_coeff.shape[1]
nelec = np.sum(mol.nelec)
h1 = np.linalg.multi_dot((mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
h2 = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
energy_mo, _ = fci.direct_spin1.FCI().kernel(h1, h2, norb, nelec)
exact_energy = energy_mo + nuc_energy
ccsd_rel_corr_error = np.abs(ccsd_energy-exact_energy)/np.abs(exact_energy-hf_energy)

# Reference data DOI: 10.1103/PhysRevB.107.205119
nqs_1e4_energy = -76.104
nqs_1e4_rel_corr_energy = np.abs(nqs_1e4_energy-exact_energy)/np.abs(exact_energy-hf_energy)
nqs_1e6_energy = -76.1155
nqs_1e6_rel_corr_energy = np.abs(nqs_1e6_energy-exact_energy)/np.abs(exact_energy-hf_energy)
sdgps_energy = -76.10338343397788
sdgps_rel_corr_error = np.abs(sdgps_energy-exact_energy)/np.abs(exact_energy-hf_energy)
pfgps_energy_M1 = -76.11168407402323
pfgps_rel_corr_error_M1 = np.abs(pfgps_energy_M1-exact_energy)/np.abs(exact_energy-hf_energy)
pfgps_energy_M4 = -76.1154282864921
pfgps_rel_corr_error_M4 = np.abs(pfgps_energy_M4-exact_energy)/np.abs(exact_energy-hf_energy)
pfgps_energy_M8 = -76.11625708288253
pfgps_rel_corr_error_M8 = np.abs(pfgps_energy_M8-exact_energy)/np.abs(exact_energy-hf_energy)

# Load CPD backflow data
df = pd.read_csv('h2o.csv')

# Correlation energy
fig, ax = plt.subplots(figsize=(9, 6))
for M, sub_df in df.query('`M` in [1, 4]').groupby('M'):
    ax.errorbar(sub_df['n_samples'], sub_df['rel_corr_error']*100, sub_df['rel_corr_errorbar']*100, marker="o", markeredgecolor="0.2", markersize=12, label=r"$\Psi^{{CPD}} (M={{{}}})$".format(M))
ax.plot([1e4, 1e6], np.array([nqs_1e4_rel_corr_energy, nqs_1e6_rel_corr_energy])*100, marker="s", markeredgecolor="0.2", markersize=12, label=r"$\Psi^{RBM}$")
ax.scatter(1e4, pfgps_rel_corr_error_M8*100, color='C3', marker='d', edgecolor="0.2", s=96, label=r"$\Psi^{GPS}(M=8)\times\left|\text{Pf}\right\rangle$")
ax.axhline(ccsd_rel_corr_error*100, color='0.4', ls='-', label="CCSD")
ax.set_xscale('log')
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(fmt)
ax.set_xlim((8e2, 1.2e6))
ax.set_ylim((0.0, nqs_1e4_rel_corr_energy*100+0.5))
chem_acc_rel_corr_error = 0.0016/np.abs(exact_energy-hf_energy)
ax.fill_betweenx([ax.get_ylim()[0], chem_acc_rel_corr_error*100], ax.get_xlim()[0], ax.get_xlim()[1], color='red', alpha=0.2, label="Chemical accuracy")
ax.set_xlabel(r'Number of samples $(N_S)$')
ax.set_ylabel('Rel. corr. energy error (%)')
handles, labels = ax.get_legend_handles_labels()
order = [4, 5, 0, 1, 2, 3]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper left")
plt.savefig("h2o_rel_corr_error.pdf", dpi=300, bbox_inches="tight")

# Systematic improvability
fig, ax = plt.subplots(2, 1, figsize=(9, 12), gridspec_kw={'hspace': 0.15})
for Ns, sub_df in df.query('`M` in [1, 2, 3, 4] and `n_samples` in [4096, 16384]').groupby('n_samples'):
    xs = 1/sub_df['M']
    ys = sub_df['rel_corr_error']*100
    p = np.polyfit(xs, ys, 1)
    xf = np.linspace(0, 1, 100)
    yf = np.polyval(p, xf)
    ax[0].errorbar(xs, ys, sub_df['rel_corr_errorbar']*100, marker="o", markeredgecolor="0.2", markersize=12, label=r"$\Psi^{{CPD}} (N_S={{{}}})$".format(Ns))
    ax[0].scatter(xf[0], yf[0], s=96, marker='d', edgecolor='0.2', zorder=10)
    ax[0].plot(xf, yf, linestyle='dashed', color='0.2')
ax[0].set_ylim((1.0, 4.2))
ax[0].set_xlim((-0.05, 1.05))
ax[0].fill_betweenx([1.0, chem_acc_rel_corr_error*100], ax[0].get_xlim()[0], ax[0].get_xlim()[1], color='red', alpha=0.2, label="Chemical accuracy")
ax[0].set_xticks(list(1./np.arange(1, 5))+[0.], labels=['1/1', '1/2', '1/3', '1/4', '0'])
ax[0].set_xlabel(r'$1/M$')
handles, labels = ax[0].get_legend_handles_labels()
handles += [plt.Line2D([], [], linestyle='dashed', markersize=10, marker='d', color='0.2')]
labels += ['Extrapolated']
order = [1, 2, 3, 0]
ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower right", bbox_to_anchor=(0.98, 0.05))

for M, sub_df in df.query('`M` in [1, 4]').groupby('M'):
    xs = 1./np.sqrt(sub_df['n_samples'])
    ys = sub_df['rel_corr_error']*100
    p = np.polyfit(xs, ys, 1)
    xf = np.linspace(0, 1./np.sqrt(1024), 100)
    yf = np.polyval(p, xf)
    ax[1].errorbar(xs, ys, sub_df['rel_corr_errorbar']*100, marker="o", markeredgecolor="0.2", markersize=12, label=r"$\Psi^{{CPD}} (M={{{}}})$".format(M))
    ax[1].scatter(xf[0], yf[0], s=96, marker='d', edgecolor='0.2', zorder=10)
    ax[1].plot(xf, yf, linestyle='dashed', color='0.2')
ax[1].set_ylim((1.0, 4.7))
ax[1].set_xlim((-0.002, 0.033))
ax[1].fill_betweenx([1.0, chem_acc_rel_corr_error*100], ax[0].get_xlim()[0], ax[0].get_xlim()[1], color='red', alpha=0.2, label="Chemical accuracy")
ax[1].set_xlabel(r'$1/\sqrt{N_S}$')
handles, labels = ax[1].get_legend_handles_labels()
handles += [plt.Line2D([], [], linestyle='dashed', markersize=10, marker='d', color='0.2')]
labels += ['Extrapolated']
order = [1, 2, 3, 0]
ax[1].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower right", bbox_to_anchor=(0.98, 0.05))

fig.supylabel('Rel. corr. energy error (%)', x=0.05, fontsize=16)
for label, ax in zip(['a', 'b'], ax):
    ax.annotate(
    label,
    xy=(0, 1), xycoords='axes fraction',
    xytext=(+0.7, -1.0), textcoords='offset fontsize', verticalalignment='top',
    fontsize=18, fontfamily='sans-serif', fontweight='bold',
    bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
plt.savefig("h2o_syst_improv.pdf", dpi=300, bbox_inches="tight")
