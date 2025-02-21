import pandas as pd
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science', 'nature', 'notebook'])
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, FormatStrFormatter


# Reference data from DOI: 10.1103/PhysRevLett.122.226401
nnb_rel_errors = {
    0.875: {
        'best': 1.4,
        'var_extrap': 0.66
    },
    1.0: {
        'best': 2.714,
        'var_extrap': 1.745
    }
}

# Load CPD backflow data
df = pd.read_csv('hubbard2d_U8.csv')

# Plot
loc = LogLocator(2)
fmt = LogFormatterSciNotation(2.0)
fig, axs = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'hspace': 0.05})

## Doped case (n = 7/8 = 0.875)
for M, sub_df in df.query('`filling` == 0.875').groupby("M"):
    axs[0].errorbar(sub_df["n_samples"], sub_df["rel_error"]*100, sub_df["rel_errorbar"]*100, marker="o", markersize=12, markeredgecolor="0.2", label=r"$\Psi^{{CPD}} (M={{{}}})$".format(M))
axs[0].axhline(nnb_rel_errors[0.875]['best'], color="0.4", linestyle="--", label=r"Neural-network backflow (NNB)")
axs[0].axhline(nnb_rel_errors[0.875]['var_extrap'], color="C3", linestyle="--", label="NNB with variance extrapolation")
axs[0].set_xscale("log")
axs[0].xaxis.set_major_locator(loc)
handles, labels = axs[0].get_legend_handles_labels()
order = [2, 3, 0, 1]
axs[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncols=2, bbox_to_anchor=(0.01, 1.01), loc="lower left")
axs[0].text(0.5, 0.78, r"Doped ($n=0.875$)", ha="center", va="center", transform=axs[0].transAxes, fontsize=18, bbox=dict(boxstyle="round", facecolor="1.0"))
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

## Half-filling (n = 1.0)
for M, sub_df in df.query('`filling` == 1.0').groupby("M"):
    axs[1].errorbar(sub_df["n_samples"], sub_df["rel_error"]*100, sub_df["rel_errorbar"]*100, marker="o", markersize=12, markeredgecolor="0.2")
axs[1].axhline(nnb_rel_errors[1.0]['best'], color="0.4", linestyle="--")
axs[1].axhline(nnb_rel_errors[1.0]['var_extrap'], color="C3", linestyle="--")
axs[1].set_xlabel("Number of samples")
axs[1].set_xscale("log")
axs[1].xaxis.set_major_locator(loc)
axs[1].xaxis.set_major_formatter(fmt)
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[1].text(0.5, 0.78, r"Half-filling ($n=1.0$)", ha="center", va="center", transform=axs[1].transAxes, fontsize=18, bbox=dict(boxstyle="round", facecolor="1.0"))
fig.supylabel("Rel. energy error (%)", x=0.05, fontsize=16)
for label, ax in zip(['a', 'b'], axs):
    ax.annotate(
    label,
    xy=(0, 1), xycoords='axes fraction',
    xytext=(+0.7, -1.0), textcoords='offset fontsize', verticalalignment='top',
    fontsize=18, fontfamily='sans-serif', fontweight='bold',
    bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
plt.savefig("hubbard2d_U8.pdf", dpi=300, bbox_inches="tight")
