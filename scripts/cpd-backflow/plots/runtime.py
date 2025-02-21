import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science', 'nature', 'notebook'])


df = pd.read_csv('runtime.csv')
groups = df.groupby('pruning_threshold')
fig, ax = plt.subplots(figsize=(9, 6))
for i, (threshold, group) in enumerate(groups):
    notna = group['runtime'].notna()
    x = group.loc[notna]['n_atoms']
    y = group.loc[notna]['runtime']
    y_err = group.loc[notna]['error']
    if threshold != 0.0:
        exponent = int(np.log10(threshold))
        label = r'Truncated local energy: $10^{{{}}} E_h$'.format(exponent)
    else:
        label = r'$\Psi^{CPD}$ with full $\hat{H}$'
    ax.errorbar(x, y, y_err, linestyle='none', marker='d', markersize=10, markeredgecolor='0.2', label=label)
    if i == 0:
        last_n = 10
    else:
        last_n = 8
    p = np.polyfit(np.log(x)[-last_n:], np.log(y)[-last_n:], 1)
    x_fit = np.linspace(x[-last_n:].min(), x[-last_n:].max(), 100)
    y_fit = np.exp(np.polyval(p, np.log(x_fit)))
    p_str = f'{round(p[0], 1)}'
    ax.plot(x_fit, y_fit, linestyle='--', color=f"C{i}", label=r"$\mathcal{O}(N^{" + p_str + "})$")
handles, labels = ax.get_legend_handles_labels()
order = [3, 4, 5, 0, 1, 2]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Mean VMC step runtime (s)')
ax.set_xlabel('Number of hydrogen atoms');
plt.savefig("runtime.pdf", dpi=300, bbox_inches="tight")
