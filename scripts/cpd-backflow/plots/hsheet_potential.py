import numpy as np
import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from scipy.optimize import curve_fit
from scipy.constants import c, h


# Load reference and CPD backflow data
n_atoms = 36
df = pd.read_csv('hsheet_cpd_backflow.csv')
data = {
    'UCCSD': pd.read_csv('hsheet_uccsd.csv'),
    'DMRG': pd.read_csv('hsheet_dmrg.csv'),
    'GPSxSD': pd.read_csv('hsheet_gpsxsd.csv'),
    'CPD(M=1)': df.query('`exchange_cutoff`.isna() and `M` == 1', engine="python"),
    'CPD(M=1, K=5)': df.query('`exchange_cutoff` == 5.0 and `M` == 1')
}

# Morse potential
def morse_potential(r, D_e, a, r_e, u):
    return D_e * (1 - np.exp(-a * (r - r_e)))**2 + u

morse_potential_params = {}
table = '| Method | $D_e~(eV)$ | $a~(\AA^{-1})$ | $r_e~(\AA)$ | $u~(eV)$ | $\omega_e~(cm^{-1})$ | $\omega_e\chi_e~(cm^{-1})$ |\n'
table += '|---|---|---|---|---|---|---|\n'
mu = 1.6738e-27 / n_atoms # reduced mass of N H atoms in kg
ev2J = 1.602e-19
ang2cm = 1e-8
for method, sub_df in data.items():
    distances = sub_df['distance'].unique()
    energies = sub_df['energy']/n_atoms
    params, cv = curve_fit(morse_potential, distances, energies, [1.0, 1.0, 1.0, 0.5])
    morse_potential_params[method] = params
    D_e, a, r_e, u = params
    we = ((a/ang2cm)/(2*np.pi*c))*np.sqrt(2*(D_e*ev2J)/mu)
    wexe = (h*c*we**2)/(4*D_e*ev2J)
    table += f"| {method} | {D_e:.3f} | {a:.3f} | {r_e:.3f} | {u:.3f} | {we:.3f} | {wexe:.3f} |\n"
console = Console()
console.print(Markdown(table))
