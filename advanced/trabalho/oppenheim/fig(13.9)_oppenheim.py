import numpy as np
import matplotlib.pyplot as plt
import pathlib

import oppenheim_utils as outils





zP, pP, kP = outils.Pz()
zV, pV, kV = outils.Vz()

circle = outils.unit_circle(100)


plot_options = dict(markersize=10, markeredgewidth=1.5, markerfacecolor='none')

fig, ax = plt.subplots(figsize=(5,5))

ax.plot(zP.real, zP.imag, 'o', markeredgecolor='tab:blue', **plot_options)
ax.plot(zV.real, zV.imag, 'o', markeredgecolor='tab:orange', **plot_options)

ax.plot(pP.real, pP.imag, 'x', markeredgecolor='tab:blue', **plot_options)
ax.plot(pV.real, pV.imag, 'x', markeredgecolor='tab:orange', **plot_options)

ax.plot(circle[0], circle[1], 'k')
ax.set_aspect('equal', 'box')

ax.set_xlabel(r'$\mathcal{Re}\{z\}$')
ax.set_ylabel(r'$\mathcal{Im}\{z\}$')

plt.tight_layout(0.2)

out_folder = outils.out_folder
plt.savefig(out_folder / 'oppenheim_fig(13.9).png', format='png', dpi=600)
plt.savefig(out_folder / 'oppenheim_fig(13.9).pdf', format='pdf')
plt.grid(True)
plt.show()











