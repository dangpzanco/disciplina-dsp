import numpy as np
import matplotlib.pyplot as plt
import pathlib

import oppenheim_utils as outils



num_samples = 101
indi = 20
n = np.arange(num_samples) - indi


u = outils.unit_step(n-1)
p = outils.pn(n)
v = outils.vn(n)

indf = indi + num_samples
x = np.convolve(p, v, mode='full')[indi:indf]

print(x.shape)


stem_options = dict(markerfmt='k.', basefmt='k,', use_line_collection=True)

fig, ax = plt.subplots(3,1,figsize=(7,10), sharex=True)

ax[0].stem(n, v, 'k', **stem_options)
ax[1].stem(n, p, 'k', **stem_options)
ax[2].stem(n, x, 'k', **stem_options)

ax[0].set_ylabel('$v[n]$', fontsize=14)
ax[1].set_ylabel('$p[n]$', fontsize=14)
ax[2].set_ylabel('$x[n]$', fontsize=14)

ax[0].axis([-20,80,-2,4])
ax[1].axis([-20,80,-0.4,1.2])
ax[2].axis([-20,80,-2,4])

for i in range(3):
    ax[i].set_xlabel('Sample number [$n$]')
    # ax[i].set_ylabel('Amplitude')
    ax[i].grid(True, which='both')

plt.tight_layout(0.2)

out_folder = outils.out_folder

plt.savefig(out_folder / 'oppenheim_fig(13.10).png', format='png', dpi=600)
plt.savefig(out_folder / 'oppenheim_fig(13.10).pdf', format='pdf')

plt.show()











