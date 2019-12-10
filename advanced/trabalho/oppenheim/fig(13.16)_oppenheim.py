import numpy as np
import numpy.fft as fftlib
import matplotlib.pyplot as plt
import pathlib

import oppenheim_utils as outils



num_samples = 1024
n = np.arange(num_samples)



N1, N2 = (200,100)
low_pass = outils.box_filter(n, N1, N2, filter_type='low')

N1, N2 = (200,100)
high_pass = outils.box_filter(n, N1, N2, filter_type='high')

fig, ax = plt.subplots(2,1,figsize=(7,10), sharex=True, sharey=True)

ax[0].plot(n, low_pass)
ax[1].plot(n, high_pass)

for i in range(2):
    ax[i].axis([0,1023,-0.5,1.5])
    ax[i].set_xlabel('Sample number [$n$]')
    ax[i].set_ylabel('Amplitude')
    ax[i].grid(True, which='both')

plt.tight_layout(0.2)
out_folder = outils.out_folder
plt.savefig(out_folder / 'oppenheim_fig(13.16).png', format='png', dpi=600)
plt.savefig(out_folder / 'oppenheim_fig(13.16).pdf', format='pdf')
plt.show()





