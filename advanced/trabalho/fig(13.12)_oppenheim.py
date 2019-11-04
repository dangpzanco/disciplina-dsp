import numpy as np
import numpy.fft as fftlib
import matplotlib.pyplot as plt
import pathlib

import oppenheim_utils as outils



num_samples = 1024
n = np.arange(num_samples)

v = outils.vn(n)
p = outils.pn(n)

V = fftlib.rfft(v)
P = fftlib.rfft(p)
X = P * V

Vlog = np.log(V)
Plog = np.log(P)
Xlog = np.log(X)

v_hat = fftlib.irfft(Vlog.real)
p_hat = fftlib.irfft(Plog.real)
x_hat = fftlib.irfft(Xlog.real)



fig, ax = plt.subplots(3,1,figsize=(7,10), sharex=True, sharey=True)
stem_options = dict(markerfmt='k.', basefmt='k,', use_line_collection=True)

ax[0].stem(n - num_samples // 2, fftlib.fftshift(v_hat), 'k', **stem_options)
ax[1].stem(n - num_samples // 2, fftlib.fftshift(p_hat), 'k', **stem_options)
ax[2].stem(n - num_samples // 2, fftlib.fftshift(x_hat), 'k', **stem_options)

for i in range(3):
    ax[i].axis([-100,100,-0.5,1.5])
    ax[i].set_xlabel('Sample number [$n$]')
    ax[i].set_ylabel('Amplitude')
    ax[i].grid(True, which='both')

plt.tight_layout(0.2)
out_folder = outils.out_folder
plt.savefig(out_folder / 'oppenheim_fig(13.12).png', format='png')
plt.savefig(out_folder / 'oppenheim_fig(13.12).pdf', format='pdf')
plt.show()





