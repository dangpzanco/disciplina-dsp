import numpy as np
import numpy.fft as fftlib
import matplotlib.pyplot as plt
import pathlib

import oppenheim_utils as outils



num_samples = 512
n = np.arange(num_samples)

v = outils.vn(n+1)
p = outils.pn(n)

V = fftlib.fft(v)
P = fftlib.fft(p)
X = P * V

Vlog = np.log(V)
Plog = np.log(P)
Xlog = np.log(X)
Xlog.imag = np.unwrap(Xlog.imag)

v_hat = fftlib.ifft(Vlog)
p_hat = fftlib.ifft(Plog)
x_hat = fftlib.ifft(Xlog)

v_hat = outils.vhatn(n - num_samples // 2)
x_hat = fftlib.fftshift(v_hat) + p_hat


fig, ax = plt.subplots(3,1,figsize=(7,10), sharex=True, sharey=True)
stem_options = dict(markerfmt='k.', basefmt='k.', use_line_collection=True)

n = n - num_samples // 2
ax[0].stem(n, v_hat, 'k', **stem_options)
# ax[0].stem(n, fftlib.fftshift(v_hat), 'k', **stem_options)
ax[1].stem(n, fftlib.fftshift(p_hat), 'k', **stem_options)
ax[2].stem(n, fftlib.fftshift(x_hat), 'k', **stem_options)


signal_name = [r'$\hat{v}[n]$', r'$\hat{p}[n]$', r'$\hat{x}[n]$']

for i in range(3):
    ax[i].axis([-100,100,-0.8,2])
    ax[i].set_xlabel('Sample number [$n$]')
    ax[i].set_ylabel(f'{signal_name[i]}')
    ax[i].grid(True, which='both')

plt.tight_layout(0.2)
out_folder = outils.out_folder
plt.savefig(out_folder / 'oppenheim_fig(13.11).png', format='png', dpi=600)
plt.savefig(out_folder / 'oppenheim_fig(13.11).pdf', format='pdf')
plt.show()





