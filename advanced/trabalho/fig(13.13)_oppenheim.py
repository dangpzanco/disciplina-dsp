import numpy as np
import numpy.fft as fftlib
import matplotlib.pyplot as plt
import pathlib

import oppenheim_utils as outils



num_samples = 1024
n = np.arange(num_samples)

p = outils.pn(n)
v = outils.vn(n)

# indf = indi + num_samples
# x_full = np.convolve(p, v, mode='full')
# x = x_full[indi:indf]
# print(x_full.shape, x.shape, indi, indf)



P = fftlib.rfft(p)
V = fftlib.rfft(v)
# X = fftlib.rfft(x)
X = P * V

Xlog = np.log(X)

Xlog2 = np.log(fftlib.rfft(outils.vn(n+1)) * fftlib.rfft(outils.pn(n)))


fig, ax = plt.subplots(3,1,figsize=(7,10))

omega = np.linspace(0, 1, Xlog.size)

ax[0].plot(omega, Xlog.real, 'k')
ax[1].plot(omega, Xlog.imag, 'k')
# ax[2].plot(omega, np.unwrap(Xlog.imag), 'k')
ax[2].plot(omega, Xlog2.imag, 'k')

ax[0].axis([0,1,-6,6])
ax[1].axis([0,1,-4,4])
ax[2].axis([0,1,-4,4])

ax[0].set_ylabel('dB')
ax[1].set_ylabel('Radians')
ax[2].set_ylabel('Radians')


for i in range(3):
    ax[i].set_xlabel(r'Relative Frequency [$2 f/f_s$]')
    ax[i].grid(True, which='both')


plt.tight_layout(0.2)
out_folder = outils.out_folder
plt.savefig(out_folder / 'oppenheim_fig(13.13).png', format='png')
plt.savefig(out_folder / 'oppenheim_fig(13.13).pdf', format='pdf')
plt.show()













