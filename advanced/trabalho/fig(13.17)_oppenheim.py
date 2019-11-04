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
Xlog.imag = np.unwrap(Xlog.imag)

v_hat = fftlib.irfft(Vlog)
p_hat = fftlib.irfft(Plog)
x_hat = fftlib.irfft(Xlog)


N1, N2 = (14,14)
low_pass = outils.box_filter(n, N1, N2, filter_type='low')
x_hat *= low_pass

# N1, N2 = (14,512)
# high_pass = outils.box_filter(n, N1, N2, filter_type='high')
# x_hat *= high_pass


Xlog_hat = fftlib.rfft(x_hat)
# Xlog_hat.imag = np.unwrap(Xlog_hat.imag)
X_hat = np.exp(Xlog_hat)
x = fftlib.irfft(X_hat)


fig, ax = plt.subplots(3,1,figsize=(7,10))
stem_options = dict(markerfmt='k.', basefmt='k,', use_line_collection=True)

omega = np.linspace(0, 1, Xlog.size)

ax[0].plot(omega, Xlog.real, 'k')
ax[0].plot(omega, Xlog_hat.real, 'k--')
ax[1].plot(omega, Xlog.imag, 'k')
ax[1].plot(omega, Xlog_hat.imag, 'k--')

ax[2].stem(n, x, 'k', **stem_options)

ax[0].axis([0,1,-6,6])
ax[1].axis([0,1,-4,4])
ax[2].axis([-10,80,-2,4])

ax[0].set_ylabel('dB')
ax[1].set_ylabel('Radians')
ax[2].set_ylabel('Amplitude')

ax[0].set_xlabel(r'Relative Frequency [$2 f/f_s$]')
ax[1].set_xlabel(r'Relative Frequency [$2 f/f_s$]')
ax[2].set_xlabel('Sample number [$n$]')


for i in range(3):
    # ax[i].axis([-100,100,-0.8,2])
    # ax[i].set_xlabel('Sample number [$n$]')
    # ax[i].set_ylabel('Amplitude')
    ax[i].grid(True, which='both')

plt.tight_layout(0.2)
out_folder = outils.out_folder
plt.savefig(out_folder / 'oppenheim_fig(13.17).png', format='png')
plt.savefig(out_folder / 'oppenheim_fig(13.17).pdf', format='pdf')
plt.show()





