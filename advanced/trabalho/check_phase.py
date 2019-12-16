import pathlib

import numpy as np
import numpy.random as rnd
import numpy.fft as fftlib
import scipy.signal as signal

import matplotlib.pyplot as plt

import cepstrum_utils as utils





num_samples = 1024
indi = 0
n = np.arange(num_samples) - indi

fs = 10e3
Tmax = num_samples / fs
echo_period = 15 / fs
echo_est = echo_period
Tc = (echo_est, Tmax - echo_est)

echo_options = dict(N0=int(echo_period * fs),
    K=3, 
    beta=0.9)

x, v, p = utils.get_example_signals(num_samples, echo_options, indi=indi)


gamma = 2


f, Xg = signal.group_delay((x,1), w=num_samples, whole=True, fs=1)

f, Xg2 = utils.grpdelay((x,1), nfft=num_samples, whole=True, fs=1)


X = fftlib.fft(x)

Xv = np.abs(X) ** gamma * np.exp(1j * gamma * np.angle(X))


x_rceps = utils.rceps(x, gamma, real=False)
x_ceps = utils.ceps(x, gamma, real=False)



fig, ax = plt.subplots(2, 1, figsize=(10,7), sharex=True)

# ax[0].plot(Xv.real)
# ax[1].plot(Xv.imag)

ax[0].plot(np.abs(Xv))
ax[1].plot(np.angle(Xv))

# ax[0].plot(x_rceps)
# ax[0].plot(x_ceps)


plt.tight_layout()
plt.show()

