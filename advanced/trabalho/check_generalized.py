import pathlib

import numpy as np
import numpy.random as rnd
import numpy.fft as fftlib
import scipy.signal as signal

import matplotlib.pyplot as plt

import cepstrum_utils as utils



gamma = 1


num_samples = 1000
P = rnd.randn(num_samples) + 1j * rnd.randn(num_samples)
H = rnd.randn(num_samples) + 1j * rnd.randn(num_samples)
X = P * H

# Pv = P ** gamma
# Hv = H ** gamma

Pv = utils.log(P, gamma)
Hv = utils.log(H, gamma)
Xv = Pv * Hv

X_hat = utils.exp(Xv, gamma)

fig, ax = plt.subplots(2,1,figsize=(10,7))

ax[0].plot(X.real)
ax[0].plot(X_hat.real)

ax[1].plot(X.imag)
ax[1].plot(X_hat.imag)

plt.show()

