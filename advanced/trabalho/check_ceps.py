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

p = utils.pn(n)
v = utils.vn(n)
# v = np.sin(2 * np.pi * 100/8e3 * n) ** 2

indf = indi + num_samples
x = np.convolve(v, p, mode='full')[indi:indf]


gamma = 2
real_mode = False
xv = utils.ceps(x, gamma, real=real_mode)

# xv += 1e-3 * rnd.randn(xv.size)

x_hat = utils.iceps(xv, gamma, real=real_mode)


fig, ax = plt.subplots(figsize=(10,7))

# ax.plot(x)
# ax.plot(x_hat)
# ax.plot(np.abs(x - x_hat))
# ax.plot(xv.imag)
ax.plot(np.abs(fftlib.fftshift(xv)))

plt.show()

