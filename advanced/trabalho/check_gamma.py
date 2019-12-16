import pathlib

import numpy as np
import numpy.random as rnd
import numpy.fft as fftlib
import scipy.signal as signal

import matplotlib.pyplot as plt

import cepstrum_utils as utils

from tqdm import trange


num_p = 2 * np.arange(7)
# num_p = 1 * np.arange(7)
num_z = num_p[::-1]

num_sequences = len(num_p)

print(num_p)
print(num_z)
# gamma = 0.5

r = 0.98
num_samples = 256
num_h = 1000
num_gamma = 200

max_gamma = 1
gamma_list = np.linspace(-max_gamma,max_gamma,num_gamma)

n = np.arange(num_samples)
delta = utils.unit_impulse(n)

# print(np.diff(gamma_list))


# num_sequences = 1

D = np.empty([num_sequences, num_h, num_gamma, num_samples//2])


for i in range(num_sequences):

    Nz, Np = (num_p[i], num_z[i])

    for j in range(num_h):

        zpk = utils.rnd_zpk(r, Nz, Np)
        sos = signal.zpk2sos(*zpk)
        h = signal.sosfilt(sos, delta)

        for k in range(num_gamma):
            gamma = gamma_list[k]
            hv = utils.rceps(h, gamma)
            
            # hv = utils.zpk2root(zpk[0], zpk[1], zpk[2], n, gamma, L=100).real
            # hv = utils.recursive_rceps(h, gamma, A=zpk[-1])

            # print(h.size)
            # print(hv.size)

            # plt.plot(n[:num_samples//2], hv)
            # # plt.plot(n[:num_samples//2], hv.imag)
            # plt.show()

            d = utils.relative_energy(hv[1:num_samples//2+1])

            D[i,j,k,:] = d

    # d_temp = D[i,:,:,:]
    # # plt.plot(gamma_list, d_temp)
    # plt.plot(n[:num_samples//2], d_temp.std(axis=0).T)
    # plt.show()



fig, ax = plt.subplots(figsize=(10,7))

# ax.plot(hv)

Dmean = D.mean(axis=1)
# Dmean = D.std(axis=1)


for i in range(num_sequences):
    ind = Dmean[i,].argmax(axis=0)

    print(ind)
    Nz, Np = (num_p[i], num_z[i])

    ax.plot(n[:num_samples//2], gamma_list[ind], label=f'{Nz}, {Np}')

plt.legend()
plt.show()











