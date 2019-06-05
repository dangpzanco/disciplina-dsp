# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.linalg as lin
from scipy.special import binom
from scipy.special import factorial
import scipy.misc as misc
import autograd

import filter_design as fd


def plotFig1(mag_type='abs', fs=2, ax1=None):

    if ax1 is None:
        fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')

    LK = [(1,1), (3,3), (5,7), (7,9), (9,13)]
    tau = 0.5
    P = 6

    omega_b = 0.35 * np.pi

    worN = np.linspace(0,1,512)
    for item in LK:
        L, K = item
        M = L - 1
        N = K - 1
        p = int((L + K + 2)/2)
        
        Nz, Dz = fd.getHz1(omega_b, M, N, tau, P, p)
        w1, h = sig.freqz(Nz, Dz, worN, fs=fs)

        print(f'L = {L}, K = {K}')
        print(f'M = {M}, N = {N}')
        print(f'p = {p}, num_eq = {int((L+K+4)/2)}')
        print(f'Nz shape = {Nz.shape}')
        print(f'Dz shape = {Dz.shape}')

        if mag_type == 'abs':
            mag = np.abs(h)
            mag_label = 'Magnitude'
        elif mag_type == 'dB':
            mag = 20 * np.log10(np.abs(h))
            mag_label = 'Magnitude [dB]'
        elif mag_type == 'loss':
            mag = 20 * np.log10(1/np.abs(h))
            mag_label = 'Attenuation [dB]'

        ax1.plot(w1, mag)

    ax1.legend([f'L = {l}, K = {k}' for l, k in LK])
    ax1.set_ylabel(mag_label, color='b')
    ax1.set_xlabel('Frequency [normalized]')
    ax1.set_xlim(0,1)
    ax1.set_xticks(np.linspace(0,1,11))

    if mag_type == 'abs':
        ax1.set_ylim(0,1.1)
    elif mag_type == 'dB':
        ax1.set_ylim(-60,5)
    elif mag_type == 'loss':
        ax1.set_ylim(-5,60)

    ax2 = ax1.twinx()
    w2, gd = sig.group_delay((1, Dz), fs=fs)
    ax2.plot(w2, gd, 'k')
    ax2.set_ylabel('Group delay', color='k')
    ax2.grid()
    ax2.axis('tight')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-1.1,1.2)

    ax2.grid(False)
    ax1.grid(True)


tau = 0.5
P = 6

L = 7
K = 9

M = L - 1
N = K - 1
p = int((L + K + 3)/2)

omega_b = 0.35 * np.pi

print(f'tau = {tau}')
print(f'P = {P}')
print(f'L = {L}, K = {K}')
print(f'M = {M}, N = {N}')
print(f'p = {p}, num_eq = {M+N+4}')
print(f'omega_b = {omega_b/np.pi}π')


# Reference 6
# fig, ax1 = plt.subplots()
# plotFilter(1, maxFlatPoly(4, 5), mag_type='loss', ax1=ax1)
# plotFilter(1, maxFlatPoly(4, 10), mag_type='loss', ax1=ax1)
# plotFilter(1, maxFlatPoly(4, 15), mag_type='loss', ax1=ax1)
# plt.show()



Nz, Dz = fd.getHz1(omega_b, M, N, tau, P, p)
print(f'Nz shape = {Nz.shape}')
print(f'Dz shape = {Dz.shape}')

# fd.plotFilter(Nz, Dz)

plotFig1(mag_type='abs')
plt.show()
