import numpy as np
import numpy.fft as libfft
import librosa

import matplotlib.pyplot as plt




def activation(x, gamma):
    x[x < 0] = 0

    y = x ** gamma

    return y


def Wfilter(x, index):
    x[:index] = 0
    return x


def Wlifter(x, index):
    x[:index] = 0
    x[x.size-index:] = 0
    return x


def network(x, g1, g2, g3, kc, nc):

    a1 = Wfilter(np.abs(libfft.rfft(x)), kc)
    z1 = activation(a1, g1)

    a2 = Wlifter(libfft.irfft(z1).real, nc)
    z2 = activation(a2, g2)

    a3 = Wfilter(libfft.rfft(z2).real, kc)
    z3 = activation(a3, g3)

    z = [z1, z2, z3]

    return z


# Signal duration and sample rate
N = 4096
n = np.arange(N)
fs = 44.1e3

# Define frequencies
num_freqs = 10
freqs = np.linspace(100,fs/4,num_freqs)

# Generate input signal
x = 0
for i in range(num_freqs):
    x += np.sin(2*np.pi*freqs[i]/fs*n + np.random.randn())
x /= num_freqs
x += np.random.randn(N) * 1e-2



# fc = 20
# tc = 1/20000

fc = 20
tc = 1/20e3
fc_idx = int(np.round(fc/fs * N))
tc_idx = int(np.round(fs*tc))

params = dict(g1=0.6, g2=0.8, g3=1, kc=fc_idx, nc=tc_idx)


z = network(x, **params)

print(z)


fig, ax = plt.subplots(2,1, sharex=True)
ax = ax.ravel()
ax[0].plot(n/fs,x)
ax[1].plot(n/fs,z[1].real)
ax[1].set_ylabel('$z_2$')
ax[1].set_xlabel('Time [s]')


fig, ax = plt.subplots(2,1, sharex=True)
ax = ax.ravel()
f = np.linspace(0,fs/2,int(N/2)+1)

ax[0].plot(f,z[0].real)
ax[0].set_ylabel('$z_1$')

ax[1].plot(f,z[2].real)
ax[1].set_ylabel('$z_3$')
ax[1].set_xlabel('Frequency [Hz]')


plt.show()



