import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as lin
import scipy.misc as misc


def plot_zpk(z, p, k, fp, fs, Amax, Amin, sample_rate=48e3):
    w, h = signal.freqz_zpk(z, p, k)
    
    plt.plot(w/np.pi*sample_rate, 20 * np.log10(abs(h)))
    plt.xscale('log')
    plt.title('Lowpass filter fit to constraints')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True, which='both', axis='both')
    plt.fill([1e2, 1e2,  fp,  fp], [-Amin, -Amax, -Amax, -Amin], '0.9', lw=0) # pass
    plt.fill([fs, fs,  1e4,  1e4], [-Amin, 0, 0, -Amin], '0.9', lw=0) # stop
    plt.axis([1e2, 1e4, -50, 1])


def stepInvTransform(z, p, k, sample_rate):

    T = 1/sample_rate
    pass




sample_rate = 48e3

fp = 1.8e3
fs = 3.5e3

Amax = 1
Amin = 42


N, Wn = signal.buttord(2*np.pi*fp, 2*np.pi*fs, Amax, Amin, analog=True)

print(f'Amax = {Amax} dB, Amin = {Amin} dB')
print(f'fp = {fp/1e3} kHz, fs = {fs/1e3} kHz')
print(f'Butterworth order = {N}')
print(f'Wn = 2π * {Wn/2/np.pi:.3f}Hz')

z, p, k = signal.butter(N, Wn, output='zpk', btype='low', analog=True)
b, a = signal.zpk2tf(z, p, k)

print(f'z = {z}')
print(f'p = {p}')
print(f'k = {k}')
print(f'b = {b}')
print(f'a = {a}')


zd, pd, kd, dt = signal.cont2discrete((z,p,k), 1/sample_rate, method='euler', alpha=None)
bd, ad, dt = signal.cont2discrete((b,a), 1/sample_rate, method='euler', alpha=None)

print(f'bd = {bd}')
print(f'ad = {ad}')
print(f'zd = {zd}')
print(f'pd = {pd}')
print(f'kd = {kd}')

plot_zpk(zd, pd, kd, fp, fs, Amax, Amin)
plt.show()
