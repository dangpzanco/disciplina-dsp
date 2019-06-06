import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as lin
import scipy.misc as misc


def plot_zpk(z, p, k, fp, fs, Amax, Amin, sample_rate=48e3):
    f, h = signal.freqz_zpk(z, p, k, fs=sample_rate)
    
    plt.plot(f, 20 * np.log10(abs(h)))
    plt.xscale('log')
    plt.title('Lowpass filter fit to constraints')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True, which='both', axis='both')
    plt.fill([1e2, 1e2,  fp,  fp], [-Amin*2, -Amax, -Amax, -Amin*2], '0.9', lw=0) # pass
    plt.fill([fs, fs,  1e4,  1e4], [-Amin, Amax, Amax, -Amin], '0.9', lw=0) # stop
    plt.axis([1e2, 1e4, -50, 1])

def plot_step(analog_system, discrete_system, num_samples=100, sample_rate=48e3, plot_error=False):

    n = np.arange(num_samples)
    _, ya = signal.step2(analog_system, T=n/sample_rate)
    _, yd = signal.dstep((*discrete_system,1/sample_rate), t=n/sample_rate)


    fig, ax1 = plt.subplots()
    ax1.set_title('Invariância ao degrau')

    ax1.plot(n,ya,'b')
    ax1.plot(n,np.ravel(yd),'k.')
    ax1.set_ylabel('Amplitude', color='b')
    ax1.set_xlabel('Amostras')
    ax1.grid(True)
    ax1.legend(['Analógico', 'Discreto'])

    if plot_error:
        ax2 = ax1.twinx()
        ax2.plot(n, np.abs(ya-np.ravel(yd)), 'r')
        ax2.set_ylabel('Erro', color='r')
        ax2.axis('tight')

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



zd, pd, kd, dt = signal.cont2discrete((z,p,k), 1/sample_rate, method='zoh', alpha=None)

print(f'zd = {zd}')
print(f'pd = {pd}')
print(f'kd = {kd}')

plot_zpk(zd, pd, kd, fp, fs, Amax, Amin)

analog_system = (z,p,k)
discrete_system = (zd,pd,kd)
plot_step(analog_system, discrete_system, num_samples=100, sample_rate=sample_rate, plot_error=True)
plt.show()




