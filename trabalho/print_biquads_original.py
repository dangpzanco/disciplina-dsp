from numba import jit, njit
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as lin
import scipy.misc as misc
import warnings
import pathlib

import design_digital as filtdesign

warnings.filterwarnings('ignore', '.*Ill-conditioned matrix.*')
warnings.filterwarnings('ignore', '.*Badly conditioned filter coefficients.*')
np.set_printoptions(linewidth=300, precision=4, floatmode='fixed', formatter={'float':lambda x: f'{x}'})


def print_biquads(sos, Qformat):
    num_biquads = sos.shape[0]
    sos = sos.copy()
    sos = np.round(sos * 2 ** Qformat[1]).astype(int)
    print('')
    for i in range(num_biquads):
        print(f'{sos[i,0]}, {sos[i,1]}, {sos[i,2]}, {sos[i,4]}, {sos[i,5]},')
    print('')

def biquads_zpk2(z, p, k):
    sos = signal.zpk2sos(z, p, 1)
    N = np.size(sos, 0)
    sos[:, :3] *= np.abs(k)**(1/N)
    sost = sos
    wN = 1024
    ho = np.zeros((N, wN))
    for m in np.arange(N):
        wo, ho[m] = signal.freqz(sos[m, 0:3], sos[m, 3:7], wN)

    print(ho[:,0])

    ht = np.prod(ho[:, 0])
    for m in np.arange(N):
        sost[m, 0:3] = sos[m, 0:3]*(ht/(ho[m, 0]*np.abs(ht)**(1/N)))

    sos_quant = np.round(sost * 2 ** 14) * 2 ** -14
    return sost, sos_quant

def biquads_zpk(z, p, k):

    sos = signal.zpk2sos(z, p, 1)
    num_biquads = sos.shape[0]

    h0 = np.zeros([num_biquads,1])
    for m in np.arange(num_biquads):
        wo, h0[m] = signal.sosfreqz(sos[m,:], worN=[0])
    h0 = h0.ravel()
    gain = np.abs(k*np.prod(h0))

    sos[:,:3] *= gain**(1/num_biquads) / h0.reshape(-1,1)
    sos_quant = np.round(sos * 2 ** 14) * 2 ** -14

    return sos, sos_quant



sample_rate = 48e3
fp = 1.8e3
fs = 3.5e3
Amax = 1
Amin = 42
spec = dict(fp=fp, fs=fs, Amax=Amax, Amin=Amin, sample_rate=sample_rate, dt=1/sample_rate)
Qformat = (2,14)
sinewave_amplitude = 1 - 2 ** -Qformat[1] # max amplitude
rnd_seed = 100
limits_samples = 1000

# filter_type = 'but'
# method = 'matched'

filter_list = ['but', 'cau']
method_list = ['bilinear', 'matched', 'zoh']

filter_dict = dict(but='Butterworth', cau='Cauer')
method_dict = dict(bilinear='Bilinear', matched='Matched Z-Transform', zoh='Zero-Order Hold')

num_freqs = 200
images_path = pathlib.Path('images')
images_path.mkdir(parents=True, exist_ok=True)

for i, filter_type in enumerate(filter_list):
    for j, method in enumerate(method_list):

        # Get filter
        analog_system, discrete_system = filtdesign.get_filter(spec, filter_type=filter_type, method=method)

        # Quantize filter
        # sos, sos_quant = filtdesign.zpk2sos_quant(discrete_system, Qformat, filter_type=filter_type)

        sos, sos_quant = biquads_zpk(*discrete_system)

        print(f'Analog filter: {filter_type} | Analog-to-Discrete Method: {method}')
        print(f'Specification: fp = {fp} Hz | fs = {fs} Hz | Amax = {Amax} dB | Amin = {Amin} dB')
        # print(f'Biquads:\n', sos)
        # print(f'Quantized biquads (Q{Qformat[0]}.{Qformat[1]}):\n', np.round(sos_quant * 2 ** Qformat[1]).astype(int))

        # print_biquads(sos, Qformat)
        print_biquads(sos_quant, Qformat)

        
        # Plot
        fig, ax = plt.subplots()
        filtdesign.plot_zpk(discrete_system, fp, fs, Amax, Amin, num_freqs=num_freqs, ax=ax, plot_focus='all')
        # filtdesign.plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=num_freqs, ax=ax, plot_focus='all')
        # ax.legend(['Discrete', 'Quantized'])
        ax.set_title(f'Frequency Reponse ({filter_dict[filter_type]}, {method_dict[method]})')
        # plt.savefig(images_path / f'all_{filter_type}-{method}.eps', format='eps')
        # plt.savefig(images_path / f'all_{filter_type}-{method}.png', format='png')

        fig, ax = plt.subplots()
        filtdesign.plot_zpk(discrete_system, fp, fs, Amax, Amin, num_freqs=num_freqs, ax=ax, plot_focus='pass')
        # filtdesign.plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=num_freqs, ax=ax, plot_focus='pass')
        # ax.legend(['Discrete', 'Quantized'])
        ax.set_title(f'Pass Band ({filter_dict[filter_type]}, {method_dict[method]})')
        # plt.savefig(images_path / f'pass_{filter_type}-{method}.eps', format='eps')
        # plt.savefig(images_path / f'pass_{filter_type}-{method}.png', format='png')

        fig, ax = plt.subplots()
        filtdesign.plot_zpk(discrete_system, fp, fs, Amax, Amin, num_freqs=num_freqs, ax=ax, plot_focus='stop')
        # filtdesign.plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=num_freqs, ax=ax, plot_focus='stop')
        # ax.legend(['Discrete', 'Quantized'])
        ax.set_title(f'Stop Band ({filter_dict[filter_type]}, {method_dict[method]})')
        # plt.savefig(images_path / f'stop_{filter_type}-{method}.eps', format='eps')
        # plt.savefig(images_path / f'stop_{filter_type}-{method}.png', format='png')
        plt.show()

