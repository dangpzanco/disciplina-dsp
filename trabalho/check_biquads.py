from numba import jit, njit
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.pyplot import axvline, axhline


import scipy.signal as signal
import scipy.linalg as lin
import scipy.misc as misc
import warnings
import pathlib

import design_digital as filtdesign

warnings.filterwarnings('ignore', '.*Ill-conditioned matrix.*')
warnings.filterwarnings('ignore', '.*Badly conditioned filter coefficients.*')
np.set_printoptions(linewidth=300, precision=4, floatmode='fixed', formatter={'float':lambda x: f'{x}'})


def print_biquads(sos):
    num_biquads = sos.shape[0]
    sos = sos.copy()
    sos = np.round(sos * 2 ** Qformat[1]).astype(int)
    print('')
    for i in range(num_biquads):
        print(f'{sos[i,0]}, {sos[i,1]}, {sos[i,2]}, {sos[i,4]}, {sos[i,5]},')
    print('')


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

filter_type = 'but'
method = 'matched'

filter_list = ['but', 'cau']
method_list = ['bilinear', 'matched', 'zoh']

filter_dict = dict(but='Butterworth', cau='Cauer')
method_dict = dict(bilinear='Bilinear', matched='Matched Z-Transform', zoh='Zero-Order Hold')

num_freqs = 200
images_path = pathlib.Path('images')
images_path.mkdir(parents=True, exist_ok=True)

for i, filter_type in enumerate(filter_list):
    for j, method in enumerate(method_list):

        # Get filter (meta)data
        filter_filename = f'type-{filter_type}_method-{method}.npz'
        filter_data = np.load(filter_filename, allow_pickle=True)

        sos = filter_data['sos']
        sos_quant = filter_data['sos_quant']
        final_spec = filter_data['final_spec'].item()
        discrete_system = signal.sos2zpk(sos)

        print(f'Analog filter: {filter_type} | Analog-to-Discrete Method: {method}')
        # print(f'Biquads:\n', sos)
        # print(f'Quantized biquads (Q{Qformat[0]}.{Qformat[1]}):\n', np.round(sos_quant * 2 ** Qformat[1]).astype(int))

        print_biquads(sos_quant)

        zd, pd, kd = signal.sos2zpk(sos)
        zq, pq, kq = signal.sos2zpk(sos_quant)

        
        # Plot
        fig, ax = plt.subplots()
        unit_circle = patches.Circle((0,0), radius=1, fill=False, color='black', ls='solid', alpha=0.1)
        ax.add_patch(unit_circle)
        ax.plot(pd.real, pd.imag, 'x', markersize=9, alpha=0.5, markeredgecolor='b')
        ax.plot(zd.real, zd.imag, 'o', markersize=9, alpha=0.5, color='none', markeredgecolor='b')
        ax.plot(pq.real, pq.imag, 'x', markersize=9, alpha=0.5, markeredgecolor='r')
        ax.plot(zq.real, zq.imag, 'o', markersize=9, alpha=0.5, color='none', markeredgecolor='r')
        ax.legend(['Discrete Poles', 'Discrete Zeros', 'Quantized Poles', 'Quantized Zeros'])
        ax.grid(True)
        ax.set_title(f'Poles and Zeros ({filter_dict[filter_type]}, {method_dict[method]})')
        plt.show()

