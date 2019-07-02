from numba import jit, njit
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as lin
import scipy.misc as misc
import warnings
import pathlib

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


def plot_boxes(fp, fs, Amax, Amin, ax=None, plot_focus='all'):

    fmin = np.floor(np.log10(fp))-1
    fmax = np.ceil(np.log10(fs))

    if ax is None:
        fig, ax = plt.subplots()
    
    if plot_focus == 'all':
        axis_focus = [1e2, 10**fmax, -50, 1]
        ax.set_xscale('log')
    elif plot_focus == 'pass':
        # axis_focus = [1000, 2500, -5, 0]
        axis_focus = [100, 2500, -5, 1]
    elif plot_focus == 'stop':
        # axis_focus = [3000, 5000, -50, -35]
        axis_focus = [2000, 5000, -80, -35]

    # Plot boxes
    box_style = dict(linewidth=2, linestyle='--', edgecolor='k', facecolor='0.9')
    ax.fill([10**fmin, 10**fmin,  fp,  fp], [-Amin*2, -Amax, -Amax, -Amin*2], **box_style) # pass
    ax.fill([fs, fs,  10**fmax,  10**fmax], [-Amin, Amax, Amax, -Amin], **box_style) # stop

    # Set plot properties
    ax.set_title('Lowpass filter')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.grid(True, which='both', axis='both')
    ax.axis(axis_focus)

def plot_response(freq, amp, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(freq, amp, linewidth=2)



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


filter_list = ['but', 'cau']
method_list = ['bilinear', 'matched', 'zoh']

filter_dict = dict(but='Butterworth', cau='Cauer')
method_dict = dict(bilinear='Bilinear', matched='Matched Z-Transform', zoh='Zero-Order Hold')

num_freqs = 200
images_path = pathlib.Path('images')
images_path.mkdir(parents=True, exist_ok=True)

for i, filter_type in enumerate(filter_list):
    for j, method in enumerate(method_list):

        # Get filter experiment data
        filter_filename = f'experiments/exp-{filter_type}_{method}.csv'
        data = pd.read_csv(filter_filename).to_dict('series')

        data['amp'] /= 1.82

        print(data)
        # exit(0)

        print(f'Analog filter: {filter_type} | Analog-to-Discrete Method: {method}')

        
        # Plot
        fig, ax = plt.subplots()
        plot_response(**data, ax=ax)
        plot_boxes(fp, fs, Amax, Amin, ax=ax, plot_focus='all')
        ax.set_title(f'Frequency Reponse ({filter_dict[filter_type]}, {method_dict[method]})')
        # plt.savefig(images_path / f'all_{filter_type}-{method}.eps', format='eps')
        # plt.savefig(images_path / f'all_{filter_type}-{method}.png', format='png')

        fig, ax = plt.subplots()
        plot_response(**data, ax=ax)
        plot_boxes(fp, fs, Amax, Amin, ax=ax, plot_focus='pass')
        ax.set_title(f'Pass Band ({filter_dict[filter_type]}, {method_dict[method]})')
        # plt.savefig(images_path / f'pass_{filter_type}-{method}.eps', format='eps')
        # plt.savefig(images_path / f'pass_{filter_type}-{method}.png', format='png')

        fig, ax = plt.subplots()
        plot_response(**data, ax=ax)
        plot_boxes(fp, fs, Amax, Amin, ax=ax, plot_focus='stop')
        ax.set_title(f'Stop Band ({filter_dict[filter_type]}, {method_dict[method]})')
        # plt.savefig(images_path / f'stop_{filter_type}-{method}.eps', format='eps')
        # plt.savefig(images_path / f'stop_{filter_type}-{method}.png', format='png')
        plt.show()

