from numba import jit, njit
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as lin
import scipy.misc as misc
import warnings

warnings.filterwarnings('ignore', '.*Ill-conditioned matrix.*')
warnings.filterwarnings('ignore', '.*Badly conditioned filter coefficients.*')
np.set_printoptions(linewidth=300, precision=4, floatmode='fixed', formatter={'float':lambda x: f'{x}'})


def plot_zpk(system, fp, fs, Amax, Amin, sample_rate=48e3, num_freqs=1024, ax=None, plot_focus='all'):

    fmin = np.floor(np.log10(fp))-1
    fmax = np.ceil(np.log10(fs))

    f = np.logspace(fmin, fmax, num_freqs)
    f, h = signal.freqz_zpk(*system, fs=sample_rate, worN=f)

    if ax is None:
        fig, ax = plt.subplots()
    
    if plot_focus == 'all':
        axis_focus = [1e2, 10**fmax, -50, 1]
        ax.set_xscale('log')
    elif plot_focus == 'pass':
        axis_focus = [1000, 2500, -5, 0]
    elif plot_focus == 'stop':
        axis_focus = [3000, 5000, -50, -35]

    # Plot data
    ax.plot(f, 20 * np.log10(np.abs(h)), linewidth=2)

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


def plot_digital(sos, Qformat, fp, fs, Amax, Amin, magnitude=0.5, sample_rate=48e3, num_freqs=256, num_samples=1000, ax=None, plot_focus='all'):

    fmin = np.floor(np.log10(fp))-1
    fmax = np.ceil(np.log10(fs))

    f = np.logspace(fmin, fmax, num_freqs)
    f, h = freqz_quant(sos, Qformat, magnitude=magnitude, 
        freq_vec=f, sample_rate=sample_rate, num_samples=num_samples)

    # f, h = signal.sosfreqz(sos, fs=sample_rate, worN=f)

    if ax is None:
        fig, ax = plt.subplots()
    
    if plot_focus == 'all':
        axis_focus = [1e2, 10**fmax, -50, 1]
        ax.set_xscale('log')
    elif plot_focus == 'pass':
        axis_focus = [1000, 2500, -5, 0]
    elif plot_focus == 'stop':
        axis_focus = [3000, 5000, -50, -35]

    # Plot data
    ax.plot(f, 20 * np.log10(np.abs(h)), linewidth=2)

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

    return f, 20 * np.log10(np.abs(h))


def freqz_quant(sos, Qformat, magnitude=0.5, freq_vec=None, sample_rate=48e3, num_freqs=256, num_samples=1000):

    if freq_vec is None:
        freq_vec = np.arange(num_freqs)/num_freqs * sample_rate/2

    f = freq_vec.copy()
    num_freqs = f.size

    time_vec = np.arange(num_samples).reshape(-1,1)
    freq_vec = freq_vec.ravel().reshape(1,-1)/sample_rate
    
    # Avoid clipping with mag < 1.0
    # mag = 0.75

    # Make complex sinusoids matrix (num_samples, num_freqs)
    x = magnitude * np.exp(1j * 2 * np.pi * freq_vec * time_vec)
    cosx = x.real
    sinx = x.imag

    # Filter for each frequency
    m, n = Qformat
    y = np.empty(x.shape, dtype=np.complex)
    for i in range(num_freqs):
        y[:,i] = filtsos_quant(sos, x[:,i], m, n)

    # y = filtsos_quant(sos, x, Qformat)

    # y = signal.sosfilt(sos, x, axis=0)
    # y = signal.sosfilt(sos, cosx, axis=0) + 1j * signal.sosfilt(sos, sinx, axis=0)

    h = (x * y.conjugate()).mean(axis=0) / magnitude**2
    # h = (cosx * y + 1j * sinx * y).mean(axis=-1)

    return f, h

def get_filter(spec, filter_type='but', method='zoh'):
    
    if filter_type.lower() in ('butterworth'):
        N, Wn = signal.buttord(2*np.pi*spec['fp'], 2*np.pi*spec['fs'], spec['Amax'], spec['Amin'], analog=True)
        z, p, k = signal.butter(N, Wn, output='zpk', btype='low', analog=True)
    elif filter_type.lower() in ('cauer' + 'elliptic'):
        N, Wn = signal.ellipord(2*np.pi*fp, 2*np.pi*spec['fs'], spec['Amax'], spec['Amin'], analog=True)
        z, p, k = signal.ellip(N, spec['Amax'], spec['Amin'], Wn, output='zpk', btype='low', analog=True)

    def matched_method(z, p, k, dt):
        zd = np.exp(z*dt)
        pd = np.exp(p*dt)
        kd = k * np.abs(np.prod(1-pd)/np.prod(1-zd) * np.prod(z)/np.prod(p))
        return zd, pd, kd, dt

    if method == 'matched':
        zd, pd, kd, dt = matched_method(z, p, k, spec['dt'])
        kd *= 1 - (1 - 10 ** (-spec['Amax']/20))/2
    else:
        zd, pd, kd, dt = signal.cont2discrete((z,p,k), spec['dt'], method=method)

    analog_system = (z,p,k)
    discrete_system = (zd,pd,kd)

    return analog_system, discrete_system

def check_limits_quant(system, spec, Qformat, magnitude=0.5, num_freqs=1000, num_samples=1000):

    f1 = np.logspace(np.floor(np.log10(spec['fp']))-1, np.log10(spec['fp']), 2*num_samples)
    f2 = np.logspace(np.log10(spec['fs']), np.log10(spec['sample_rate']/2), num_samples)
    f = np.hstack([f1, f2])
    # num_samples *= 2
    # fmin = np.floor(np.log10(spec['fp']))-1
    # fmax = np.ceil(np.log10(spec['fs']))
    # f = np.logspace(fmin, fmax, num_freqs)

    # f, h = signal.freqz_zpk(*system, fs=spec['sample_rate'], worN=f)
    _, sos = zpk2sos_quant(system, Qformat)
    f, h = freqz_quant(sos, Qformat, magnitude=magnitude,
        freq_vec=f, sample_rate=spec['sample_rate'], num_samples=num_samples)
    Hdb = 20 * np.log10(np.abs(h))

    pass_band = Hdb[f <= spec['fp']]
    stop_band = Hdb[f >= spec['fs']]
    pass_band_faults = (pass_band < -spec['Amax']).sum() + (pass_band > 0).sum()
    stop_band_faults = (stop_band > -spec['Amin']).sum()

    total_faults = pass_band_faults + stop_band_faults + system[1].size

    # print(pass_band_faults, stop_band_faults, system[1].size)

    # plot_digital(sos, Qformat, spec['fp'], spec['fs'], spec['Amax'], spec['Amin'], 
    #              sample_rate=48e3, num_freqs=num_freqs, num_samples=num_samples, ax=None, plot_focus='all')
    # plt.show()

    return total_faults

def optimize_filter_quant(spec, Qformat=(2,14), magnitude=0.5, filter_type='but', min_order=None, method='zoh', num_iters=50, limits_samples=1000):
    original_spec = spec.copy()

    Amax_vec = rnd.uniform(0.1, spec['Amax'], num_iters)
    Amin_vec = rnd.uniform(spec['Amin'], 1.5*spec['Amin'], num_iters)
    spec_vec = np.vstack([Amax_vec, Amin_vec])

    if min_order is None:
        if filter_type.lower() in ('butterworth'):
            min_order = 9
        elif filter_type.lower() in ('cauer' + 'elliptic'):
            min_order = 4

    faults = np.inf
    for i in range(num_iters):
        spec['Amax'] = spec_vec[0,i]
        spec['Amin'] = spec_vec[1,i]
        asys, dsys = get_filter(spec, filter_type=filter_type, method=method)
        total_faults = check_limits_quant(dsys, original_spec, Qformat, 
            magnitude=magnitude, num_freqs=limits_samples, num_samples=1000)
        total_faults -= min_order

        filter_order = dsys[1].size
        print(f"Amax: {spec['Amax']:.3f} dB |",
            f"Amin: {spec['Amin']:.3f} dB |",
            f"Mag. faults: {total_faults:3} of {limits_samples} |",
            f"Filter order: {filter_order}")

        if faults > total_faults:
            faults = total_faults
            analog_system, discrete_system = (asys, dsys)
        
        if faults <= 0:
            analog_system, discrete_system = (asys, dsys)
            return analog_system, discrete_system, spec

    return analog_system, discrete_system, spec




@njit
def biquad_quant(b, a, x, m, n):
    """ Quantized biquad filter (single stage, Direct Form I)
    Input:
        b [array] - Numerator: [b0, b1, b2]
        a [array] - Denominator: [1, a1, a2]
        x [array] - Input signal
        Qformat [tuple] - (m,n):
            m [int] - Integer bits
            n [int] - Fractional bits
    Output:
        y [array] = Filtered signal: Y(z) = B(z)/A(z)*X(z)

    """

    num_samples = x.shape[0]
    buffx = np.zeros(3, dtype=x.dtype)
    buffy = np.zeros(2, dtype=x.dtype)
    y = np.zeros(x.shape, dtype=x.dtype)

    # Direct form I
    for i in range(num_samples):
        buffx[2] = buffx[1]
        buffx[1] = buffx[0]
        buffx[0] = x[i]
        
        valx = quantizer_1d((b * buffx).sum(), m, n)
        valy = quantizer_1d((a * buffy).sum(), m, n)
        y[i] = quantizer_1d(valx - valy, m, n)
        
        buffy[1] = buffy[0]
        buffy[0] = y[i]

    return y


def zpk2sos_quant(discrete_system, Qformat):
    sos = signal.zpk2sos(*discrete_system, pairing='nearest')
    
    # Trick for higher accuracy on small numerator coefficients
    non_zeros = sos[0,:3] > 0
    b_factor = np.prod(sos[0,:3][non_zeros]) ** (1/non_zeros.sum())
    sos[0,:3] /= b_factor
    sos[:,:3] *= b_factor ** (1/sos.shape[0])

    sos_quant = quantizer_real(sos, Qformat)

    return sos, sos_quant

@njit
def filtsos_quant(sos, x, m, n):

    y = x.copy()
    for i in range(sos.shape[0]):
        b = sos[i,:3]
        a = sos[i,4:]
        y = biquad_quant(b, a, y, m, n)

    return y

def quantizer_real(x, Qformat):
    """ 
    Input:
    x [array] - Input signal
    Qformat [tuple] - (m,n):
        m [int] - Integer bits
        n [int] - Fractional bits
    
    e.g. Q16.16 -> 32bit fixed-point: m = 16, n = 16
    reference: https://en.wikipedia.org/wiki/Q_(number_format)
    
    Output:
    y - sinal quantizado
    
    butt, cheb1 -> m = 3
    cheb2, ellip -> m = 2
    """

    m, n = Qformat

    # Quantize signal
    M = 2 ** n
    y = np.round(x * M) / M

    # Clip the signal
    x_max = 2 ** (m - 1)
    max_value = (1 - 1/M) * x_max
    min_value = -x_max

    y = np.clip(y, min_value, max_value)

    return y

@jit(nopython=True)
def quantizer_1d(x, m, n):
    # Quantize signal
    M = 2 ** n
    y = (round(x.real * M) + 1j * round(x.imag * M)) / M
    # y = np.round(x * M) / M

    # Clip the signal
    x_max = 2 ** (m - 1)
    max_value = (1 - 1/M) * x_max
    min_value = -x_max

    # Clip real part
    if y.real < min_value:
        yr = min_value
    elif y.real > max_value:
        yr = max_value
    else:
        yr = y.real

    # Clip imag part
    if y.imag < min_value:
        yi = min_value
    elif y.imag > max_value:
        yi = max_value
    else:
        yi = y.imag
    
    # Join parts
    y = yr + 1j * yi

    return y

sample_rate = 48e3
fp = 1.8e3
fs = 3.5e3
Amax = 1
Amin = 42
spec = dict(fp=fp, fs=fs, Amax=Amax, Amin=Amin, sample_rate=sample_rate, dt=1/sample_rate)
Qformat = (2,14)
sinewave_amplitude = 0.5
rnd_seed = 0
limits_samples = 1000

filter_type = 'cau'
method = 'matched'

# Consistent results
rnd.seed(rnd_seed)

# Get filter coefficients
analog_system, discrete_system, final_spec = optimize_filter_quant(spec, 
    filter_type=filter_type, method=method, limits_samples=limits_samples, Qformat=Qformat, magnitude=sinewave_amplitude)

# Quantize filter
sos, sos_quant = zpk2sos_quant(discrete_system, Qformat)

# Get filter (meta)data
filter_data = dict(fp=fp, fs=fs, Amax=Amax, Amin=Amin, sample_rate=sample_rate, spec=spec, 
    sos=sos, sos_quant=sos_quant, final_spec=final_spec, filter_type=filter_type, method=method, 
    limits_samples=limits_samples, Qformat=Qformat, magnitude=sinewave_amplitude, seed=rnd_seed)

# Save filter (meta)data
out_filename = f'type-{filter_type}_method-{method}.npz'
np.savez(out_filename, **filter_data)


print(f'Biquads:\n', sos)
print(f'Quantized biquads (Q{Qformat[0]}.{Qformat[1]}):\n', np.round(sos * 2 ** Qformat[1]).astype(int))

plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=256, plot_focus='all')
plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=256, plot_focus='pass')
plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=256, plot_focus='stop')
plt.show()


