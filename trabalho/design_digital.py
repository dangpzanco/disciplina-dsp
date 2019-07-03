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
        axis_focus = [100, 2500, -5, 1]
    elif plot_focus == 'stop':
        axis_focus = [2000, 5000, -80, -35]

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

    if ax is None:
        fig, ax = plt.subplots()
    
    if plot_focus == 'all':
        axis_focus = [1e2, 10**fmax, -50, 1]
        ax.set_xscale('log')
    elif plot_focus == 'pass':
        axis_focus = [100, 2500, -5, 1]
    elif plot_focus == 'stop':
        axis_focus = [2000, 5000, -80, -35]

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


def freqz_quant(sos, Qformat, magnitude=None, freq_vec=None, sample_rate=48e3, num_freqs=256, num_samples=1000):

    if freq_vec is None:
        freq_vec = np.arange(num_freqs)/num_freqs * sample_rate/2

    if magnitude is None:
        magnitude = 1 - 2 ** -Qformat[-1]

    sos_quant = (sos * 32768/2).astype('int16')

    f = freq_vec.copy()
    num_freqs = f.size

    time_vec = np.arange(num_samples).reshape(-1,1)
    freq_vec = freq_vec.ravel().reshape(1,-1)

    # Make complex sinusoids matrix (num_samples, num_freqs)
    x = magnitude * np.exp(1j * 2 * np.pi * freq_vec/sample_rate * time_vec)
    cosx = np.round(x.real * 32768).astype('int16')
    sinx = np.round(x.imag * 32768).astype('int16')

    # Filter for each frequency
    m, n = Qformat
    yr = np.empty(x.shape)
    yi = np.empty(x.shape)
    for i in range(num_freqs):
        yr[:,i] = filtsos_quant(sos_quant, cosx[:,i], m, n)
        yi[:,i] = filtsos_quant(sos_quant, sinx[:,i], m, n)

    x = (cosx + 1j * sinx)/32768
    y = (yr + 1j * yi)/32768

    steady_ind = 200
    h = (x[steady_ind:,] * y[steady_ind:,].conjugate()).mean(axis=0) / magnitude**2

    epsilon = 2 ** -Qformat[-1]
    h[np.abs(h) < epsilon] = epsilon

    return f, h

def get_filter(spec, filter_type='but', method='zoh'):

    wp = 2*np.pi*spec['fp']
    ws = 2*np.pi*spec['fs']

    if method == 'bilinear':
        wp = 2/spec['dt'] * np.arctan(wp * spec['dt']/2)
        ws = 2/spec['dt'] * np.arctan(ws * spec['dt']/2)
    
    if filter_type.lower() in ('butterworth'):
        N, Wn = signal.buttord(wp, ws, spec['Amax'], spec['Amin'], analog=True)
        z, p, k = signal.butter(N, Wn, output='zpk', btype='low', analog=True)
    elif filter_type.lower() in ('cauer' + 'elliptic'):
        N, Wn = signal.ellipord(wp, ws, spec['Amax'], spec['Amin'], analog=True)
        z, p, k = signal.ellip(N, spec['Amax'], spec['Amin'], Wn, output='zpk', btype='low', analog=True)

    def matched_method(z, p, k, dt):
        zd = np.exp(z*dt)
        pd = np.exp(p*dt)
        kd = k * np.abs(np.prod(1-pd)/np.prod(1-zd) * np.prod(z)/np.prod(p))
        return zd, pd, kd, dt

    if method == 'matched':
        zd, pd, kd, dt = matched_method(z, p, k, spec['dt'])
        kd *= 1 - (1 - 10 ** (-spec['Amax']/20))/5
    else:
        zd, pd, kd, dt = signal.cont2discrete((z,p,k), spec['dt'], method=method)

    analog_system = (z,p,k)
    discrete_system = (zd,pd,kd)

    return analog_system, discrete_system

def check_limits_quant(sos, spec, Qformat, magnitude=0.5, num_freqs=1000, num_samples=1000):

    fmin = np.floor(np.log10(spec['fp']))-1
    fmax = np.ceil(np.log10(spec['fs']))
    f = np.logspace(fmin, fmax, num_freqs)

    f, h = freqz_quant(sos, Qformat, magnitude=magnitude, freq_vec=f, sample_rate=spec['sample_rate'], num_samples=num_samples)
    Hdb = 20 * np.log10(np.abs(h))

    pass_band = Hdb[f <= spec['fp']]
    stop_band = Hdb[f >= spec['fs']]
    pass_band_faults = (pass_band < -spec['Amax']).sum() + (pass_band > 0).sum()
    stop_band_faults = (stop_band > -spec['Amin']).sum()

    total_faults = pass_band_faults + stop_band_faults

    return total_faults

def optimize_filter_quant(spec, Qformat=(2,14), magnitude=0.5, filter_type='but', min_order=None, method='zoh', num_iters=50, limits_samples=1000, num_samples=1000):
    original_spec = spec.copy()

    Amax_vec = rnd.uniform(0.1, spec['Amax'], num_iters)
    Amin_vec = rnd.uniform(spec['Amin'], 1.5*spec['Amin'], num_iters)
    fs_vec = rnd.uniform(spec['fp'], spec['fs'], num_iters)
    fp_vec = rnd.uniform(spec['fp'], spec['fs'], num_iters)
    spec_vec = np.vstack([Amax_vec, Amin_vec, fp_vec, fs_vec])

    if min_order is None:
        if filter_type.lower() in ('butterworth'):
            min_order = 10
        elif filter_type.lower() in ('cauer' + 'elliptic'):
            min_order = 4

    faults = np.inf
    for i in range(num_iters):
        spec['Amax'] = spec_vec[0,i]
        spec['Amin'] = spec_vec[1,i]

        asys, dsys = get_filter(spec, filter_type=filter_type, method=method)

        filter_order = dsys[1].size
        if filter_order != min_order:
            continue

        _, sos = zpk2sos_quant(dsys, Qformat, filter_type=filter_type)
        total_faults = check_limits_quant(sos, original_spec, Qformat, 
            magnitude=magnitude, num_freqs=limits_samples, num_samples=num_samples)
        total_faults += filter_order - min_order

        print(f"#{i:3} | Amax: {spec['Amax']:.3f} dB |", f"Amin: {spec['Amin']:.3f} dB |",
            f"fp: {spec['fp']/1e3:.3f} kHz |", f"fs: {spec['fs']/1e3:.3f} kHz |",
            f"Mag. faults: {total_faults:3} of {limits_samples} |",
            f"Filter order: {filter_order}")


        if faults > total_faults:
            faults = total_faults
            analog_system, discrete_system = (asys, dsys)
        
        if faults <= 0:
            analog_system, discrete_system = (asys, dsys)
            return analog_system, discrete_system, spec

    return analog_system, discrete_system, spec


def zpk2sos_quant(discrete_system, Qformat, filter_type):
    z, p, k = discrete_system
    sos = signal.zpk2sos(z, p, 1)
    num_biquads = sos.shape[0]

    h0 = np.zeros([num_biquads,1], dtype=np.complex)
    for m in np.arange(num_biquads):
        wo, h0[m] = signal.sosfreqz(sos[m,:], worN=[0])
    h0 = np.abs(h0).ravel()
    gain = np.abs(k*np.prod(h0))

    sos[:,:3] *= gain**(1/num_biquads) / h0.reshape(-1,1)
    sos_quant = np.round(sos * 2 ** Qformat[-1]) * 2 ** -Qformat[-1]

    return sos, sos_quant


@njit
def filtsos_quant(sos, x, m, n):

    num_samples = x.shape[0]
    num_biquads = sos.shape[0]
    buff = np.zeros((num_biquads,5), dtype=np.int16)
    y = np.zeros(num_samples)

    for i in range(num_samples):


        y_temp = cast_int16(x[i])
        for k in range(sos.shape[0]):
            b = sos[k,:3]
            a = sos[k,4:]
            # buffx[k,0] = y[i]
            buff[k,0] = y_temp

            valx = np.int32(0)
            for j in range(b.size):
                valx = valx + np.int32(b[j]) * np.int32(buff[k,j])

            valy = np.int32(0)
            for j in range(a.size):
                valy = valy + np.int32(a[j]) * np.int32(buff[k,j+b.size])

            y_temp = cast_int16(2.0*(valx - valy)/2**15)

            buff[k,2] = buff[k,1]
            buff[k,1] = buff[k,0]

            buff[k,b.size+1] = buff[k,b.size]
            buff[k,b.size] = y_temp

        y[i] = y_temp

    return y

@njit
def cast_int16(x):
    # Clipping limits
    max_value = np.int16(32767)
    min_value = np.int16(-32768)

    # Clip output
    if x < min_value:
        return min_value
    elif x > max_value:
        return max_value
    else:
        return np.int16(x)


if __name__ == '__main__':

    sample_rate = 48e3
    fp = 1.8e3
    fs = 3.5e3
    Amax = 1
    Amin = 42
    spec = dict(fp=fp, fs=fs, Amax=Amax, Amin=Amin, sample_rate=sample_rate, dt=1/sample_rate)
    # Qformat = (3,13)
    Qformat = (2,14)
    sinewave_amplitude = 1 - 2 ** -Qformat[1] # max amplitude
    rnd_seed = 100
    limits_samples = 1000

    filter_type = 'but'
    # filter_type = 'cau'
    method = 'bilinear'
    # method = 'matched'
    # method = 'zoh'

    # Consistent results
    rnd.seed(rnd_seed)

    # Get filter coefficients
    analog_system, discrete_system, final_spec = optimize_filter_quant(spec, num_iters=200, num_samples=1000,
        filter_type=filter_type, method=method, limits_samples=limits_samples, Qformat=Qformat, magnitude=sinewave_amplitude)

    # Quantize filter
    sos, sos_quant = zpk2sos_quant(discrete_system, Qformat, filter_type=filter_type)

    # discrete_system = signal.sos2zpk(sos_quant)

    # Get filter (meta)data
    filter_data = dict(fp=fp, fs=fs, Amax=Amax, Amin=Amin, sample_rate=sample_rate, spec=spec, 
        sos=sos, sos_quant=sos_quant, final_spec=final_spec, filter_type=filter_type, method=method, 
        limits_samples=limits_samples, Qformat=Qformat, magnitude=sinewave_amplitude, seed=rnd_seed)

    # Save filter (meta)data
    out_filename = f'type-{filter_type}_method-{method}.npz'
    np.savez(out_filename, **filter_data)

    print(f'Biquads:\n', sos)
    print(f'Quantized biquads (Q{Qformat[0]}.{Qformat[1]}):\n', np.round(sos_quant * 2 ** Qformat[1]).astype(int))

    num_freqs = 200

    fig, ax = plt.subplots()
    plot_zpk(discrete_system, fp, fs, Amax, Amin, num_freqs=num_freqs, ax=ax, plot_focus='all')
    plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=num_freqs, ax=ax, plot_focus='all')

    fig, ax = plt.subplots()
    plot_zpk(discrete_system, fp, fs, Amax, Amin, num_freqs=num_freqs, ax=ax, plot_focus='pass')
    plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=num_freqs, ax=ax, plot_focus='pass')

    fig, ax = plt.subplots()
    plot_zpk(discrete_system, fp, fs, Amax, Amin, num_freqs=num_freqs, ax=ax, plot_focus='stop')
    plot_digital(sos_quant, Qformat, fp, fs, Amax, Amin, magnitude=sinewave_amplitude, num_freqs=num_freqs, ax=ax, plot_focus='stop')
    plt.show()


