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
        # axis_focus = [1000, 2500, -5, 0]
        axis_focus = [100, 2500, -5, 1]
    elif plot_focus == 'stop':
        # axis_focus = [3000, 5000, -50, -35]
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

    # f, h = signal.sosfreqz(sos, fs=sample_rate, worN=f)

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
    
    # Avoid clipping with mag < 1.0
    # mag = 0.75

    # Make complex sinusoids matrix (num_samples, num_freqs)
    x = magnitude * np.exp(1j * 2 * np.pi * freq_vec/sample_rate * time_vec)
    cosx = np.round(x.real * 32768).astype('int16')
    sinx = np.round(x.imag * 32768).astype('int16')

    # Filter for each frequency
    m, n = Qformat
    yr = np.empty([sos.shape[0], *x.shape])
    yi = np.empty([sos.shape[0], *x.shape])
    for i in range(num_freqs):
        yr[:,:,i] = filtsos_quant(sos_quant, cosx[:,i], m, n)
        yi[:,:,i] = filtsos_quant(sos_quant, sinx[:,i], m, n)

    x = (cosx + 1j * sinx)/32768
    y = (yr + 1j * yi)/32768

    # y = filtsos_quant(sos, x, Qformat)

    # y = signal.sosfilt(sos, x, axis=0)
    # y = signal.sosfilt(sos, cosx, axis=0) + 1j * signal.sosfilt(sos, sinx, axis=0)


    steady_ind = 200
    # steady_ind = 1000
    # h = (x[steady_ind:,] * y[steady_ind:,].conjugate()).mean(axis=0) / magnitude**2
    h = yr[:,steady_ind:,].max(axis=1)/32768.0
    # h = (cosx * y + 1j * sinx * y).mean(axis=-1)

    # print(h)

    freq_ind = np.argmin(np.abs(freq_vec - 100))
    print(freq_ind, freq_vec[0,freq_ind])
    plt.figure()
    plt.plot(sinx[:,freq_ind])
    plt.plot(yi[:,:,freq_ind].T)

    fig, ax = plt.subplots()
    ax.plot(freq_vec.ravel(), 20 * np.log10(np.abs(h.T)), linewidth=2)
    plt.show()

    exit(0)

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
        kd *= 1 - (1 - 10 ** (-spec['Amax']/20))/2
    else:
        zd, pd, kd, dt = signal.cont2discrete((z,p,k), spec['dt'], method=method)

    analog_system = (z,p,k)
    discrete_system = (zd,pd,kd)

    return analog_system, discrete_system

def check_limits_quant(sos, spec, Qformat, magnitude=0.5, num_freqs=1000, num_samples=1000):

    # f1 = np.logspace(np.floor(np.log10(spec['fp']))-1, np.log10(spec['fp']), 2*num_samples)
    # f2 = np.logspace(np.log10(spec['fs']), np.log10(spec['sample_rate']/2), num_samples)
    # f = np.hstack([f1, f2])
    # num_samples *= 2
    fmin = np.floor(np.log10(spec['fp']))-1
    fmax = np.ceil(np.log10(spec['fs']))
    f = np.logspace(fmin, fmax, num_freqs)

    # f, h = signal.freqz_zpk(*system, fs=spec['sample_rate'], worN=f)
    # sos_debug, sos = zpk2sos_quant(system, Qformat)
    f, h = freqz_quant(sos, Qformat, magnitude=magnitude,
        freq_vec=f, sample_rate=spec['sample_rate'], num_samples=num_samples)
    Hdb = 20 * np.log10(np.abs(h))

    pass_band = Hdb[f <= spec['fp']]
    stop_band = Hdb[f >= spec['fs']]
    pass_band_faults = (pass_band < -spec['Amax']).sum() + (pass_band > 0).sum()
    stop_band_faults = (stop_band > -spec['Amin']).sum()

    total_faults = pass_band_faults + stop_band_faults

    # print(pass_band_faults, stop_band_faults, system[1].size)

    # print('sos_debug\n', sos_debug)
    # print('sos\n', sos)

    # fig, ax = plt.subplots()
    # plot_zpk(system, spec['fp'], spec['fs'], spec['Amax'], spec['Amin'], 
    #              sample_rate=48e3, num_freqs=num_freqs, ax=ax, plot_focus='pass')
    # plot_digital(sos, Qformat, spec['fp'], spec['fs'], spec['Amax'], spec['Amin'], magnitude=magnitude,
    #              sample_rate=48e3, num_freqs=num_freqs, num_samples=num_samples, ax=ax, plot_focus='pass')
    # plt.show()

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

    # if magnitude > 0.5:
    #     min_order += 1

    faults = np.inf
    for i in range(num_iters):
        spec['Amax'] = spec_vec[0,i]
        spec['Amin'] = spec_vec[1,i]
        # spec['fp'] = spec_vec[2,i]
        # spec['fs'] = spec_vec[3,i]
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
        buffx[0] = quantizer_1d(x[i], m, n)
        
        valx = quantizer_1d((b * buffx).sum(), m, n)
        valy = quantizer_1d((a * buffy).sum(), m, n)
        y[i] = quantizer_1d(valx - valy, m, n)
        
        buffy[1] = buffy[0]
        buffy[0] = y[i]

    return y


def zpk2sos_quant(discrete_system, Qformat, filter_type):
    z, p, k = discrete_system
    sos = signal.zpk2sos(z, p, 1, pairing='nearest')
    # sos = signal.zpk2sos(z, p, k, pairing='keep_odd')

    # Trick for higher accuracy on small numerator coefficients

    # h0 = np.abs(z.sum()/p.sum())

    num_biquads = sos.shape[0]

    sos_quant = sos.copy()

    k_factor = 1.0 + np.arange(sos.shape[0])[::-1]
    k_factor /= k_factor.sum()
    # sos_quant[:,:3] *= (k) ** (k_factor.reshape(-1,1))
    sos_quant[:,:3] *= (k) ** (1/num_biquads)

    # h0 = np.empty([num_biquads,1], dtype=np.complex)
    # for i in range(num_biquads):
    #     w, h0[i] = signal.freqz(sos[i,:3], sos[i,3:], worN=[0])
    # h0 = np.abs(h0).ravel()
    # ht = np.prod(h0)

    # for i in range(num_biquads):
    #     sos_quant[:,:3] *= ht/(h0[i] * ht ** (1/num_biquads))

    # print(h0)
    # exit(0)

    # if filter_type == 'but':

    #     # Geometric mean of the first biquad's B(z=1) [b0 b1 b2]
    #     non_zeros = np.abs(sos_quant[0,:3]) > 0
    #     b_factor = np.abs(np.prod(sos_quant[0,:3][non_zeros])) ** (1/non_zeros.sum())
    #     # b_factor = k
    #     # b_factor = np.abs(sos_quant[0,:3]).max()
    #     sos_quant[0,:3] /= b_factor

    #     # Increasing gain (k) thru the biquad stages
    #     k_factor = 1.0 + np.arange(sos.shape[0])[::-1]
    #     # k_factor = np.ones(sos.shape[0])
    #     k_factor /= k_factor.sum()
    #     sos_quant[:,:3] *= b_factor ** (k_factor.reshape(-1,1))

    #     print(k)
    #     print(sos_quant)
    #     print(k_factor)


    # if filter_type == 'cau':
    #     # sos_quant[0,:3] /= k
    #     # # sos_quant[:,:3] *= k ** (1/sos_quant.shape[0])
    #     # non_zeros2 = np.abs(sos_quant[:,:3]) > 0
    #     # a_factor = np.abs(np.prod(sos_quant[:,:3][non_zeros2], axis=-1)) ** (1/non_zeros2.sum(axis=-1))
    #     # sos_quant[:,:3] *= k ** (a_factor.reshape(-1,1)/a_factor.sum())
    #     # print(a_factor/a_factor.sum())
        
    #     # Geometric mean of the first biquad's B(z=1) [b0 b1 b2]
    #     non_zeros = np.abs(sos_quant[0,:3]) > 0
    #     b_factor = np.abs(np.prod(sos_quant[0,:3][non_zeros])) ** (1/non_zeros.sum())
    #     # b_factor = np.abs(sos_quant[0,:3]).max()

    #     sos_quant[0,:3] /= b_factor

    #     k_factor = 1.0 + np.arange(sos.shape[0])[::-1]
    #     k_factor /= k_factor.sum()
    #     sos_quant[:,:3] *= b_factor ** (k_factor.reshape(-1,1))

        # sos_quant[0,:3] *= 1e-2
        # sos_quant[1,:3] *= k/1e-2

    # print('k:', k)
    # print('sos_quant:\n', sos_quant)

    sos_quant = quantizer_real(sos_quant, Qformat)


    return sos, sos_quant

@njit
def filtsos_quant2(sos, x, m, n):

    y = x.copy()
    for i in range(sos.shape[0]):
        b = sos[i,:3]
        a = sos[i,4:]
        y = biquad_quant(b, a, y, m, n)

    return y

@njit
def filtsos_quant(sos, x, m, n):

    num_samples = x.shape[0]
    num_biquads = sos.shape[0]
    buffx = np.zeros((num_biquads,3), dtype=np.int16)
    buffy = np.zeros((num_biquads,2), dtype=np.int16)
    # y = np.zeros(x.shape, dtype=np.int16)
    y = np.zeros((num_biquads,num_samples), dtype=np.int16)

    for i in range(num_samples):


        y[0,i] = x[i]
        for k in range(sos.shape[0]):
            b = sos[k,:3]
            a = sos[k,4:]
            buffx[k,0] = y[k,i]

            
            # valx = (b * buffx[k,:]).sum()
            # valy = (a * buffy[k,:]).sum()

            valx = np.int32(0)
            for j in range(b.size):
                valx = valx + np.int32(b[j]) * np.int32(buffx[k,j])

            valy = np.int32(0)
            for j in range(a.size):
                valy = valy + np.int32(a[j]) * np.int32(buffy[k,j])

            y[k,i] = cast_int16((valx - valy) >> 15)            

            buffx[k,2] = buffx[k,1]
            buffx[k,1] = buffx[k,0]
            # buffx[k,0] = x[i]

            buffy[k,1] = buffy[k,0]
            buffy[k,0] = y[k,i]

        # if i == 10:
        #     print(b,a)
        #     print('Buffers:\n', buffx, buffy)
        #     print('num, den:', valx, valy)
        #     exit(0)

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

@njit
def cast_int32(x):
    # Clipping limits
    max_value = np.int16(2**31 - 1)
    min_value = np.int16(-2**31)

    # Clip output
    if x < min_value:
        return min_value
    elif x > max_value:
        return max_value
    else:
        return np.int32(x)

@jit(nopython=True)
def quantizer_int16(x):
    # Quantize signal
    n = 15
    m = 1
    M = 2 ** n

    y = round(x * M) / M

    # Clipping limits
    x_max = 2 ** (m - 1)
    max_value = (1 - 1/M) * x_max
    min_value = -x_max

    # Clip output
    if y < min_value:
        y = min_value
    elif y > max_value:
        y = max_value

    return y


@jit(nopython=True)
def quantizer_int32(x):
    # Quantize signal
    n = 31
    m = 1
    M = 2 ** n

    y = round(x * M) / M

    # CLipping limits
    x_max = 2 ** (m - 1)
    max_value = (1 - 1/M) * x_max
    min_value = -x_max

    # Clip output
    if y < min_value:
        y = min_value
    elif y > max_value:
        y = max_value

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

    y = round(x * M) / M

    # CLipping limits
    x_max = 2 ** (m - 1)
    max_value = (1 - 1/M) * x_max
    min_value = -x_max

    # Clip output
    if y < min_value:
        y = min_value
    elif y > max_value:
        y = max_value

    return y

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
    # sinewave_amplitude = 0.5
    # rnd_seed = 0
    rnd_seed = 100
    limits_samples = 1000

    filter_type = 'but'
    method = 'bilinear'

    # Consistent results
    rnd.seed(rnd_seed)

    # Get filter coefficients
    analog_system, discrete_system, final_spec = optimize_filter_quant(spec, num_iters=20, num_samples=1000,
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

    # Exemplo Matsuo smb://home/public/Walter/pds_filtr
    # a = ['0x0166','0xFEC1','0x0166','0x8D53','0x33FB','0x0B60','0xEC4C','0x0B60','0x880D','0x3B94']
    # vint = np.vectorize(lambda x: int(x,16))
    # b = vint(a).reshape(2,5)
    # c = -(b & 0x8000) | (b & 0x7fff)
    # sos_quant = np.empty([2,6])
    # sos_quant[:,:3] = c[:,:3] * 1.0
    # sos_quant[:,3] = 16384
    # sos_quant[:,4:] = c[:,3:] * 1.0
    # sos_quant /= 2 ** Qformat[1]

    # sos_quant[0,:3] /= discrete_system[-1]
    # # sos_quant[:,:3] *= discrete_system[-1] ** (1/sos_quant.shape[0])
    # sos_quant[1:,:3] *= discrete_system[-1] ** (1/(sos_quant.shape[0]-1))
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


