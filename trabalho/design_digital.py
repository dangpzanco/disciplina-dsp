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

rnd.seed(0)

def plot_zpk(system, fp, fs, Amax, Amin, sample_rate=48e3, num_samples=1024, ax=None, plot_focus='all'):

    fmin = np.floor(np.log10(fp))-1
    fmax = np.ceil(np.log10(fs))

    f = np.logspace(fmin, fmax, num_samples)
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

def matched_method(z, p, k, dt):

    zd = np.exp(z*dt)
    pd = np.exp(p*dt)
    kd = k * np.abs(np.prod(1-pd)/np.prod(1-zd) * np.prod(z)/np.prod(p))

    return zd, pd, kd, dt

def quantizer(x, Qformat, underflow_clip=False):
    """ 
    Input:
    x [array] - Input signal
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

    if underflow_clip:
        x_min = 2 ** -(n - 1)
        x[np.abs(x) < x_min] = x_min

    # Quantize signal
    M = 2 ** n
    y = np.round(x * M) / M

    # Clip the signal
    x_max = 2 ** (m - 1)
    max_value = (1 - 1/M) * x_max
    min_value = -x_max
    y = np.clip(y, min_value, max_value)

    return y


def biquad_quant(b, a, x, Qformat=(1,16)):
    # SINGLE_BIQUAD_QUANT Filtro biquad forma direta I quantizado
    #    Implementação de um estágio quantizado de filtragem IIR com biquadradas

    num_samples = x.shape[0]
    buffx = np.zeros(3)
    buffy = np.zeros(2)
    y = np.zeros(x.shape)

    # Forma direta I
    for i in range(num_samples):
        
        buffx[1:] = buffx[:1]
        buffx[0] = x[i]
        
        valx = quantizer((b * buffx).sum(), Qformat)
        valy = quantizer((a * buffy).sum(), Qformat)
        
        y[i] = quantizer(valx - valy, Qformat)
        
        buffy[1] = buffy[0]
        buffy[0] = y[i]

    return y


def biquad_quant2(b, a, x, Qformat=(1,16)):
    # SINGLE_BIQUAD_QUANT Filtro biquad forma direta I quantizado
    #    Implementação de um estágio quantizado de filtragem IIR com biquadradas

    num_samples = x.shape[0]
    y = np.zeros(x.shape)
    # w = np.zeros([x.shape[0] + 2, x.shape[1:]])
    buff = np.zeros(3)


    # Forma direta I
    for i in range(num_samples):
        
        buff[1:] = buff[:1]
        buff[0] = x[i] - quantizer((a * buff[1:]).sum(), Qformat)
        y[i] = quantizer((b * buff).sum(), Qformat)
        

    return y

def sos2sos_quant(sos, Qformat):
    non_zeros = sos[0,:3] > 0
    b_factor = np.prod(sos[0,:3][non_zeros]) ** (1/non_zeros.sum())
    sos[0,:3] /= b_factor
    sos[:,:3] *= b_factor ** (1/sos.shape[0])

    sos = quantizer(sos, Qformat)

    return sos


def filtsos_quant(sos, x, Qformat):

    y = x
    for i in range(sos.shape[0]):
        
        b = quantizer(sos[i,:3], Qformat)
        a = quantizer(sos[i,4:], Qformat)
        y = biquad_quant(b, a, y, Qformat)

    return y


def freqz_quant(sos, freq_vec=None, sample_rate=48e3, num_freqs=1000):

    if freq_vec is None:
        freq_vec = np.arange(num_freqs)/num_freqs * sample_rate/2



    return f, h


def get_filter(spec, filter_type='but', method='zoh'):
    
    if filter_type.lower() in ('butterworth'):
        N, Wn = signal.buttord(2*np.pi*spec['fp'], 2*np.pi*spec['fs'], spec['Amax'], spec['Amin'], analog=True)
        z, p, k = signal.butter(N, Wn, output='zpk', btype='low', analog=True)
    elif filter_type.lower() in ('cauer' + 'elliptic'):
        N, Wn = signal.ellipord(2*np.pi*fp, 2*np.pi*spec['fs'], spec['Amax'], spec['Amin'], analog=True)
        z, p, k = signal.ellip(N, spec['Amax'], spec['Amin'], Wn, output='zpk', btype='low', analog=True)

    if method == 'matched':
        zd, pd, kd, dt = matched_method(z, p, k, spec['dt'])
        kd *= 1 - (1 - 10 ** (-spec['Amax']/20))/2
    else:
        zd, pd, kd, dt = signal.cont2discrete((z,p,k), spec['dt'], method=method)

    analog_system = (z,p,k)
    discrete_system = (zd,pd,kd)

    return analog_system, discrete_system

def check_limits(system, spec, num_samples=1000):

    f1 = np.logspace(np.floor(np.log10(spec['fp']))-1, np.log10(spec['fp']), 2*num_samples)
    f2 = np.logspace(np.log10(spec['fs']), np.log10(spec['sample_rate']/2), num_samples)
    f = np.hstack([f1, f2])

    f, h = signal.freqz_zpk(*system, fs=sample_rate, worN=f)
    Hdb = 20 * np.log10(np.abs(h))

    pass_band = Hdb[f <= spec['fp']]
    pass_band_faults = (pass_band < -spec['Amax']).sum() + (pass_band > 0).sum()
    stop_band_faults = (Hdb[f >= spec['fs']] > -spec['Amin']).sum()

    total_faults = pass_band_faults + stop_band_faults + system[1].size

    return total_faults


def optimize_filter(spec, filter_type='but', min_order=None, method='zoh', num_samples=1000, limits_samples=1000):
    # fp, fs, Amax, Amin, sample_rate
    original_spec = spec.copy()
    # Amax_vec = np.linspace(spec['Amax'], 1e-3, num_samples)
    # Amin_vec = np.linspace(spec['Amin'], 2*spec['Amin'], num_samples)
    Amax_vec = rnd.uniform(1e-3, spec['Amax'], num_samples)
    Amin_vec = rnd.uniform(spec['Amin'], 2*spec['Amin'], num_samples)
    Avec = np.vstack([Amax_vec, Amin_vec])

    if min_order is None:
        if filter_type.lower() in ('butterworth'):
            min_order = 9
        elif filter_type.lower() in ('cauer' + 'elliptic'):
            min_order = 4

    faults = np.inf
    for i in range(num_samples):
        spec['Amax'] = Avec[0,i]
        spec['Amin'] = Avec[1,i]
        asys, dsys = get_filter(spec, filter_type=filter_type, method=method)
        total_faults = check_limits(dsys, original_spec, num_samples=limits_samples)
        total_faults -= min_order

        filter_order = dsys[1].size
        print(total_faults, filter_order)

        if faults > total_faults:
            faults = total_faults
            analog_system, discrete_system = (asys, dsys)
        
        if faults == 0:
            analog_system, discrete_system = (asys, dsys)
            return analog_system, discrete_system, spec
    
    print(faults, discrete_system[1].size)

    return analog_system, discrete_system, spec



sample_rate = 48e3

fp = 1.8e3
fs = 3.5e3

Amax = 1
Amin = 42

spec = dict(fp=fp, fs=fs, Amax=Amax, Amin=Amin, sample_rate=sample_rate, dt=1/sample_rate)





analog_system, discrete_system, final_spec = optimize_filter(spec, filter_type='cau', method='bilinear', num_samples=1000)
print(final_spec)

print(discrete_system[-1])

# pair_mode = 'keep_odd'
pair_mode = 'nearest'
sos = signal.zpk2sos(*discrete_system, pairing=pair_mode)
print(f'Biquads of shape {sos.shape}:\n', sos)

non_zeros = sos[0,:3] > 0
b_factor = np.prod(sos[0,:3][non_zeros]) ** (1/non_zeros.sum())
# print(sos[:,:3]/b_factor)
# sos[:,:3] = sos[:,:3] * b_factor ** (1/sos.shape[0])
sos[0,:3] /= b_factor
sos[:,:3] *= b_factor ** (1/sos.shape[0])

sos = quantizer(sos, (2,14), underflow_clip=False)

print('Quantized biquads:\n', sos)
print('B Factor:\n', b_factor, b_factor ** (1/sos.shape[0]))

discrete_system = signal.sos2zpk(sos)

plot_zpk(discrete_system, fp, fs, Amax, Amin, plot_focus='all')
plot_zpk(discrete_system, fp, fs, Amax, Amin, plot_focus='pass')
plot_zpk(discrete_system, fp, fs, Amax, Amin, plot_focus='stop')
plt.show()



