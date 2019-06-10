import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as lin
import scipy.misc as misc
import warnings

warnings.filterwarnings('ignore', '.*Ill-conditioned matrix.*')
warnings.filterwarnings('ignore', '.*Badly conditioned filter coefficients.*')

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


def plot_time(analog_system, discrete_system, response='step', num_samples=100, sample_rate=48e3, plot_error=False):

    n = np.arange(num_samples)
    t = n/sample_rate
    fig, ax1 = plt.subplots()

    if response == 'impulse':
        ta, ya = signal.impulse2(analog_system, T=t)
        _, yd = signal.dimpulse((*discrete_system,1/sample_rate), t=t)
        ax1.set_title('Resposta ao impulso')

    elif response == 'step':
        ta, ya = signal.step2(analog_system, T=t)
        _, yd = signal.dstep((*discrete_system,1/sample_rate), t=t)
        ax1.set_title('Resposta ao degrau')

    elif response == 'ramp':
        ramp_factor = 1
        ta = np.arange(int(num_samples*ramp_factor))/(ramp_factor*sample_rate)
        print(ta.shape)
        ta, ya, xa = signal.lsim2(analog_system, U=ta, T=ta, printmessg=True)

        _, yd = signal.dlsim((*discrete_system,1/sample_rate), u=t, t=t, x0=None)
        ax1.set_title('Resposta a rampa')


    ax1.plot(ta*sample_rate,ya,'b')
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

def matched_method(z, p, k, dt):

    zd = np.exp(z*dt)
    pd = np.exp(p*dt)
    kd = k * np.abs(np.prod(1-pd)/np.prod(1-zd) * np.prod(z)/np.prod(p))

    return zd, pd, kd, dt

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
    # fp, fs, Amax=1, Amin=42, sample_rate=48e3

    f1 = np.logspace(np.floor(np.log10(spec['fp']))-1, np.log10(spec['fp']), 2*num_samples)
    f2 = np.logspace(np.log10(spec['fs']), np.log10(spec['sample_rate']/2), num_samples)
    # f1 = np.linspace(0, spec['fp'], 2*num_samples)
    # f2 = np.linspace(spec['fs'], spec['sample_rate']/2, num_samples)
    f = np.hstack([f1, f2])

    f, h = signal.freqz_zpk(*system, fs=spec['sample_rate'], worN=f)
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





analog_system, discrete_system, final_spec = optimize_filter(spec, filter_type='cau', method='matched', num_samples=1000)
print(final_spec)

print(discrete_system[-1])


plot_zpk(discrete_system, fp, fs, Amax, Amin, plot_focus='all')
plot_zpk(discrete_system, fp, fs, Amax, Amin, plot_focus='pass')
plot_zpk(discrete_system, fp, fs, Amax, Amin, plot_focus='stop')
# plot_time(analog_system, discrete_system, response='step', num_samples=100, sample_rate=sample_rate, plot_error=False)
plt.show()




