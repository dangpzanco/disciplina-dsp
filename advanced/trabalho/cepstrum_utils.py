import pathlib

import numpy as np
import numpy.random as rnd
import numpy.fft as fftlib
import scipy.signal as signal
from scipy.special import binom

import matplotlib.pyplot as plt


eps = 1e-6

def unit_impulse(n):
    d = 1*(n == 0)

    return d


def unit_step(n):
    u = 1*(n >= 0)

    return u

def relative_energy(h):

    h2 = h ** 2
    d = np.cumsum(h2) / h2.sum()

    return d

def rnd_unit_circle(N=None):

    if N is None:
        theta = 2 * np.pi * rnd.rand()
    else:
        theta = 2 * np.pi * rnd.rand(N)

    x = np.exp(1j * theta)

    return x

def rnd_zpk(r, Nz, Np):

    z = rnd_unit_circle(Nz//2)
    z = np.append(z, z.conj())
    
    p = rnd_unit_circle(Np//2)
    p = np.append(p, p.conj())

    k = r

    return z, p, k

def zpk2root(z, p, k, n, gamma, L=None):

    if L is None:
        L = 10

    num_z = len(z)
    num_p = len(p)

    x = np.zeros(n.size, dtype=np.complex)

    for k in range(num_z):
        temp_sum = 0
        for l in range(L):
            temp_sum += binom(gamma, l) * (-z[k]) ** l * unit_impulse(n-l)
        x += unit_impulse(n)
        x += temp_sum

    for k in range(num_p):
        temp_sum = 0
        for l in range(L):
            temp_sum += binom(-gamma, l) * (-p[k]) ** l * unit_impulse(n-l)
        x += unit_impulse(n)
        x += temp_sum

    return x


def recursive_rceps(x, gamma, A=1):
    N = x.size
    samples = np.arange(N)
    xv = np.zeros(N)

    xv[0] = A ** gamma
    xv[1] = gamma * x[1] * xv[0] / x[0]
    for n in range(2,N):
        xv[n] = gamma * x[n] * xv[0] / x[0]
        temp_sum = 0
        for k in range(1,n):
            temp_sum += k/(n*x[0]) * (gamma * x[k] * xv[n-k] - xv[k] * x[n-k])
        xv[n] += temp_sum

    return xv


def pn(n, beta=0.9, N0=15, K=3):
    # p = unit_impulse(n) + beta * unit_impulse(n - N0) + beta ** 2 * unit_impulse(n - 2*N0)

    p = 0
    for i in range(K):
        p += (beta ** i) * unit_impulse(n - i*N0)

    return p

def vn(n, b0=0.98, b1=1, r=0.9, theta=np.pi/6):
    
    def w(n, r=r, theta=theta):
        w = 0.5 * (r ** n) * (np.sin(theta) ** -2) * (np.cos(theta * n) - np.cos(theta * (n+2))) * unit_step(n)
        return w
 
    w0 = w(n)
    w1 = w(n-1)

    v = b0 * w0 + b1 * w1

    return v

def nextpow2(A):
    return int(np.ceil(np.log2(A)))


def log(a, gamma=0):
    if np.abs(gamma) < eps:
        b = np.log(a)
    else:
        b = (a ** gamma - 1) / gamma

    return b

def exp(a, gamma=0):
    if np.abs(gamma) < eps:
        # a.imag = np.unwrap(a.imag)
        b = np.exp(a)
    else:
        b = (a * gamma + 1) ** (1/gamma)

    return b


def ceps(x, gamma, real=False, unwrap=False):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    X = fft_func(x)

    if np.abs(gamma) < eps:
        Xv = np.log(X)
        # Xv.imag = np.unwrap(Xv.imag)
    else:
        # Xv = (X ** gamma - 1) / gamma
        Xv = np.abs(X) ** gamma * np.exp(1j * gamma * np.angle(X))
        Xv = (Xv - 1) / gamma

    # Xv.imag = np.unwrap(Xv.imag)
    xv = ifft_func(Xv)

    return xv


def iceps(xv, gamma, real=False):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    Xv = fft_func(xv)

    if np.abs(gamma) < eps:
        X = np.exp(Xv)
    else:
        X = (Xv * gamma + 1) ** (1/gamma)

    x = ifft_func(X)

    return x



def rceps(x, gamma, real=False):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    X = fft_func(x)

    if np.abs(gamma) < eps:
        Xv = np.log(X)
        # Xv.imag = np.unwrap(Xv.imag)
    else:
        # Xv = X ** gamma
        Xv = np.abs(X) ** gamma * np.exp(1j * gamma * np.angle(X))

    xv = ifft_func(Xv)

    return xv


def irceps(xv, gamma, real=False):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    Xv = fft_func(xv)

    if np.abs(gamma) < eps:
        X = np.exp(Xv)
    else:
        X = Xv ** (1/gamma)
        # X = np.abs(Xv) ** (1/gamma) * np.exp(1j / gamma * np.angle(Xv))

    x = ifft_func(X)

    return x

def box_filter(N, N1, N2, filter_type='low'):

    if filter_type is 'low':
        mask = np.ones(N)
        mask[N1:N-N2] = 0
    else:
        mask = np.zeros(N)
        mask[N1:N-N2] = 1

    return mask


def box_filter2(N, N1, N2, filter_type='low'):

    if filter_type is 'low':
        mask = np.ones(N)
        mask[N1:N2+1] = 0
    else:
        mask = np.zeros(N)
        mask[N1:N2+1] = 1

    return mask



def deconvolve(x, gamma, fs=8e3, Tc=None, ceps_type='log'):

    if Tc is None:
        Tc = (10e-3, 100e-3)

    # x[n] = p[n] * v[n]

    N1, N2 = np.round(np.array(Tc) * fs).astype(int)

    low_mask = box_filter2(x.size, N1, N2, filter_type='low')
    high_mask = box_filter2(x.size, N1, N2, filter_type='high')


    if ceps_type is 'root':
        xv = rceps(x, gamma, real=False)
        v_est = irceps(xv * low_mask, gamma, real=False).real
        p_est = irceps(xv * high_mask, gamma, real=False).real
    elif ceps_type is 'log':
        xv = ceps(x, gamma, real=False)
        v_est = iceps(xv * low_mask, gamma, real=False).real
        p_est = iceps(xv * high_mask, gamma, real=False).real
    else:
        print(f'ceps_type [{ceps_type}] not implemented.')


    return v_est, p_est


def get_example_signals(num_samples, echo_options, indi=0):

    n = np.arange(num_samples) - indi

    p = pn(n, **echo_options)
    v = vn(n+1)

    indf = indi + num_samples
    x = np.convolve(v, p, mode='full')[indi:indf]

    return x, v, p


def mse(x, x_est, use_log=True):

    mse_value = ((x - x_est) ** 2).mean()

    if use_log:
        return np.log(mse_value)
    else:
        return mse_value



def grpdelay(system, nfft=1024, whole=True, fs=1):

    b = system[0]
    a = system[1]

    b = np.array(b)
    a = np.array(a)
    w = fs * np.arange(nfft) / nfft;

    # try:
    #     a = np.fliplr(a)

    # order of a(z)
    oa = a.size - 1
    # order of c(z)
    oc = oa + b.size - 1

    print(b.size, a.size)

    # c(z) = b(z)*a(1/z)*z^(-oa)
    c = np.convolve(b, a, mode='full')

    # derivative of c wrt 1/z
    cr = c * np.arange(oc+1)
    num = fftlib.fft(cr, nfft)
    den = fftlib.fft(c, nfft)
    
    ind = np.abs(den) < eps
    num[ind] = 0
    den[ind] = 1

    gd = (num / den).real - oa

    return w, gd

