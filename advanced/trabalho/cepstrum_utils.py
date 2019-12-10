import pathlib

import numpy as np
import numpy.fft as fftlib
import scipy.signal as signal

import matplotlib.pyplot as plt


eps = 1e-6

def unit_impulse(n):
    d = 1*(n == 0)

    return d


def unit_step(n):
    u = 1*(n >= 0)

    return u


def pn(n, beta=0.9, N0=15, num_echoes=2):
    # p = unit_impulse(n) + beta * unit_impulse(n - N0) + beta ** 2 * unit_impulse(n - 2*N0)

    p = 0
    for i in range(num_echoes+1):
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

def ceps(x, gamma, real=True, fft_size='auto'):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    if fft_size is None:
        N = 2 ** nextpow2(x.size)
    else:
        N = x.size

    X = fft_func(x, N)

    if np.abs(gamma) < eps:
        Xv = np.log(X)
    else:
        Xv = (X ** gamma - 1) / gamma

    Xv.imag = np.unwrap(Xv.imag)
    xv = ifft_func(Xv)

    return xv


def iceps(xv, gamma, real=True):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    Xv = fft_func(xv)

    if np.abs(gamma) < eps:
        # Xv.imag = np.unwrap(Xv.imag)
        X = np.exp(Xv)
    else:
        X = (Xv * gamma + 1) ** (1/gamma)

    x = ifft_func(X)

    return x



def root_cepstrum(x, gamma, real=True):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    X = fft_func(x)

    Xv = X ** gamma

    xv = ifft_func(Xv)

    return xv





def rceps(x, gamma, real=True, fft_size='auto'):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    if fft_size is None:
        N = 2 ** nextpow2(x.size)
    else:
        N = x.size

    X = fft_func(x, N)

    if np.abs(gamma) < eps:
        Xv = np.log(X)
    else:
        Xv = (X ** gamma - 1) / gamma

    Xv.imag = np.unwrap(Xv.imag)
    xv = ifft_func(Xv)

    return xv


def irceps(xv, gamma, real=True):
    
    if real:
        fft_func = fftlib.rfft
        ifft_func = fftlib.irfft
    else:
        fft_func = fftlib.fft
        ifft_func = fftlib.ifft

    Xv = fft_func(xv)

    if np.abs(gamma) < eps:
        # Xv.imag = np.unwrap(Xv.imag)
        X = np.exp(Xv)
    else:
        X = (Xv * gamma + 1) ** (1/gamma)

    x = ifft_func(X)

    return x






