import numpy as np
import matplotlib.pyplot as plt
import pathlib
import scipy.signal as signal


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


def Pz(beta=0.9, N0=15, num_echoes=2):

    
    b = np.zeros(num_echoes*N0+1)
    for i in range(num_echoes+1):
        b[i*N0] = beta ** i

    a = 1

    z, p, k = signal.tf2zpk(b, a)

    return z, p, k


def Vz(b0=0.98, b1=1, r=0.9, theta=np.pi/6):

    z = np.empty(1, dtype=np.complex)
    p = np.empty(2, dtype=np.complex)

    z[0] = -b1/b0
    p[0] = r * np.exp(1j*theta)
    p[1] = r * np.exp(-1j*theta)
    k = b0
    
    return z, p, k

def unit_circle(N, r=1):

    theta = np.linspace(0, 2*np.pi, N)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def box_filter(n,N1,N2, filter_type='low'):

    N = n.size
    
    if filter_type is 'low':
        mask = np.ones(N)
        mask[N1:N-N2] = 0
    else:
        mask = np.zeros(N)
        mask[N1:N-N2] = 1


    return mask


out_folder = pathlib.Path('images')
out_folder.mkdir(parents=True, exist_ok=True)




