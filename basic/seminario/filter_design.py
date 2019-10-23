# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.linalg as lin
from scipy.special import binom
from scipy.special import factorial
import scipy.misc as misc
import autograd


def maxFlatPoly(tau, P):
    a = np.empty(P+1)
    i = np.arange(P+1)
    for k in range(P+1):
        a[k] = (-1) ** k * binom(P, k) * np.prod((2*tau + i)/(2*tau + k + i))
    a /= a.sum()

    return a

def absDz(omega, ak):
    mag = np.abs(np.sum(ak * np.exp(-1j * omega * np.arange(ak.size))))
    return mag

def D_derivative(k, tau, P):
    func = lambda x: absDz(x, maxFlatPoly(tau, P))

    if k < 1:
        return func(0.0)
    else:
        d = autograd.grad(func)

    if k > 1:
        for i in range(1,k):
            d = autograd.grad(d)

    return d(0.0)

def Marray(M, p):
    a = np.arange(p + 1).reshape(1,-1)
    b = np.arange(int(M/2)).reshape(-1,1) + 1
    
    x = a ** (2*b) * (-1) ** b
    
    return x

def Narray(N, p):
    a = np.arange(p + 1).reshape(1,-1)
    b = np.arange(int(N/2)).reshape(-1,1) + 1
    
    x = a ** (2*b) * (-1) ** b * (-1) ** a
    
    return x

def getA1_matrix(wb, M, N, p):

    Mmatrix = Marray(M, p)
    Nmatrix = Narray(N, p)
    a = np.arange(p+1)
    onevec = np.ones(p+1)

    A = np.vstack([onevec, (-1) ** a, np.cos(wb * a), Mmatrix, Nmatrix])

    print(f'A.shape: {A.shape}')

    return A

def getD1I_array(wb, M, N, tau, P):
    d = np.zeros(3 + int(M/2) + int(N/2))
    func = lambda x: absDz(x, maxFlatPoly(tau, P))
    d[0] = func(0)
    d[1] = 0
    d[2] = np.sqrt(0.5)*func(wb)

    for k in range(int(M/2)):
        d[3+k] = D_derivative(2*(k+1), tau, P)

    return d

def b1_to_Nz(b1):
    return np.hstack([np.flipud(b1)[:-1]/2,b1[0],b1[1:]/2])

def getHz1(wb, M, N, tau, P, p):
    A1 = getA1_matrix(wb, M, N, p)
    d1I = getD1I_array(wb, M, N, tau, P)
    b1, _, _, _ = np.linalg.lstsq(A1,d1I)
    # b1 = np.linalg.pinv(A1) @ d1I
    # Q, R = np.linalg.qr(A1)
    # b1 = np.linalg.inv(R) @ (Q.T @ d1I)
    # error = A1 @ b1 - d1I

    a1 = maxFlatPoly(tau, p)

    Dz = maxFlatPoly(tau, p)
    Nz = b1_to_Nz(b1)

    Hz = (Nz, Dz)

    return Hz

def plotFilter(b, a, mag_type='abs', fs=2, ax1=None):
    w1, h = sig.freqz(b, a, fs=fs)
    w2, gd = sig.group_delay((b, a), fs=fs)

    if ax1 is None:
        fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')

    # ax1.plot(w1, 20 * np.log10(abs(h)), 'b')
    # ax1.plot(w1[:-1], (abs(h[:-1]) - abs(h[1:]))/(w1[1]-w1[0]), 'b')

    if mag_type == 'abs':
        mag = np.abs(h)
        mag_label = 'Magnitude'
    elif mag_type == 'dB':
        mag = 20 * np.log10(np.abs(h))
        mag_label = 'Magnitude [dB]'
    elif mag_type == 'loss':
        mag = 20 * np.log10(1/np.abs(h))
        mag_label = 'Attenuation [dB]'

    ax1.plot(w1, mag, 'b')
    ax1.set_ylabel(mag_label, color='b')
    ax1.set_xlabel('Frequency [normalized]')

    ax2 = ax1.twinx()
    ax2.plot(w2, gd, 'g')
    ax2.set_ylabel('Group delay', color='g')
    ax2.grid()
    ax2.axis('tight')
    if mag_type == 'loss':
        ax2.set_ylim(-5,5)

# tau = 0.5
# P = 6

# L = 7
# K = 9

# M = L - 1
# N = K - 1
# p = int((L + K + 1)/2)

# omega_b = 0.35 * np.pi

# print(f'tau = {tau}')
# print(f'P = {P}')
# print(f'L = {L}, K = {K}')
# print(f'M = {M}, N = {N}')
# print(f'p = {p}, num_eq = {M+N+4}')
# print(f'omega_b = {omega_b/np.pi}π')


# # Reference 6
# # fig, ax1 = plt.subplots()
# # plotFilter(1, maxFlatPoly(4, 5), mag_type='loss', ax1=ax1)
# # plotFilter(1, maxFlatPoly(4, 10), mag_type='loss', ax1=ax1)
# # plotFilter(1, maxFlatPoly(4, 15), mag_type='loss', ax1=ax1)
# # plt.show()



# A1 = getA1_matrix(omega_b, M, N, p).T
# d1I = getD1I_array(omega_b, M, N, tau, P)
# b1, _, _, _ = np.linalg.lstsq(A1,d1I)
# # b1 = np.linalg.pinv(A1) @ d1I
# # Q, R = np.linalg.qr(A1)
# # b1 = np.linalg.inv(R) @ (Q.T @ d1I)
# error = A1 @ b1 - d1I

# a1 = maxFlatPoly(tau, p)

# Dz = maxFlatPoly(tau, p)
# Nz = b1_to_Nz(b1)

# np.set_printoptions(precision=3, suppress=True, linewidth=600)
# # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# print(A1.shape, '\n', A1)
# print(d1I.shape, d1I)
# print(b1.shape, b1)
# print(np.sum(error ** 2))

# # print(Nz)
# # exit(0)

# # plotFilter(b1, a1)
# # plotFilter(1, a1)
# # plotFilter(b1, 1)
# # plotFilter(1, Dz)
# # plotFilter(Nz, 1)
# plotFilter(Nz, Dz)
# plt.show()



