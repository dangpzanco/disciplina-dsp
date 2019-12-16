import pathlib

import numpy as np
import numpy.random as rnd
import numpy.fft as fftlib
import scipy.signal as signal

import matplotlib.pyplot as plt

import cepstrum_utils as utils


out_folder = pathlib.Path('images')
out_folder.mkdir(parents=True, exist_ok=True)
plt.style.use('grayscale')


num_samples = 1024
indi = 0
n = np.arange(num_samples) - indi

fs = 10e3
Tmax = num_samples / fs
echo_period = 15 / fs
echo_est = echo_period
Tc = (echo_est, Tmax - echo_est)

echo_options = dict(N0=int(echo_period * fs),
    K=3, 
    beta=0.9)

x, v, p = utils.get_example_signals(num_samples, echo_options, indi=indi)


gamma = 0.9
num_gamma = 1000
gamma_list = np.linspace(-2, 2, num_gamma)
# ind_log = np.argmin(np.abs(gamma_list))
# gamma_list[ind_log] = 0


v_log, p_log = utils.deconvolve(x, 0, fs, Tc, ceps_type='log')
mse_log = [utils.mse(v, v_log), utils.mse(p, p_log)]

mse_glog = np.empty([num_gamma, 2])
mse_root = np.empty([num_gamma, 2])
for i in range(num_gamma):

    gamma = gamma_list[i]

    v_est, p_est = utils.deconvolve(x, gamma, fs, Tc, ceps_type='log')
    mse_glog[i,0] = utils.mse(v, v_est)
    mse_glog[i,1] = utils.mse(p, p_est)

    v_est, p_est = utils.deconvolve(x, gamma, fs, Tc, ceps_type='root')
    mse_root[i,0] = utils.mse(v, v_est)
    mse_root[i,1] = utils.mse(p, p_est)


fig, ax = plt.subplots(2, 1, figsize=(10,7), sharex=True, sharey=True)
label_list = ['log MSE ($v[n]$)', 'log MSE ($p[n]$)']


for i in range(ax.size):
    ax[i].plot(gamma_list, mse_root[:,i], '-', label='Spectral Root Cepstrum', linewidth=2)
    ax[i].plot(gamma_list, mse_glog[:,i], '-.', label='Generalized Cepstrum', linewidth=4)
    ax[i].plot(0, mse_log[i], 'x', label='Cepstrum', markersize=15)
    ax[i].set_ylabel(label_list[i])
    ax[i].set_xlabel(r'$\gamma$', fontsize=14)
    ax[i].legend()
    ax[i].grid(True)

ax[0].axis([-2,2,-20,5])

fig.tight_layout()
fig.savefig(out_folder / 'gamma_grid.png', format='png', dpi=600)
fig.savefig(out_folder / 'gamma_grid.pdf', format='pdf')

# plt.show()
# exit()


ind_root = np.nanargmin(mse_root[:,0])
ind_glog = np.nanargmin(mse_glog[:,0])
gamma_root = gamma_list[ind_root]
gamma_glog = gamma_list[ind_glog]

v_root, p_root = utils.deconvolve(x, gamma_root, fs, Tc, ceps_type='root')
v_glog, p_glog = utils.deconvolve(x, gamma_glog, fs, Tc, ceps_type='log')

# exit()


fig, ax = plt.subplots(3, 1, figsize=(10,7), sharex=True, sharey=True)
stem_options = dict(markerfmt='k.', basefmt='k,', use_line_collection=True)

ax[0].stem(n, v, 'k', label='Original Signal', **stem_options)
ax[0].plot(n, v_root, '-', label='Estimate')
ax[0].set_ylabel('$v[n]$', fontsize=14)
ax[0].set_title(f'Spectral Root Cepstrum ($\\gamma$ = {gamma_root:.3f}) | logMSE = {mse_root[ind_root,0]:.3f}')

ax[1].stem(n, v, 'k', label='Original Signal', **stem_options)
ax[1].plot(n, v_glog, '-', label='Estimate')
ax[1].set_ylabel('$v[n]$', fontsize=14)
ax[1].set_title(f'Generalized Cepstrum ($\\gamma$ = {gamma_glog:.3f}) | logMSE = {mse_glog[ind_glog,0]:.3f}')

ax[2].stem(n, v, 'k', label='Original Signal', **stem_options)
ax[2].plot(n, v_log, '-', label='Estimate')
ax[2].set_ylabel('$v[n]$', fontsize=14)
ax[2].set_title(f'Cepstrum | logMSE = {mse_log[0]:.3f}')

ax[0].axis([-indi,100,None,None])

for i in range(ax.size):
    ax[i].legend()
    ax[i].grid(True)

fig.tight_layout()
fig.savefig(out_folder / 'logmse_v.png', format='png', dpi=600)
fig.savefig(out_folder / 'logmse_v.pdf', format='pdf')



ind_root = np.nanargmin(mse_root[:,1])
ind_glog = np.nanargmin(mse_glog[:,1])
gamma_root = gamma_list[ind_root]
gamma_glog = gamma_list[ind_glog]
print(gamma_root, gamma_glog)
_, p_root = utils.deconvolve(x, gamma_root, fs, Tc, ceps_type='root')
_, p_glog = utils.deconvolve(x, gamma_glog, fs, Tc, ceps_type='log')




fig, ax = plt.subplots(3, 1, figsize=(10,7), sharex=True, sharey=True)
stem_options = dict(markerfmt='k.', basefmt='k,', use_line_collection=True)

ax[0].stem(n, p, 'k', label='Original Signal', **stem_options)
ax[0].plot(n, p_root, '-', label='Estimate')
ax[0].set_ylabel('$p[n]$', fontsize=14)
ax[0].set_title(f'Spectral Root Cepstrum ($\\gamma$ = {gamma_root:.3f}) | logMSE = {mse_root[ind_root,1]:.3f}')

ax[1].stem(n, p, 'k', label='Original Signal', **stem_options)
ax[1].plot(n, p_glog, '-', label='Estimate')
ax[1].set_ylabel('$p[n]$', fontsize=14)
ax[1].set_title(f'Generalized Cepstrum ($\\gamma$ = {gamma_glog:.3f}) | logMSE = {mse_glog[ind_glog,1]:.3f}')


ax[2].stem(n, p, 'k', label='Original Signal', **stem_options)
ax[2].plot(n, p_log, '-', label='Estimate')
ax[2].set_ylabel('$p[n]$', fontsize=14)
ax[2].set_title(f'Cepstrum | logMSE = {mse_log[1]:.3f}')


ax[0].axis([-indi,100,None,None])

for i in range(ax.size):
    ax[i].legend()
    ax[i].grid(True)

fig.tight_layout()
fig.savefig(out_folder / 'logmse_p.png', format='png', dpi=600)
fig.savefig(out_folder / 'logmse_p.pdf', format='pdf')





# ax.plot(x)
# ax.plot(x_est)
# ax.plot(np.abs(x - x_est))
# ax.plot(xv.real)
# ax.plot(np.abs(fftlib.fftshift(xv)))

plt.show()

