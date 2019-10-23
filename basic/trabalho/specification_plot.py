import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import matplotlib as mpl
# mpl.rc('text', usetex=True)


sample_rate = 48e3
fp = 1000
fs = 2000
Amax = 5
Amin = 10
fmin = 0
fmax = 3e3



fig, ax = plt.subplots(figsize=(9,7))
axis_focus = [1e2, fmax, -15, 1]

# Plot boxes
box_style = dict(linewidth=2, linestyle='--', edgecolor='k', facecolor='0.9')
ax.fill([fmin, fmin,  fp,  fp], [-Amin*2, -Amax, -Amax, -Amin*2], **box_style) # pass
ax.fill([fs, fs,  fmax,  fmax], [-Amin, Amax, Amax, -Amin], **box_style) # stop

# Low pass filter example
N, Wn = signal.buttord(fp/sample_rate*2.5, fs/sample_rate*2, Amax, Amin)
b, a = signal.butter(N, Wn, btype='low')
w, h = signal.freqz(b,a,fs=sample_rate)
ax.plot(w, 20*np.log10(np.abs(h)), linewidth=2, color='k')

# Set ticks
plt.xticks(ticks=[fp,fs], labels=['$f_p$','$f_s$'], fontsize=18)
ax.plot([fs,fs],[-Amin,-30], 'k--')
plt.yticks(ticks=[-Amax,-Amin], labels=['$-A_{max}$','$-A_{min}$'], fontsize=18)
ax.plot([0,fs],[-Amin,-Amin], 'k--')


# Set plot properties
ax.set_title('Low-pass Filter Specifications', fontsize=13)
ax.set_xlabel('Frequency [Hz]', fontsize=13)
ax.set_ylabel('Amplitude [dB]', fontsize=13)
ax.axis(axis_focus)
plt.savefig('images/example_spec.eps', format='eps')
plt.savefig('images/example_spec.png', format='png')
plt.show()



# sample_rate = 48e3
# fp = 1.8e3
# fs = 3.5e3
# Amax = 1
# Amin = 42
# fmin = 0
# fmax = 48e3

fig, ax = plt.subplots(figsize=(9,7))
axis_focus = [1e2, fmax, -15, 0]

# Plot boxes
box_style = dict(linewidth=2, linestyle='--', edgecolor='k', facecolor='0.9')
ax.fill([fmin, fmin,  fp,  fp], [-Amin*2, -Amax, -Amax, -Amin*2], **box_style) # pass
ax.fill([fs, fs,  fmax,  fmax], [-Amin, Amax, Amax, -Amin], **box_style) # stop
ax.fill([0, 0, sample_rate, sample_rate], [0,0,0,0], **box_style) # stop

# Set ticks
plt.xticks(ticks=[fp,fs,sample_rate], labels=['$1800$','$3500$', '$\\frac{1}{\\Delta t}$'], fontsize=18)
plt.yticks(ticks=[0,-Amax,-Amin], labels=['$0$','$-1$','$-42$'], fontsize=18)
ax.plot([fs,fs],[-Amin,-30], 'k--')
ax.plot([0,fs],[-Amin,-Amin], 'k--')


# Set plot properties
ax.set_title('Filter Specification', fontsize=13)
ax.set_xlabel('Frequency [Hz]', fontsize=13)
ax.set_ylabel('Amplitude [dB]', fontsize=13)
ax.axis(axis_focus)
plt.savefig('images/specification.eps', format='eps')
plt.savefig('images/specification.png', format='png')
plt.show()



