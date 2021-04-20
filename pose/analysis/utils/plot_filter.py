from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

b, a = signal.butter(4,2.5, analog=True)
w, h = signal.freqs(b, a)

plt.semilogx(w, 20 *np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
# plt.axvline(100, color='green') # cutoff frequency
plt.show()
