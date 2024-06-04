# A basic usage example of the pcm analyzer class
from signalAnalyzer import signalAnalyzer
import matplotlib.pyplot as plt

# *******  Create a test signal ***********************************************************
import sys
sys.path.append("test")
from pcmVector import pcmSignal

sampleRate=1e6                                      # 1 MHz sampling rate
adcResolution=12                                    # 12-bit ADC resolution
signal = pcmSignal(frequency= 356e3,                # 356 kHz signal frequency
                     amplitude=0,                   # 0 dBFS signal amplitude
                     sampling_rate=sampleRate,
                     adcResolution=adcResolution,
                     harmonic_levels=[-60, -60],    # 2nd and 3rd harmonic levels
                     nrSamples=8192)                # 8192 samples
# ****************************************************************************************

# ******  Analyze the signal *************
# Initialize an object of the signalAnalyzer class with the test signal as input
analyzer = signalAnalyzer(signal.pcmVector, sampleRate, adcResolution)

# get SNR
snr = analyzer.getSNR()
print(f'SNR: {snr:.2f} dB')

# get ENOB
enob = analyzer.getENOB()
print(f'ENOB: {enob:.2f} bits')

# print all the signal specs
analyzer.printAll()

# Plot the power spectrum
analyzer.plotPowerSpectrum()
plt.show()
