
# import pcm signal from csv file data_21.csv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from signalAnalyzer import signalAnalyzer


def getZeroCrossings(data, interpolationFactor=1):
    # get zero crossings
    # interpolate the signal to get a better resolution
    data = np.interp(np.linspace(0, len(data), len(data)*interpolationFactor), np.arange(len(data)), data,)
    # find zero crossings
    zeroCrossings = np.where(np.diff(np.sign(data))>0)[0]+1 # positive going zero crossings indexes in the signal add 1 to get the index of the zero crossing

        
    return zeroCrossings

def plotSignalAndZeroCrossings(pcmVector, zeroCrossings):
    # plot the signal and zero crossings

    samplesToPlot = 100
    # plot start and end of the pcm signal is horizontal subplots
    fig, axs = plt.subplots(2)
    axs[0].plot(pcmVector[:samplesToPlot])
    axs[0].set_title('PCM Signal Start')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Amplitude')
    # add 'x' marks at the sample locations
    axs[0].plot(np.arange(samplesToPlot), pcmVector[:samplesToPlot], 'x')
    # add red dots at the zero crossings
    axs[0].plot(zeroCrossings[:5], pcmVector[zeroCrossings[:5]], 'ro')
    axs[0].grid()

    axs[1].plot(pcmVector[-samplesToPlot:])
    axs[1].set_title('PCM Signal End')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid()
    plt.show()
    pass
        

def getFrequency(data, adcSampleRate):
    interpolationFactor = 1000
    adcSampleRate = adcSampleRate*interpolationFactor
    
    zeroCrossings = getZeroCrossings(data)
    zeroCrossingsFrequency = adcSampleRate/np.diff(zeroCrossings)
    avgFrequency = np.mean(zeroCrossingsFrequency)
    return avgFrequency

# read csv file. delimiter is ","
pcmVector = pd.read_csv('data/data_24.csv', delimiter=',', header=None)
pcmVector = pcmVector.values.flatten()[:8192]
print('Number of samples in the signal: ', len(pcmVector))

adcResolution = 12
adcSampleRate = 1e6

# remove the DC offset from the signal
# pcmVector = pcmVector + 2**(adcResolution-1)
# pcmVector = (pcmVector - (pcmVector.max() + pcmVector.min())/2).astype(np.float32)
dcOffset = np.mean(pcmVector)
print('DC offset of the signal: ', dcOffset)
pcmVector = pcmVector - dcOffset

# print max and min values of the signal
print('Max value of the signal: ', pcmVector.max())
print('Min value of the signal: ', pcmVector.min())


zeroCrossings = getZeroCrossings(pcmVector)

# # number of samples per period
# samplesPerPeriod = np.mean(np.diff(zeroCrossings)[1:-1])
# print('Number of samples per period: ', samplesPerPeriod)
# print('Frequency from number of samples per period: ', adcSampleRate/samplesPerPeriod)

# # if odd number of zero crossings, remove the last one
# if len(zeroCrossings)%2 == 1:
#     zeroCrossings = zeroCrossings[:-1]
# print('Number of zero crossings in the signal: ', len(zeroCrossings))

# # get the index of the last zero crossing
# indexLast = zeroCrossings[-1]
# print('Index of the last rising zero crossing in the signal: ', indexLast)
# indexFirst = zeroCrossings[0]

# # crop the signal to the last zero crossing odd zero crossing
# pcmVector = pcmVector[:indexLast]

plotSignalAndZeroCrossings(pcmVector, zeroCrossings)



analysis = signalAnalyzer(pcmVector, adcSampleRate, adcResolution)
spectrum = analysis.getPowerSpectrum()
# plt.plot(spectrum.frequency, spectrum.level)
# plt.show()

analysis.printAll()

analysis.plotPowerSpectrum()
plt.show()


# analysis.printAll()



                                    