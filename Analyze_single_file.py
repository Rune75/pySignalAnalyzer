
# import pcm signal from csv file data_21.csv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from pcmAnalyzer import pcmAnalyzer


def getZeroCrossings(data, interpolationFactor=1):
    # get zero crossings
    # interpolate the signal to get a better resolution
    data = np.interp(np.linspace(0, len(data), len(data)*interpolationFactor), np.arange(len(data)), data,)
    # find zero crossings
    zeroCrossings = np.where(np.diff(np.sign(data))>0)[0]+0 # positive going zero crossings indexes in the signal add 1 to get the index of the zero crossing

        
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
pcmVector = pd.read_csv('data/data_21.csv', delimiter=',', header=None)
pcmVector = pcmVector.values.flatten()[:8192]
print('Number of samples in the signal: ', len(pcmVector))

adcResolution = 12
adcSampleRate = 1e6

# remove the DC offset from the signal
# pcmVector = pcmVector + 2**(adcResolution-1)
pcmVector = pcmVector - (pcmVector.max() + pcmVector.min())/2

# print max and min values of the signal
print('Max value of the signal: ', pcmVector.max())
print('Min value of the signal: ', pcmVector.min())


zeroCrossings = getZeroCrossings(pcmVector)
# rotate the pcm signal to the first zero crossing
#pcmVector = np.roll(pcmVector, -zeroCrossings[0])
#zeroCrossings = zeroCrossings - zeroCrossings[0]

# number of samples per period
samplesPerPeriod = np.mean(np.diff(zeroCrossings)[1:-1])
print('Number of samples per period: ', samplesPerPeriod)
print('Frequency from number of samples per period: ', adcSampleRate/samplesPerPeriod)

# number of periods in the signal
periods = len(pcmVector)//samplesPerPeriod
print('integer number of periods in the signal: ', periods)
numberSamplesNeeded = periods*samplesPerPeriod
print('Number of samples needed to have integer number of periods: ', numberSamplesNeeded)


# print number of zero crossings
print('Number of zero crossings in the signal: ', len(zeroCrossings))
# if odd number of zero crossings, remove the last one
if len(zeroCrossings)%2 != 1:
    zeroCrossings = zeroCrossings[:-1]
print('Number of zero crossings in the signal: ', len(zeroCrossings))

# get the index of the last zero crossing
index = zeroCrossings[-1]
print('Index of the last rising zero crossing in the signal: ', index)

# crop the signal to the last zero crossing
pcmVector = pcmVector[:index+1]
print('Number of samples in the signal: ', len(pcmVector))


index = zeroCrossings[0]
print('Index of the first rising zero crossing in the signal: ', index)

plotSignalAndZeroCrossings(pcmVector, zeroCrossings)



analysis = pcmAnalyzer(pcmVector, adcSampleRate, adcResolution)
spectrum = analysis.getPowerSpectrum()
plt.plot(spectrum.frequency, spectrum.level)
plt.show()

# analysis.plotPowerSpectrum()
# plt.show()


# analysis.printAll()



                                    