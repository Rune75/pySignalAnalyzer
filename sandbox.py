
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
    pass
        

def getFrequency(data, adcSampleRate):
    interpolationFactor = 1000
    adcSampleRate = adcSampleRate*interpolationFactor
    
    zeroCrossings = getZeroCrossings(data)
    zeroCrossingsFrequency = adcSampleRate/np.diff(zeroCrossings)
    avgFrequency = np.mean(zeroCrossingsFrequency)
    return avgFrequency

# read csv file. delimiter is ","
pcmVector = pd.read_csv('data/data_15.csv', delimiter=',', header=None)
pcmVector = pcmVector.values.flatten()[:8192]
print('Number of samples in the signal: ', len(pcmVector))

adcResolution = 12
adcSampleRate = 1e6


# remove the DC offset from the signal
pcmVector = pcmVector - np.mean(pcmVector)
data = pcmVector

zeroCrossings = getZeroCrossings(data)

# get the frequency of the signal
avgFrequency = getFrequency(data, adcSampleRate)
print('Average frequency of the signal: ', avgFrequency)

index = zeroCrossings[0]
print('Index of the first rising zero crossing in the signal: ', index)




analysis = pcmAnalyzer(data, adcSampleRate, adcResolution)
analysis.plotPowerSpectrum()
# analysis.printAll()



                                    