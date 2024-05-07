
# import pcm signal from csv file data_21.csv
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from pcmAnalyzer import pcmAnalyzer


def getZeroCrossings(data, interpolationFactor=1):
    # get zero crossings
    # interpolate the signal to get a better resolution
    data = np.interp(np.linspace(0, len(data), len(data)*interpolationFactor), np.arange(len(data)), data,)
    # find zero crossings
    zeroCrossings = np.where(np.diff(np.sign(data))>0)[0]+1 # positive going zero crossings indexes in the signal add 1 to get the index of the zero crossing
    
    return zeroCrossings


adcResolution = 12
adcSampleRate = 1e6


pp = PdfPages('analysisResults.pdf')
# read and plot al the data files
for i in range(1, 25):
    pcmVector = pd.read_csv('data/data_' + str(i) + '.csv', delimiter=',', header=None)
    pcmVector = pcmVector.values.flatten()[:8192]
    
    # remove the DC offset from the signal
    pcmVector = pcmVector - np.mean(pcmVector)
    
    # get zero crossings
    zeroCrossings = getZeroCrossings(pcmVector)
    # rotate the pcm signal to the first zero crossing
    pcmVector = np.roll(pcmVector, -zeroCrossings[0])
    
    # run spectral analysis
    analysis = pcmAnalyzer(pcmVector, adcSampleRate, adcResolution)
    
    # plot the analysis results to the pdf file
    pp.savefig(analysis.plotPowerSpectrum())

pp.close()

# open the pdf file
os.system('evince analysisResults.pdf')


                                    