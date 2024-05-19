
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcmAnalyzer import pcmAnalyzer

adcResolution = 12
adcSampleRate = 1e6


pp = PdfPages('analysisResults.pdf')
# read and plot al the data files
for i in range(1, 25):
    pcmVector = pd.read_csv('data/data_' + str(i) + '.csv', delimiter=',', header=None)
    pcmVector = pcmVector.values.flatten()[:8192]
    
    # remove the DC offset from the signal
    pcmVector = pcmVector - np.mean(pcmVector)
    
    # run spectral analysis
    analysis = pcmAnalyzer(pcmVector, adcSampleRate, adcResolution)
    
    # plot the analysis results to the pdf file.
    ax = analysis.plotPowerSpectrum()
    # add text to the plot
    ax.text(0.7, 0.1, 'File: data_' + str(i) + '.csv')
    pp.savefig(ax.figure, bbox_inches='tight')
    
pp.close()

# open the pdf file
os.system('evince analysisResults.pdf')


                                    