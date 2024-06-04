
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signalAnalyzer import signalAnalyzer


def get25LastFiles(folder_path):
    # Get all the filenames in the folder
    filenames = os.listdir(folder_path)

    # order the filenames in ascending order by the number in the filename
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=False)

    # Get the 25 highest numbered filenames
    return sorted_filenames


adcResolution = 12
adcSampleRate = 1e6


pp = PdfPages('analysisResults.pdf')
# read and plot al the data files
for i in range(0, 26):

    # Load the PCM data from NI
    folder_path = '/home/rune/work/githubRune75/signalAnalyzer/data/ATE'
    files = get25LastFiles(folder_path)
    pcmVector = pd.read_csv(folder_path + '/' + files[i-1], delimiter=',', header=None)
    
    pcmVector = pcmVector.values.flatten()[:8192]
    
    # remove the DC offset from the signal
    pcmVector = pcmVector - np.mean(pcmVector)
    
    # run spectral analysis
    analysis = signalAnalyzer(pcmVector, adcSampleRate, adcResolution)
    
    # plot the analysis results to the pdf file.
    ax = analysis.plotPowerSpectrum()
    # add text to the plot
    ax.text(0.7, 0.1, 'File: data_' + str(i) + '.csv')
    pp.savefig(ax.figure, bbox_inches='tight')
    
pp.close()

# open the pdf file
os.system('evince analysisResults.pdf')


                                    
