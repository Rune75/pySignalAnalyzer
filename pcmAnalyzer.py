#!/usr/bin/env python3


# Define a class for the PCM signal analysis with:
# input signal: the PCM signal and required input parameters for sampled signal analysis
# output: the analysis results including Distortion, Noise, SMR and dynamic range preformance using typical adc preformance metrics
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pcmVector import pcmSignal

# define a variable of type pcmSignal
# inputSignal = pcmSignal()
               
class pcmAnalyzer:
    def __init__(self, signal=None, sampling_rate=None, adcResolution=None): 
        self.signal = signal
        self.sampling_rate =sampling_rate
        self.adcResolution = adcResolution
        
        self.adcFS = 2 ** (adcResolution - 1)
        self.num_samples = len(signal)
        self.power_spectrum = self.spectrum()
        self.powerSpectrumLinear = self.spectrum()
        self.magnitude_spectrum = self.spectrum()
        self.fundamental = self.peak()
        self.quantization_noise = 20 * np.log10(1 / (2 ** (self.adcResolution - 1)))
        self.harmonics = []
        self.THD = None
        self.THDN = None
        self.THDN_FS = None
        self.SNR = None
        self.SINAD = None
        self.SINAD_FS = None
        self.SFDR = None
        self.SFDR_FS = None
        self.ENOB = None
        self.ENOB_FS = None       
        self.noisePower = None
        self.harmonicPower = None
        self.totNoiseAndDistortion = None
        
    class peak:
        def __init__(self, frequency=None, power=None):
            self.frequency = frequency
            self.power = power

    class spectrum:
        def __init__(self, frequency=None, level=None):
            self.frequency = frequency
            self.level = level
    
    def getMagnitudeSpectrum(self):
        if self.magnitude_spectrum.frequency is None:
            # Calculate the magnitude spectrum of the signal
            fft_result = np.fft.fft(self.signal)                # FFT of the signal
            magnitude_spectrum = np.abs(fft_result) / self.num_samples   # Magnitude spectrum
            # compensate level for half side of the spectrum
            magnitude_spectrum = magnitude_spectrum * 2
            # convert the magnitude spectrum to dBFS
            self.magnitude_spectrum.level = 20 * np.log10(magnitude_spectrum / self.adcFS + 1e-100)[:self.num_samples // 2]
            # Create frequency axis
            self.magnitude_spectrum.frequency = np.fft.fftfreq(self.num_samples, d=1/self.sampling_rate)[:self.num_samples // 2]
            
        return self.magnitude_spectrum
                    
    def getPowerSpectrum(self):
        if self.power_spectrum.frequency is None:
            # calculate power spectrum from the magnitude spectrum given in dBFS
            self.power_spectrum.frequency = self.getMagnitudeSpectrum().frequency
            self.power_spectrum.level = 10 ** (self.getMagnitudeSpectrum().level / 20)
            self.power_spectrum.level = self.power_spectrum.level ** 2
            self.power_spectrum.level = 10 * np.log10(self.power_spectrum.level)
            
        return self.power_spectrum
            
        
    def plotPowerSpectrum(self):
        # Plot the power spectrum of the signal
        plt.figure()
        plt.plot(self.getPowerSpectrum().frequency, self.getPowerSpectrum().level)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Power (dBFS)')
        plt.title('Power Spectrum')
        
        # set the y-axis limits to the average noise power - 10 dBFS and 10 dBFS
        plt.ylim([self.getNoisePower() - 40, 10])
        
        # format the x-axis to show the frequency in kHz
        plt.xticks(np.arange(0, self.sampling_rate / 2 + 1e5, 1e5), 
                   [f'{f/1e3:.0f}' for f in np.arange(0, self.sampling_rate / 2 + 1e5, 1e5)])
        
        # include the fundamental frequency in the plot marked with a green ring
        plt.plot(self.getFundamental().frequency, self.getFundamental().power, 'go')
        plt.text(self.getFundamental().frequency, self.getFundamental().power, 
                 f' {self.getFundamental().power:.1f}', fontsize=10, color='green')
    
        
        # inlude harmonics in the plot marked with red dots, the harmonic power, and harmonic index
        harmonics = self.getHarmonics(10)
        for harmonic in harmonics:
            i = harmonics.index(harmonic) + 2
            plt.plot(harmonic.frequency, harmonic.power, 'ro')
            plt.text(harmonic.frequency, harmonic.power, f'_H{i}: {harmonic.power:.1f}', fontsize=10, color='red')

        plt.grid()
        
        # Add textbox including the printouts in a table format
        text = [
            ['Param', 'Value'],
            ['Noise Power:', f'{self.getNoisePower():.1f} dBFS'],
            ['Harmonic power THD:', f'{self.getTHD(5):.1f} dBFS'],
            ['THD+N FS:', f'{self.getTHDN_FS():.1f} dBFS'],
            ['THD+N:', f'{self.getTHDN():.1f} dBc'],
            ['SINAD FS:', f'{self.getSINAD_FS():.1f} dBFS'],
            ['SINAD:', f'{self.getSINAD():.1f} dBc'],
            ['SFDR FS:', f'{self.getSFDR_FS():.1f} dBFS'],
            ['ENOB FS:', f'{self.getENOB_FS():.1f} bits'],
            ['ENOB:', f'{self.getENOB():.1f} bits'],
            ['THD:', f'{self.getTHD():.1f} dB']
            ]
        
        # table font size
        plt.rc('font', size=15)
        # create a table with the text in the upper left corner of the plot
        # the table is 60% of the plot width and 50% of the plot height
        table = plt.table(cellText=text, loc='upper left', cellLoc='left', bbox=[0, 0.4, 0.6, 0.6])
        
        # make the edges of the table white
        cells = table.properties()['children']
        for cell in cells:
            cell.set_edgecolor('white')
            
        
        # show the plot
        plt.show()
        
    
    def getFundamental(self):
        if self.fundamental.frequency is None:
            # find the peaks of the power spectrum
            peaks, _ = find_peaks(self.getPowerSpectrum().level, height=-100)
            # find the peak with the highest power
            self.fundamental.power = max(self.getPowerSpectrum().level[peaks])
            index = np.where(self.getPowerSpectrum().level == self.fundamental.power)[0][0]
            # find the frequency of the peak
            self.fundamental.frequency = self.getPowerSpectrum().frequency[index]
        return self.fundamental
        
    def findPeak(self, frequency):
        # find the highest peak in the power spectrum within +-3 frequency bins of the given frequency
        # find the closest frequency bin to the given frequency
        index = np.argmin(np.abs(self.getPowerSpectrum().frequency - frequency))
        
        # find the peak with the highest power within +-3 frequency bins of the given frequency
        top = self.peak()
        top.power = max(self.getPowerSpectrum().level[index - 2: index + 2])
        # find the frequency of the peak
        top.frequency = self.getPowerSpectrum().frequency[np.where(self.getPowerSpectrum().level == top.power)[0][0]]    
        return top
    
    def getAlias(self, harmonic_frequency):
        # find the alias of a harmonic frequency
        Nyquist = self.sampling_rate / 2
        alias = abs(harmonic_frequency - self.sampling_rate)
        while alias > Nyquist:
            alias -= self.sampling_rate
        return abs(alias)
    
    def getHarmonics(self, number_of_harmonics=0):
        if (number_of_harmonics > 0):
            if (len(self.harmonics) < number_of_harmonics):
                # find the harmonics of the fundamental frequency including aliases even if they are aliased multiple times.
                # find the harmonics of the fundamental frequency
                fundamental = self.getFundamental()
                harmonics = [fundamental.frequency * i for i in range(2, number_of_harmonics + 1)]
            
                # find the aliases of the harmonics
                self.harmonics = []
                for harmonic in harmonics:
                    frequency = self.getAlias(harmonic)
                    power = self.findPeak(frequency).power
                    self.harmonics.append(self.peak(frequency, power))
                    #print(f"Harmonic: {frequency/1e3:.3f} kHz, Power: {power:.1f} dBFS")
            self.harmonics = self.harmonics[:number_of_harmonics]
        return self.harmonics
    
    
    def getNoisePower(self):
        if self.noisePower is None:
            # calculate the noise power in the power spectrum
            # remove the fundamental frequency and its harmonics from the power spectrum to estimate the noise power
            # remove the +- 1 frequency bins around the frequency of the fundamental frequency and its harmonics
            # replace the power of the removed bins with the average power of the bins before and after the removed bins
            pwrSpectrum = self.getPowerSpectrum().level.copy()
            peakArray = self.getHarmonics(100).copy()
            peakArray.append(self.getFundamental())
            
            for peak in peakArray:
                index = np.where(self.getPowerSpectrum().frequency == peak.frequency)[0][0]
                pwrSpectrum[index - 1: index + 1] = np.mean([pwrSpectrum[index - 2], pwrSpectrum[index + 2]])
            # calculate the noise power as the average power of the power spectrum
            
            
            pwrSpectrum = 10 ** (pwrSpectrum / 10)
            self.noisePower = 10 * np.log10(np.sum(pwrSpectrum))
            #self.noisePower = np.mean(pwrSpectrum)
            
        return self.noisePower
    
    def getTHD(self, number_of_harmonics=0):
        # THD is defined as the sum of the powers of all harmonics
        if self.THD is None:
            self.THD = sum([10 ** (harmonic.power / 10) for harmonic in self.getHarmonics(number_of_harmonics)])
            self.THD = 10 * np.log10(self.THD)
            
        return self.THD
    
    def getTHDN_FS(self):
        # THD+N is defined as the sum of the powers of all harmonics and noise
        # Noise power is calculated as the average power of the power spectrum excluding the fundamental frequency and its harmonics
        # The unit of the Noise power is dBFS and the unit of the THD is dBFS
        if self.THDN_FS is None:
            # convert the noise power from dBFS to linear scale
            noise_power_linear = 10 ** (self.getNoisePower() / 10)
            
            # Convert the THD from dBFS to linear scale
            THD_linear = 10 ** (self.getTHD(100) / 10)
            
            # calculate the THD+N in linear scale
            THDN_linear = (THD_linear + noise_power_linear)
            #THDN_linear = np.sqrt(THD_linear ** 2 + noise_power_linear)
            
            # convert the THD+N back to dBFS
            self.THDN_FS = 10 * np.log10(THDN_linear)
            
        return self.THDN_FS
    
    def getTHDN(self):
        if self.THDN is None:
            self.THDN = self.getTHDN_FS() - self.getFundamental().power
        return self.THDN
    
            
    def getSINAD(self):
        # SINAD is defined as the ratio of the power of the fundamental frequency to the sum of the powers of all other frequencies
        # SINAD = 10 * log10(P1 / (P2 + P3 + ... + Pn + Pn+1 + ...))
        # where P1 is the power of the fundamental frequency and P2, P3, ..., Pn, Pn+1, ... are the powers of the harmonics and noise
        if self.SINAD is None:
            ND_power = self.getTHDN()
            self.SINAD = self.getFundamental().power - ND_power
        return self.SINAD
    
    def getSINAD_FS(self):
        # SINAD_FS is defined as the ratio of the power of the fundamental frequency to the sum of the powers of all other frequencies in dBFS
        # SINAD_FS = 10 * log10(P1 / (P2 + P3 + ... + Pn + Pn+1 + ...))
        # where P1 is the power of the fundamental frequency and P2, P3, ..., Pn, Pn+1, ... are the powers of the harmonics and noise
        if self.SINAD_FS is None:
            ND_power = self.getTHDN_FS()
            self.SINAD_FS = self.getFundamental().power - ND_power
        return self.SINAD_FS
    
    def getSFDR_FS(self):
        # SFDR_FS is defined as the ratio of the power of the fundamental frequency to the power of the highest harmonic in dBFS
        # SFDR_FS = P1 - Pn
        # where P1 is the power of the fundamental frequency and Pn is the power of the highest harmonic
        if self.SFDR_FS is None:
            self.SFDR_FS = self.getFundamental().power - max([harmonic.power for harmonic in self.getHarmonics(100)])
        return self.SFDR_FS
            
    
    def getENOB_FS(self):
        # ENOB_FS is defined as the ratio of the SINAD to the quantization noise level in dBFS
        # ENOB_FS = (SINAD - 1.76) / 6.02
        if self.ENOB_FS is None:
            self.ENOB_FS = (self.getSINAD_FS() - 1.76) / 6.02
            
        return self.ENOB_FS
    
    
    def getENOB(self):
        # ENOB is defined as the ratio of the SINAD to the quantization noise level in dBc
        # ENOB = (SINAD - 1.76) / 6.02
        if self.ENOB is None:
            self.ENOB = (self.getSINAD() - 1.76) / 6.02
            
        return self.ENOB
    
    
    def printAll(self):
        # print results from the signal analysis class functions
        print("\n************* Signal Analysis results: *************")
        # print the fundamental frequency and power
        print("Fundamental:")
        print (f"Frequency = {self.getFundamental().frequency/1e3:.3f} kHz, Power = {self.getFundamental().power:.1f} dBFS")
        # print the harmonic indexes, frequencies, and powers in a tabular format.
        # use constant width columns for the table
        print("\n************* Harmonics: *************")
        print(f"{'Index':<6}{'Frequency[kHz]':<15}{'Power[dBFS]':<15}")
        for i, harmonic in enumerate(self.getHarmonics(5)):
            print(f"{i+2:<6}{harmonic.frequency/1e3:<15.3f}{harmonic.power:<15.1f}")
            
        print(f"\n{'Noise Power:':<21}{self.getNoisePower():<5.1f} dBFS")
        print(f"{len(self.getHarmonics())}{' harmonics power:':<18} {self.getTHD(5):<5.1f} dBFS")
        print(f"{'THD+N FS:':<21}{self.getTHDN_FS():<5.1f} dBFS")
        print(f"{'THD+N:':<21}{self.getTHDN():<5.1f} dBc")
        print(f"{'SINAD FS:':<21}{self.getSINAD_FS():<5.1f} dBFS")
        print(f"{'SINAD:':<21}{self.getSINAD():<5.1f} dBc")
        print(f"{'SFDR FS:':<21}{self.getSFDR_FS():<5.1f} dBFS")
        print(f"{'ENOB FS:':<21}{self.getENOB_FS():<5.1f} bits")
        print(f"{'ENOB:':<21}{self.getENOB():<5.1f} bits")
        print("\n************* End of Signal Analysis results *************")
        
        
def main():
    #plt.close('all')
    
    #from testData import pcmSignal as generate_pcm_signal_with_harmonics
    pcmSgl = pcmSignal(frequency=227e3, 
                       amplitude=0, 
                       sampling_rate=2e6, 
                       adcResolution=8,
                       harmonic_levels=[-60, -60, -60, -60]
                       )
    
    pcmSgl.printSignalSpecs()
    #pcmSgl.plotPCMVector()
    
    # initialize an object of the pcmSignalAnalyser class with the generated pcmSgl as input
    analysis = pcmAnalyzer(pcmSgl.pcmVector, pcmSgl.sampling_rate, pcmSgl.adcResolution)
    #analysis.getMagnitudeSpectrum()
    #analysis.getPowerSpectrum()
    #3analysis.plotPowerSpectrum()
    # analysis.getFundamental()
    # print(f"Fundamental frequency: {analysis.getFundamental().frequency/1e3:.3f} kHz, Power: {analysis.getFundamental().power:.1f} dBFS")
    # analysis.getHarmonics(5)
    # 
    # print the results of the signal analysis
    #analysis.printAll()
    
    # plot the power spectrum of the signal
    analysis.plotPowerSpectrum()
     
if __name__ == '__main__':
    main()
    