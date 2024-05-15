#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pcmVector import pcmSignal
from scipy.fft import fft, fftfreq

# The pcmAnalyzer class is used to analyze the power spectrum of a PCM signal               
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
        self.noiseSpectrum = self.spectrum()
        self.fundamental = self.peak()
        self.quantization_noise = 20 * np.log10(1 / (2 ** (self.adcResolution - 1)))
        self.harmonics = []
        self._N = None
        self.THD = None
        self.THDN = None
        self.THDN_FS = None
        self.SNR = None
        self.SNDR = None
        self.SNDR_FS = None
        self.SFDR = None
        self.SFDR = None
        self.ENOB = None
        self.ENOB_FS = None       
        self.noisePower = None
        self.harmonicPower = None
        self.totNoiseAndDistortion = None
        self.fftNoiseFloor = self.quantization_noise - 10 * np.log10(self.num_samples/2)
#        self.fftNoiseFloor = self.getSNR_FS() - 10 * np.log10(self.num_samples/2)
        # print(f"FFT noise floor: {self.fftNoiseFloor:.1f} dBFS")
        
    class peak:
        def __init__(self, frequency=None, power=None):
            self.frequency = frequency
            self.power = power
            
        def __str__(self):
            return f"Frequency: {self.frequency}, Power: {self.power}"
    
    # inherit the peak class and add an index to the new class harmonic
    class harmonic(peak): 
        def __init__(self, frequency=None, power=None, index=None):
            super().__init__(frequency, power)
            self.index = index        # add harmonic index to the peak class
    
        
        def __str__(self):
            return f"Index: {self.index}, Frequency: {self.frequency}, Power: {self.power}"


    class spectrum:
        def __init__(self, frequency=None, level=None):
            self.frequency = frequency
            self.level = level
            
    
    def getMagnitudeSpectrum_numpy(self):
        if self.magnitude_spectrum.frequency is None:
            # Calculate the magnitude spectrum of the signal
            fft_result = np.fft.fft(self.signal)
            magnitude_spectrum = np.abs(fft_result) / self.num_samples      # Magnitude spectrum
            
            # compensate level for half side of the spectrum
            magnitude_spectrum = magnitude_spectrum * 2
            
            # convert the magnitude spectrum to dBFS
            self.magnitude_spectrum.level = 20 * np.log10(magnitude_spectrum / self.adcFS + 1e-1000)[:self.num_samples // 2]
            # Create frequency axis
            self.magnitude_spectrum.frequency = np.fft.fftfreq(self.num_samples, d=1/self.sampling_rate)[:self.num_samples // 2]
            
        return self.magnitude_spectrum
    
    def getMagnitudeSpectrum(self):
        # calculate the magnitude spectrum of the signal unsing scipy fft
        if self.magnitude_spectrum.frequency is None:
            # Calculate the magnitude spectrum of the signal
            fft_result = fft(self.signal)
            magnitude_spectrum = np.abs(fft_result) / self.num_samples
            
            # compensate level for half side of the spectrum
            magnitude_spectrum = magnitude_spectrum * 2
            
            # convert the magnitude spectrum to dBFS
            self.magnitude_spectrum.level = 20 * np.log10(magnitude_spectrum / self.adcFS + 1e-100)[:int(self.num_samples / 2 + 1)]
            
            self.magnitude_spectrum.level[0] = self.magnitude_spectrum.level[0]/2
            
            # Create frequency axis
            fstep = self.sampling_rate / self.num_samples # freq interval for each frequency bin (sampling frequency divided by number of samples)
            
            # Create freq steps –> x-axis for frequency spectrum
            self.magnitude_spectrum.frequency = np.linspace(0, (self.num_samples-1) * fstep, self.num_samples) 
            
            self.magnitude_spectrum.frequency = self.magnitude_spectrum.frequency[:int(self.num_samples / 2 + 1)]
            
            #self.magnitude_spectrum.frequency = fftfreq(self.num_samples, d=1/self.sampling_rate)[:int(self.num_samples /2 + 1)]
            
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
        # Plot the power spectrum of the signal and add a table with the results of the signal analysis in two subplots
        fig = plt.figure(figsize=(12, 6))
        
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2,1])
        ax1 = plt.subplot(gs[0])
        
        ax1.plot(self.getPowerSpectrum().frequency, self.getPowerSpectrum().level)
        ax1.set_title('Power Spectrum')
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Power [dBFS]')
        
        # set the y-axis limits to the average noise power - 10 dBFS and 10 dBFS
        plt.ylim([self.fftNoiseFloor, 10])
        
        # format the x-axis to show the frequency in kHz
        plt.xticks(np.arange(0, self.sampling_rate / 2 + 1e5, 1e5), 
                   [f'{f/1e3:.0f}' for f in np.arange(0, self.sampling_rate / 2 + 1e5, 1e5)])
        
        # include the fundamental frequency in the plot marked with a green ring and print the power level and frequency in kHz
        fundamental = self.getFundamental()
        plt.plot(fundamental.frequency, fundamental.power, 'go')
        plt.text(fundamental.frequency, fundamental.power, f'{fundamental.power:.1f} ' f'{fundamental.frequency / 1e3:.3f}kHz', fontsize=10, color='green')
        
        # inlude harmonics in the plot marked with red small dots, the harmonic power, and harmonic index
        harmonics = self.getHarmonics(100)
        # get the 20 harmonics with the highest power
        harmonics = sorted(harmonics, key=lambda x: x.power, reverse=True)[:30]
        
        for harmonic in harmonics:
            if harmonic.power > self.fftNoiseFloor:
                plt.plot(harmonic.frequency, harmonic.power, 'ro', markersize=2)
                plt.text(harmonic.frequency, harmonic.power, f'H{harmonic.index}', fontsize=6, color='red')
                
        plt.grid()
        ax2 = plt.subplot(gs[1])
        ax2.axis('off')
        ax2.set_title('Analysis Results')
        
        # Add textbox including the printouts in a table format
        text = [
            ['Param', 'Value'],
            ['Fundamental power:', f'{self.getFundamental().power:.1f} dBFS'],
            ['N_FS:', f'{self.get_N():.1f} dBFS'],
            ['THD:', f'{self.getTHDN():.1f} dBc'],
            ['THD+N FS:', f'{self.getTHDN_FS():.1f} dBFS'],
            ['SNR:', f'{self.getSNR():.1f} dBc'],
            ['SNDR FS:', f'{self.getSNDR_FS():.1f} dBFS'],
            ['SNDR:', f'{self.getSNDR():.1f} dBc'],
            ['SFDR:', f'{self.getSFDR():.1f} dBc'],
            ['ENOB FS:', f'{self.getENOB_FS():.1f} bits'],
            ['ENOB:', f'{self.getENOB():.1f} bits'],
            # print the 5 highest harmonics
            ['Harmonics:', ''],
            ['  Index', 'Power'],
        ]
        
        for harmonic in harmonics[:5]:
            text.append([f'  H{harmonic.index}', f'{harmonic.power:.1f} dBFS'])
            
                
        # create a table with the text in the upper left corner of the plot
        table = plt.table(cellText=text, loc='upper left', cellLoc='left', colWidths=[0.7, 0.3])
        # table font size
        plt.rc('font', size=13)
        
        #plt.tight_layout()
            
        #plt.show()
        # plot the noise spectrum of the signal in a subplot
        ax2 = plt.subplot(gs[2])
        plt.plot(self.noiseSpectrum.frequency, self.noiseSpectrum.level)
        plt.title('Noise Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [dBFS]')
        plt.ylim([self.fftNoiseFloor-20, self.fftNoiseFloor+40])
        plt.grid()
        return fig
        
    
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
    
    def getHarmonics(self, number_of_harmonics=None):
        if (number_of_harmonics is not None):
            if (len(self.harmonics) < number_of_harmonics):
                # find the harmonics of the fundamental frequency including aliases even if they are aliased multiple times.
                fundamental = self.getFundamental()
                harmonics = [fundamental.frequency * i for i in range(2, number_of_harmonics + 1)]
            
                # find the aliases of the harmonics
                self.harmonics = []
                # for
                for i, harmonic in enumerate(harmonics):
                    frequency = self.getAlias(harmonic)
                    power = self.findPeak(frequency).power
                    index = i + 2
                    self.harmonics.append(self.harmonic(frequency, power, index))
                    #print(f"Harmonic: {frequency/1e3:.3f} kHz, Power: {power:.1f} dBFS, Index: {index}")
            self.harmonics = self.harmonics[:number_of_harmonics]
        return self.harmonics
    
    
    def getNoisePower(self, excl_nrOfHarmonics):
        #if self.noisePower is None:
        # calculate the noise power in the power spectrum
        # remove the fundamental frequency and its harmonics from the power spectrum to estimate the noise power
        # remove the +- 1 frequency bins around the frequency of the fundamental frequency and its harmonics
        # replace the power of the removed bins with the average power of the bins before and after the removed bins
        pwrSpectrum = self.getPowerSpectrum().level.copy()
        peakArray = self.getHarmonics(excl_nrOfHarmonics)
        peakArray.append(self.getFundamental())
        
        # rotate the peakArray to start with the fundamental frequency
        peakArray = sorted(peakArray, key=lambda x: x.frequency)
        
        for peak in peakArray:
            print(f"Peak: {peak.frequency/1e3:.3f} kHz, Power: {peak.power:.1f} dBFS")
            index = np.where(self.getPowerSpectrum().frequency == peak.frequency)
            print(f"Index: {index}")
            # remove the +- 1 frequency bins around the frequency of the fundamental frequency and its harmonics
            pwrSpectrum[index - 10: index + 10] = -150 # replace the power of the removed bins with -150 dBFS
            
            #pwrSpectrum[index - 1: index + 1] = np.mean([pwrSpectrum[index - 2], pwrSpectrum[index + 2]]) # replace the power of the removed bins with the average power of the bins before and after the removed bins
        
        # calculate the noise power as the average power of the power spectrum
        pwrSpectrumLin = 10 ** (pwrSpectrum / 10)
        self.noisePower = 10 * np.log10(np.sum(pwrSpectrumLin))
        
        return self.noisePower, pwrSpectrum
    
    def get_N(self):
        if self._N is None:
            
            self.noiseSpectrum.frequency = self.getPowerSpectrum().frequency
            self._N, self.noiseSpectrum.level = self.getNoisePower(50)
        return self._N
        
    def getSNR(self):
        # Signal-to-Noise Ratio (SNR) is the ratio of the power of the fundamental
        # to the power of the noise excluding the harmonics
        
        # SNR_FS = 10 * log10(P1 / Pn)
        # where P1 is the power of the fundamental frequency and Pn is the power of the noise
        if self.SNR is None:
            self.SNR = self.getFundamental().power - self.get_N()
        return self.SNR
    
    def getSNR_FS(self):
        if self.SNR is None:
            self.SNR = self.get_N()
        return self.SNR
    
    def getTHD(self, number_of_harmonics=50):
        # THD is defined as the sum of the powers of all harmonics
        if self.THD is None:
            self.THD = sum([10 ** (harmonic.power / 10) for harmonic in self.getHarmonics(number_of_harmonics)])
            self.THD = 10 * np.log10(self.THD)
            
        return self.THD
            
    def getTHDN_FS_old(self, number_of_harmonics=100):
        # THD+N is defined as the sum of the powers of all harmonics and noise
        # Noise power is calculated as the average power of the power spectrum excluding the fundamental frequency and its harmonics
        # The unit of the Noise power is dBFS and the unit of the THD is dBFS
        if self.THDN_FS is None:
            # convert the noise power from dBFS to linear scale
            _N_Linear = 10 ** (self.get_N() / 10)
            
            # Convert the THD from dBFS to linear scale
            THD_linear = 10 ** (self.getTHD(number_of_harmonics) / 10)
            
            # calculate the THD+N in linear scale
            THDN_linear = (THD_linear + _N_Linear)
            #THDN_linear = np.sqrt(THD_linear ** 2 + noise_power_linear)
            
            # convert the THD+N back to dBFS
            self.THDN_FS = 10 * np.log10(THDN_linear)
            
        return self.THDN_FS
    
    
    def getTHDN_FS(self, number_of_harmonics=100):
        # THD+N_FS is defined as the sum of the powers of all harmonics and noise
        if self.THDN_FS is None:
            self.THDN_FS, d = self.getNoisePower(0)
        return self.THDN_FS
    
    
    def getTHDN(self):
        if self.THDN is None:
            self.THDN = self.getTHDN_FS() - self.getFundamental().power
        return self.THDN
    
            
    def getSNDR(self):
        # Signal-to-Noise-and-Distortion (SNDR, or S/(N + D)
        # is the ratio of the rms signal amplitude to the mean value of the 
        # root-sum-square (rss) of all other spectral components, including 
        # harmonics, but excluding dc.
        
        if self.SNDR is None:
            ND_power = self.getTHDN()
            self.SNDR = self.getFundamental().power - ND_power
        return self.SNDR
    
    
    def getSNDR_FS(self):
        # SNDR_FS is defined as the ratio of the power of the fundamental frequency to the sum of the powers of all other frequencies in dBFS
        # SNDR_FS = 10 * log10(P1 / (P2 + P3 + ... + Pn + Pn+1 + ...))
        # where P1 is the power of the fundamental frequency and P2, P3, ..., Pn, Pn+1, ... are the powers of the harmonics and noise
        if self.SNDR_FS is None:
            ND_power = self.getTHDN_FS()
            self.SNDR_FS = self.getFundamental().power - ND_power
        return self.SNDR_FS
    
    
    def getSFDR(self):
        # SFDR is defined as the ratio of the power of the fundamental frequency to the power of the highest harmonic in dBFS
        # SFDR = P1 - Pn
        # where P1 is the power of the fundamental frequency and Pn is the power of the highest harmonic
        if self.SFDR is None:
            self.SFDR = self.getFundamental().power - max([harmonic.power for harmonic in self.getHarmonics(100)])
        return self.SFDR
            
    
    def getENOB_FS(self):
        # ENOB_FS is defined as the ratio of the SNDR to the quantization noise level in dBFS
        # ENOB_FS = (SNDR - 1.76) / 6.02
        if self.ENOB_FS is None:
            # signalAmplitude = 1908 #20 ** (self.getFundamental().power / 20) * self.adcFS
            
            # signalPowerFS = 20 * np.log10(self.adcFS / signalAmplitude)
            # print (f"Signal amplitude: {signalAmplitude:.1f}")
            # print (f"ADC full scale: {self.adcFS:.0f}")
            # print (f"Signal power FS: {signalPowerFS:.1f} dBFS")
            self.ENOB_FS = (self.getSNDR_FS() - 1.76 - self.getFundamental().power) / 6.02
            
        return self.ENOB_FS
    
    
    def getENOB(self):
        # ENOB is defined as the ratio of the SNDR to the quantization noise level in dBc
        # ENOB = (SNDR - 1.76) / 6.02
        if self.ENOB is None:
            self.ENOB = (self.getSNDR() - 1.76 - self.getFundamental().power) / 6.02   
        return self.ENOB
    
    
    def printAll(self):
        # print results from the signal analysis class functions
        print("\n************* Signal Analysis results: *************")
        # print the fundamental frequency and power
        print("Fundamental:")
        print (f"Frequency = {self.getFundamental().frequency/1e3:.6f} kHz, Power = {self.getFundamental().power:.1f} dBFS")
        # print the harmonic indexes, frequencies, and powers in a tabular format.
        # use constant width columns for the table
        print("\n************* Harmonics: *************")
        print(f"{'Index':<6}{'Frequency':<15}{'Power':<10}")
        for harmonic in self.getHarmonics(5):
            print(f"{harmonic.index:<6}{harmonic.frequency/1e3:<15.6f}{harmonic.power:<10.1f}")
                
        print(f"\n{'N:':<21}{self.get_N():<5.1f} dBFS")
        print(f"{len(self.getHarmonics())}{' harmonics power:':<18} {self.getTHD(5):<5.1f} dBFS")
        print(f"{'THD+N FS:':<21}{self.getTHDN_FS():<5.1f} dBFS")
        print(f"{'THD+N:':<21}{self.getTHDN():<5.1f} dBc")
        print(f"{'SNDR FS:':<21}{self.getSNDR_FS():<5.1f} dBFS")
        print(f"{'SNDR:':<21}{self.getSNDR():<5.1f} dBc")
        print(f"{'SFDR FS:':<21}{self.getSFDR():<5.1f} dBFS")
        print(f"{'ENOB FS:':<21}{self.getENOB_FS():<5.1f} bits")
        print(f"{'ENOB:':<21}{self.getENOB():<5.1f} bits")
        print("\n************* End of Signal Analysis results *************")
        
        
def main():
    #plt.close('all')
    
    #from testData import pcmSignal as generate_pcm_signal_with_harmonics
    pcmSgl = pcmSignal(frequency= 10e3,
                       amplitude=0, 
                       sampling_rate=1e6, 
                       adcResolution=12,
                       harmonic_levels=[-60, -65, -70, -75, -80]
                       )
    
    pcmSgl.printSignalSpecs()
    #pcmSgl.plotPCMVector()
    
    # initialize an object of the pcmSignalAnalyser class with the generated pcmSgl as input
    analysis = pcmAnalyzer(pcmSgl.pcmVector, pcmSgl.sampling_rate, pcmSgl.adcResolution)
    
    # print the results of the signal analysis
    analysis.printAll()
    # plot the power spectrum of the signal
    analysis.plotPowerSpectrum()
    
    
    plt.show()
    
    
    
     
if __name__ == '__main__':
    main()
    