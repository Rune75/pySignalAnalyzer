#!/usr/bin/env python3


# Define a class for the PCM signal analysis with:
# input signal: the PCM signal and required input parameters for sampled signal analysis
# output: the analysis results including Distortion, Noise, SMR and dynamic range preformance using typical adc preformance metrics
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from testData import pcmSignal

# define a variable of type pcmSignal
inputSignal = pcmSignal
               
class pcmAnalyzer:
    class peak:
        def __init__(self, frequency=None, power=None):
            self.frequency = frequency
            self.power = power

    class powerSpectrum:
        def __init__(self, frequency=None, power=None):
            self.frequency = frequency
            self.power = power

    
    def __init__(self, inputSignal): 
        # Remove the redundant line that initializes the inputSignal variable
        self.signal = inputSignal.pcmVector
        self.sampling_rate = inputSignal.sampling_rate
        self.adcResolution = inputSignal.adcResolution
        self.adcFS = 2 ** inputSignal.adcResolution - 1
        self.amplitude = inputSignal.amplitude
        self.power_spectrum = self.powerSpectrum()
        self.fundamental = self.peak()
        self.quantization_noise = 20 * np.log10(1 / (2 ** (self.adcResolution - 1)))
        # self.harmonics is a list of peak objects
        # initialize the list of harmonics
        self.harmonics = []
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
  
        
    def getPowerSpectrum(self):
        if self.power_spectrum.frequency is None:
            # calculate power spectrum
            # frequency vector
            self.power_spectrum.frequency = np.fft.fftfreq(len(self.signal), 1/self.sampling_rate)
            # scale with input sample length
            power_spectrum = np.abs(np.fft.fft(self.signal)) / len(self.signal)
            # Convert to dBFS
            power_spectrum = 20 * np.log10(power_spectrum / (self.adcFS / 1) + 1e-10)
            # convert to one sided spectrum
            power_spectrum = power_spectrum[:len(self.power_spectrum.frequency)//2] + 6   # only positive frequencies are valid
            self.power_spectrum.frequency = self.power_spectrum.frequency[:len(self.power_spectrum.frequency)//2]   # only positive frequencies are valid   
            self.power_spectrum.power = power_spectrum
            
        return self.power_spectrum
        
    def plotPowerSpectrum(self):
        # Plot the power spectrum of the signal
        plt.figure()
        plt.plot(self.getPowerSpectrum().frequency, self.getPowerSpectrum().power)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dBFS)')
        plt.title('Power Spectrum of the Signal')
        # qunatization noise level for a sampled signal with given ADC resolution and amplitude is calculated as:
        # 20 * log10(1 / (2 ** (adcResolution - 1))) dBFS
        quantization_noise = 20 * np.log10(1 / (2 ** (self.adcResolution - 1)))
        print(f"Quantization Noise Level: {quantization_noise} dBFS")
        
        # set the y-axis limits to the quantization noise level and the full scale level of the ADC
        plt.ylim(quantization_noise - 60, 10)
        # format the x-axis to show the frequency in kHz
        plt.xticks(np.arange(0, self.sampling_rate / 2 + 1e5, 1e5), [f'{f/1e3:.0f}' for f in np.arange(0, self.sampling_rate / 2 + 1e5, 1e5)])
        
        # include the fundamental frequency in the plot marked with a green ring
        plt.plot(self.getFundamental().frequency, self.getFundamental().power, 'go')
        
        # inlude harmonics in the plot marked with red dots
        for harmonic in self.getHarmonics():
            plt.plot(harmonic.frequency, harmonic.power, 'ro')
        
        plt.grid()
        plt.show()
        
    
    def getFundamental(self):
        if self.fundamental.frequency is None:
            # find the peaks of the power spectrum
            peaks, _ = find_peaks(self.getPowerSpectrum().power, height=-100)
            # find the peak with the highest power
            self.fundamental.power = max(self.getPowerSpectrum().power[peaks])
            index = np.where(self.getPowerSpectrum().power == self.fundamental.power)[0][0]
            # find the frequency of the peak
            self.fundamental.frequency = self.getPowerSpectrum().frequency[index]
        return self.fundamental
        
    def findPeak(self, frequency):
        # find the highest peak in the power spectrum within +-3 frequency bins of the given frequency
        # find the index of the given frequency
        index = np.where(self.getPowerSpectrum().frequency == frequency)[0][0]
        # find the peak with the highest power within +-3 frequency bins of the given frequency
        top = self.peak()
        top.power = max(self.getPowerSpectrum().power[index - 2: index + 2])
        # find the frequency of the peak
        top.frequency = self.getPowerSpectrum().frequency[np.where(self.getPowerSpectrum().power == top.power)[0][0]]    
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
            self.harmonics = self.harmonics[:number_of_harmonics]
        return self.harmonics
    
    
    def getNoisePower(self):
        if self.noisePower is None:
            # calculate the noise power in the power spectrum
            # remove the fundamental frequency and its harmonics from the power spectrum to estimate the noise power
            # remove the +- 1 frequency bins around the frequency of the fundamental frequency and its harmonics
            # replace the power of the removed bins with the average power of the bins before and after the removed bins
            pwrSpectrum = self.getPowerSpectrum().power.copy()
            for harmonic in self.getHarmonics(100):
                index = np.where(self.getPowerSpectrum().frequency == harmonic.frequency)[0][0]
                pwrSpectrum[index - 1: index + 2] = np.mean([pwrSpectrum[index - 2], pwrSpectrum[index + 2]])
            # calculate the noise power as the average power of the power spectrum
            self.noisePower = np.mean(pwrSpectrum)
            
        return self.noisePower
    
    def getHarmonicPower(self, number_of_harmonics=0):
        # calculate the total power of the harmonics
        if self.harmonicPower is None:
            self.harmonicPower = sum([10 ** (harmonic.power / 10) for harmonic in self.getHarmonics(number_of_harmonics)])
            self.harmonicPower = 10 * np.log10(self.harmonicPower)
            
        return self.harmonicPower
    
    def getTHDN_FS(self):
        if self.THDN_FS is None:

            # Convert dBFS to linear scale
            fundamental_power = 10 ** (self.getFundamental().power / 20)
            harmonic_powers = [10 ** (harmonic.power / 20) for harmonic in self.getHarmonics(100)]
            
            # Calculate THD
            thd = np.sqrt(np.sum(np.array(harmonic_powers) ** 2) / fundamental_power ** 2) # THD in linear scale

            # Calculate noise power
            noise_power = 10 ** (self.getNoisePower() / 10)
            
            # Calculate THD+N in dBFS
            self.THDN_FS = 20 * np.log10(np.sqrt(thd ** 2 + noise_power))

        return self.THDN_FS
    
    def getTHDN(self):
        if self.THDN is None:
            # Convert THDN_FS from dBFS to dBc
            self.THDN = self.getTHDN_FS() - self.getFundamental().power 
        return self.THDN
    
     
    def getSINAD_FS(self):
        # SINAD_FS is defined as the ratio of the power of the fundamental frequency to the sum of the powers of all other frequencies
        # SINAD_FS = 10 * log10(P1 / (P2 + P3 + ... + Pn + Pn+1 + ...))
        # the difference between SINAD_FS and SINAD is that SINAD_FS is referenced to full scale 
        # and SINAD is referenced to the fundamental power
        if self.SINAD_FS is None:
            fundamental_power = 10 ** (self.getFundamental().power / 10)
            harmonic_powers = [10 ** (harmonic.power / 10) for harmonic in self.getHarmonics(100)]
            self.SINAD_FS = 10 * np.log10(fundamental_power / sum(harmonic_powers))
        return self.SINAD_FS
    
    
    def getSINAD(self):
        # SINAD is defined as the ratio of the power of the fundamental frequency to the sum of the powers of all other frequencies
        # SINAD = 10 * log10(P1 / (P2 + P3 + ... + Pn + Pn+1 + ...))
        # the difference between SINAD and SINAD_FS is that SINAD_FS is referenced to full scale 
        # and SINAD is referenced to the fundamental power. to calculate SINAD from SINAD_FS and fundamental power:
        # SINAD = SINAD_FS + fundamental power
        if self.SINAD is None:
            self.SINAD = self.getSINAD_FS() + self.getFundamental().power
        return self.SINAD
    
    
    def getSFDR_FS(self):
        # SFDR is defined as the ratio of the power of the fundamental frequency to the power of the largest harmonic
        # SFDR = 10 * log10(P1 / Pn)
        # where P1 is the power of the fundamental frequency and Pn is the power of the largest harmonic
        if self.SFDR_FS is None:
            fundamental_power = 10 ** (self.getFundamental().power / 10)
            harmonics_power = [10 ** (harmonic.power / 10) for harmonic in self.getHarmonics(100)]
            self.SFDR_FS = 10 * np.log10(fundamental_power / max(harmonics_power))
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
        print(f"{len(self.getHarmonics())}{' harmonics power:':<18} {self.getHarmonicPower(5):<5.1f} dBFS")
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
    pcmSgl = pcmSignal(frequency=179e3, amplitude=-6, sampling_rate=500e3,
                       adcResolution=12, harmonic_levels=[-80, -72, -95, -90, -85, -50, -60, -70])
    pcmSgl.printSignalSpecs()
    #pcmSgl.plotPCMVector()
    
    # initialize an object of the pcmSignalAnalyser class with the generated pcmSgl as input
    analysis = pcmAnalyzer(pcmSgl)
    #analysis.getPowerSpectrum()
    #analysis.getFundamental()
    #analysis.getHarmonics(5)
    
    # print the results of the signal analysis
    analysis.printAll()
    
    # plot the power spectrum of the signal
    #analysis.plotPowerSpectrum()
     
if __name__ == '__main__':
    main()
    
    
    