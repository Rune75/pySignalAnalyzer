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
        power = None
        frequency = None

    
    def __init__(self, inputSignal): 
        # Remove the redundant line that initializes the inputSignal variable
        self.signal = inputSignal.pcmVector
        self.sampling_rate = inputSignal.sampling_rate
        self.adcResolution = inputSignal.adcResolution
        self.adcFS = 2 ** inputSignal.adcResolution - 1
        self.amplitude = inputSignal.amplitude
        self.power_spectrum = self.powerSpectrum
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
        self.ENOB = None
        self.ENOB_FS = None       
        self.noisePower = None 
  
        
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
    
    def getHarmonics(self, number_of_harmonics=3):
        if len(self.harmonics) == 0 or len(self.harmonics) < number_of_harmonics:
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
                
        return self.harmonics[:number_of_harmonics]
    
    
    def getNoisePower(self):
        if self.noisePower is None:
            # calculate the noise power in the power spectrum
            # remove the fundamental frequency and its harmonics from the power spectrum to estimate the noise power
            # remove the +- 1 frequency bins around the frequency of the fundamental frequency and its harmonics
            # replace the power of the removed bins with the average power of the bins before and after the removed bins
            power_spectrum = self.getPowerSpectrum().power
            fundamental = self.getFundamental()
            harmonics = self.getHarmonics(50)
            for harmonic in [fundamental] + harmonics:
                index = np.where(self.getPowerSpectrum().frequency == harmonic.frequency)[0][0]
                power_spectrum[index - 1: index + 2] = np.mean(power_spectrum[index - 2: index + 3])
            # calculate the noise power as the average power of the bins with power less than -60 dBFS            
            self.noisePower = np.mean(power_spectrum[power_spectrum < -60])
            # plot the power spectrum of the noise
            plt.figure()
            plt.plot(self.getPowerSpectrum().frequency, power_spectrum)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dBFS)')
            plt.title('Power Spectrum of the Noise')
            plt.grid()
            plt.show()
            
        return self.noisePower
    
    
    
    def getTHDN_FS(self):
        if self.THDN_FS is None:

            # Convert dBFS to linear scale
            fundamental_power = 10 ** (self.getFundamental().power / 20)
            harmonic_powers = [10 ** (harmonic.power / 20) for harmonic in self.getHarmonics(5)]
            print(f"Harmonic Powers = {harmonic_powers}")
            

            # Calculate THD
            thd = np.sqrt(np.sum(np.array(harmonic_powers) ** 2) / fundamental_power ** 2) # THD in linear scale

            # Estimate noise power (for simplicity, assuming all other bins contribute equally to noise)
            noise_power = 10 ** (-90 / 20)  # Assuming noise floor is at -90 dBFS

            # Calculate THD+N in dBFS
            self.THDN_FS = 20 * np.log10(np.sqrt(thd ** 2 + noise_power))

        return self.THDN_FS
    
    def getTHDN(self):
        if self.THDN is None:
            # THD+N in dBc relates to self.THDN_FS as follows:
            # THD+N = 10 * log10((P2 + P3 + ... + Pn) / P1) = 10 * log10((P2 + P3 + ... + Pn) / P1) - 10 * log10(P1)
            # THD+N = THD+N_FS - 10 * log10(P1)
            print(f"THD+N_FS = {self.getTHDN_FS()}")
            print(f"Fundamental Power = {self.getFundamental().power}")
            self.THDN = self.getTHDN_FS() - self.getFundamental().power
        return self.THDN
    
    def getSINAD_FS(self):
        # SINAD is defined as the ratio of the power of the fundamental frequency to the sum of the powers of all other frequencies
        # SINAD = 10 * log10(P1 / (P2 + P3 + ... + Pn))
        # where P1 is the power of the fundamental frequency and P2, P3, ..., Pn are the powers of the harmonics
        # q: what about noise?
        # a: noise is included in the power of the harmonics
        if self.SINAD_FS is None:
            fundamental_power = 10 ** (self.getFundamental().power / 10)
            harmonics_power = sum([10 ** (harmonic.power / 10) for harmonic in self.getHarmonics(100)])
            self.SINAD_FS = 10 * np.log10(fundamental_power / harmonics_power)
        return self.SINAD_FS
    
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
        for i, harmonic in enumerate(self.getHarmonics()):
            print(f"{i+2:<6}{harmonic.frequency/1e3:<15.3f}{harmonic.power:<15.1f}")
        print('************* ************ *************')
        
        print(f"\nTHD+N = {self.getTHDN():.1f} dBc")
        print(f"THD+N_FS = {self.getTHDN_FS():.1f} dBFS")
        print(f"SINAD_FS = {self.getSINAD_FS():.1f} dBFS")
        print(f"Quantization Noise Level = {self.quantization_noise:.1f} dBFS")
        print(f"Noise Power = {self.getNoisePower():.1f} dBFS")
        
        print("\n************* End of Signal Analysis results *************")
        
def main():
    #from testData import pcmSignal as generate_pcm_signal_with_harmonics
    pcmSgl = pcmSignal(frequency=179e3, amplitude=0, sampling_rate=500e3, adcResolution=12, harmonic_levels=[-80, -72, -95, -90, -85
                                                                                                             ])
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
    
    
    