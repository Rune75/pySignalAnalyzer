#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from scipy.fftpack import fft
from scipy.signal import find_peaks

# The signalAnalyzer class is used to analyze the power spectrum of a PCM signal               
class signalAnalyzer:
    def __init__(self, signal=None, sampling_rate=None, adcResolution=None): 
        self._signal = signal
        self._sampling_rate =sampling_rate
        self._adcResolution = adcResolution
        
        self._adcFS = 2 ** (adcResolution - 1)
        self._num_samples = len(signal)
        self._power_spectrum = self._spectrum()
        self._powerSpectrumLinear = self._spectrum()
        self._magnitude_spectrum = self._spectrum()
        self._noiseSpectrum = self._spectrum()
        self._fundamental = self._peak()
        self._fftwindow = self._window('blackmanharris', self._num_samples)
        self._quantization_noise = 20 * np.log10(1 / (2 ** (self._adcResolution - 1)))
        self._harmonics = []
        self._N = None
        self._THD_FS = None
        self._THD = None
        self._THDN = None
        self._THDN_FS = None
        self._SNR = None
        self._SNDR = None
        self._SNDR_FS = None
        self._SFDR = None
        self._SFDR = None
        self._ENOB = None
        self._ENOB_FS = None       
        self._noisePower = None
        self._harmonicPower = None
        self._totNoiseAndDistortion = None
        self._fftNoiseFloor = self._quantization_noise - 10 * np.log10(self._num_samples/2)
        
    class _peak:
        def __init__(self, frequency=None, power=None):
            self._frequency = frequency
            self._power = power
            
        def __str__(self):
            return f"Frequency: {self._frequency}, Power: {self._power}"
    
    # inherit the peak class and add an index to the new class harmonic
    class _harmonic(_peak): 
        def __init__(self, frequency=None, power=None, index=None):
            super().__init__(frequency, power)
            self._index = index        # add harmonic index to the peak class
    
        
        def __str__(self):
            return f"Index: {self._index}, Frequency: {self._frequency}, Power: {self._power}"


    class _spectrum:
        def __init__(self, frequency=None, level=None):
            self._frequency = frequency
            self._level = level
            
    class _window:
        def __init__(self, type, size):
            self._type = type
            self._size = size
            self._window = sg.get_window(type, self._size)
            
            # # window losses
            self._enbw = size * np.sum(np.abs(self._window) ** 2) / np.abs(np.sum(self._window)) ** 2 # Equivalent Noise Bandwidth
            # scalloping loss for the specific window
            self._maxScallopingLoss = self._getMaxScaplopingLoss()
            # coherent power gain cpg
            self._cpg = 20 * np.log10(np.mean(self._window))
            self._enbwCorr = 10 * np.log10(self._enbw)
            
        
        def _getMaxScaplopingLoss(self):
            # Generate the complex exponential term
            n = np.arange(len(self._window))
            complex_exp = np.exp(-1j * np.pi / self._size * n)
            
            # Calculate the numerator and denominator
            numerator = np.sum(self._window * complex_exp)
            denominator = np.sum(self._window)
            
            # Calculate the scalloping loss
            scalloping_loss = numerator / denominator
            
            return 20 * np.log10(np.abs(scalloping_loss))
                    
        
        def printWindowSpecs(self):
            # print window specs in a table format
            print("\n************* Window specs: *************")
            print(f"{'Window type:':<35}{self._type:<10}")
            print(f"{'Window size:':<35}{self._size:<10}")
            print(f"{'Equivalent Noise Bandwidth ENBW:':<35}{self._enbw:<10.3f}")
            print(f"{'ENBW Correction:':<35}{self._enbwCorr:<10.3f}")
            print(f"{'Max Scalloping Loss:':<35}{self._maxScallopingLoss:<10.3f}")
            print(f"{'Coherent Power Gain CPG:':<35}{self._cpg:<10.3f}")
            print("\n")
            
        def plotWindow(self):
            # plot the window function
            plt.figure()
            plt.plot(self._window)
            plt.title(f'{self._type} window')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.show()
            

    def getMagnitudeSpectrum(self):
        # calculate the magnitude spectrum of the signal unsing scipy fft
        if self._magnitude_spectrum._frequency is None:
            
            #apply a window to the signal before calculating the fft
            self._signal = self._signal * self._fftwindow._window
            
            # Calculate the magnitude spectrum of the signal
            fft_result = fft(self._signal)
            magnitude_spectrum = np.abs(fft_result) / self._num_samples
            
            # compensate level for half side of the spectrum
            magnitude_spectrum = magnitude_spectrum * 2
            
            # convert the magnitude spectrum to dBFS
            self._magnitude_spectrum._level = 20 * np.log10(magnitude_spectrum / self._adcFS + 1e-100)[:int(self._num_samples / 2 + 1)]
            
            self._magnitude_spectrum._level[0] = self._magnitude_spectrum._level[0]/2
            
            # Create frequency axis
            fstep = self._sampling_rate / self._num_samples # freq interval for each frequency bin (sampling frequency divided by number of samples)
            
            # Create freq steps –> x-axis for frequency spectrum
            self._magnitude_spectrum._frequency = np.linspace(0, (self._num_samples-1) * fstep, self._num_samples) 
            
            # keep only the first half of the spectrum
            self._magnitude_spectrum._frequency = self._magnitude_spectrum._frequency[:int(self._num_samples / 2 + 1)]
            
            # remove the DC component
            self._magnitude_spectrum._level[0] = -150
            
        return self._magnitude_spectrum
    
                    
    def getPowerSpectrum(self):
        if self._power_spectrum._frequency is None:
            # calculate power spectrum from the magnitude spectrum given in dBFS
            self._power_spectrum._frequency = self.getMagnitudeSpectrum()._frequency
            self._power_spectrum._level = 10 ** (self.getMagnitudeSpectrum()._level / 20)
            self._power_spectrum._level = self._power_spectrum._level ** 2
            self._power_spectrum._level = 10 * np.log10(self._power_spectrum._level)
            
            # compensate the power spectrum for the coherent power gain
            self._power_spectrum._level = self._power_spectrum._level - self._fftwindow._cpg
            
        return self._power_spectrum
            
        
    def plotPowerSpectrum(self):
        # Plot the power spectrum of the signal and add a table with the results of the signal analysis in two subplots
        fig = plt.figure(figsize=(12, 6))
        
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2,1])
        ax1 = plt.subplot(gs[0])
        
        ax1.plot(self.getPowerSpectrum()._frequency, self.getPowerSpectrum()._level)
        ax1.set_title('Power Spectrum')
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Power [dBFS]')
        
        # set the y-axis limits to the average noise power - 10 dBFS and 10 dBFS
        plt.ylim([self._fftNoiseFloor-10, 10])
        
        # format the x-axis to show the frequency in kHz
        plt.xticks(np.arange(0, self._sampling_rate / 2 + 1e5, 1e5), 
                   [f'{f/1e3:.0f}' for f in np.arange(0, self._sampling_rate / 2 + 1e5, 1e5)])
        
        # include the fundamental frequency in the plot marked with a green ring and print the power level and frequency in kHz
        fundamental = self.getFundamental()
        plt.plot(fundamental._frequency, fundamental._power, 'go')
        plt.text(fundamental._frequency, fundamental._power, f'{fundamental._power:.1f} ' f'{fundamental._frequency / 1e3:.3f}kHz', fontsize=10, color='green')
        
        # inlude harmonics in the plot marked with red small dots, the harmonic power, and harmonic index
        harmonics = self.getHarmonics(100)
        # get the 20 harmonics with the highest power
        harmonics = sorted(harmonics, key=lambda x: x._power, reverse=True)[:30]
        
        for harmonic in harmonics:
            if harmonic._power > self._fftNoiseFloor:
                plt.plot(harmonic._frequency, harmonic._power, 'ro', markersize=2)
                plt.text(harmonic._frequency, harmonic._power, f'H{harmonic._index}', fontsize=6, color='red')
               
        plt.grid()
        ax2 = plt.subplot(gs[1])
        ax2.axis('off')
        ax2.set_title('Analysis Results')
        
        # Add textbox including the printouts in a table format
        text = [
            ['Param', 'Value'],
            ['Fundamental power:', f'{self.getFundamental()._power:.1f} dBFS'],
            # print the fulscale referenced results first
            ['N_FS:', f'{self.get_N():.1f} dBFS'],
            ['THD_FS:', f'{self.getTHD_FS():.1f} dBFS'],
            ['THD+N FS:', f'{self.getTHDN_FS():.1f} dBFS'],
            ['SNR_FS:', f'{self.getSNR_FS():.1f} dBFS'],
            ['SNDR FS:', f'{self.getSNDR_FS():.1f} dBFS'],
            ['ENOB FS:', f'{self.getENOB_FS():.1f} bits'],
            # print the rest of the results in the same order
            ['THD:', f'{self.getTHD():.1f} dBc'],
            ['THD+N:', f'{self.getTHDN():.1f} dBc'],
            ['SNR:', f'{self.getSNR():.1f} dBc'],
            ['SNDR:', f'{self.getSNDR():.1f} dBc'],
            ['SFDR:', f'{self.getSFDR():.1f} dBc'],
            ['ENOB:', f'{self.getENOB():.1f} bits'],
            # print the 5 highest harmonics
            ['Harmonics:', ''],
            ['  Index', 'Power'],
        ]
        
        for harmonic in harmonics[:5]:
            text.append([f'  H{harmonic._index}', f'{harmonic._power:.1f} dBFS'])
            
                
        # create a table with the text in the upper left corner of the plot
        plt.table(cellText=text, loc='upper left', cellLoc='left', colWidths=[0.7, 0.3])
        plt.rc('font', size=13)
        
        # plot the noise spectrum of the signal in a subplot
        ax2 = plt.subplot(gs[2])
        plt.plot(self._noiseSpectrum._frequency, self._noiseSpectrum._level)
        #plt.title('Noise Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Noise [dBFS]')
        plt.ylim([self._fftNoiseFloor-10, self._fftNoiseFloor+30])
        plt.grid()
        return fig
        
    
    def getFundamental(self):
        if self._fundamental._frequency is None:
            # find the peaks of the power spectrum
            peaks, _ = find_peaks(self.getPowerSpectrum()._level, height=-100)
            # find the peak with the highest power
            self._fundamental._power = max(self.getPowerSpectrum()._level[peaks])
            index = np.where(self.getPowerSpectrum()._level == self._fundamental._power)[0][0]
            # find the frequency of the peak
            self._fundamental._frequency = self.getPowerSpectrum()._frequency[index]
        return self._fundamental
        
    def findPeak(self, frequency):
        # find the highest peak in the power spectrum
        # find the closest frequency bin to the given frequency
        index = np.argmin(np.abs(self.getPowerSpectrum()._frequency - frequency))
        
        # find the peak with the highest power within +-2 frequency bins of the given frequency
        top = self._peak()
        top._power = max(self.getPowerSpectrum()._level[index - 2: index + 2])
        top._frequency = self.getPowerSpectrum()._frequency[np.where(self.getPowerSpectrum()._level == top._power)[0][0]]    

        return top
    
    def getAlias(self, harmonic_frequency):
        # find the alias of a harmonic frequency
        Nyquist = self._sampling_rate / 2
        alias = abs(harmonic_frequency - self._sampling_rate)
        while alias > Nyquist:
            alias -= self._sampling_rate
        return abs(alias)
    
    def getHarmonics(self, number_of_harmonics=None):
        if (number_of_harmonics is not None):
            if (len(self._harmonics) < number_of_harmonics):
                # find the harmonics of the fundamental frequency including aliases
                # even if they are aliased multiple times.
                fundamental = self.getFundamental()
                harmonics = [fundamental._frequency * i for i in range(2, number_of_harmonics + 1)]
            
                # find the aliases of the harmonics
                self._harmonics = []
                for i, harmonic in enumerate(harmonics):
                    frequency = self.getAlias(harmonic)
                    power = self.findPeak(frequency)._power
                    index = i + 2
                    self._harmonics.append(self._harmonic(frequency, power, index))
            self._harmonics = self._harmonics[:number_of_harmonics]
        return self._harmonics
    
    
    def getNoisePower(self, excl_nrOfHarmonics):
        # calculate the noise power in the power spectrum
        pwrSpectrum = self.getPowerSpectrum()._level.copy()  # make a copy of the power spectrum
        peakArray = self.getHarmonics(excl_nrOfHarmonics)   # get the harmonics excluding the first n harmonics
        peakArray.append(self.getFundamental())             # append the fundamental frequency to the peakArray
        
        # rotate the peakArray to start with the fundamental frequency
        peakArray = sorted(peakArray, key=lambda x: x._frequency)
        
        for peak in peakArray:
            index = np.argmin(np.abs(self.getPowerSpectrum()._frequency - peak._frequency))
        
        # rotate the peakArray to start with the fundamental frequency
        peakArray = sorted(peakArray, key=lambda x: x._frequency)
        
        for peak in peakArray:
            index = np.argmin(np.abs(self.getPowerSpectrum()._frequency - peak._frequency))
            pwrSpectrum[index - 5: index + 5] = -150 # cancel the peak powers from the Noise calculation
            
        # calculate the noise power as the average power of the power spectrum
        pwrSpectrumLin = 10 ** (pwrSpectrum / 10)
        self._noisePower = 10 * np.log10(np.sum(pwrSpectrumLin))
        # Add window losses to the noise power 
        self._noisePower = self._noisePower - self._fftwindow._enbwCorr # add equivalent noise bandwidth correction
        # coher power gain is already compensated for in the power spectrum
        return self._noisePower, pwrSpectrum
    
    def get_N(self):
        if self._N is None:
            
            self._noiseSpectrum._frequency = self.getPowerSpectrum()._frequency
            self._N, self._noiseSpectrum._level = self.getNoisePower(50)
        return self._N
        
    def getSNR(self):
        # Signal-to-Noise Ratio (SNR) is the ratio of the power of the fundamental
        # to the power of the noise excluding the harmonics
        if self._SNR is None:
            self._SNR = self.getFundamental()._power - self.get_N()
        return self._SNR
    
    def getSNR_FS(self):
        if self._SNR is None:
            self._SNR = self.get_N()
        return self._SNR
    
    def getTHD_FS(self, number_of_harmonics=50):
        # THD is defined as the sum of the powers of all harmonics
        if self._THD_FS is None:
            THD = 0
            for harmonic in self.getHarmonics(number_of_harmonics):
                # calculate the THD as the sum of the linear powers of the harmonics
                THD += 10 ** (harmonic._power / 10)
            # convert the THD to dBFS
            self._THD_FS = 10 * np.log10(THD)
        return self._THD_FS
    
    def getTHD(self):
        # THD is defined as the sum of the powers of all harmonics
        if self._THD is None:
            self._THD = self.getTHD_FS() - self.getFundamental()._power
        return self._THD
    
    def getTHDN_FS_old(self, number_of_harmonics=100):
        # THD+N is defined as the sum of the powers of all harmonics and noise
        # Noise power is calculated as the average power of the power spectrum excluding the fundamental frequency and its harmonics
        # The unit of the Noise power is dBFS and the unit of the THD is dBFS
        if self._THDN_FS is None:
            # convert the noise power from dBFS to linear scale
            _N_Linear = 10 ** (self.get_N() / 10)
            
            # Convert the THD from dBFS to linear scale
            THD_linear = 10 ** (self.getTHD_FS(number_of_harmonics) / 10)
            
            # calculate the THD+N in linear scale
            THDN_linear = (THD_linear + _N_Linear)
            
            # convert the THD+N back to dBFS
            self._THDN_FS = 10 * np.log10(THDN_linear)
            
        return self._THDN_FS
    
    
    def getTHDN_FS(self, number_of_harmonics=100):
        # THD+N_FS is defined as the sum of the powers of all harmonics and noise
        if self._THDN_FS is None:
            self._THDN_FS, d = self.getNoisePower(0)
        return self._THDN_FS
    
    
    def getTHDN(self):
        if self._THDN is None:
            self._THDN = self.getTHDN_FS() - self.getFundamental()._power
        return self._THDN
    
            
    def getSNDR(self):
        # Signal-to-Noise-and-Distortion (SNDR, or S/(N + D)
        # is the ratio of the rms signal amplitude to the mean value of the 
        # root-sum-square (rss) of all other spectral components, including 
        # harmonics, but excluding dc.
        
        if self._SNDR is None:
            ND_power = self.getTHDN()
            self._SNDR = self.getFundamental()._power - ND_power
        return self._SNDR
    
    
    def getSNDR_FS(self):
        # SNDR_FS is defined as the ratio of the power of the fundamental frequency to the sum of the powers of all other frequencies in dBFS
        # SNDR_FS = 10 * log10(P1 / (P2 + P3 + ... + Pn + Pn+1 + ...))
        # where P1 is the power of the fundamental frequency and P2, P3, ..., Pn, Pn+1, ... are the powers of the harmonics and noise
        if self._SNDR_FS is None:
            ND_power = self.getTHDN_FS()
            self._SNDR_FS = self.getFundamental()._power - ND_power
        return self._SNDR_FS
    
    
    def getSFDR(self):
        # SFDR is defined as the ratio of the power of the fundamental frequency to the power of the highest harmonic in dBFS
        # SFDR = P1 - Pn
        # where P1 is the power of the fundamental frequency and Pn is the power of the highest harmonic
        if self._SFDR is None:
            self._SFDR = self.getFundamental()._power - max([harmonic._power for harmonic in self.getHarmonics(100)])
        return self._SFDR
            
    
    def getENOB_FS(self):
        # ENOB_FS is defined as the ratio of the SNDR to the quantization noise level in dBFS
        # ENOB_FS = (SNDR - 1.76) / 6.02
        if self._ENOB_FS is None:
            # signalAmplitude = 1908 #20 ** (self.getFundamental()._power / 20) * self._adcFS
            
            # signalPowerFS = 20 * np.log10(self._adcFS / signalAmplitude)
            # print (f"Signal amplitude: {signalAmplitude:.1f}")
            # print (f"ADC full scale: {self._adcFS:.0f}")
            # print (f"Signal power FS: {signalPowerFS:.1f} dBFS")
            self._ENOB_FS = (self.getSNDR_FS() - 1.76 - self.getFundamental()._power) / 6.02
            #self._ENOB_FS = (self.getSNDR_FS() - 1.76) / 6.02
        return self._ENOB_FS
    
    
    def getENOB(self):
        # ENOB is defined as the ratio of the SNDR to the quantization noise level in dBc
        # ENOB = (SNDR - 1.76) / 6.02
        if self._ENOB is None:
            self._ENOB = (self.getSNDR() - 1.76 - self.getFundamental()._power) / 6.02   
            #self._ENOB = (self.getSNDR() - 1.76) / 6.02   

        return self._ENOB
    
    
    def printAll(self):
        # print results from the signal analysis class functions
        print("\n************* Signal Analysis results: *************")
        # print the fundamental frequency and power
        print("Fundamental:")
        print (f"Frequency = {self.getFundamental()._frequency/1e3:.6f} kHz, Power = {self.getFundamental()._power:.1f} dBFS")
        # print the harmonic indexes, frequencies, and powers in a tabular format.
        # use constant width columns for the table
        print("\n************* Harmonics: *************")
        print(f"{'Index':<6}{'Frequency':<15}{'Power':<10}")
        for harmonic in self.getHarmonics(5):
            print(f"{harmonic._index:<6}{harmonic._frequency/1e3:<15.6f}{harmonic._power:<10.1f}")
        # print fulscale results first
        print('\n********* Fullscale results: **********')
        print(f"{'N:':<21}{self.get_N():<5.1f} dBFS")
        print(f"{'THD_FS:':<21}{self.getTHD_FS():<5.1f} dBFS")
        print(f"{'THD+N FS:':<21}{self.getTHDN_FS():<5.1f} dBFS")
        print(f"{'SNR_FS:':<21}{self.getSNR_FS():<5.1f} dBFS")
        print(f"{'SNDR FS:':<21}{self.getSNDR_FS():<5.1f} dBFS")
        print(f"{'ENOB FS:':<21}{self.getENOB_FS():<5.1f} bits")
        # print the rest of the results in the same order
        print('\n**** Fundamental referenced Results: ****')
        print(f"{'THD:':<21}{self.getTHD():<5.1f} dBc")
        print(f"{'THD+N:':<21}{self.getTHDN():<5.1f} dBc")
        print(f"{'SNR:':<21}{self.getSNR():<5.1f} dBc")
        print(f"{'SNDR:':<21}{self.getSNDR():<5.1f} dBc")
        print(f"{'SFDR:':<21}{self.getSFDR():<5.1f} dBc")
        print(f"{'ENOB:':<21}{self.getENOB():<5.1f} bits")
        print("\n")
        
        
def main():
    # create usage example text
    usage = """A basic usage example of the pcm analyzer class
    # A basic usage example of the pcm analyzer class
    import numpy as np
    from signalAnalyzer import signalAnalyzer
    import matplotlib.pyplot as plt
    
    # *******  Create a test signal ***********************************************************
    import sys
    sys.path.append("test")             # add the test folder to the path
    from pcmVector import pcmSignal     # import the pcmSignal class from the pcmVector module
    
    sampleRate=1e6                                      # 1 MHz sampling rate
    adcResolution=12                                    # 12-bit ADC resolution
    signal = pcmSignal(frequency= 356e3,                # 356 kHz signal frequency
                         amplitude=0,                   # 0 dBFS signal amplitude
                         sampling_rate=sampleRate,
                         adcResolution=adcResolution,
                         harmonic_levels=[-60, -60],    # 2nd and 3rd harmonic levels
                         nrSamples=8192)                # 8192 samples
    # ****************************************************************************************
    
    # ******  Analyze the signal *************
    # Initialize an object of the signalAnalyzer class with the test signal as input
    analyzer = signalAnalyzer(signal.pcmVector, sampleRate, adcResolution)
    
    analyzer.printAll()
    analyzer.plotPowerSpectrum()
    plt.show()
    """
    
    print(usage)
    
if __name__ == "__main__":
    main()
    
    