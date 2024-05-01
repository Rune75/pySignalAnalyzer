import numpy as np
import matplotlib.pyplot as plt
        

class pcmSignal:
    def __init__(self, frequency=55000, amplitude=100, sampling_rate=1e6, adcResolution=16, harmonic_levels=[-80, -76, -83], nrSamples=8192):
        self.frequency = frequency
        self.sampling_rate = sampling_rate
        self.adcResolution = adcResolution
        self.adcFS = 2 ** (adcResolution -1)
        self.amplitude = amplitude
        self.harmonic_levels = harmonic_levels
        self.nrSamples = nrSamples
        self.pcmMaxRes = []
        self.quantization_noise = []
        self.pcmVector = []
        self.pcmVector = self.getPCMVector()
        
        
        #self.plotPCMVector()

    def getPCMVector(self):
        if len(self.pcmVector) == 0:
            
            # Adjust input frequency for fulfilling coherent sampling condition
            # find integer number of periods to fit in the total sample length from self.nrSamples
            number_of_periods = self.nrSamples / self.sampling_rate * self.frequency
            number_of_periods = np.round(number_of_periods) # round to the nearest integer
            self.frequency = self.sampling_rate * number_of_periods / self.nrSamples # adjust the frequency to fit the number of periods
        
            
            # Generate time vector
            time = np.arange(self.nrSamples) / self.sampling_rate
            
            # convert the amplitude from dBFS, with full scale set by the ADC resolution, to linear
            self.amplitude = 10 ** (self.amplitude / 20)
            
            # convert to full scale ratio
            self.amplitude = float(self.amplitude * self.adcFS)
            
            # Generate the waveform with given amplitude, and frequency
            waveform = self.amplitude * np.sin(2 * np.pi * self.frequency * time)
            
            # Add harmonics to the waveform
            for i in range(len(self.harmonic_levels)):
                H_freq = (i + 2) * self.frequency
                H_lvl = 10 ** (self.harmonic_levels[i] / 20) * self.amplitude
                waveform = waveform + H_lvl * np.sin(2 * np.pi * H_freq * time)
            self.pcmMaxRes = waveform
            # get waveform amplitude
            self.amplitude = np.max(np.abs(waveform))
            
            # Apply quantization to the signal
            # stepSize = np.linspace(-10*self.adcFS, 10*self.adcFS, 2 ** self.adcResolution)
            self.pcmVector = self.quantize(waveform, 1)
        return self.pcmVector
    
    def quantize(self, vaveform, stepSize):
        # Quantize the waveform to the nearest step
        waveform = np.round(vaveform / stepSize) * stepSize
        return waveform
    
        
    def quantize_old(self,x, S):
        X = x.reshape((-1,1))
        S = S.reshape((1,-1))
        dists = abs(X-S)
        
        nearestIndex = dists.argmin(axis=1)
        quantized = S.flat[nearestIndex]
        
        return quantized.reshape(x.shape)
    
    def getQuantizationNoise(self):
        if len(self.quantization_noise) == 0:
            # Get the quantization noise vector from the generated signal by subtracting
            # the signal before quantization from the signal after quantization. cast the signal to float
            self.quantization_noise = self.pcmMaxRes - self.getPCMVector().astype(float)
            
        return self.quantization_noise
    
        
    def getSignalPowerdBFS(self):
        # Get the signal power in dBFS
        # The signal power is the power of the signal before quantization
    
        signal_power = np.mean(np.abs(self.pcmVector) ** 2)
        
        fullScalePower = (self.adcFS)**2 / 2 # full scale power in linear scale
        
        signal_power_dBFS = 10 * np.log10(signal_power / fullScalePower)
        
        return signal_power_dBFS
    
            
    def getQuantizationNoisePowerdBFS(self):
        # Get the quantization noise power in dBFS
        # The quantization noise power is the power of the quantization noise signal
        # in dBFS (decibels full scale)
        quantization_noise_power = 10 * np.log10(np.mean(abs(self.getQuantizationNoise()) ** 2)/ (self.adcFS ** 2 / 2))
        
        return quantization_noise_power
    
    def getSNR(self):
        # Get the signal to noise ratio in dB
        # The signal to noise ratio is the power of the signal before quantization
        # divided by the power of the quantization noise signal
        snr = self.getSignalPowerdBFS() - self.getQuantizationNoisePowerdBFS()
        return snr
    
    def GetIdealSNR(self):
        # Get the optimal signal to noise ratio in dB
        ideal_snr = 6.02 * self.adcResolution + 1.76
        return ideal_snr
    
    def plotPCMVector(self):
        # In horizontal 1 x 2 plot, plot the first period of the generated signal with harmonics in the first subplot
        # and the last period of the generated signal with harmonics in the second subplot.
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(self.pcmVector[:int(self.sampling_rate / self.frequency)])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('First Period of the Signal')
        plt.grid()
    
        plt.subplot(2, 2, 2)
        plt.plot(self.pcmVector[-int(self.sampling_rate / self.frequency):])
        plt.xlabel('Sample')
        plt.title('Last Period of the Signal')
        plt.grid()
        
        plt.subplot(2, 2, 3)
        plt.plot(self.getQuantizationNoise())
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Quantization Noise')
        plt.grid()
        
        length = 20
        plt.subplot(2, 2, 4)
        plt.plot(self.pcmMaxRes[:length])
        plt.plot(self.getPCMVector()[:length])
        
        plt.plot(self.quantization_noise[:length])
        plt.grid()
        plt.show()
        
        plt.show()
    
    def printSignalSpecs(self):
        # Print the signal specifications
        print("\nSignal Specifications:")
        print("*********************")
        # Print table of signal specifications
        print(f'Frequency: {self.frequency:.1f} Hz')
        print(f'Sampling Rate: {self.sampling_rate:.1f} Hz')
        print(f'ADC Resolution: {self.adcResolution} bits') 
        print(f'Amplitude: {self.amplitude:.1f}')
        print(f'Harmonic Levels: {self.harmonic_levels}')
        print(f'Number of Samples: {self.nrSamples}')
        print(f'Signal Power: {self.getSignalPowerdBFS():.1f} dBFS')
        print(f'Quantization Noise Power: {self.getQuantizationNoisePowerdBFS():.1f} dBFS')
        print(f'SNR: {self.getSNR():.1f} dB')
        print(f'Ideal SNR: {self.GetIdealSNR():.1f} dB')
        
            
        print("************************")
        
        
def main():
    pcmSgl = pcmSignal(frequency=10e3, 
                       amplitude=0, 
                       sampling_rate=1e6, 
                       adcResolution=2, 
                       harmonic_levels=[]
                       )
    
    pcmSgl.printSignalSpecs()
    pcmSgl.plotPCMVector()
    
    
if __name__ == "__main__":
    main()  
    
    
        
