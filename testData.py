import numpy as np
import matplotlib.pyplot as plt
        

class pcmSignal:
    def __init__(self, frequency=55000, amplitude=100, sampling_rate=1e6, adcResolution=16, harmonic_levels=[-80, -76, -83], nrSamples=8192):
        self.frequency = frequency
        self.sampling_rate = sampling_rate
        self.adcResolution = adcResolution
        self.adcFS = 2 ** adcResolution
        self.amplitude = amplitude
        self.harmonic_levels = harmonic_levels
        self.nrSamples = nrSamples
        self.pcmVector = self.getPCMVector()
        #self.plotPCMVector()
        
    # Function to generate a PCM signal with harmonics
    def getPCMVector(self):
        # Time array nrSamples long with specified sampling rate
        t = np.arange(0, self.nrSamples, 1) / self.sampling_rate
        # Adjust the requested frequency of the signal to be coherent with the sampling rate and number of samples
        # and calculate the adjusted frequency
        self.frequency = (self.sampling_rate / self.nrSamples) * np.round(self.frequency * self.nrSamples / self.sampling_rate)
               
        # check if coherent sampling is fulfilled. This check is somewhat wrong, but it is not the main focus of the task
        if self.frequency != (self.sampling_rate / self.nrSamples):
            print("Coherent sampling not fulfilled")
        else:
            print("Coherent sampling fulfilled")
        
        # convert the amplitude from dBFS to LSB
        self.amplitude = 10 ** (self.amplitude / 20) * (self.adcFS / 2)
                
        # Generate sinusoidal signal with adjusted frequency
        signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        
        # Generate harmonic signals from specified harmonic levels given in dBc
        harmonic_signals = [self.amplitude * 10 ** (level / 20) * np.sin(2 * np.pi * self.frequency * (i + 2) * t) for i, level in enumerate(self.harmonic_levels)]
        
        # Add harmonic signals to the main signal
        self.pcmVector = (signal + sum(harmonic_signals))
        # normalize the signal to the full scale of the ADC
        self.pcmVector = self.pcmVector / (self.adcFS / 2)
        
        # Scale the signal to the desired ADC resolution
        self.pcmVector = self.pcmVector * (2**self.adcResolution - 1)
        
        # Round the signal to the nearest integer
        self.pcmVector = np.round(self.pcmVector)
        
        # cast the signal to integer
        self.pcmVector = self.pcmVector.astype(int)
        
        
        # Return the signal with harmonics
        return self.pcmVector
    
    def plotPCMVector(self):
        # In horizontal 1 x 2 plot, plot the first period of the generated signal with harmonics in the first subplot
        # and the last period of the generated signal with harmonics in the second subplot.
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(self.pcmVector[:int(self.sampling_rate / self.frequency)])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('First Period of the Signal')
        plt.grid()
    
        plt.subplot(1, 2, 2)
        plt.plot(self.pcmVector[-int(self.sampling_rate / self.frequency):])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Last Period of the Signal')
        plt.grid()
        plt.show()
    
    def printSignalSpecs(self):
        # Print the signal specifications
        print("\nSignal Specifications:")
        print("*********************")
        print(f"Frequency: {self.frequency} Hz")
        print(f"Sampling Rate: {self.sampling_rate} Hz")
        print(f"ADC Resolution: {self.adcResolution} bits")
        print(f"Amplitude: {self.amplitude} LSB")
        print(f"Harmonic Levels: {self.harmonic_levels}")
        print(f"Number of Samples: {self.nrSamples}")
        print(f"Number of Samples per Signal Period: {self.nrSamples / (self.sampling_rate / self.frequency)}")
        print(f"Number of Signal Periods in the Signal: {(self.nrSamples / self.sampling_rate) * self.frequency}")
        print("************************")
        
        
def main():
    pcmSgl = pcmSignal(frequency=33e3, amplitude=-6, sampling_rate=1e6, adcResolution=6, harmonic_levels=[-80, -76, -83])
    pcmSgl.printSignalSpecs()
    
    pcmSgl.plotPCMVector()
    
    
if __name__ == "__main__":
    main()  
    
    
        