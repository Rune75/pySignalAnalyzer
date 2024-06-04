import unittest
# import the module to be tested from the parent directory
from pcmVector import pcmSignal # Import the class to be used for generating test data
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to the path
from signalAnalyzer import signalAnalyzer



class TestsignalAnalyzer(unittest.TestCase):
    def setUp(self):
        # Set up any necessary test data or objects
        # Create a test signal

        self.inputFrequency= 10e3
        self.inputAmplitude=0 
        self.sampleRate=1e6
        self.adcResolution=12
        self.inputHarmonic_levels=[-60, -60] #, -70, -75, -80]
        self.nrSamples=8192
                                
        
        self.signal = pcmSignal(self.inputFrequency,
                                self.inputAmplitude,
                                self.sampleRate,
                                self.adcResolution,
                                self.inputHarmonic_levels,
                                self.nrSamples
                                )
        
        number_of_periods = self.nrSamples / self.sampleRate * self.inputFrequency
        number_of_periods = np.round(number_of_periods) # round to the nearest integer
        self.inputFrequency = self.sampleRate * number_of_periods / self.nrSamples # adjust the frequency to fit the number of periods
        
        
        pass

    def tearDown(self):
        # Clean up any resources used by the test
        pass

    def test_getFundamental(self):
        # Test the getFundamental() method
        # Create an instance of signalAnalyzer
        analyzer = signalAnalyzer(self.signal.pcmVector,
                               self.sampleRate,
                               self.adcResolution)
        
        # Set the expected values
        expected_frequency = self.inputFrequency
        expected_power = 0
        
        # Call the method and assert the results
        result = analyzer.getFundamental()
        print(f'Fundamental frequency: {result.frequency:.2f} Hz')
        print(f'Fundamental power: {result.power:.2f} dBFS')
        self.assertEqual(result.frequency, expected_frequency)
        self.assertAlmostEqual(result.power, expected_power, delta=0.1)
    
    
    def test_getTHD(self):
        # Test the getTHD() method
        # Create an instance of signalAnalyzer
        analyzer = signalAnalyzer(self.signal.pcmVector,
                               self.sampleRate,
                               self.adcResolution)
        
        # Set the expected values
        expected_THD = -57
        
        # Call the method and assert the results
        result = analyzer.getTHD()
        self.assertAlmostEqual(result, expected_THD, delta=0.1)
        
    def test_getHarmonics(self):
        # Test the getHarmonics() method
        # Create an instance of signalAnalyzer
        analyzer = signalAnalyzer(self.signal.pcmVector,
                               self.sampleRate,
                               self.adcResolution)
        
        result = analyzer.getHarmonics(5)
        
        # Set the expected values
        for i in range(len(self.inputHarmonic_levels)):
            expected_harmonic_freq = (i + 2) * self.inputFrequency
            expected_harmonic_power = self.inputHarmonic_levels[i]
            print(f'Harmonic {i+1} frequency: {expected_harmonic_freq:.2f} Hz')
            print(f'Harmonic {i+1} power: {expected_harmonic_power:.2f} dBFS')
            self.assertAlmostEqual(result[i].frequency, expected_harmonic_freq, delta=0.1)
            self.assertAlmostEqual(result[i].power, expected_harmonic_power, delta=0.1)
            
    def test_windowing(self):
        # Test the windowing() method
        # Create an instance of signalAnalyzer
        analyzer = signalAnalyzer(self.signal.pcmVector,
                               self.sampleRate,
                               self.adcResolution)
        
        window = analyzer.window('hamming', self.nrSamples)
        
        #print (window)
        window.printWindowSpecs()
    
        #window.plotWindow()
        
        # Set the expected values for the Hamming window
        cpg = -5.35
        scalloping_loss = -1.75
        enbw = 1.36
        
        self.assertAlmostEqual(window.cpg, cpg, delta=0.01)
        self.assertAlmostEqual(window.maxScallopingLoss, scalloping_loss, delta=0.01)
        self.assertAlmostEqual(window.enbw, enbw, delta=0.01)
        
        
    # Add more test methods for other methods in the signalAnalyzer class
    

if __name__ == '__main__':
    unittest.main()