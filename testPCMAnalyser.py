import unittest
from analyserClass import pcmAnalyser
from testData import pcmSignal as pcm_data

PCMV = pcm_data()
# Print the PCM signal parameters
print(PCMV.__dict__)

analysis = pcmAnalyser(PCMV)



#analyser = pcmSignalAnalyser(PCMV)
