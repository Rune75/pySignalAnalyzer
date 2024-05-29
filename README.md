## pySignalAnalyzer
pySignalAnalyzer is a python class implementation of a sampled signal analyzer used for testing ADC signal quality or other sampled signals.

The output of the analysis includes: 
- THD
- THD+N
- N
- SNR
- SNDR / SINAD
- SFDR
- ENOB
- Funamental and harmonic levels
- Power spectrum plot

Example usage can be found in the jupyter notebook [README.ipynb](./README.ipynb) or the [basicExample.py](./basicExample.py)

Dependencies:
- numpy
- scipy
- matplotlib

Limitations

The class is currently missing subtraction of scaloping loss, so we rely on having a fairly coherent input signal frequency. 

