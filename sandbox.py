def getAlias(harmonic_frequency, sample_rate):
    
    Nyquist = sample_rate / 2
    alias = abs(harmonic_frequency - sample_rate)
    while alias > Nyquist:
        alias -= sample_rate
    return abs(alias)

harmonic_frequency = 1344
sample_rate = 3000

aliased_frequency = getAlias(harmonic_frequency, sample_rate)
print(f"The aliased frequency is: {aliased_frequency}")
