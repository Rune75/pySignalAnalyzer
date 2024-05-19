#Author : Mathuranathan Viswanathan for gaussianwaves.com
#Date: 13 Sept 2020
#Script to calculate equivalent noise bandwidth (ENBW) for some well known window functions

import numpy as np
import pandas as pd
from scipy import signal

def equivalent_noise_bandwidth(window):
    #Returns the Equivalent Noise BandWidth (ENBW)
    return len(window) * np.sum(window**2) / np.sum(window)**2

def get_enbw_windows():
    #Return ENBW for all the following windows as a dataframe
    window_names = ['boxcar','barthann','bartlett','blackman','blackmanharris','bohman','cosine','exponential','flattop','hamming','hann','nuttall','parzen','triang']
    
    df = pd.DataFrame(columns=['Window','ENBW (bins)','ENBW correction (dB)'])
    for window_name in window_names:
        method_name = window_name
        func_to_run = getattr(signal.windows, method_name) #map window names to window functions in scipy package
        L = 16384 #Number of points in the output window
        window = func_to_run(L) #call the functions
        
        enbw = equivalent_noise_bandwidth(window) #compute ENBW
        
        df = df._append({'Window': window_name.title(),'ENBW (bins)':round(enbw,3),'ENBW correction (dB)': round(10*np.log10(enbw),3)},ignore_index=True)
        
    return df

df = get_enbw_windows() #call the function
# display the dataframe 
print(df)

