"""
Process raw data into design matrix
This uses V5 
@author: Ines
"""
#%%

import os
import numpy as np
from one.api import ONE
import pickle
import pandas as pd
import brainbox.behavior.wheel as wh
from scipy.stats import zscore
from brainbox.io.one import SessionLoader
import scipy.interpolate as interpolate
from joblib import Parallel, delayed
from scipy.fftpack import fft, ifft, fftshift
from sklearn.preprocessing import StandardScaler, Normalizer
import gc


from one.api import ONE
one = ONE(mode='remote')

#%%

""" PARAMETERS """

bin_size = 0.017  # np.round(1/60, 3)  # No binning, number indicates sampling rate
video_type = 'left'    
first_90 = False  # Use full sessions

# Wavelet decomposition
f = np.array([.25, .5, 1, 2, 4, 8, 16])
omega0 = 5

#%%

""" Load BWM data post-QC """
prefix = '/home/ines/repositories/'
# prefix = '/Users/ineslaranjeira/Documents/Repositories/'
save_path = prefix + '/representation_learning_variability/paper-individuality/'
filename = '1_bwm_qc_07-10-2025'

bwm_query = pickle.load(open(save_path+filename, "rb"))
sessions =  bwm_query['eid'].unique()



#%%

