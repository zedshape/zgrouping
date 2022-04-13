"""
Utility modules of Z-Grouping
============================
submitted to ECML-PKDD 2022
- pyts is used for symbolic aggregate approximation
"""

from pyts.approximation import SymbolicAggregateApproximation
import numpy as np

def znorm(samples):
  return (samples - samples.mean(axis=1)[:,...,np.newaxis]) / samples.std(axis=1)[:,...,np.newaxis]

def createChannels(data):
    lettermatrices = {}
    letters = np.unique(data)
    
    for l in letters:
        lettermatrices[l] = (data == l).astype(np.int32)
    return lettermatrices

def SAXify(data, n_bins = 5, glob=True):
    sax = SymbolicAggregateApproximation(n_bins = n_bins, alphabet='ordinal')
    
    if glob == True:
        globalvals = np.concatenate(data)
        data_new = sax.fit_transform([globalvals])
        data_new = data_new.reshape(data.shape)
    else:
        data_new = sax.fit_transform(data)
    return data_new