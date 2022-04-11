"""
Synthetic dataset generator modules of Z-Profile
============================
submitted to SDM 2022
- default values are the ones used in the experiments of the paper
"""

import numpy as np 

def createSyntheticData(c = 20, tc = 50, tl=365,
                        freq = [2,4,6,8], lineheights=[2, 4, 8], lineranges = [30,60,90,120,150,180], amp = [2,4,6,8], 
                        no_outliers = 10, outlier_size=15):
    samples = []
    metas = []

    """
    :param c: number of classes
    :param tc: the number of class members for each class
    :param tl: length of each time series
    :param no_outliers: number of outliers
    :param outlier_size: outlier size
    :param amp: amplitudes
    :param lineranges: length of straight lines
    :param lineheights: height of straight lines
    """

    for classno in range(c):
        s = syntheticSinusoidal(tc=tc, tl=tl, freq = freq, lineranges = lineranges, lineheights = lineheights,
                                amp = amp, no_outliers=no_outliers, outlier_size=outlier_size)
        samples += s
        metas += [classno] * tc

    # samples contain the raw time series values
    samples = np.array(samples, dtype=np.float)
    # metas contain the metadata for each time series
    metas = np.array(metas, dtype=np.int32)

    return samples, metas

def syntheticSinusoidal(tc=30, tl=365, freq = [2,4,6,8], lineheights=[2, 4, 8], lineranges = [30,60,90,120,150,180], amp = [2,4,6,8], no_outliers=10, outlier_size=10):
    """
    :param tc: the number of class members for each class
    :param tl: length of each time series
    :param no_outliers: number of outliers
    :param outlier_size: outlier size
    :param amp: amplitudes
    :param lineranges: length of straight lines
    :param lineheights: height of straight lines
    """
    Fs = tl
    sample = tl # length
    
    classvalues = []

    # define frequency
    f = np.random.choice(freq) # frequency
    weight = np.random.choice(amp) # weight factor

    # generate patterns
    for _ in range(tc):
        x = np.arange(sample)
        a = np.sin(2 * np.pi * f * (x) / Fs) * weight
        irange_chosen = np.random.choice(lineranges)
        # 2. is it going up or down and how much
        patterns = [np.random.rand(irange_chosen)+i for i in lineheights] + [np.random.rand(irange_chosen)-i for i in lineheights]
        #patterns = [np.random.rand(irange_chosen)+2, np.random.rand(irange_chosen)-2, np.random.rand(irange_chosen)+4,
        #            np.random.rand(irange_chosen)-4, np.random.rand(irange_chosen)+8, np.random.rand(irange_chosen)-8]

        pick = np.random.choice(len(patterns))
        place = np.random.choice(int(tl - irange_chosen))
        
        a[place:place + int(irange_chosen)] = patterns[pick]

        # 4. add some outliers
        for __ in range(no_outliers):
            outliers = [np.random.rand(1)[0]+outlier_size, np.random.rand(1)[0]-outlier_size]
            outlier = np.random.choice(outliers)
            random_place = np.random.choice(tl)
            a[random_place] = outlier
        
        # 5. add some very small noise to make it more realistic..
        a_final = a + np.random.normal(0, 0.3, a.shape)
        classvalues.append(a_final)

    return classvalues