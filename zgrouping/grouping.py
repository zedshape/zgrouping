"""
Grouping modules of Z-Grouping
============================
submitted to ECML-PKDD 2022
"""

import numpy as np
from .matrix import createMatrixCandidate
from collections import Counter

def createLocalGroupings(matrices, alpha=0.9, K=30, debug=False):
    """
    create local gropuings by applying semigeometric tiling
    :param alpha: a purity score (See Sec. 2).
    :param K: restrict the number of local groupings for improving runtime performance. Practical-use only. This parameter is not used in the experiment.
    """
    metadata = []
    tilevalues = []
    
    for matrix in matrices:
        results = createMatrixCandidate(matrices[matrix].copy(), alpha=alpha, K=K, debug=debug)
        for result in results:
            metadata.append(result[:2] + (matrix,))
            tilevalues.append(result[-1])
            
    timeranges = np.array(metadata)
    members = np.array(tilevalues)
    indices = np.argsort(timeranges[:, 0])
    
    return timeranges[indices], members[indices].T


def createGroupings(matrices, metas = None, alpha=0.9, eta=1.5, K=30, accept=False, c=None, debug=False):
    """
    create local groupings and associations
    
    :param matrices: SAXified matrices (from SAXIfy function from util.py).
    :param metas: a (numpy) list of metadata for each time series sample (only used if accept=True).
    :param alpha: a purity score.
    :param eta: an acceptance score to accept the value or not (only used if accept=True).
    :param K: restrict the number of local groupings for improving runtime performance (cut based on the purity score). This parameter is not used in the experiment.
    :param accept: enable accepting groupings if the dataset has global grouping information.
    :param c: global grouping information of our target (only used if accept=True).
    """
    R = []
    G = []

    if debug == True:
        print("[DEBUG] BEGIN Local grouping generation")
    
    L_timeranges, L_members = createLocalGroupings(matrices=matrices, alpha=alpha, K=K, debug=debug)
    
    if debug == True:
        print("[DEBUG] BEGIN Association generation")

    # profile chain only allows near-consecutive tiles
    tilesets = createMatrixCandidate(L_members.copy(), alpha=alpha, K=K)
    
    # DATA STRUCTURE FORMING
    for tileset in tilesets:
        
        min_val = np.min(L_timeranges[tileset[0]:tileset[1], 0])
        max_val = np.max(L_timeranges[tileset[0]:tileset[1], 1])

        association = {"members": tileset[-1], "range": (min_val, max_val)}

        if accept == True:
            dist = Counter(metas[tileset[-1]])
            if dist[c] >= ((metas == c).sum() * eta * (len(metas[tileset[-1]])/len(metas))):
                classes = {k for k, v in dist.items() if v > ((metas == k).sum() * eta * (len(metas[tileset[-1]])/len(metas)))} - {c}
                association["classes"] = classes

                if len(association["classes"]) != 0:
                    G.append(association)
        else:    
            if len(association["members"]) != 0:
                G.append(association)

    for tileset in tilesets:
        for tileidx in range(tileset[0], tileset[1]):
            if accept == True:
                dist_tile = Counter(metas[L_members.T[tileidx]])
                if dist_tile[c] >= (len(metas[L_members.T[tileidx]])/len(metas))*eta*(metas == c).sum():
                    tileclasses = {k for k, v in dist_tile.items() if v > (eta * (metas == k).sum() * (len(metas[L_members.T[tileidx]])/len(metas)))} - {c}
                    grouping = {"members": L_members.T[tileidx], "classes": list(tileclasses), "range": L_timeranges[tileidx]}
                    if len(grouping["classes"]) != 0:
                        R.append(grouping)
            else:
                grouping = {"members": L_members.T[tileidx], "range": L_timeranges[tileidx]}
            
                if len(grouping["members"]) != 0:
                    R.append(grouping)
    
    return R, G
