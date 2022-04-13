import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_score
import pickle
from zgrouping import grouping, matrix, utils
import warnings
warnings.filterwarnings('ignore')

# EDIT REQUIRED: PICK ONE DATASET TO PERFORM THE EXPERIMENT
# =================================
f = "datasets/synthetic_data.pickle"
# =================================

# PARAMETERS
# =================================
N_SPLITS = 10
N_BINS_GROUP = [3, 5, 10]
ALPHAS = [0.8, 0.9, 1]
ACCPT_SCORES = [1, 1.5, 2]
T_RANGES = [30, 60, 180]
KS = [3, 5, 10]
CUTOFFS = [0.1, 0.2, 0.3]
wholetraining = True
active_save = False
debug = True
averaging = False
# =================================

mean_kmeans_ts_mse = {}
mean_kmeans_ts_mae = {}
mean_kmeans_count = {}
mean_grouping_mse = {}
mean_grouping_mae = {}
count_grouping = {}

final_result = {}

def knnExperiment_SAXified(c, metas, samples, targetval, samples_SAX, BASE=360, K=10, time_range=30, wholetraining = True, averaging = False, debug=False):
    knn_means_mse = []
    knn_means_mae = []
    
    for i in range(int(BASE / time_range)):
        range_start = i*time_range
        range_end = (i+1)*time_range

        samples_range = samples[:, range_start:range_end]
        targetval_range = targetval[range_start:range_end]

        samples_range_SAX = samples_SAX[:, range_start:range_end]
        # get sample excluding our own class
        metas_without = metas[metas != c]
        
        #get sample same to ours
        samples_with_SAX = samples_range_SAX[metas == c]
        samples_without_SAX = samples_range_SAX[metas != c]

        #pick K randomly (FROM TRAINING SET)
        picked_indices = np.random.choice(samples_with_SAX.shape[0], np.min([K, samples_with_SAX.shape[0]-1]), replace=False)

        #FIND NEIGHBORS FROM TRAINING SET
        knn = NearestNeighbors(n_neighbors=K)
        knn.fit(samples_without_SAX)
        knns = knn.kneighbors(samples_with_SAX[picked_indices], return_distance=False)

        #CHECK THE CLASSES
        classes = np.unique(metas_without[np.unique(np.array(knns).flatten())])
        
        if wholetraining == True:
            distval = samples_range[np.isin(metas, classes)]
        else:
            distval = samples_range[np.unique(np.array(knns).flatten())]

        if averaging == True:
            distval = distval.mean(axis=0)

        diff = distval - targetval_range
        knn_mean_mse = np.nanmean((diff)**2)
        knn_mean_mae = np.nanmean(np.abs(diff))

        knn_means_mse.append(knn_mean_mse)
        knn_means_mae.append(knn_mean_mae)

    return np.mean(knn_means_mse), np.mean(knn_means_mae)

def kmeansExperiment_SAXified(c, kmeans, targetval_range, targetval_range_SAX, samples_range, metas, wholetraining = True, averaging = False, debug=False):
    kmeans_means_mse = []
    kmeans_means_mae = []

    predict = kmeans.predict([targetval_range_SAX])
    
    # retreive indices of the training set in the cluster
    classes = set(metas[kmeans.labels_ == predict[0]]) - {c}
    
    if len(classes) != 0:
        if wholetraining == True:
            distval = samples_range[np.isin(metas, list(classes))]
        else:
            distval = samples_range[np.logical_and((kmeans.labels_ == predict), np.isin(metas, list(classes)))]
        if averaging == True:
            distval = distval.mean(axis=0)

        diff = distval - targetval_range
        kmeans_mean_mse = ((diff)**2).mean()
        kmeans_mean_mae = np.abs(diff).mean()
        kmeans_means_mse.append(kmeans_mean_mse)
        kmeans_means_mae.append(kmeans_mean_mae)

    return np.mean(kmeans_means_mse), np.mean(kmeans_means_mae)

def knnExperiment(c, metas, samples, targetval, BASE=360, K=10, time_range=30, wholetraining = True, averaging = False, debug=False):
    knn_means_mse = []
    knn_means_mae = []
    
    for i in range(int(BASE / time_range)):
        range_start = i*time_range
        range_end = (i+1)*time_range

        samples_range = samples[:, range_start:range_end]
        targetval_range = targetval[range_start:range_end]
        # get sample excluding our own class
        metas_without = metas[metas != c]
        
        #get sample same to ours
        samples_with = samples_range[metas == c]
        samples_without = samples_range[metas != c]

        #pick K randomly (FROM TRAINING SET)
        picked_indices = np.random.choice(samples_with.shape[0], np.min([K, samples_with.shape[0]-1]), replace=False)

        #FIND NEIGHBORS FROM TRAINING SET
        knn = NearestNeighbors(n_neighbors=K)
        knn.fit(samples_without)
        knns = knn.kneighbors(samples_with[picked_indices], return_distance=False)

        #CHECK THE CLASSES
        classes = np.unique(metas_without[np.unique(np.array(knns).flatten())])
        
        if wholetraining == True:
            distval = samples_range[np.isin(metas, classes)]
        else:
            distval = samples_range[np.unique(np.array(knns).flatten())]

        if averaging == True:
            distval = distval.mean(axis=0)

        diff = distval - targetval_range
        knn_mean_mse = np.nanmean((diff)**2)
        knn_mean_mae = np.nanmean(np.abs(diff))

        knn_means_mse.append(knn_mean_mse)
        knn_means_mae.append(knn_mean_mae)

    return np.mean(knn_means_mse), np.mean(knn_means_mae)

def kmeansExperiment(c, kmeans, targetval_range, samples_range, metas, wholetraining = True, averaging = False, debug=False):
    kmeans_means_mse = []
    kmeans_means_mae = []

    predict = kmeans.predict([targetval_range])
    
    # retreive indices of the training set in the cluster
    classes = set(metas[kmeans.labels_ == predict[0]]) - {c}
    
    if len(classes) != 0:
        if wholetraining == True:
            distval = samples_range[np.isin(metas, list(classes))]
        else:
            distval = samples_range[np.logical_and((kmeans.labels_ == predict), np.isin(metas, list(classes)))]
        if averaging == True:
            distval = distval.mean(axis=0)

        diff = distval - targetval_range
        kmeans_mean_mse = ((diff)**2).mean()
        kmeans_mean_mae = np.abs(diff).mean()
        kmeans_means_mse.append(kmeans_mean_mse)
        kmeans_means_mae.append(kmeans_mean_mae)

    return np.mean(kmeans_means_mse), np.mean(kmeans_means_mae)

def createGroupingExperimentWithAllow(cs, associations, metas, metadata, acceptance_score, groupings, wholetraining = True, binary=False, length=0):
    G_list = {}
    R_list = {}

    for c in cs:
        G = []
        R = []
        if binary == False:

            for cidx in associations:
                
                dist = Counter(metas[cidx[-1]])
                if dist[c] >= ((metas == c).sum() * acceptance_score * (len(metas[cidx[-1]])/len(metas))):

                    classes = {k for k, v in dist.items() if v >= ((metas == k).sum() * acceptance_score * (len(metas[cidx[-1]])/len(metas)))} - {c}
                    
                    min_val = np.min(metadata[cidx[0]:cidx[1], 0])
                    max_val = np.max(metadata[cidx[0]:cidx[1], 1])
                    
                    if wholetraining == False:
                        members = np.logical_and(cidx[-1], np.isin(metas, list(classes)))
                    else:
                        members = np.isin(metas, list(classes))

                    association = {"classes": list(classes), "range": (min_val, max_val), "members": members }
                    if len(classes) != 0:
                        G.append(association)

                for pidx in range(cidx[0], cidx[1]):
                    dist_tile = Counter(metas[groupings.T[pidx]])
                    if dist_tile[c] >= (len(metas[groupings.T[pidx]])/len(metas))*acceptance_score*(metas == c).sum():
                        classes = {k for k, v in dist_tile.items() if v >= (acceptance_score * (metas == k).sum() * (len(metas[groupings.T[pidx]])/len(metas)))} - {c}

                        if wholetraining == False:
                            members = np.logical_and(groupings.T[pidx], np.isin(metas, list(classes)))
                        else:
                            members = np.isin(metas, list(classes))
                            
                        grouping = {"classes": list(classes), "range": metadata[pidx], "members": members }
                        if len(grouping["classes"]) != 0:
                            R.append(grouping)
        else:
            for pidx in range(groupings.shape[1]):
                dist_tile = Counter(metas[groupings.T[pidx]])
                if dist_tile[c] >= (len(metas[groupings.T[pidx]])/len(metas))*acceptance_score*(metas == c).sum():
                    classes = {k for k, v in dist_tile.items() if v >= (acceptance_score * (metas == k).sum() * (len(metas[groupings.T[pidx]])/len(metas)))} - {c}

                    if wholetraining == False:
                        members = np.logical_and(groupings.T[pidx], np.isin(metas, list(classes)))
                    else:
                        members = np.isin(metas, list(classes))
                        
                    grouping = {"classes": list(classes), "range": metadata[pidx], "members": members }
                    if len(grouping["classes"]) != 0:
                        R.append(grouping)

        G_list[c] = G
        R_list[c] = R

    return G_list, R_list


def localGroupingExperiment(associations, groupings, samples, targetval, BASE=360, averaging = False, binary=False, wholetraining = True, length=0, gap=0.1, debug=False):
    baseline = np.empty(BASE)
    baseline[:] = np.nan

    if binary == False:
        for association in associations:
            distval = samples[association['members'], association["range"][0]:association["range"][1]]
            emptyval = np.empty([distval.shape[0], BASE])
            emptyval[:,:] = np.nan
            emptyval[:, association["range"][0]:association["range"][1]] = distval
            baseline = np.vstack((baseline, emptyval))

    for grouping in groupings:
        distval = samples[grouping['members'], grouping["range"][0]:grouping["range"][1]]
        emptyval = np.empty([distval.shape[0], BASE])
        emptyval[:,:] = np.nan
        emptyval[:, grouping["range"][0]:grouping["range"][1]] = distval
        baseline = np.vstack((baseline, emptyval))
    
    if averaging == True:
        baseline = np.nanmean(baseline, axis = 0)
    
    diff = baseline - targetval
    error_MSE = ((diff)**2)
    error_MAE = np.abs(diff)

    return np.nanmean(error_MSE), np.nanmean(error_MAE), np.count_nonzero(~np.isnan(np.nanmean(baseline, axis = 0)))


print("Z-GROUPING EXPERIMENT")
print("CHOSEN FILE:", f)

with open(f, 'rb') as handle:
    data = pickle.load(handle)

X = utils.znorm(data['X'])
y = data['y']
BASE = X.shape[1]

fold = 1
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

# for k-means-FLEX with SAX 
X_SAX = utils.SAXify(X, n_bins = 5, glob=True)

for train_index, test_index in skf.split(X, y):

    samples = X[train_index]
    samples_test = X[test_index]
    metas = y[train_index]
    metas_test = y[test_index]
    
    # SAX applied values (numbers)
    samples_SAX = X_SAX[train_index]
    samples_test_SAX = X_SAX[test_index]

    print(f"Processing FOLD {fold}...")
    fold = fold + 1

    for n_bins in N_BINS_GROUP:
        X_sax = utils.SAXify(samples, n_bins = n_bins, glob=True)
        matrices = utils.createChannels(X_sax)
        length = matrices[0].shape[1]

        if n_bins == 2:
            del matrices[0]
        
        binary = True if n_bins == 2 else False

        for alpha in ALPHAS:
            metadata, groupings_candidates = grouping.createLocalGroupings(matrices, alpha=alpha)
            associations_candidates = matrix.createMatrixCandidate(groupings_candidates.copy(), alpha=alpha)
                
            for acceptance_score in ACCPT_SCORES:
            
                associations, groupings = createGroupingExperimentWithAllow(np.unique(metas), associations_candidates, metas, metadata, acceptance_score, groupings_candidates, wholetraining = wholetraining, binary=binary, length=0)

                tmp_mse = []
                tmp_mae = []
                tmp_count = []

                for idx in range(samples_test.shape[0]):
                    c = metas_test[idx]
                    targetval = samples_test[idx]
                    error_MSE, error_MAE, count_all = localGroupingExperiment(associations[c], groupings[c], samples, targetval, BASE=BASE, binary=binary)
                
                    tmp_mse.append(error_MSE)
                    tmp_mae.append(error_MAE)
                    tmp_count.append(count_all)
        
                if (n_bins, alpha, acceptance_score) not in mean_grouping_mse:
                    mean_grouping_mse[(n_bins, alpha, acceptance_score)] = []
                    mean_grouping_mae[(n_bins, alpha, acceptance_score)] = []
                    count_grouping[(n_bins, alpha, acceptance_score)] = []
        
                mean_grouping_mse[(n_bins, alpha, acceptance_score)].append(np.nanmean(tmp_mse))
                mean_grouping_mae[(n_bins, alpha, acceptance_score)].append(np.nanmean(tmp_mae))
                count_grouping[(n_bins, alpha, acceptance_score)].append(np.nanmean(tmp_count) / BASE)
        
    for SAXop in [False, True]:
        for cutoff in CUTOFFS:
            tmp_kmeans_mse = []
            tmp_kmeans_mae = []
            tmp_count = []
        
            # PHASE 1: 
            selected_k_means_list = []
            for time_range in T_RANGES:
                for i in range(int(BASE / time_range * 2)):
                    range_start = i*time_range # start time
                    range_end = np.min([(i+2)*time_range, samples.shape[1]]) 
                    
                    samples_range = samples[:, range_start:range_end]
                    samples_range_SAX = samples_SAX[:, range_start:range_end]
                    
                    if samples_range.shape[1] == 0:
                        continue

                    for k_km in KS:
                        kmeans = KMeans(n_clusters=k_km)
                        cluster_labels = kmeans.fit_predict(samples_range_SAX)
                        silhouette_avg = silhouette_score(samples_range_SAX, cluster_labels)
                        if silhouette_avg >= cutoff:
                            selected_k_means = kmeans
                            selected_k_means_list.append((range_start, range_end, selected_k_means))

            # PHASE 2:
            kmeans_means_mse = []
            kmeans_means_mae = []
            kmeans_count = []

            for idx in range(samples_test.shape[0]):
            
                c = metas_test[idx]
                targetval = samples_test[idx]
                
                baseline = np.empty(BASE)
                baseline[:] = np.nan
                baseline_MAE = np.empty(BASE)
                baseline_MAE[:] = np.nan

                # We get the indices and ranges and valdulate the error
                for kmeansTuple in selected_k_means_list:
                    range_start = kmeansTuple[0]
                    range_end = kmeansTuple[1]
                    targetval_range = targetval[range_start:range_end]
                    samples_range = samples[:, range_start:range_end]
                    samples_test_range = samples_test[:, range_start:range_end]
                    samples_test_with = samples_test_range[metas_test == c]
                    kmeans = kmeansTuple[2]
                    K_type = 10

                    if SAXop == False:
                        samples_test_range = samples_test[:, range_start:range_end]
                    else:
                        samples_test_range = samples_test_SAX[:, range_start:range_end]

                    if K_type <= samples_test_with.shape[0]:
                        picked_indices = np.random.choice(samples_test_with.shape[0], K_type, replace=False)
                    else:
                        picked_indices = np.random.choice(samples_test_with.shape[0], samples_test_with.shape[0], replace=False)

                    predicts = kmeans.predict(samples_test_with[picked_indices])
                    counter = Counter(predicts)
                    top_freq, _  = counter.most_common(1)[0]
                    classes = set(metas[kmeans.labels_ == top_freq]) - {c}

                    if len(classes) != 0:
                        distval = samples_range[np.isin(metas, list(classes))]

                    diff = distval - targetval_range
                    baseline[range_start:range_end] = ((diff)**2).mean(axis=0)
                    baseline_MAE[range_start:range_end] = np.abs(diff).mean(axis=0)

                kmeans_means_mse.append(np.nanmean(baseline))
                kmeans_means_mae.append(np.nanmean(baseline_MAE))
                kmeans_count.append(np.count_nonzero(~np.isnan(baseline)))
                    
            if (cutoff, SAXop) not in mean_kmeans_ts_mse:
                mean_kmeans_ts_mse[(cutoff, SAXop)] = []
                mean_kmeans_ts_mae[(cutoff, SAXop)] = []
                mean_kmeans_count[(cutoff, SAXop)] = []
                mean_kmeans_ts_mse[(cutoff, SAXop)].append(np.nanmean(kmeans_means_mse))
                mean_kmeans_ts_mae[(cutoff, SAXop)].append(np.nanmean(kmeans_means_mae))
                mean_kmeans_count[(cutoff, SAXop)].append(np.nanmean(kmeans_count) / BASE)
        
    if debug == True:
        for k, v in mean_grouping_mse.items():   
            print(k, np.nanmean(v), np.nanmean(mean_grouping_mae[k]), np.nanmean(count_grouping[k]))

        for k, v in mean_kmeans_ts_mse.items():    
            print(k, np.nanmean(v), np.nanmean(mean_kmeans_ts_mae[k]), np.nanmean(mean_kmeans_count[k]))

    if active_save == True:
        final_result = [
            mean_kmeans_ts_mse,
            mean_kmeans_ts_mae,
            mean_kmeans_count,
            mean_grouping_mse,
            mean_grouping_mae,
            count_grouping
        ]

    with open('results.pickle', 'wb') as handle:
        pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

final_result = [
    mean_kmeans_ts_mse,
    mean_kmeans_ts_mae,
    mean_kmeans_count,
    mean_grouping_mse,
    mean_grouping_mae,
    count_grouping
]

with open('results.pickle', 'wb') as handle:
    pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)