import math
from diptest.diptest import diptest
#import modality
import matplotlib.pyplot as plt
from src.dist_util import distance
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.random import uniform
import pdb
from random import sample
ackerman_cutoff = 0.05

def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X, 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if math.isnan(H):
        H = 0
 
    return H

def ackerman_dist(instance):
    distances = distance(instance)
    if len(distances) > 70000:
        distances = np.random.choice(distances, 70000)
    out = diptest(distances)
    return out[1] < ackerman_cutoff,out[1]

def ackerman_score(features,labels,feature_names):
    indx = feature_names.index('pnorm_dip_p_value')
    cor = 0
    for i in range(len(features)):
        assert len(features[i]) == len(feature_names), print(len(features[i]), len(feature_names))
        c = features[i][indx] < ackerman_cutoff
        l = labels[i] == 0.0
        if c == l:
            cor += 1
    return cor/len(features)

def hopkins_test(instance):
    h = hopkins(instance.points)
    return h > 0.75,7


def feature_eval(features,labels,target,feature_names):
    assert len(features) > 0 and len(features[0]) == len(feature_names)
    indx = feature_names.index(target)
    best = co = float('-inf')
    for cutoff in [0.01, 0.05, 0.1, 0.2 ,0.3, 0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
        tot = 0
        for i in range(len(features)):
            c = features[i][indx] < cutoff
            #c_gt = features[i][indx] > cutoff
            l = labels[i] == 0.0
            if c == l:
                tot += 1
        s = tot / len(features)
        if s > best:
            best = s
            co = cutoff
    return best,co