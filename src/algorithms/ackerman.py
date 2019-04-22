import math
from diptest.diptest import diptest
import modality
import matplotlib.pyplot as plt
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

def euclid_distance(a,b):
    assert len(a) == len(b) and len(a) > 0
    ret = 0.0
    for i in range(len(a)):
        ret += (a[i] - b[i]) * (a[i] - b[i])
    return math.sqrt(ret)

def ackerman(instance):
    distances = []
    for i in range(len(instance.points)):
        for j in range(i+1,len(instance.points)):
            distances.append(euclid_distance(instance.points[i],instance.points[j]))
    distances = np.array(distances)

    if len(distances) > 70000:
        distances = np.random.choice(distances, 70000)
    plt.hist(distances)
    plt.show()
    out = diptest(distances)
    print(out,hopkins(instance.points))
    return out[1] < ackerman_cutoff