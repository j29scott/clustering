import math
import numpy as np

# def entropy(X):
#     ret = 0.0
#     for v in X:
#         ret += v * math.log2(v)
#     ret = -ret
#     return ret

def _euclid_distance(a,b,dim):
    assert len(a) == len(b) == dim and len(a) > 0
    ret = 0.0
    for i in range(dim):
        ret += (a[i] - b[i]) * (a[i] - b[i])
    return math.sqrt(ret)

def _pnorm_distance(a,b,dim):
    assert len(a) == len(b) == dim and len(a) > 0
    ret = 0.0
    for i in range(dim):
        ret += (a[i] - b[i]) * (a[i] - b[i])
    return ret ** 1.0/dim

def _cheb_distance(a,b,dim):
    assert len(a) == len(b) == dim and len(a) > 0
    ret = float('-inf')
    for i in range(dim):
        ret = max(ret, abs(a[i] - b[i]))
    return ret

def _taxi_distance(a,b,dim):
    assert len(a) == len(b) == dim and len(a) > 0
    ret = 0.0
    for i in range(dim):
        ret += abs(a[i] - b[i])
    return ret

def _minkowski_distance(a,b,dim):
    assert len(a) == len(b) == dim and len(a) > 0
    ret = 0.0
    for i in range(dim):
        ret += abs(a[i] - b[i]) ** dim
    return ret ** 1.0/dim

dist_funcs = {
    'eucld'     :_euclid_distance,
    'pnorm'     :_pnorm_distance,
    'cheb'      :_cheb_distance,
    'taxi'      :_taxi_distance,
    'minkowski' :_minkowski_distance
} 

def distance(instance,metric='eucld'):
    distances = []
    avg = 0.0
    assert len(instance.points) > 0
    dim = len(instance.points[0])
    for i in range(len(instance.points)):
        for j in range(i+1,len(instance.points)):
            distances.append(dist_funcs[metric](instance.points[i],instance.points[j],dim))
            avg += distances[-1]
    distances = np.array(distances)
    avg /= len(distances)
    std = 0.0
    for d in distances:
        std += (d - avg) * (d - avg)
    std /= len(distances)
    std = math.sqrt(std)

    return distances,avg,std