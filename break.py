from src.gen import break_gen
from src.dist_util import distance
from diptest.diptest import diptest
import sys
import matplotlib.pyplot as plt

best = 0
other = 'minkowski'
for i in range(100000):
    inst = break_gen()
    distances_euclid,mean_euclid,dev_euclid = distance(inst,'eucld') 
    out = diptest(distances_euclid)
    distances_other,mean_other,dev_other = distance(inst,other)
    out2 = diptest(distances_other)

    #print(len(distances_euclid), len(distances_other))
    cur = (out2[1] - out[1])
    if cur < best:
        print(out,out2)
        plt.title("p values --- Euclid = " + str(round(out[1],2)) + ", " + other + " = " + str(round(out2[1],2) ))
        plt.scatter(inst.points[:,0],inst.points[:,1])
        plt.show()
        plt.hist([distances_euclid,distances_other],label=['euclid',other],bins=30)
        plt.legend()
        plt.title("p values --- Euclid = " + str(round(out[1],2)) + ", " + other + " = " + str(round(out2[1],2) ))
        plt.show()
    best = min(best,cur)
    sys.stdout.flush()