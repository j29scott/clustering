import src.settings as settings
import src.dist_util as dist_util
from src.algorithms.ackerman import hopkins
from src.util import printer
import pdb
from diptest.diptest import diptest
from sklearn import linear_model
import numpy as np

def instance2features(instance,return_names=False):
    features = []
    feature_names = []
    for dist in settings.distances:
        distances,mean,dev = dist_util.distance(instance,dist)
        features.append(mean)
        feature_names.append(dist + "_mean")
        features.append(dev)
        feature_names.append(dist + "_dev")
        for test in settings.modality_tests:
            if test == 'dip':
                if len(distances) > 70000:
                    distances = np.random.choice(distances, 70000)
                out = diptest(distances)
                if 'dip_stat' in settings.modality_tests[test]:
                    features.append(out[0])
                    feature_names.append('dip_stat')
                if 'p_value' in settings.modality_tests[test]:
                    features.append(out[1])
                    feature_names.append('dip_p_value')

    for feat in settings.additional_statistics:
        if feat == 'hopkins':
            features.append(hopkins(instance.points))
            feature_names.append('hopkins')
    
    
    
    assert len(features) == len(feature_names),print(len(features) ,len(feature_names))
    if not return_names:
        return features
    return features,feature_names


class Classifier:
    def __init__(self):
        self.model = None
    def train(self,X,Y):
        features = []
        labels = []

        for i in range(len(X)):
            printer("Computing features: " +str(i/len(X) * 100.0))
            features.append(instance2features(X[i]))
            if Y[i] == True:
                labels.append(1.0)
            elif Y[i] == False:
                labels.append(2.0)
            else:
                assert False
            pdb.set_trace()

        self.model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3).fit(features,labels)
        
    def predict(self,X,Y=None):
        features = []
        labels = []
        for i in range(len(X)):
            printer("Computing features: " +str(i/len(X) * 100.0))
            features.append(instance2features(X[i]))
            if Y[i] == True:
                labels.append(1.0)
            elif Y[i] == False:
                labels.append(2.0)
            else:
                assert False
        classes = self.model.predict(features)
        pred = []
        for c in classes:
            if c == 1.0:
                pred.append(True)
            elif c == 2.0:
                pred.append(False)
            else:
                assert False
        if Y == None:
            return pred
        assert len(pred) == len(Y)
        n = 0
        for i in range(len(Y)):
            if pred[i] == Y[i]:
                n += 1
        return pred,n/len(Y)

