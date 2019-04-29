import pdb
import ast
from sklearn import preprocessing,svm

class NoiseStat:
    def __init__(self,data_set='db/427/noise_data.list'):
        file = open('db/427/noise_data.list')
        inputs = []
        for line in file.readlines():
            x = ast.literal_eval(line)
            inputs.append(x)
        features = []
        labels = []
        for i in range(len(inputs)):
            features.append(inputs[i][:-1])
            labels.append(inputs[i][-1])
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(features)
        features = self.scaler.transform(features)
        self.model = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
            gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
            tol=0.001, verbose=False).fit(features,labels)

    def predict(self,features):
        scaled_features = self.scaler.transform(features)
        ret = self.model.predict(scaled_features)
        if len(features) == 1:
            return ret[0]
        return ret

    def noise_test(self,features,labels):
        scaled_features = self.scaler.transform(features)
        ns = self.model.predict(scaled_features)
        best = co = float('-inf')
        for cutoff in [0.1, 0.2 ,0.3, 0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
            tot = 0
            for i in range(len(ns)):
                c = ns[i] < cutoff
                l = labels[i] == 0.0
                if c == l:
                    tot += 1
            s = tot / len(ns)
            if s > best:
                best = s
                co = cutoff
        return best,co