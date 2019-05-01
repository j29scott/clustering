import pdb
import ast
from sklearn import preprocessing,svm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

class NoiseStat:
    def __init__(self,data_set='db/combined_noise_data_2.pylist'):
        file = open(data_set,'r')
        self.model = {}
        inputs = []
        for line in file.readlines():
            line = line.replace("array([","")
            line = line.replace("])","")
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
        self.model['dnn'] = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(features,labels)
        self.model['svm'] = SVR(gamma='auto').fit(features,labels)
        self.model['lm'] = LinearRegression().fit(features,labels)
    def predict(self,features,model='dnn'):
        scaled_features = self.scaler.transform(features)
        ret = self.model[model].predict(scaled_features)
        if len(features) == 1:
            return ret[0]
        return ret

    def noise_test(self,features,labels,model='dnn'):
        cm = {}
        scaled_features = self.scaler.transform(features)
        ns = self.model[model].predict(scaled_features)
        best = co = float('-inf')
        for cutoff in [0.01, 0.05, 0.1, 0.2 ,0.3, 0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
            tot = 0
            conf_mat = {}
            conf_mat[0.0] = {0.0:0,1.0:0}
            conf_mat[1.0] = {0.0:0,1.0:0}
            for i in range(len(ns)):
                c = ns[i] < cutoff
                l = labels[i] == 0.0
                if c == l:
                    tot += 1
                if ns[i] < cutoff:
                    p = 0.0
                    conf_mat[p][labels[i]] += 1
            s = tot / len(ns)
            if s > best:
                best = s
                co = cutoff
                cm = conf_mat
        return best,co,cm