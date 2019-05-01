import numpy as np
import sys
import ast
import random
import matplotlib.pyplot as plt
from src.util import printer
from src.gen import instance_gen
from src.algorithms.ackerman import ackerman_score
from src.algorithms.classifier import  instance2features
import pdb
from sklearn import linear_model
from sklearn.svm import SVR
from src.instance.uniform_random import Uniform_Random
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt 

data_set='db/combined_noise_data_2.pylist'
file = open(data_set,'r')
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
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features = scaler.transform(features)

N = len(features)

nn = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(features[:N//2],labels[:N//2]).predict(features[N//2+1:])
svm = SVR(gamma='scale', C=1.0, epsilon=0.2).fit(features[:N//2],labels[:N//2]).predict(features[N//2+1:])
lm = LinearRegression().fit(features[:N//2],labels[:N//2]).predict(features[N//2+1:])

plt.scatter(nn,labels[N//2+1:],label='DNN',s=5)
plt.scatter(svm,labels[N//2+1:],label='SVM',s=5)
plt.scatter(lm,labels[N//2+1:],label='Linear Regression',s=2)
plt.scatter(np.arange(0,1,0.001),np.arange(0,1,0.001))
plt.legend()


nn = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(features[:N//2],labels[:N//2]).score(features[N//2+1:],labels[N//2+1:])
svm = SVR(gamma='scale', C=1.0, epsilon=0.2).fit(features[:N//2],labels[:N//2]).score(features[N//2+1:],labels[N//2+1:])
lm = LinearRegression().fit(features[:N//2],labels[:N//2]).score(features[N//2+1:],labels[N//2+1:])
print(nn,svm,lm)
plt.title("Actual Noise vs Predicted Noise")
plt.xlabel("Predicted noise ratio")
plt.ylabel("Actual noise ratio")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()