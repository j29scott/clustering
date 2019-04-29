import numpy as np
import sys
import ast
import random
import matplotlib.pyplot as plt
from src.util import printer
from src.gen import instance_gen
from src.algorithms.ackerman import ackerman_score
from src.algorithms.classifier import Classifier, instance2features
import pdb
from sklearn import linear_model
from sklearn.svm import SVC
from src.instance.uniform_random import Uniform_Random
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt 

inputs = []

feature_names = instance2features(Uniform_Random(N=10),return_names=True)[1]

for line in sys.stdin.readlines():
    x = ast.literal_eval(line)
    inputs.append(x)
random.shuffle(inputs)
N = len(inputs)
features = []
labels = []
for i in range(N):
    features.append(inputs[i][:-1])
    labels.append(inputs[i][-1])

features = preprocessing.scale(features)

_svm = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False).fit(features[:N//2],labels[:N//2]).score(features[N//2+1:],labels[N//2+1:])

pred = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False).fit(features[:N//2],labels[:N//2]).predict(features[N//2+1:])



print(_svm)
plt.scatter(labels[N//2+1:],pred)
plt.show()

_lm = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3).fit(features[:N//2],labels[:N//2]).score(features[N//2+1:],labels[N//2+1:])
print(_lm)
pred = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3).fit(features[:N//2],labels[:N//2]).predict(features[N//2+1:])
plt.scatter(labels[N//2+1:],pred)
plt.show()

