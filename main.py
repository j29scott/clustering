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
from src.algorithms.noise_stat import NoiseStat
from sklearn.ensemble import RandomForestClassifier

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
for i in range(N):
    if labels[i]:
        labels[i] = 0.0
    else:
        labels[i] = 1.0

print("Without Noise Stat")
scaled_features = preprocessing.scale(features)
lm  = linear_model.SGDClassifier(max_iter=1000, tol=1e-3).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
svm = SVC(gamma='auto').fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
rf = RandomForestClassifier(n_estimators=10).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
ack = ackerman_score(scaled_features[:N//2],labels[:N//2],feature_names)



print("Ackerman: " + str(ack))
print("SVM     : " + str(svm))
print("LM      : " + str(lm))
print("RF      : " + str(lm))
print("")
print("Improvement: " + str(100.0 * (max(lm,svm,rf) - ack)/ack))
print()

print("With Noise Stat")

ns_model = NoiseStat()
print(ns_model.noise_test(features,labels))

ns = ns_model.predict(features)
for i in range(len(features)):
    features[i].append(ns[i])
scaled_features = preprocessing.scale(features)
lm  = linear_model.SGDClassifier(max_iter=1000, tol=1e-3).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
svm = SVC(gamma='auto').fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
rf = RandomForestClassifier(n_estimators=10).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])


print()
#ack = ackerman_score(features[:N//2],labels[:N//2],feature_names)
print("Ackerman: " + str(ack))
print("SVM     : " + str(svm))
print("LM      : " + str(lm))
print("RF      : " + str(lm))
print("")
print("Improvement: " + str(100.0 * (max(lm,svm,rf) - ack)/ack))
print()