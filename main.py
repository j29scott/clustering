import numpy as np
import sys
import ast
import random
import matplotlib.pyplot as plt
from src.util import printer
from src.gen import instance_gen
from src.algorithms.ackerman import ackerman_score,feature_eval
from src.algorithms.classifier import  instance2features,get_feature_names
import pdb
from sklearn import linear_model
from sklearn.svm import SVC
from src.instance.uniform_random import Uniform_Random
from sklearn import preprocessing
from src.algorithms.noise_stat import NoiseStat
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def rel_diff(a,b):
    return (b-a)/a

inputs = []

feature_names = get_feature_names()
print(feature_names)



for line in sys.stdin.readlines():
    line = line.replace("array([","")
    line = line.replace("])","")
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


target = 'eucld_dip_p_value'
print(target,feature_eval(features,labels,target,feature_names))

target = 'pnorm_dip_p_value'
print(target,feature_eval(features,labels,target,feature_names))

target = 'cheb_dip_p_value'
print(target,feature_eval(features,labels,target,feature_names))

target = 'taxi_dip_p_value'
print(target,feature_eval(features,labels,target,feature_names))

target = 'minkowski_dip_p_value'
print(target,feature_eval(features,labels,target,feature_names))

print(len(features[0]))
print("Without Noise Stat")
scaled_features = preprocessing.scale(features)
lm  = linear_model.SGDClassifier(max_iter=1000, tol=1e-3).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
svm = SVC(gamma='auto').fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
rf = RandomForestClassifier(n_estimators=10).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
ack = ackerman_score(scaled_features[:N//2],labels[:N//2],feature_names)
nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])

imp_without = rel_diff(ack,max(lm,svm,rf,nn))

print("Ackerman: " + str(ack))
print("SVM     : " + str(svm))
print("LM      : " + str(lm))
print("RF      : " + str(lm))
print("NN      : " + str(nn))
print("")
print("Improvement: " + str(100.0 * imp_without))
print()

print("With Noise Stat")

ns_model = NoiseStat()
print("dnn", ns_model.noise_test(features,labels,'dnn'))
print("svm", ns_model.noise_test(features,labels,'svm'))
print("lm" , ns_model.noise_test(features,labels,'lm'))
ns = ns_model.predict(features)
for i in range(len(features)):
    features[i].append(ns[i])
scaled_features = preprocessing.scale(features)
lm  = linear_model.SGDClassifier(max_iter=1000, tol=1e-3).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
svm = SVC(gamma='auto').fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
rf = RandomForestClassifier(n_estimators=10).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])
nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(scaled_features[:N//2],labels[:N//2]).score(scaled_features[N//2+1:],labels[N//2+1:])

imp_with = rel_diff(ack,max(lm,svm,rf,nn))
imp_diff = rel_diff(imp_without,imp_with)

print()
#ack = ackerman_score(features[:N//2],labels[:N//2],feature_names)
print("Ackerman: " + str(ack))
print("SVM     : " + str(svm))
print("LM      : " + str(lm))
print("RF      : " + str(lm))
print("NN      : " + str(nn))
print("")
print("Net Improvement : " + str(100.0 * imp_with))
print("NS Improvement  : " + str(100.0 * imp_diff))