from src.gen import instance_gen 
from src.algorithms.classifier import instance2features

import sys


while True:
    instance,clusterable = instance_gen()
    features = instance2features(instance)
    features.append(clusterable)
    print(features)
    sys.stdout.flush()