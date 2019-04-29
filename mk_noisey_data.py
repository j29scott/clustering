from src.gen import instance_rand_noise_gen 
from src.algorithms.classifier import instance2features

import sys


while True:
    instance,noise = instance_rand_noise_gen()
    features = instance2features(instance)
    features.append(noise)
    print(features)
    sys.stdout.flush()