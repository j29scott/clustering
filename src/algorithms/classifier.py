import src.settings as settings
import src.dist_util as dist_util
from src.algorithms.ackerman import hopkins
from src.util import printer
import pdb
from diptest.diptest import diptest
from sklearn import linear_model
import numpy as np
import modality
import random
from sklearn.decomposition import PCA

def get_feature_names():
    feature_names = []
    if settings.use_N_DIM:
        feature_names.append('dim')
        feature_names.append('N')
    for dist in settings.distances:
        feature_names.append(dist + "_mean")
        feature_names.append(dist + "_dev")
        for test in settings.modality_tests:
            if test == 'dip':
                if 'dip_stat' in settings.modality_tests[test]:
                    feature_names.append(dist + '_dip_stat')
                if 'p_value' in settings.modality_tests[test]:
                    feature_names.append(dist + '_dip_p_value')

            if test == 'silverman':
                if 'p_value' in settings.modality_tests[test]:
                    feature_names.append(dist + '_silv_p_value')

    for feat in settings.additional_statistics:
        if feat == 'hopkins':
            feature_names.append('hopkins')
    
    
    return feature_names

def instance2features(instance):
    features = []
    if settings.use_N_DIM:
        features.append(instance.dim)
        features.append(len(instance.points))  
    for dist in settings.distances:
        distances,mean,dev = dist_util.distance(instance,dist)
        features.append(mean)
        features.append(dev)
        for test in settings.modality_tests:
            if test == 'dip':
                out = diptest(distances)
                if 'dip_stat' in settings.modality_tests[test]:
                    features.append(out[0])
                if 'p_value' in settings.modality_tests[test]:
                    features.append(out[1])

            if test == 'silverman':
                if 'p_value' in settings.modality_tests[test]:
                    out = modality.silverman_bwtest(np.random.choice(distances,250),alpha=0.05)
                    assert isinstance(out,float)
                    features.append(out)

    for feat in settings.additional_statistics:
        if feat == 'hopkins':
            features.append(hopkins(instance.points))
        

    return features