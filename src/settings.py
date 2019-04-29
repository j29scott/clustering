clamp = 1.0
verb = 1
distances = [
    'eucld'     ,
    'pnorm'     ,
    'cheb'      ,
    'taxi'      ,
    'minkowski' ,
    'PCA'
]

modality_tests = {
    'dip': ['dip_stat', 'p_value'],
    'silverman' : ['p_value']
}

additional_statistics = [
    'hopkins'
]

classifier = 'linear' ##['linear','svm','knn', 'perceptron']