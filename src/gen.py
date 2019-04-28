import random
from src.instance.gaussian import Gaussian
from src.instance.uniform_random import Uniform_Random


def instance_gen(N=1):
    instances = []
    labels = []
    for i in range(N):
        c = random.choice(['clusterable', 'not clusterable'])
        if c == 'clusterable':
                labels.append(True)

                instances.append(Gaussian(dim=random.choice(range(2,10)),
                                num_clusters=random.choice(range(2,10)),
                                N=random.choice(range(50,1000)),
                                noise_frac= random.random()/3.0
                        ))
        else:
            labels.append(False)

            c = random.choice(['single_gauss','uniform'])
            if c == 'single_gauss':
                instances.append(Gaussian(dim=random.choice(range(2,10)),
                            num_clusters=1,
                            N=random.choice(range(50,1000)),
                            noise_frac= random.random()
                    ))
            elif c == 'uniform':
                instances.append(Uniform_Random(dim=random.choice(range(2,10)),N=random.choice(range(50,1000))))
    if N==1:
        return instances[0],labels[0]
    return instances,labels
