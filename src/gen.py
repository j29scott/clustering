import random
from src.instance.gaussian import Gaussian
from src.instance.uniform_random import Uniform_Random


def instance_gen(NGEN=1):
    instances = []
    labels = []
    for i in range(NGEN):
        c = random.choice(['clusterable', 'not clusterable'])
        if c == 'clusterable':
                labels.append(True)

                instances.append(Gaussian(dim=random.choice(range(2,10)),
                                num_clusters=random.choice(range(2,10)),
                                N=random.choice(range(50,1000)),
                                noise_frac= random.random()
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
    if NGEN==1:
        return instances[0],labels[0]
    return instances,labels


def instance_rand_noise_gen(N=1):
    instances = []
    labels = []

    for i in range(N):
        labels.append(random.random())

        instances.append(Gaussian(dim=random.choice(range(2,10)),
                        num_clusters=random.choice(range(2,10)),
                        N=random.choice(range(50,1000)),
                        noise_frac= labels[-1]
                ))

    if N==1:
        return instances[0],labels[0]
    return instances,labels

def break_gen():
    return Gaussian(dim=2,
                                num_clusters=random.choice(range(2,3+1)),
                                N=250,
                                noise_frac= 0.10 + random.random()/10.0
            )