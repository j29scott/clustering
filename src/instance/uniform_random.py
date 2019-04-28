from src.instance.instance import Instance
import src.settings as settings
import numpy as np
class Uniform_Random(Instance):
    def __init__(self,dim=2,N=1000):
        self.dim = dim
        self.N = N
        self.points = []
        self.gen()

    def gen(self):
        for i in range(self.N):
            self.points.append(2.0 * settings.clamp * np.random.random(self.dim) - settings.clamp)
        self.points = np.array(self.points)