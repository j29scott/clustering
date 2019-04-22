from src.instance.instance import Instance
import src.settings as settings
import numpy as np
class Gaussian(Instance):
    def __init__(self,dim=6,N=1000,num_clusters=5,noise_frac = 0.2, normal_dev = 0.2):
        self.dim = dim
        self.N = N
        self.num_clusters = num_clusters
        self.noise_frac = noise_frac

        self.cluster_centers = []
        self.normal_dev = normal_dev
        self.points = []
        self.gen()

    def gen(self):
        for i in range(self.num_clusters):
            self.cluster_centers.append(2.0 * settings.clamp * np.random.random(self.dim) - settings.clamp)
        for i in range(self.num_clusters):
            for j in range(round((1.0 - self.noise_frac) * self.N * 1.0 / self.num_clusters)):
                self.points.append(self.cluster_centers[i] + np.random.normal(0.0,self.normal_dev,self.dim))
        for i in range(round(self.N * self.noise_frac)):
            self.points.append(2.0 * settings.clamp * np.random.random(self.dim) - settings.clamp)
        self.points = np.array(self.points)