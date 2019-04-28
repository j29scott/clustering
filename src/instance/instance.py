
class Instance:
    def __init__(self):

        self.dim = dim
        self.N = N
        self.clusterable = None
        self.num_clusters = num_clusters
        self.noise_frac = noise_frac
        self.points = []
        self.cluster_centers = []
        self.normal_dev = normal_dev

        for i in range(self.num_clusters):
            self.cluster_centers.append(2.0 * self.clamp * np.random.random(self.dim) - self.clamp)
        for i in range(self.num_clusters):
            for j in range(round((1.0 - self.noise_frac) * self.N * 1.0 / self.num_clusters)):
                self.points.append(self.cluster_centers[i] + np.random.normal(0.0,self.normal_dev,self.dim))
        for i in range(round(self.N * self.noise_frac)):
            self.points.append(2.0 * self.clamp * np.random.random(self.dim) - self.clamp)
        self.points = np.array(self.points)