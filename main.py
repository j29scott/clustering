import numpy as np
import matplotlib.pyplot as plt
from diptest.diptest import diptest
from src.algorithms.ackerman import ackerman
import math

from src.instance.gaussian import Gaussian
from src.instance.uniform_random import Uniform_Random




x = Gaussian()
print(ackerman(x))
y = Gaussian(num_clusters=1)
print(ackerman(y))
z = Uniform_Random()
print(ackerman(z))


