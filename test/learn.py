from __future__ import print_function
import torch
import numpy as np

x = np.random.rand(4, 3)
print(x)
print(x.shape)

x = x[:, [0, 2]]
print(x)
print(x.shape)