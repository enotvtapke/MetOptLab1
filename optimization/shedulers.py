import numpy as np

def exp_decay(initial_lr = 0.1, k = 0.01):
   return lambda iteration: initial_lr * np.exp(-k * iteration)