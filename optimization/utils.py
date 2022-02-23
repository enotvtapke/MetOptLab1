import numpy as np


def grad(fun, x, h=1e-5):
    if not callable(fun):
        raise ValueError("fun should be callable")
    dim = len(x)
    g = np.zeros(dim)
    step = np.zeros(dim)
    for i in range(dim):
        step[i] = h
        g[i] = (fun(x + step) - fun(x - step)) / (2 * h)
        step[i] = 0
    return g
