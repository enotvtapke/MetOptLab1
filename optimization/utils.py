from inspect import signature

import numpy as np

def grad(fun, x, h=1e-5):
    if not callable(fun):
        raise ValueError("fun should be callable")
    dim = len(signature(fun).parameters)
    g = np.zeros(dim)
    for i in range(dim):
        step = np.zeros(dim)
        step[i] = h
        g[i] = (fun(*(x + step)) - fun(*(x - step))) / (2 * h)
    return g
