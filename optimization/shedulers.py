import numpy as np

from optimization import utils


def exp_decay(initial_lr=0.1, k=0.01):
    return lambda iteration: initial_lr * np.exp(-k * iteration)


def _wolfe_2(fun, x, p, alpha, grad, c2):
    return np.matmul(utils.grad(fun, x + alpha * p), p) >= c2 * np.matmul(grad, p)

def _wolfe_1(fun, x, p, alpha, grad, c1):
    return fun(x + alpha * p) <= fun(x) + c1 * alpha * np.matmul(grad, p)

def wolfe_conditions(fun, x, p, grad=None):
    if grad is None:
        grad = utils.grad(fun, x)
    c1 = 1e-4
    c2 = 0.9
    lr = 1e-2
    alpha = lr
    while not (_wolfe_1(fun, x, p, alpha, grad, c1) and _wolfe_2(fun, x, p, alpha, grad, c2)):
        alpha += lr
    return alpha