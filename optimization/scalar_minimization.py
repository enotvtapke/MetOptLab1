import numpy as np
import optimization.utils as utils
from optimization.sсhedulers import exp_increase


def dichotomy(fun, bounds, stopping_criterion, delta=0.05, log=False):
    a, b = bounds
    step = 0
    points = []
    while True:
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta
        if fun(x1) < fun(x2):
            b = x2
        else:
            a = x1
        points.append((a + b) / 2)
        if stopping_criterion(step, points[step]):
            return np.array(points) if log else (a + b) / 2
        # if step > max_iter:
        #     return points if log else (a + b) / 2
        step += 1


def find_unimodal_interval(fun, a, initial_lr=0.05, max_lr=5):
    grad = utils.grad
    if grad(fun, a) > 0:
        raise ValueError("Function should decrease in point a.")
    lr = np.array([initial_lr])
    b = a
    while grad(fun, b) <= 0:
        b = b + lr
        lr = min(exp_increase(lr, 0.5), max_lr)
    return np.array([a, b])


# def scalar_wolfe_conditions(alpha, fun, x, direction, grad=None, c1=1e-4, c2=0.8):
#     if grad is None:
#         grad = utils.grad
#     lr = 1e-2
#     alpha = lr
#     grad_x = grad(fun, x)
#     while not (_wolfe_1(fun, x, direction, alpha, grad_x, c1) and _wolfe_2(fun, x, direction, alpha, grad, grad_x, c2)):
#         alpha = exp_increase(alpha)
#     return alpha


def _grad(fun, h=1e-5):
    return lambda x: utils.grad(fun, x, h)
