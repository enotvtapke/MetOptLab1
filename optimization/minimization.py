import optimization as opt


def to_scalar(fun, init_point, direction):
    return lambda x: fun(*(init_point + x * direction))


def min_in_direction(fun, init_point, direction, method="dichotomy"):
    g = to_scalar(fun, init_point, direction)
    bounds = opt.find_unimodal_interval(g, 0)
    if method == "dichotomy":
        g_min = opt.dichotomy(g, bounds, max_iter=2)
    else:
        raise ValueError("Unknown method")
    return init_point + direction * g_min
