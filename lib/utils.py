import torch
import numpy as np


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def arity(token):
    binary = ['add', 'mul', 'sub', 'div']
    unary = ['sin', 'cos', 'sqr', 'cub']
    nullary = ['x', 'const']
    nullary += [str(i) for i in range(-9, 10)]

    if token in binary:
        return 2
    elif token in unary:
        return 1
    elif token in nullary:
        return 0
    else:
        raise Exception(f"Unknown operation: {token}")


def get_cumulative_rewards(rewards, gamma=0.99):
    g_t = 0
    cum_rewards = []
    for r in reversed(rewards):
        g_t = r + gamma * g_t
        cum_rewards.append(g_t)
    cum_rewards.reverse()
    return cum_rewards


def context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)
    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)
    num_context : int
        Number of context points.
    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[0]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[None, locations[:num_context], :]
    y_context = y[None, locations[:num_context], :]
    x_target = x[None, locations, :]
    y_target = y[None, locations, :]
    return x_context, y_context, x_target, y_target
