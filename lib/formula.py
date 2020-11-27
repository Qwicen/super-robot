import numpy as np
from scipy.optimize import minimize


def prefix_to_infix(tokens):
    tokens = tokens.copy()
    return _prefix_to_infix(tokens)


def _prefix_to_infix(tokens):
    token = tokens.pop(0)
    if token == 'add':
        return f'({_prefix_to_infix(tokens)} + {_prefix_to_infix(tokens)})'
    elif token == 'sub':
        return f'{_prefix_to_infix(tokens)} - {_prefix_to_infix(tokens)}'
    elif token == 'mul':
        return f'{_prefix_to_infix(tokens)} * {_prefix_to_infix(tokens)}'
    elif token == 'div':
        return f'{_prefix_to_infix(tokens)} / {_prefix_to_infix(tokens)}'
    elif token == 'sin':
        return f'sin({_prefix_to_infix(tokens)})'
    elif token == 'cos':
        return f'cos({_prefix_to_infix(tokens)})'
    elif token == 'sqr':
        return f'(({_prefix_to_infix(tokens)}) ** 2)'
    elif token == 'cub':
        return f'(({_prefix_to_infix(tokens)}) ** 3)'
    else:
        return token


def eval_formula(formula, x, const=None):
    from numpy import sin, cos
    return eval(formula)


def optimize_consts(formula, n_const, X, y):
    def mse_loss(const):
        y_pred = eval_formula(formula, X, const)
        return np.mean((y_pred - y) ** 2)

    opt_result = minimize(mse_loss, np.ones(n_const))
    return opt_result.fun, opt_result.x
