import numpy as np
from scipy.optimize import minimize


def prefix_to_infix(tokens):
    const_cnt = -1
    tokens = [tok if tok != 'const' else tok + f'[{(const_cnt := const_cnt + 1)}]' for tok in tokens]
    return _prefix_to_infix(tokens), const_cnt + 1


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
    else:
        return token


def eval_formula(formula, x, const=None):
    from numpy import sin, cos
    return eval(formula)


def optimize_consts(formula, n_const, data):
    def mse_loss(const):
        y = eval_formula(formula, data.X_train, const)
        return np.mean((y - data.y_train) ** 2)

    opt_result = minimize(mse_loss, [1] * n_const)
    return opt_result.fun, opt_result.x
