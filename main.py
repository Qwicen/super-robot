import lib
import torch
import torch.nn.functional as F
import numpy as np


def sample_formula(model, hard=False):
    samples, logits = model.forward(hard=hard)
    prefix = [alphabet[x] for x in samples]

    formula = lib.prefix_to_infix(prefix)
    y_pred = lib.eval_formula(formula, X.numpy())
    mse = np.mean((y_pred - y.numpy()) ** 2)
    return formula, mse, samples, logits

if __name__ == '__main__':
    alphabet = ['add', 'mul', 'sin', 'cos', 'sqr', 'x'] + [str(i) for i in range(-5, 6)]
    decoder = lib.LSTMDecoder(alphabet)
    opt = torch.optim.Adam(decoder.parameters())
    data = lib.Data(coeff_range=(-5, 5), degree=2, num_samples=1, num_points=100, seed=11)
    X, y = next(iter(data))
    mse_history = []
    # state = torch.load('exp_seed11_poly2.pth')
    # decoder.load_state_dict(state)
    for ep in range(1_000_000):
        formula, mse, samples, logits = sample_formula(decoder)
        rewards = [0] * len(logits)
        rewards[-1] = 1 / (1 + np.sqrt(mse))
        cumulative_rewards = lib.get_cumulative_rewards(rewards, gamma=0.9)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_for_actions = torch.sum(log_probs * F.one_hot(samples, len(alphabet)), dim=1)

        entropy = - (probs * log_probs).sum(-1).mean()
        J_hat = torch.mean(log_probs_for_actions * torch.as_tensor(cumulative_rewards))
        loss = - J_hat - entropy * 5e-2

        loss.backward()
        opt.step()
        opt.zero_grad()

        mse_history.append(mse)
        if ep % 10_000 == 0:
            print(rewards[-1], '\t', np.mean(mse_history[-10_000:]), formula)

    torch.save(decoder.state_dict(), 'exp_seed11_poly2.pth')

    print('==========================================')
    for _ in range(15):
        formula, mse, _, _ = sample_formula(decoder)
        print(formula, ';\t', mse)

    print('==========================================')
    formula, mse, _, _ = sample_formula(decoder, hard=True)
    print(formula, ';\t', mse)

