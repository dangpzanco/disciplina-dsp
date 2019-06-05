import autograd.numpy as np
import autograd


def func(x):
    y = np.exp(x)

    return y


d = autograd.grad(func)

print(d(0.0))

