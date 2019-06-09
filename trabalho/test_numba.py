from numba import jit
import numpy as np
np.set_printoptions(formatter={'float':lambda x: f'{x:.3f}'})

from numba import jit, njit
from numba.types import float64, int64
import numpy as np

# @jit(float64[:](float64[:], int64, float64[:]),nopython=True)
@jit(nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)

@jit(nopython=True)
def numpy_quantizer(x, n):
    """ 
    Input:
        x [array] - Input signal
        n [int] - Fractional bits
    Output:
        y - Quantized signal
    """

    # Quantize signal
    M = 2 ** n
    # y = np.round_(x * M, 0, None) / M
    y = np.empty(x.shape)
    rnd1(x * M, 0, y)
    y /= M

    return y

def floop_quantizer(x, n):

    # Quantize signal
    M = 2 ** n
    y = np.empty(x.size)
    for i in range(x.size):
        y[i] = round(x[i] * M) / M

    return y

n = 2
x = np.linspace(-1,1,8)

y1 = numpy_quantizer(x, n)
y2 = floop_quantizer(x, n)

print('Input:            ', x)
print('Output (np.round):', y1)
print('Output (for loop):', y2)
print('Difference:       ', y1-y2)

