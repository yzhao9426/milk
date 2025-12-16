import numpy as np
from scipy.optimize import differential_evolution

def sphere(x):
    return np.sum(x**2)

bounds = [(-5, 5)] * 5

result = differential_evolution(sphere, bounds, maxiter=20, polish=True)

print("DE result:")
print("x* =", result.x)
print("f(x*) =", result.fun)
