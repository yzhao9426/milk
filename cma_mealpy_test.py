import numpy as np
from mealpy.evolutionary_based import CMAES

lb = [-5] * 5
ub = [5] * 5

def sphere(solution):
    return np.sum(np.square(solution))

problem = {
    "fit_func": sphere,
    "lb": lb,
    "ub": ub,
    "minmax": "min",
}

model = SimpleCMAES(
    problem,
    epoch=20,
    pop_size=20,
)

best_x, best_f = model.train()

print("CMA-ES result:")
print("x* =", best_x)
print("f(x*) =", best_f)

