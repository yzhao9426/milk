import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO

class SphereProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=5, n_obj=1, n_constr=0, xl=-5.0, xu=5.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x))

problem = SphereProblem()
algorithm = PSO(pop_size=20)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 20),
    verbose=True,
)

print("PSO finished")
print("x* =", res.X)
print("f(x*) =", res.F)
