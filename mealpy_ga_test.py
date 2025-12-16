import numpy as np
from mealpy import FloatVar, GA

def objective_function(solution):
    # 簡單 sphere 測試函數：f(x) = sum(x^2)
    return np.sum(np.square(solution))

# 定義 5 維連續變數，每一維在 [-5, 5]
bounds = FloatVar(
    lb=[-5.0] * 5,
    ub=[5.0] * 5,
    name="x"
)

problem = {
    "obj_func": objective_function,  # 目標函數
    "bounds": bounds,                # 搜索空間
    "minmax": "min",                 # 最小化問題
}

# 建立 GA 模型（用 BaseGA，最標準的版本）
model = GA.BaseGA(
    epoch=50,        # 疊代次數，可以先用少一點
    pop_size=20,     # 族群大小
    pc=0.9,          # 交配概率
    pm=0.05          # 突變概率
)

# solve() 是核心 API
g_best = model.solve(problem)

print("GA (MealPy) result:")
print("x* =", g_best.solution)
print("f(x*) =", g_best.target.fitness)
