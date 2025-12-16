import numpy as np
from scipy.optimize import differential_evolution, minimize
from multiprocessing import Pool


# === 目標函數：一個有很多 local minima 的玩具函數 ===
def objective_function(x):
    # x 是長度 3 的向量
    return (x[0] - 2)**2 + np.sin(x[1] * 3) + (x[2] + 1)**4 + np.cos(x[0] * x[2])


# 單一次「DE + SLSQP」優化流程（之後會被平行呼叫）
def run_single_optimization(seed):
    np.random.seed(seed)

    # 全域搜尋：DE（global opt）
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    de_result = differential_evolution(
        objective_function,
        bounds,
        seed=seed,
        polish=False,      # 不在 DE 裡面做 local polish，留給 SLSQP
        updating="deferred",
        workers=1,         # 我們用 multiprocessing 自己平行，多跑幾個 seed
    )

    # 區域搜尋：SLSQP（local opt）
    x0 = de_result.x
    local_result = minimize(objective_function, x0, method="SLSQP")

    return local_result.fun, local_result.x


if __name__ == "__main__":
    num_processes = 4   # 同時用幾個 process（對應一個 node 上的 core 數）
    num_runs = 10       # 做幾個獨立的 DE+SLSQP run（不同 seed）

    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_optimization, range(num_runs))

    # 從多個平行 run 中挑最好的
    best_fun = float("inf")
    best_x = None
    for fun, x in results:
        if fun < best_fun:
            best_fun = fun
            best_x = x

    print("Global minimum found:", best_fun)
    print("Optimal parameters:", best_x)
