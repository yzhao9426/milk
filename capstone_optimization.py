import time
import math
import numpy as np
from numba import cuda

# ============================================================
# 0. Global Settings
# ============================================================

# 每個 unit = 250 顆蛋 = 250 維
DIM = 250

# 搜索範圍（可以視情況調整）
LOW = -5.0
HIGH = 5.0

# Rastrigin function 的 global optimum
GLOBAL_OPTIMUM = 0.0

# 明確指定：每個 CUDA block 使用 128 個 thread（128 "cores"）
NUM_CORES = 128


# ============================================================
# 1. GPU Objective Function: 多峰 "egg-box" 地形 (Rastrigin)
# ============================================================

@cuda.jit
def rastrigin_kernel(pop, fitness):
    """
    每個 thread 負責計算一個 candidate solution（= 一個 250 維 unit）的 fitness。
    這是典型的 data-parallel SIMD/SIMT：128 個 threads 同時執行相同指令、處理不同個體。
    pop: (pop_size, DIM) on device
    fitness: (pop_size,) on device
    """
    i = cuda.grid(1)  # global thread index
    if i < pop.shape[0]:
        d = pop.shape[1]
        total = 10.0 * d
        for j in range(d):
            x = pop[i, j]
            total += x * x - 10.0 * math.cos(2.0 * math.pi * x)
        fitness[i] = total


def gpu_evaluate(pop_host):
    """
    在 GPU 上用 128 threads per block 併行計算 population fitness。
    pop_host: numpy array, shape (pop_size, DIM), dtype float32
    return: numpy array, shape (pop_size,), dtype float32
    """
    pop_host = pop_host.astype(np.float32)
    pop_gpu = cuda.to_device(pop_host)
    fitness_gpu = cuda.device_array(pop_host.shape[0], dtype=np.float32)

    threads_per_block = NUM_CORES           # 128 cores
    blocks = (pop_host.shape[0] + threads_per_block - 1) // threads_per_block

    rastrigin_kernel[blocks, threads_per_block](pop_gpu, fitness_gpu)
    cuda.synchronize()

    return fitness_gpu.copy_to_host()


# ============================================================
# 2. Egg Units: 1 unit = 250 eggs，生成很多 units
# ============================================================

def generate_unit():
    """
    生成一個 unit = 250 顆蛋，每顆蛋高度在 [LOW, HIGH] 之間。
    """
    return np.random.uniform(LOW, HIGH, size=DIM).astype(np.float32)


def generate_units(n_units=1_000_000):
    """
    生成 n_units 個不同的 units。
    注意：1_000_000 * 250 * 4 bytes ≈ 1 GB 記憶體，視你的環境調整。
    """
    units = np.random.uniform(LOW, HIGH, size=(n_units, DIM)).astype(np.float32)
    return units


def init_population_around_unit(unit, pop_size=256, noise_std=0.5):
    """
    在給定的 unit 周圍初始化一個 population。
    這對應到「隨機 pick 一個起點，然後在附近開始搜索」的概念。
    """
    base = np.tile(unit, (pop_size, 1))
    noise = np.random.normal(0.0, noise_std, size=base.shape).astype(np.float32)
    pop = base + noise
    pop = np.clip(pop, LOW, HIGH)
    return pop


# ============================================================
# 3. Differential Evolution (DE)
# ============================================================

def run_DE(pop_init, max_iter=100, F=0.5, CR=0.9):
    """
    Differential Evolution，使用 GPU 計算 fitness。
    pop_init: (pop_size, DIM) initial population
    return: best_value, runtime, history(list of best per iteration)
    """
    pop = pop_init.copy()
    pop_size, dim = pop.shape

    # 初始 fitness
    fitness = gpu_evaluate(pop)
    history = []
    start = time.perf_counter()

    rng = np.random.default_rng()

    for t in range(max_iter):
        trial_pop = np.empty_like(pop)

        for i in range(pop_size):
            # 隨機選 3 個不同的個體 a, b, c
            idxs = rng.choice(pop_size, size=3, replace=False)
            a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]

            # Mutation
            mutant = a + F * (b - c)

            # Crossover
            j_rand = rng.integers(0, dim)
            trial = np.empty(dim, dtype=np.float32)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i, j]

            # 邊界處理
            trial = np.clip(trial, LOW, HIGH)
            trial_pop[i] = trial

        # 評估 trial population
        trial_fitness = gpu_evaluate(trial_pop)

        # Selection
        for i in range(pop_size):
            if trial_fitness[i] <= fitness[i]:
                pop[i] = trial_pop[i]
                fitness[i] = trial_fitness[i]

        best_iter = float(np.min(fitness))
        history.append(best_iter)

    runtime = time.perf_counter() - start
    best = float(np.min(fitness))
    return best, runtime, history


# ============================================================
# 4. Particle Swarm Optimization (PSO)
# ============================================================

def run_PSO(pop_init, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization，使用 GPU 計算 fitness。
    pop_init: (pop_size, DIM)
    """
    pos = pop_init.copy()
    pop_size, dim = pos.shape

    rng = np.random.default_rng()

    # 初始化 velocity
    v = rng.normal(0.0, 0.1, size=pos.shape).astype(np.float32)

    # 初始 fitness
    fitness = gpu_evaluate(pos)

    pbest_pos = pos.copy()
    pbest_val = fitness.copy()

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = pos[gbest_idx].copy()
    gbest_val = float(fitness[gbest_idx])

    history = []
    start = time.perf_counter()

    for t in range(max_iter):
        # 更新 velocity 與 position
        r1 = rng.random(size=pos.shape).astype(np.float32)
        r2 = rng.random(size=pos.shape).astype(np.float32)

        cognitive = c1 * r1 * (pbest_pos - pos)
        social = c2 * r2 * (gbest_pos - pos)

        v = w * v + cognitive + social

        # 更新位置
        pos = pos + v
        pos = np.clip(pos, LOW, HIGH)

        # 評估新位置
        fitness = gpu_evaluate(pos)

        # 更新 pbest
        improved = fitness < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = fitness[improved]

        # 更新 gbest
        min_idx = int(np.argmin(fitness))
        if fitness[min_idx] < gbest_val:
            gbest_val = float(fitness[min_idx])
            gbest_pos = pos[min_idx].copy()

        history.append(gbest_val)

    runtime = time.perf_counter() - start
    best = gbest_val
    return best, runtime, history


# ============================================================
# 5. Genetic Algorithm (GA)
# ============================================================

def tournament_selection(pop, fitness, k=3, rng=None):
    """
    簡單的 tournament selection。
    回傳被選中的個體 index。
    """
    if rng is None:
        rng = np.random.default_rng()
    idxs = rng.choice(len(pop), size=k, replace=False)
    best_idx = idxs[np.argmin(fitness[idxs])]
    return best_idx


def run_GA(pop_init, max_iter=100, crossover_rate=0.9,
           mutation_rate=0.1, mutation_std=0.1):
    """
    Real-coded Genetic Algorithm，使用 GPU 計算 fitness。
    pop_init: (pop_size, DIM)
    """
    pop = pop_init.copy()
    pop_size, dim = pop.shape
    rng = np.random.default_rng()

    # 初始 fitness
    fitness = gpu_evaluate(pop)

    history = []
    start = time.perf_counter()

    for t in range(max_iter):
        new_pop = np.empty_like(pop)

        # elitism: 保留最好的一個
        elite_idx = int(np.argmin(fitness))
        elite = pop[elite_idx].copy()
        new_pop[0] = elite

        # 產生剩下的子代
        for i in range(1, pop_size):
            # selection
            p1_idx = tournament_selection(pop, fitness, k=3, rng=rng)
            p2_idx = tournament_selection(pop, fitness, k=3, rng=rng)
            parent1 = pop[p1_idx]
            parent2 = pop[p2_idx]

            child = parent1.copy()

            # crossover (簡單 arithmetic crossover)
            if rng.random() < crossover_rate:
                alpha = rng.random()
                child = alpha * parent1 + (1.0 - alpha) * parent2

            # mutation
            if rng.random() < mutation_rate:
                noise = rng.normal(0.0, mutation_std, size=dim).astype(np.float32)
                child = child + noise

            child = np.clip(child, LOW, HIGH)
            new_pop[i] = child

        pop = new_pop

        # 評估
        fitness = gpu_evaluate(pop)
        best_iter = float(np.min(fitness))
        history.append(best_iter)

    runtime = time.perf_counter() - start
    best = float(np.min(fitness))
    return best, runtime, history


# ============================================================
# 6. 實驗主迴圈：隨機 pick unit 起點，三個演算法比較
# ============================================================

def run_experiments(
    N_runs=10,
    pop_size=256,
    max_iter=100,
    n_units=1_000_000,   # 照教授說的做 1M units；如果記憶體不夠，先改小
    noise_std=0.5,
):
    """
    N_runs: 每個演算法跑幾次（不同隨機起點），用來做統計（平均 accuracy / runtime）
    pop_size: 每個演算法的 population size
    max_iter: 每次演算法迭代次數
    n_units: 先生成多少個不同的 egg units，之後每次 run 隨機 pick 一個
    """
    print(f"Generating {n_units} units (each with {DIM} eggs)...")
    units = generate_units(n_units)

    results = {
        "DE":  {"best": [], "runtime": [], "history": []},
        "PSO": {"best": [], "runtime": [], "history": []},
        "GA":  {"best": [], "runtime": [], "history": []},
    }

    rng = np.random.default_rng()

    for r in range(N_runs):
        print(f"\n=== Run {r+1} / {N_runs} ===")

        # 隨機挑一個 unit 當作起始中心（教授「隨機 pick 起點」）
        unit_idx = int(rng.integers(0, n_units))
        start_unit = units[unit_idx]
        print(f"  Using unit #{unit_idx} as start point.")

        # 在這個 unit 周圍建立 population
        pop0 = init_population_around_unit(
            start_unit,
            pop_size=pop_size,
            noise_std=noise_std
        )

        # -----------------------------
        # Differential Evolution
        # -----------------------------
        print("  [DE] running...")
        best_de, time_de, hist_de = run_DE(pop0, max_iter=max_iter)
        results["DE"]["best"].append(best_de)
        results["DE"]["runtime"].append(time_de)
        results["DE"]["history"].append(hist_de)
        print(f"  [DE] best = {best_de:.6f}, time = {time_de:.3f}s")

        # -----------------------------
        # PSO
        # -----------------------------
        print("  [PSO] running...")
        best_pso, time_pso, hist_pso = run_PSO(pop0, max_iter=max_iter)
        results["PSO"]["best"].append(best_pso)
        results["PSO"]["runtime"].append(time_pso)
        results["PSO"]["history"].append(hist_pso)
        print(f"  [PSO] best = {best_pso:.6f}, time = {time_pso:.3f}s")

        # -----------------------------
        # GA
        # -----------------------------
        print("  [GA] running...")
        best_ga, time_ga, hist_ga = run_GA(pop0, max_iter=max_iter)
        results["GA"]["best"].append(best_ga)
        results["GA"]["runtime"].append(time_ga)
        results["GA"]["history"].append(hist_ga)
        print(f"  [GA] best = {best_ga:.6f}, time = {time_ga:.3f}s")

    return results


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    # 先用較小參數測試，確定沒問題再放大
    N_RUNS = 20         # 測試時可用 3，正式實驗可以 10 或 30
    POP_SIZE = 512
    MAX_ITER = 200      # 測試用 50，正式可以 100 或更多
    N_UNITS = 100_000_000  # 測試用，HPC 足夠再改 1_000_000

    print("Starting capstone optimization experiments (GPU-only, 128-thread blocks)...")

    results = run_experiments(
        N_runs=N_RUNS,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        n_units=N_UNITS,
        noise_std=0.5,
    )

    np.save("capstone_results.npy", results, allow_pickle=True)
    print("\nAll experiments finished. Results saved to capstone_results.npy")

