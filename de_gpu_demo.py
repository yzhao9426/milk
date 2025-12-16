import time
import numpy as np
from gpu_herd_eval import gpu_eval_population

# -----------------------------
# Problem setup: cows + mixes
# -----------------------------

def make_cow_data(n_cows=250, seed=42):
    rng = np.random.default_rng(seed)
    bw = rng.normal(650.0, 80.0, size=n_cows).astype(np.float32)   # body weight
    dmi = rng.normal(23.0, 3.0, size=n_cows).astype(np.float32)    # dry matter intake
    milk = rng.normal(35.0, 5.0, size=n_cows).astype(np.float32)   # milk yield
    return bw, dmi, milk


# -----------------------------
# DE hyperparameters
# -----------------------------

DIM = 3            # number of mixes
POP_SIZE = 128     # population size
F = 0.8            # mutation factor
CR = 0.9           # crossover rate
N_GEN = 100        # number of generations


def init_population(pop_size, dim, rng):
    """Initialize population of mix fractions (each row sums to 1)."""
    pop = rng.random((pop_size, dim)).astype(np.float32)
    pop /= pop.sum(axis=1, keepdims=True)
    return pop


def repair(x):
    """Repair vector: clip to >=0, renormalize to sum=1."""
    x = np.maximum(x, 1e-8)
    s = x.sum()
    if s <= 0:
        x[:] = 1.0 / len(x)
    else:
        x /= s
    return x


def de_optimize(bw, dmi, milk, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)

    # 初始化 population
    pop = init_population(POP_SIZE, DIM, rng)

    # 初始 cost（一次丟給 GPU 算）
    costs = gpu_eval_population(pop, bw, dmi, milk)
    best_idx = np.argmin(costs)
    best_cost = float(costs[best_idx])
    best_vec = pop[best_idx].copy()

    print(f"[Init] best cost = {best_cost:.4f}, mix = {best_vec}")

    t0 = time.perf_counter()

    for gen in range(1, N_GEN + 1):
        trial_pop = np.empty_like(pop)

        # DE/rand/1/bin
        for i in range(POP_SIZE):
            # 隨機選三個不同索引
            idxs = np.arange(POP_SIZE)
            rng.shuffle(idxs)
            a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]

            mutant = a + F * (b - c)

            # crossover
            cross_mask = rng.random(DIM) < CR
            # 至少讓一個維度來自 mutant
            if not cross_mask.any():
                cross_mask[rng.integers(0, DIM)] = True

            trial = np.where(cross_mask, mutant, pop[i])
            trial = repair(trial)
            trial_pop[i] = trial

        # 用 GPU 一次算整個 trial population 的 cost
        trial_costs = gpu_eval_population(trial_pop, bw, dmi, milk)

        # selection
        improved = trial_costs < costs
        pop[improved] = trial_pop[improved]
        costs[improved] = trial_costs[improved]

        # 更新 global best
        gen_best_idx = np.argmin(costs)
        gen_best_cost = float(costs[gen_best_idx])
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_vec = pop[gen_best_idx].copy()

        if gen % 10 == 0 or gen == 1:
            print(f"[Gen {gen:3d}] best cost = {best_cost:.4f}, mix = {best_vec}")

    total_time = time.perf_counter() - t0
    return best_cost, best_vec, total_time


if __name__ == "__main__":
    # 產生一批固定的牛，之後 PSO / GA 也可以用同一批來比較
    bw, dmi, milk = make_cow_data(n_cows=250, seed=42)

    print("Running DE on GPU with:")
    print(f"  POP_SIZE = {POP_SIZE}, N_GEN = {N_GEN}, DIM = {DIM}")

    best_cost, best_vec, total_time = de_optimize(bw, dmi, milk)

    print("\n==== Final DE result (GPU) ====")
    print(f"Best herd cost: {best_cost:.6f}")
    print(f"Best mix fractions: {best_vec}")
    print(f"Total DE runtime: {total_time:.3f} s")
