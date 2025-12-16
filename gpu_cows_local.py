import numpy as np
import time
from numba import cuda

# -------------------------
# 假想的 cow local objective
# -------------------------
# 對每一頭牛，我們假設有這些參數：
#   bw[i]   : 體重 (kg)
#   dmi[i]  : 乾物攝入量 (kg/d)
#   milk[i] : 產奶量 (kg/d)
#
# 我們假設有 3 個 mix，global 解決的是 ingredient 組成，
# local 這邊只看「每頭牛吃的 3 個 mix 的比例」，用一個已知的 mix vector 來算成本。
#
# 這裡的 cost 先用「假公式」：
#   cost_i = feed_cost_per_kg * dmi[i]
#            + 0.01 * (milk_target - milk[i])**2
#            + nutrient_penalty
#
# nutrient_penalty 也用一個簡單的東西模擬：mix 組合不平衡就加罰。

FEED_COST_PER_KG = 0.30
MILK_TARGET = 35.0

@cuda.jit
def cow_cost_kernel(bw, dmi, milk, mix_vec, costs):
    """
    GPU kernel：
    每個 thread 算一頭牛的 cost。

    bw, dmi, milk: 1D device arrays, 長度 n_cows
    mix_vec: 1D device array, 長度 n_mix (例如 3)
    costs: 1D device array, 長度 n_cows (output)
    """
    i = cuda.grid(1)
    n = bw.size
    if i < n:
        bw_i = bw[i]
        dmi_i = dmi[i]
        milk_i = milk[i]

        # 飼料成本
        feed_cost = FEED_COST_PER_KG * dmi_i

        # 產奶偏離目標的罰則
        milk_penalty = 0.01 * (MILK_TARGET - milk_i) * (MILK_TARGET - milk_i)

        # 假的「mix 不平衡罰則」：|mix1 - mix2| + |mix2 - mix3| ...
        penalty = 0.0
        for j in range(mix_vec.size - 1):
            diff = mix_vec[j] - mix_vec[j + 1]
            if diff < 0:
                diff = -diff
            penalty += diff

        # 再加一個簡單的「體重 & DMI」平衡罰則
        # 讓高 BW 但低 DMI 的牛多一點 penalty
        ratio = dmi_i / (bw_i + 1e-6)
        if ratio < 0.02:
            penalty += (0.02 - ratio) * 100.0

        costs[i] = feed_cost + milk_penalty + penalty


def cpu_cow_costs(bw, dmi, milk, mix_vec):
    """
    CPU 版對照實作，用同樣的公式。
    """
    n = len(bw)
    costs = np.empty(n, dtype=np.float32)
    for i in range(n):
        bw_i = bw[i]
        dmi_i = dmi[i]
        milk_i = milk[i]

        feed_cost = FEED_COST_PER_KG * dmi_i
        milk_penalty = 0.01 * (MILK_TARGET - milk_i) ** 2

        penalty = 0.0
        for j in range(len(mix_vec) - 1):
            diff = mix_vec[j] - mix_vec[j + 1]
            penalty += abs(diff)

        ratio = dmi_i / (bw_i + 1e-6)
        if ratio < 0.02:
            penalty += (0.02 - ratio) * 100.0

        costs[i] = feed_cost + milk_penalty + penalty

    return costs


def main():
    # ---------------------
    # 1. 生成假資料 (cows)
    # ---------------------
    n_cows = 250  # 這個對應 Mark 說的「每個 global 解 250 頭牛」
    rng = np.random.default_rng(42)

    bw = rng.normal(650.0, 80.0, size=n_cows).astype(np.float32)    # 體重
    dmi = rng.normal(23.0, 3.0, size=n_cows).astype(np.float32)     # DMI
    milk = rng.normal(35.0, 5.0, size=n_cows).astype(np.float32)    # 產奶

    # ---------------------
    # 2. 假設一組 mix 組成
    # ---------------------
    # 例如：3 個 mix 的比例，這裡先固定一組
    mix_vec = np.array([0.4, 0.35, 0.25], dtype=np.float32)

    # ---------------------
    # 3. CPU baseline
    # ---------------------
    t0 = time.perf_counter()
    cpu_costs = cpu_cow_costs(bw, dmi, milk, mix_vec)
    cpu_time = time.perf_counter() - t0
    print(f"[CPU] computed {n_cows} cows in {cpu_time:.6f} s")

    # ---------------------
    # 4. GPU 計算
    # ---------------------
    # 把資料搬到 GPU
    bw_dev = cuda.to_device(bw)
    dmi_dev = cuda.to_device(dmi)
    milk_dev = cuda.to_device(milk)
    mix_dev = cuda.to_device(mix_vec)
    costs_dev = cuda.device_array_like(bw)  # same length & dtype

    threads_per_block = 128
    blocks_per_grid = (n_cows + threads_per_block - 1) // threads_per_block

    t1 = time.perf_counter()
    cow_cost_kernel[blocks_per_grid, threads_per_block](bw_dev, dmi_dev, milk_dev, mix_dev, costs_dev)
    cuda.synchronize()
    gpu_time = time.perf_counter() - t1

    gpu_costs = costs_dev.copy_to_host()
    print(f"[GPU] computed {n_cows} cows in {gpu_time:.6f} s")

    # ---------------------
    # 5. 比較 CPU / GPU 結果
    # ---------------------
    max_abs_diff = np.max(np.abs(cpu_costs - gpu_costs))
    print(f"Max |CPU - GPU| difference: {max_abs_diff:.6e}")
    print("Sample (first 5 cows):")
    for i in range(5):
        print(f"cow {i:3d}: cpu={cpu_costs[i]:.6f}, gpu={gpu_costs[i]:.6f}")


if __name__ == "__main__":
    main()
