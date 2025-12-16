import numpy as np
import time
from numba import cuda

FEED_COST_PER_KG = 0.30
MILK_TARGET = 35.0

@cuda.jit
def eggcart_cost_kernel(bw, dmi, milk, mix_matrix, costs):
    """
    2D GPU kernel:
      - axis 0: egg cart / candidate index
      - axis 1: cow index

    bw, dmi, milk: 1D arrays, length n_cows
    mix_matrix: 2D array, shape (n_carts, n_mix)
    costs: 2D array, shape (n_carts, n_cows)
    """
    cart_idx, cow_idx = cuda.grid(2)

    n_carts = mix_matrix.shape[0]
    n_cows = bw.size

    if cart_idx < n_carts and cow_idx < n_cows:
        bw_i = bw[cow_idx]
        dmi_i = dmi[cow_idx]
        milk_i = milk[cow_idx]

        # feed cost
        feed_cost = FEED_COST_PER_KG * dmi_i

        # milk deviation penalty
        diff_milk = MILK_TARGET - milk_i
        milk_penalty = 0.01 * diff_milk * diff_milk

        # mix imbalance penalty (|mix[j] - mix[j+1]| sum)
        penalty = 0.0
        for j in range(mix_matrix.shape[1] - 1):
            a = mix_matrix[cart_idx, j]
            b = mix_matrix[cart_idx, j + 1]
            diff = a - b
            if diff < 0:
                diff = -diff
            penalty += diff

        # DMI/BW penalty if ratio too low
        ratio = dmi_i / (bw_i + 1e-6)
        if ratio < 0.02:
            penalty += (0.02 - ratio) * 100.0

        costs[cart_idx, cow_idx] = feed_cost + milk_penalty + penalty


def cpu_eggcart_costs(bw, dmi, milk, mix_matrix):
    """
    CPU 對照版：
      mix_matrix: shape (n_carts, n_mix)
    回傳：
      herd_costs: shape (n_carts,) 每一個 cart 的 herd 總 cost
    """
    n_carts, n_mix = mix_matrix.shape
    n_cows = len(bw)
    herd_costs = np.empty(n_carts, dtype=np.float32)

    for c in range(n_carts):
        cart_mix = mix_matrix[c, :]
        total = 0.0
        for i in range(n_cows):
            bw_i = bw[i]
            dmi_i = dmi[i]
            milk_i = milk[i]

            feed_cost = FEED_COST_PER_KG * dmi_i
            milk_penalty = 0.01 * (MILK_TARGET - milk_i) ** 2

            penalty = 0.0
            for j in range(n_mix - 1):
                penalty += abs(cart_mix[j] - cart_mix[j + 1])

            ratio = dmi_i / (bw_i + 1e-6)
            if ratio < 0.02:
                penalty += (0.02 - ratio) * 100.0

            total += feed_cost + milk_penalty + penalty

        herd_costs[c] = total

    return herd_costs


def main():
    rng = np.random.default_rng(42)

    # ---------------------
    # 1. 牛群資料
    # ---------------------
    n_cows = 250
    bw = rng.normal(650.0, 80.0, size=n_cows).astype(np.float32)
    dmi = rng.normal(23.0, 3.0, size=n_cows).astype(np.float32)
    milk = rng.normal(35.0, 5.0, size=n_cows).astype(np.float32)

    # ---------------------
    # 2. 多組 egg carts（global candidates）
    # ---------------------
    n_carts = 1000         # 你可以改成 10000, 50000, 100000 來拉高工作量
    n_mix = 3              # 先假設 3 個 mix

    mix_matrix = rng.random((n_carts, n_mix)).astype(np.float32)
    # 正規化成每一行 sum = 1
    mix_matrix /= mix_matrix.sum(axis=1, keepdims=True)

    # ---------------------
    # 3. CPU baseline
    # ---------------------
    print(f"Computing CPU herd costs for {n_carts} carts × {n_cows} cows ...")
    t0 = time.perf_counter()
    cpu_herd_costs = cpu_eggcart_costs(bw, dmi, milk, mix_matrix)
    cpu_time = time.perf_counter() - t0
    print(f"[CPU] total time: {cpu_time:.3f} s")

    # ---------------------
    # 4. GPU 計算：每個 (cart, cow) 一個 thread
    # ---------------------
    bw_dev = cuda.to_device(bw)
    dmi_dev = cuda.to_device(dmi)
    milk_dev = cuda.to_device(milk)
    mix_dev = cuda.to_device(mix_matrix)
    costs_dev = cuda.device_array((n_carts, n_cows), dtype=np.float32)

    # 2D grid/block 設定
    threads_per_block = (16, 16)   # 16 × 16 = 256 threads per block
    blocks_per_grid_x = (n_carts + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n_cows + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    print(f"Launching GPU kernel with grid={blocks_per_grid}, block={threads_per_block} ...")

    t1 = time.perf_counter()
    eggcart_cost_kernel[blocks_per_grid, threads_per_block](bw_dev, dmi_dev, milk_dev, mix_dev, costs_dev)
    cuda.synchronize()
    gpu_time_kernel = time.perf_counter() - t1

    costs = costs_dev.copy_to_host()    # shape (n_carts, n_cows)
    # 對每一個 cart，把 250 頭牛的 cost 加總成 herd cost
    gpu_herd_costs = costs.sum(axis=1)

    print(f"[GPU] kernel + copy time: {gpu_time_kernel:.3f} s")

    # ---------------------
    # 5. 結果比較
    # ---------------------
    max_abs_diff = np.max(np.abs(cpu_herd_costs - gpu_herd_costs))
    print(f"Max |CPU - GPU| herd cost difference: {max_abs_diff:.6e}")

    print("Sample of first 5 carts (herd cost):")
    for c in range(5):
        print(f"cart {c:3d}: cpu={cpu_herd_costs[c]:.3f}, gpu={gpu_herd_costs[c]:.3f}")


if __name__ == "__main__":
    main()
