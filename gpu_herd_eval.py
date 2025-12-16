import numpy as np
from numba import cuda
import time

FEED_COST_PER_KG = 0.30
MILK_TARGET = 35.0

@cuda.jit
def eggcart_cost_kernel(bw, dmi, milk, mix_matrix, costs):
    """
    GPU kernel:
      axis 0: cart index (global candidate)
      axis 1: cow index
    bw, dmi, milk: shape (n_cows,)
    mix_matrix: shape (n_carts, n_mix)
    costs: shape (n_carts, n_cows)
    """
    cart_idx, cow_idx = cuda.grid(2)

    n_carts = mix_matrix.shape[0]
    n_cows = bw.size

    if cart_idx < n_carts and cow_idx < n_cows:
        bw_i = bw[cow_idx]
        dmi_i = dmi[cow_idx]
        milk_i = milk[cow_idx]

        # 基本 feed cost
        feed_cost = FEED_COST_PER_KG * dmi_i

        # milk 偏離目標的懲罰
        diff_milk = MILK_TARGET - milk_i
        milk_penalty = 0.01 * diff_milk * diff_milk

        # mix 平滑性懲罰（相鄰 mix fraction 差太大）
        penalty = 0.0
        for j in range(mix_matrix.shape[1] - 1):
            a = mix_matrix[cart_idx, j]
            b = mix_matrix[cart_idx, j + 1]
            diff = a - b
            if diff < 0:
                diff = -diff
            penalty += diff

        # DMI/BW ratio 太低的懲罰
        ratio = dmi_i / (bw_i + 1e-6)
        if ratio < 0.02:
            penalty += (0.02 - ratio) * 100.0

        costs[cart_idx, cow_idx] = feed_cost + milk_penalty + penalty


def gpu_eval_population(mix_matrix, bw, dmi, milk, return_time=False):
    """
    在 GPU 上評估一整個 population（many egg carts）。

    mix_matrix: np.ndarray, shape (n_carts, n_mix)
    bw, dmi, milk: np.ndarray, shape (n_cows,)

    回傳:
      herd_costs: shape (n_carts,) 每個 cart 的 herd cost
      （如果 return_time=True，也回傳 GPU 計算時間）
    """
    mix_matrix = np.asarray(mix_matrix, dtype=np.float32)
    bw = np.asarray(bw, dtype=np.float32)
    dmi = np.asarray(dmi, dtype=np.float32)
    milk = np.asarray(milk, dtype=np.float32)

    n_carts, _ = mix_matrix.shape
    n_cows = bw.size

    # 傳到 GPU
    bw_dev = cuda.to_device(bw)
    dmi_dev = cuda.to_device(dmi)
    milk_dev = cuda.to_device(milk)
    mix_dev = cuda.to_device(mix_matrix)
    costs_dev = cuda.device_array((n_carts, n_cows), dtype=np.float32)

    # 2D grid/block 設定
    threads_per_block = (16, 16)
    blocks_per_grid_x = (n_carts + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n_cows + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    t0 = time.perf_counter()
    eggcart_cost_kernel[blocks_per_grid, threads_per_block](
        bw_dev, dmi_dev, milk_dev, mix_dev, costs_dev
    )
    cuda.synchronize()
    gpu_time = time.perf_counter() - t0

    costs = costs_dev.copy_to_host()
    herd_costs = costs.sum(axis=1)

    if return_time:
        return herd_costs, gpu_time
    else:
        return herd_costs


if __name__ == "__main__":
    # 這個 main 只是測試 GPU evaluator 用的，不是必要的
    rng = np.random.default_rng(42)
    n_cows = 250
    n_carts = 1000
    n_mix = 3

    bw = rng.normal(650.0, 80.0, size=n_cows).astype(np.float32)
    dmi = rng.normal(23.0, 3.0, size=n_cows).astype(np.float32)
    milk = rng.normal(35.0, 5.0, size=n_cows).astype(np.float32)

    mix_matrix = rng.random((n_carts, n_mix)).astype(np.float32)
    mix_matrix /= mix_matrix.sum(axis=1, keepdims=True)

    herd_costs, t_gpu = gpu_eval_population(mix_matrix, bw, dmi, milk, return_time=True)
    print(f"GPU herd costs computed for {n_carts} carts × {n_cows} cows in {t_gpu:.3f} s")
    print("Sample herd costs:", herd_costs[:5])
