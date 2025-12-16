import numpy as np
import time
from numba import cuda

@cuda.jit
def square_kernel(x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = x[i] * x[i]

def main():
    print("cuda.is_available:", cuda.is_available())
    try:
        cuda.detect()
    except Exception as e:
        print("cuda.detect() error:", e)

    n = 10_000_000
    x_host = np.linspace(0, 1, n).astype(np.float32)
    y_host = np.zeros_like(x_host)

    x_dev = cuda.to_device(x_host)
    y_dev = cuda.device_array_like(x_host)

    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    start = time.perf_counter()
    square_kernel[blocks_per_grid, threads_per_block](x_dev, y_dev)
    cuda.synchronize()
    end = time.perf_counter()

    y_host = y_dev.copy_to_host()

    print(f"GPU square kernel done for n={n}")
    print(f"Time elapsed: {end - start:.4f} seconds")
    print(f"Sample: x[12345]={x_host[12345]:.6f}, y[12345]={y_host[12345]:.6f}")

if __name__ == "__main__":
    main()
