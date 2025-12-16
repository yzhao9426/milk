import time
import sys
from multiprocessing import Pool

# 重複模擬 "local optimization for one cow"
def simulate_one_cow(_):
    # 模擬 1–2 毫秒計算 (跟 Mark 的 local opt 類似)
    t0 = time.time()
    while time.time() - t0 < 0.001:  # 1 ms
        pass
    return 1

def run_benchmark(num_cows, workers):
    print(f"Benchmark start: {num_cows} cows with {workers} workers")
    t0 = time.time()

    with Pool(processes=workers) as p:
        p.map(simulate_one_cow, range(num_cows))

    total_time = time.time() - t0
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average per cow: {total_time/num_cows:.6f} seconds\n")


if __name__ == "__main__":
    workers = int(sys.argv[1])

    print(f"Running benchmark with {workers} cores\n")

    tests = [
        (10000, "10k"),
        (100000, "100k"),
        (1000000, "1M")
    ]

    for num_cows, label in tests:
        print("="*60)
        run_benchmark(num_cows, workers)
