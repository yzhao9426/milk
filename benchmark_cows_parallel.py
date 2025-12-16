import argparse
import math
import time
from multiprocessing import Pool


def local_work(cow_id: int) -> float:
    """
    模擬一頭牛的 local optimization 工作。
    這裡不用真的跑 SLSQP，只用一堆數學運算來占 CPU。
    你之後可以把這裡換成真正的 SLSQP local solve。
    """
    x = cow_id * 0.0001 + 1.2345
    total = 0.0
    # 這個迴圈決定「每頭牛」的重度，覺得太快就把 3_000 調大，太慢就調小
    for i in range(3_000):
        x = math.sin(x) + math.cos(x * 0.5) + math.tan(x * 0.1)
        total += x * x
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncows", type=int, required=True,
                        help="Number of 'cows' (tasks) to simulate")
    parser.add_argument("--workers", type=int, required=True,
                        help="Number of worker processes (cores) to use")
    args = parser.parse_args()

    n_cows = args.ncows
    n_workers = args.workers

    print(f"Benchmark start: {n_cows} cows with {n_workers} workers")

    start = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        results = pool.map(local_work, range(n_cows))
    end = time.perf_counter()

    elapsed = end - start
    print(f"Total time: {elapsed:.3f} seconds")
    print(f"Average per cow: {elapsed / n_cows:.6f} seconds")

    # 寫一個小小的 log 檔，方便整理結果
    with open("benchmark_results.txt", "a") as f:
        f.write(f"{n_cows},{n_workers},{elapsed:.6f}\n")


if __name__ == "__main__":
    main()
