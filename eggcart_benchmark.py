import argparse
import math
import time
from multiprocessing import Pool


def tiny_task(task_id: int, iters: int) -> float:
    """
    一個「很小」的工作，用來模擬一個非常簡短的 local solve。
    iters 越小，單個 task 越輕，越能凸顯平行 overhead。
    """
    x = task_id * 0.0001 + 1.2345
    total = 0.0
    for _ in range(iters):
        x = math.sin(x) + math.cos(x * 0.5)
        total += x
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntasks", type=int, required=True,
                        help="要跑多少個小任務（eggs）")
    parser.add_argument("--workers", type=int, required=True,
                        help="Pool 裡面有幾個 process（核心數）")
    parser.add_argument("--iters", type=int, default=200,
                        help="每個小任務裡的迴圈次數（越小越輕）")
    args = parser.parse_args()

    n_tasks = args.ntasks
    n_workers = args.workers
    iters = args.iters

    print(f"Egg-cart benchmark start:")
    print(f"  tasks   = {n_tasks}")
    print(f"  workers = {n_workers}")
    print(f"  iters   = {iters} (per task)\n")

    start = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        # 用 starmap 給每個 task 傳同一個 iters 參數
        results = pool.starmap(tiny_task, ((i, iters) for i in range(n_tasks)))
    end = time.perf_counter()

    elapsed = end - start
    print(f"Total time: {elapsed:.3f} seconds")
    print(f"Average per task: {elapsed / n_tasks:.8f} seconds")

    # 把結果記錄下來方便之後分析
    with open("eggcart_results.txt", "a") as f:
        f.write(f"{n_tasks},{n_workers},{iters},{elapsed:.6f}\n")


if __name__ == "__main__":
    main()
