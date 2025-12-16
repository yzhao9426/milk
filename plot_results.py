import numpy as np
import matplotlib.pyplot as plt

def load_results(path="capstone_results.npy"):
    data = np.load(path, allow_pickle=True).item()
    return data

def summarize_results(results):
    for algo in ["DE", "PSO", "GA"]:
        best = np.array(results[algo]["best"])
        runtime = np.array(results[algo]["runtime"])

        print(f"=== {algo} ===")
        print(f"  runs          : {len(best)}")
        print(f"  best (mean)   : {best.mean():.4f}  ± {best.std():.4f}")
        print(f"  runtime (mean): {runtime.mean():.4f} s  ± {runtime.std():.4f} s")
        print()

def plot_best_bar(results, out="best_bar.png"):
    algos = ["DE", "PSO", "GA"]
    means = []
    stds = []
    for algo in algos:
        best = np.array(results[algo]["best"])
        means.append(best.mean())
        stds.append(best.std())

    x = np.arange(len(algos))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, algos)
    plt.ylabel("Final best value")
    plt.title("Accuracy comparison (lower is better)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_runtime_bar(results, out="runtime_bar.png"):
    algos = ["DE", "PSO", "GA"]
    means = []
    stds = []
    for algo in algos:
        rt = np.array(results[algo]["runtime"])
        means.append(rt.mean())
        stds.append(rt.std())

    x = np.arange(len(algos))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, algos)
    plt.ylabel("Runtime (s)")
    plt.title("Performance comparison (runtime)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_convergence(results, out="convergence.png"):
    algos = ["DE", "PSO", "GA"]

    plt.figure()

    for algo in algos:
        histories = results[algo]["history"]   # list of lists
        # 對齊成相同長度（取最短的那一個長度）
        min_len = min(len(h) for h in histories)
        trimmed = np.array([h[:min_len] for h in histories])
        mean_curve = trimmed.mean(axis=0)
        plt.plot(mean_curve, label=algo)

    plt.xlabel("Iteration")
    plt.ylabel("Best value")
    plt.title("Convergence curves (mean over runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

if __name__ == "__main__":
    results = load_results("capstone_results.npy")

    summarize_results(results)
    plot_best_bar(results)
    plot_runtime_bar(results)
    plot_convergence(results)

    print("Plots saved: best_bar.png, runtime_bar.png, convergence.png")

