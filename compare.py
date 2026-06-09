import time
import numpy as np
import matplotlib.pyplot as plt

from common import make_env, extract_mdp
from VI import value_iteration
from PI import policy_iteration

GAMMAS = [0.5, 0.9, 0.99, 0.999]
THETA = 1e-4
PLOT_PATH = "compare.png"


def vi_backups(nS, nA, iterations):
    """VI does |S|^2 * |A| Bellman backups per sweep."""
    return iterations * (nS**2) * nA


def pi_backups(nS, nA, iterations):
    """
    PI per each outer round:
      - greedy improvement: |S|^2 * |A|
      - policy evaluation: ~|S|^3
    """
    per_round = (nS**2) * nA + (nS**3)
    return iterations * per_round


def run_comparison():
    env = make_env(is_slippery=True)
    nS, nA, P, ncol = extract_mdp(env)

    results = {
        "vi": {"iters": [], "time": [], "backups": []},
        "pi": {"iters": [], "time": [], "backups": []},
    }

    for gamma in GAMMAS:
        # --- Value iteration ---
        t0 = time.perf_counter()
        _, _, vi_iters = value_iteration(P, gamma, THETA, nS, nA)
        vi_time = time.perf_counter() - t0
        vi_bk = vi_backups(nS, nA, vi_iters)

        # --- Policy iteration ---
        t0 = time.perf_counter()
        _, _, pi_iters = policy_iteration(P, gamma, nS, nA)
        pi_time = time.perf_counter() - t0
        pi_bk = pi_backups(nS, nA, pi_iters)

        results["vi"]["iters"].append(vi_iters)
        results["vi"]["time"].append(vi_time)
        results["vi"]["backups"].append(vi_bk)
        results["pi"]["iters"].append(pi_iters)
        results["pi"]["time"].append(pi_time)
        results["pi"]["backups"].append(pi_bk)

        print(f"gamma={gamma}")
        print(f"  VI: iters={vi_iters} time={vi_time:2f}s backups={vi_bk}")
        print(f"  PI: iters={pi_iters} time={pi_time:2f}s backups={pi_bk}")
        print()

    return results


def make_plot(results, path=PLOT_PATH):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(GAMMAS))
    ax.plot(x, results["vi"]["iters"], "o-", label="Value iteration")
    ax.plot(x, results["pi"]["iters"], "s-", label="Policy iteration")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(g) for g in GAMMAS])
    ax.set_xlabel(r"discount factor $\gamma$")
    ax.set_ylabel("iterations to converge")
    ax.set_title("VI vs PI: iterations to converge vs gamma (FrozenLake-v1, 4x4)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")


def main():
    results = run_comparison()
    make_plot(results)


if __name__ == "__main__":
    main()
