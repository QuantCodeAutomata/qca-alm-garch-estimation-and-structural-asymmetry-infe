"""
Master runner: executes all six ALM-GARCH experiments sequentially.

Run:
    python run_all_experiments.py

Each experiment writes its own outputs to results/.  A final RESULTS.md
summary is written after all experiments complete.
"""
import sys
import os
import time
import warnings
import traceback

warnings.filterwarnings("ignore")

sys.path.insert(0, "/workspace/project")
sys.path.insert(0, "/workspace/project/src")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _run(exp_name: str, main_fn) -> tuple[bool, float]:
    t0 = time.time()
    print(f"\n{'#' * 72}")
    print(f"# {exp_name}")
    print(f"{'#' * 72}\n")
    try:
        main_fn()
        elapsed = time.time() - t0
        print(f"\n[{exp_name}] ✓ completed in {elapsed:.1f}s")
        return True, elapsed
    except Exception:
        elapsed = time.time() - t0
        print(f"\n[{exp_name}] ✗ FAILED after {elapsed:.1f}s:")
        traceback.print_exc()
        return False, elapsed


def main():
    # --- warm up numba once before any experiment ---
    print("Warming up numba JIT compilation …")
    import numpy as np
    from src.alm_garch import estimate_alm_garch

    rng_warm = np.random.default_rng(0)
    r_warm = rng_warm.normal(0, 0.01, 300)
    _ = estimate_alm_garch(r_warm, model="joint_symmetric", n_starts=1, seed=0)
    print("  numba warmed up.\n")

    # --- import experiment mains ---
    from exp.exp_1_estimation import main as exp1_main
    from exp.exp_2_stability import main as exp2_main
    from exp.exp_3_robustness import main as exp3_main
    from exp.exp_4_montecarlo import main as exp4_main
    from exp.exp_5_forecasting import main as exp5_main
    from exp.exp_6_insample import main as exp6_main

    experiments = [
        ("Exp 1 – ALM-GARCH Estimation & LR Tests",     exp1_main),
        ("Exp 2 – Foster-Lyapunov Stability Certificate", exp2_main),
        ("Exp 3 – Robustness Analysis",                  exp3_main),
        ("Exp 4 – Monte Carlo Size & Power",             exp4_main),
        ("Exp 5 – Out-of-Sample Forecasting",            exp5_main),
        ("Exp 6 – In-Sample Model Comparison",           exp6_main),
    ]

    statuses = []
    t_total_start = time.time()

    for name, fn in experiments:
        ok, elapsed = _run(name, fn)
        statuses.append((name, ok, elapsed))

    total = time.time() - t_total_start
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    for name, ok, t in statuses:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}  ({t:.1f}s)")
    print(f"\nTotal wall-clock time: {total:.1f}s")


if __name__ == "__main__":
    main()
