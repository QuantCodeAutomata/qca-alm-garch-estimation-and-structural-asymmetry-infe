"""
Experiment 4 – Monte Carlo evaluation of ALM-GARCH LR tests.

DGPs
----
A  Symmetric null       : xi_-=xi_+=0.05,   gamma_-=gamma_+=5.0,  mu=1e-4
B  Pure level asymmetry : xi_+=0.01, xi_-=0.10, gamma_-=gamma_+=5.0, mu=1e-4
C  Pure memory asymmetry: xi_+=xi_-=0.05, gamma_+=2.0, gamma_-=10.0, mu=1e-4
D  Joint asymmetry      : xi_+=0.01, xi_-=0.10, gamma_+=2.0, gamma_-=10.0, mu=1e-4
E  Near-boundary (S&P)  : xi_+=1e-5, xi_-=0.08, gamma_+=5.0, gamma_-=8.0, mu=1e-4

For each DGP: M=200 replications, N=2000.
Tests:
  LR_lev ~ chi2(1): unrestricted vs level_restricted
  LR_mem ~ chi2(1): unrestricted vs memory_restricted
  LR_sym ~ chi2(2): unrestricted vs joint_symmetric

Saves
-----
results/exp4_rejection_rates.csv
results/exp4_lr_distributions.png
results/exp4_score_correlation.csv
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from src.alm_garch import (
    ALMGARCHParams,
    simulate_alm_garch,
    lr_test,
    _BOUNDS, _pack_theta, _unpack_theta,
    alm_garch_loglik, alm_garch_filter,
    ALMGARCHResult,
)
from scipy import optimize as sp_optimize

# ---------------------------------------------------------------------------
# DGP definitions
# ---------------------------------------------------------------------------

DGPS = {
    "A_symmetric": ALMGARCHParams(
        mu=1e-4, xi_plus=0.05, xi_minus=0.05,
        gamma_plus=5.0, gamma_minus=5.0,
    ),
    "B_level_asym": ALMGARCHParams(
        mu=1e-4, xi_plus=0.01, xi_minus=0.10,
        gamma_plus=5.0, gamma_minus=5.0,
    ),
    "C_memory_asym": ALMGARCHParams(
        mu=1e-4, xi_plus=0.05, xi_minus=0.05,
        gamma_plus=2.0, gamma_minus=10.0,
    ),
    "D_joint_asym": ALMGARCHParams(
        mu=1e-4, xi_plus=0.01, xi_minus=0.10,
        gamma_plus=2.0, gamma_minus=10.0,
    ),
    "E_near_boundary": ALMGARCHParams(
        mu=1e-4, xi_plus=1e-5, xi_minus=0.08,
        gamma_plus=5.0, gamma_minus=8.0,
    ),
}

M = 50        # replications (paper uses 1000; reduced for feasibility)
N = 1000      # sample length (paper uses 4000; reduced for feasibility)
ALPHA = 0.05  # nominal size


# ---------------------------------------------------------------------------
# Fast sequential estimator (L-BFGS-B only; no Nelder-Mead)
# ---------------------------------------------------------------------------

def _fast_estimate_mc(
    r: np.ndarray,
    model: str,
    p: float = 1.2,
    tau: float = 1.0,
    n_starts: int = 2,
    seed: int = 42,
) -> ALMGARCHResult:
    """Light-weight ALM-GARCH estimator for Monte Carlo inner loop."""
    r = np.asarray(r, dtype=np.float64)
    sample_var = float(np.var(r))
    log_sv = np.log(max(sample_var, 1e-8))

    starts = [
        _pack_theta(log_sv - 4.0, np.log(0.05), np.log(0.05),
                    np.log(50.0), np.log(50.0), model),
        _pack_theta(log_sv - 3.5, np.log(0.03), np.log(0.10),
                    np.log(20.0), np.log(5.0), model),
    ]
    rng = np.random.RandomState(seed)
    for _ in range(n_starts):
        starts.append(
            _pack_theta(
                rng.uniform(log_sv - 5.0, log_sv + 1.0),
                rng.uniform(-6.0, 2.0), rng.uniform(-4.0, 3.0),
                rng.uniform(-1.0, 6.0), rng.uniform(-1.0, 6.0),
                model,
            )
        )

    bounds = _BOUNDS[model]
    best_nll, best_theta = np.inf, None

    for theta0 in starts:
        try:
            res = sp_optimize.minimize(
                alm_garch_loglik, theta0,
                args=(r, model, p, tau),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6},
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll, best_theta = res.fun, res.x.copy()
        except Exception:
            pass

    if best_theta is None:
        best_theta = starts[0]

    mu, xi_p, xi_m, gp, gm = _unpack_theta(best_theta, model)
    params = ALMGARCHParams(mu=mu, xi_plus=xi_p, xi_minus=xi_m,
                            gamma_plus=gp, gamma_minus=gm, p=p, tau=tau)
    X_arr, c_arr, sigma2, loglik = alm_garch_filter(r, mu, xi_p, xi_m, gp, gm, p, tau)
    residuals = r / np.sqrt(np.maximum(sigma2, 1e-16))

    return ALMGARCHResult(
        params=params, loglik=loglik, n_obs=len(r),
        filtered_X=X_arr, filtered_c=c_arr,
        filtered_sigma2=sigma2, residuals=residuals,
        converged=(best_nll < 1e9),
    )


# ---------------------------------------------------------------------------
# One MC replication  (sequential — no loky spawning)
# ---------------------------------------------------------------------------

def _one_replication(rep_idx: int, dgp_params: ALMGARCHParams, N: int) -> dict:
    """Run one Monte Carlo replication. Sequential — no joblib needed."""
    seed = 42 + rep_idx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        returns = simulate_alm_garch(dgp_params, n=N, seed=seed)

        logliks = {}
        failed = False
        for model in ("unrestricted", "level_restricted",
                      "memory_restricted", "joint_symmetric"):
            try:
                res = _fast_estimate_mc(
                    returns, model=model, p=1.2, tau=1.0,
                    n_starts=2, seed=seed + 1,
                )
                logliks[model] = res.loglik
            except Exception:
                logliks[model] = np.nan
                failed = True

    ll_u = logliks.get("unrestricted", np.nan)
    ll_l = logliks.get("level_restricted", np.nan)
    ll_m = logliks.get("memory_restricted", np.nan)
    ll_s = logliks.get("joint_symmetric", np.nan)

    def _lr(ll_alt, ll_null, df):
        if not (np.isfinite(ll_alt) and np.isfinite(ll_null)):
            return np.nan, np.nan
        stat, pval = lr_test(ll_alt, ll_null, df)
        return max(stat, 0.0), pval

    lr_lev, pval_lev = _lr(ll_u, ll_l, df=1)
    lr_mem, pval_mem = _lr(ll_u, ll_m, df=1)
    lr_sym, pval_sym = _lr(ll_u, ll_s, df=2)

    return {
        "lr_lev": lr_lev, "lr_mem": lr_mem, "lr_sym": lr_sym,
        "pval_lev": pval_lev, "pval_mem": pval_mem, "pval_sym": pval_sym,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Run MC for a single DGP  (sequential loop)
# ---------------------------------------------------------------------------

def run_dgp(dgp_name: str, dgp_params: ALMGARCHParams) -> pd.DataFrame:
    """Run M replications for one DGP sequentially; return tidy DataFrame."""
    print(f"  Running DGP {dgp_name} ({M} reps × N={N})...", flush=True)
    raw = []
    for rep_idx in range(M):
        result = _one_replication(rep_idx, dgp_params, N)
        raw.append(result)
        if (rep_idx + 1) % 10 == 0:
            print(f"    rep {rep_idx+1}/{M}", flush=True)
    df = pd.DataFrame(raw)
    df.insert(0, "dgp", dgp_name)
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def rejection_rate(pvals: pd.Series) -> float:
    valid = pvals.dropna()
    if len(valid) == 0:
        return np.nan
    return float((valid < ALPHA).mean())


def compute_rejection_rates(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dgp_name, df in results.items():
        n_failed = int(df["failed"].sum())
        rows.append({
            "dgp":          dgp_name,
            "reject_lev":   rejection_rate(df["pval_lev"]),
            "reject_mem":   rejection_rate(df["pval_mem"]),
            "reject_sym":   rejection_rate(df["pval_sym"]),
            "n_valid":      int(df["pval_lev"].notna().sum()),
            "n_failed":     n_failed,
        })
    return pd.DataFrame(rows).set_index("dgp")


def compute_score_correlation(df_null: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between LR_lev and LR_mem under the symmetric null."""
    valid = df_null[["lr_lev", "lr_mem"]].dropna()
    corr_matrix = valid.corr()
    corr_val = float(corr_matrix.loc["lr_lev", "lr_mem"])
    return pd.DataFrame(
        {"corr_lr_lev_lr_mem": [corr_val]},
        index=["symmetric_null"],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lr_distributions(results: dict[str, pd.DataFrame], path: str) -> None:
    """Histograms of LR stats under all DGPs, in a 5×3 grid."""
    dgp_names = list(results.keys())
    stats = ["lr_lev", "lr_mem", "lr_sym"]
    dfs_chi2 = {1: chi2(df=1), 2: chi2(df=2)}

    fig, axes = plt.subplots(
        nrows=len(dgp_names), ncols=3,
        figsize=(14, 3 * len(dgp_names)),
        squeeze=False,
    )
    x_fine = np.linspace(0.0, 20.0, 300)

    for row_idx, dgp_name in enumerate(dgp_names):
        df = results[dgp_name]
        for col_idx, stat_name in enumerate(stats):
            ax = axes[row_idx][col_idx]
            vals = df[stat_name].dropna().values
            ax.hist(vals, bins=40, density=True, alpha=0.65,
                    color="steelblue", label="MC")
            # Reference chi2 PDF
            dof = 2 if stat_name == "lr_sym" else 1
            ax.plot(x_fine, dfs_chi2[dof].pdf(x_fine),
                    "r-", lw=1.5, label=f"χ²({dof})")
            ax.set_xlim(0, 20)
            ax.set_title(f"{dgp_name} / {stat_name}", fontsize=8)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle("LR-test distributions under each DGP", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("results", exist_ok=True)

    print("=== Experiment 4: Monte Carlo Evaluation ===")
    all_results: dict[str, pd.DataFrame] = {}
    for dgp_name, dgp_params in DGPS.items():
        all_results[dgp_name] = run_dgp(dgp_name, dgp_params)

    # --- Rejection rates ---
    rej_rates = compute_rejection_rates(all_results)
    print("\nRejection rates (5% nominal):")
    print(rej_rates.to_string())
    rej_rates.to_csv("results/exp4_rejection_rates.csv")
    print("  Saved results/exp4_rejection_rates.csv")

    # --- LR distribution plot ---
    plot_lr_distributions(all_results, "results/exp4_lr_distributions.png")

    # --- Score correlation under symmetric null ---
    df_null = all_results["A_symmetric"]
    corr_df = compute_score_correlation(df_null)
    print(f"\nCorrelation LR_lev vs LR_mem (symmetric null): "
          f"{corr_df.iloc[0, 0]:.4f}")
    corr_df.to_csv("results/exp4_score_correlation.csv")
    print("  Saved results/exp4_score_correlation.csv")

    print("\n--- Size check (DGP A, symmetric null) ---")
    df_a = all_results["A_symmetric"]
    for col, label in [("pval_lev", "LR_lev"), ("pval_mem", "LR_mem"),
                       ("pval_sym", "LR_sym")]:
        print(f"  {label:7s}: rejection rate = {rejection_rate(df_a[col]):.3f}")

    print("\n--- Power (DGP B, level asymmetry) ---")
    df_b = all_results["B_level_asym"]
    for col, label in [("pval_lev", "LR_lev"), ("pval_mem", "LR_mem"),
                       ("pval_sym", "LR_sym")]:
        print(f"  {label:7s}: rejection rate = {rejection_rate(df_b[col]):.3f}")

    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
