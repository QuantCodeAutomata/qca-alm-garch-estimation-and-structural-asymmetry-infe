"""
Experiment 6 – In-sample model comparison.

Models fitted on the full sample for each asset:
  1. ALM-GARCH (unrestricted, p=1.2)       — 5 free parameters
  2. LM-GARCH  (joint_symmetric, p=1.2)    — 3 free parameters
  3. GARCH(1,1) via arch                   — n_params from fitted result
  4. GJR-GARCH  via arch
  5. EGARCH     via arch
  6. FIGARCH    via arch  (falls back to GARCH if unavailable)

Information criterion:
  BIC = -2 · loglik + k · log(N)

arch log-likelihoods are in percent-squared units (data scaled ×100).
Conversion to raw-return units: loglik_raw = loglik_arch + N · log(100).
This makes ALM-GARCH and arch BICs directly comparable.

Saves
-----
results/exp6_insample_comparison.csv
results/exp6_bic_ranking.png
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

from src.alm_garch import estimate_alm_garch
from src.benchmarks import fit_garch_benchmark
from data.data_loader import load_all_assets

# Number of free params for each ALM-GARCH variant
_ALM_K = {
    "unrestricted":  5,
    "joint_symmetric": 3,
}

# ---- arch models to fit ----
_ARCH_MODELS = ["GARCH", "GJR-GARCH", "EGARCH", "FIGARCH"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bic(loglik: float, k: int, n: int) -> float:
    return -2.0 * loglik + k * np.log(n)


def fit_alm_model(returns: np.ndarray, model: str, p: float = 1.2) -> dict:
    """Fit an ALM-GARCH model and return loglik/BIC."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = estimate_alm_garch(returns, model=model, p=p, tau=1.0, n_starts=5, seed=42)
    n = res.n_obs
    k = _ALM_K[model]
    ll = res.loglik
    return {
        "loglik": ll,
        "k":      k,
        "n":      n,
        "bic":    bic(ll, k, n),
        "converged": res.converged,
    }


def fit_arch_model(returns: pd.Series, vol_model: str) -> dict:
    """Fit a GARCH-family model via arch and convert loglik to raw-return units."""
    n = len(returns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = fit_garch_benchmark(returns, vol_model=vol_model)
            # arch loglik is in percent-squared; convert to raw-return units
            ll_raw = float(res.loglikelihood) + n * np.log(100.0)
            k = int(len(res.params))
            return {
                "loglik": ll_raw,
                "k":      k,
                "n":      n,
                "bic":    bic(ll_raw, k, n),
                "converged": True,
            }
        except Exception as exc:
            return {
                "loglik":    np.nan,
                "k":         np.nan,
                "n":         n,
                "bic":       np.nan,
                "converged": False,
                "error":     str(exc),
            }


# ---------------------------------------------------------------------------
# Per-asset fitting
# ---------------------------------------------------------------------------

def fit_all_models(asset: str, returns: pd.Series) -> list[dict]:
    r_arr = np.asarray(returns, dtype=float)
    n = len(r_arr)
    rows = []

    print(f"  {asset} (N={n})")

    # ALM-GARCH unrestricted
    print("    ALM-GARCH (unrestricted)...", end=" ", flush=True)
    d = fit_alm_model(r_arr, "unrestricted")
    rows.append({"asset": asset, "model": "ALM-GARCH", **d})
    print(f"loglik={d['loglik']:.1f}  BIC={d['bic']:.1f}")

    # LM-GARCH (joint_symmetric)
    print("    LM-GARCH (joint_symmetric)...", end=" ", flush=True)
    d = fit_alm_model(r_arr, "joint_symmetric")
    rows.append({"asset": asset, "model": "LM-GARCH", **d})
    print(f"loglik={d['loglik']:.1f}  BIC={d['bic']:.1f}")

    # GARCH-family via arch
    for arch_name in _ARCH_MODELS:
        print(f"    {arch_name}...", end=" ", flush=True)
        d = fit_arch_model(returns, arch_name)
        rows.append({"asset": asset, "model": arch_name, **d})
        if np.isfinite(d.get("bic", np.nan)):
            print(f"loglik={d['loglik']:.1f}  BIC={d['bic']:.1f}")
        else:
            print("FAILED")

    return rows


# ---------------------------------------------------------------------------
# Ranking and plotting
# ---------------------------------------------------------------------------

def rank_by_bic(df: pd.DataFrame) -> pd.DataFrame:
    """Add a BIC rank column (1 = best) within each asset."""
    df = df.copy()
    df["bic_rank"] = (
        df.groupby("asset")["bic"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return df


def plot_bic_ranking(df: pd.DataFrame, path: str) -> None:
    """Heatmap of BIC ranks: rows = models, columns = assets."""
    pivot = df.pivot_table(index="model", columns="asset",
                           values="bic_rank", aggfunc="first")

    model_order = ["ALM-GARCH", "LM-GARCH", "GARCH", "GJR-GARCH", "EGARCH", "FIGARCH"]
    model_order = [m for m in model_order if m in pivot.index]
    pivot = pivot.reindex(model_order)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), 5))
    cmap = plt.get_cmap("RdYlGn_r", int(pivot.max().max()))
    im = ax.imshow(pivot.values.astype(float), aspect="auto",
                   cmap=cmap, vmin=0.5, vmax=pivot.max().max() + 0.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("BIC Rankings (1 = best model per asset)", fontsize=11)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")

    fig.colorbar(im, ax=ax, label="BIC rank", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("results", exist_ok=True)

    print("=== Experiment 6: In-sample Model Comparison ===")
    print("Loading data...")
    assets_returns = load_all_assets()

    all_rows = []
    for asset, returns in assets_returns.items():
        rows = fit_all_models(asset, returns)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = rank_by_bic(df)

    # Save
    cols_out = ["asset", "model", "loglik", "k", "n", "bic", "bic_rank", "converged"]
    cols_out = [c for c in cols_out if c in df.columns]
    df[cols_out].to_csv("results/exp6_insample_comparison.csv", index=False)
    print("\nSaved results/exp6_insample_comparison.csv")

    print("\nIn-sample comparison table:")
    print(df[cols_out].to_string(index=False))

    # BIC ranking heatmap
    plot_bic_ranking(df, "results/exp6_bic_ranking.png")

    # Print best model per asset
    print("\nBest model per asset (lowest BIC):")
    best = df.loc[df.groupby("asset")["bic"].idxmin()]
    for _, row in best.iterrows():
        print(f"  {row['asset']:12s}: {row['model']:12s}  BIC={row['bic']:.1f}")

    print("\nExperiment 6 complete.")


if __name__ == "__main__":
    main()
