"""
Experiment 2: Foster-Lyapunov Stability Certificate
=====================================================
Loads fitted unrestricted ALM-GARCH parameters (from exp1_parameters.csv if
available, else re-estimates), computes the minimax stability certificate M*
for every asset, classifies results, and saves diagnostic plots and tables.

Outputs (all in results/)
-------------------------
  exp2_stability_table.csv  — M*, λ*, c_argsup, certificate_satisfied,
                               classification for each asset
  exp2_stability_curves.png — S_δ(c), Λ_δ(c), and envelope curves
  exp2_stability.json       — full per-asset diagnostics
"""

import sys
sys.path.insert(0, '/workspace/project')
sys.path.insert(0, '/workspace/project/src')

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.alm_garch import ALMGARCHParams, estimate_alm_garch
from src.stability import compute_stability_certificate, stability_table

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAMS_CSV = os.path.join(RESULTS_DIR, 'exp1_parameters.csv')

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------

_BORDERLINE_THRESHOLD = -0.005   # M* in (-0.005, 0) → 'borderline'


def classify(M_star: float) -> str:
    if M_star < _BORDERLINE_THRESHOLD:
        return 'verified'
    elif M_star < 0.0:
        return 'borderline'
    else:
        return 'not_covered'


# ---------------------------------------------------------------------------
# Load or re-estimate parameters
# ---------------------------------------------------------------------------

def _load_params_from_csv(path: str) -> dict:
    """Read unrestricted model rows and return {asset: ALMGARCHParams}."""
    df = pd.read_csv(path)
    df_u = df[df['model'] == 'unrestricted'].copy()
    params_dict = {}
    for _, row in df_u.iterrows():
        params_dict[row['asset']] = ALMGARCHParams(
            mu=float(row['mu']),
            xi_plus=float(row['xi_plus']),
            xi_minus=float(row['xi_minus']),
            gamma_plus=float(row['gamma_plus']),
            gamma_minus=float(row['gamma_minus']),
            p=float(row['p']),
            tau=float(row['tau']),
        )
    return params_dict


def load_or_estimate_params() -> dict:
    """Return {asset_name: ALMGARCHParams} for the unrestricted model."""
    if os.path.exists(PARAMS_CSV):
        print(f"Loading parameters from {PARAMS_CSV} ...")
        return _load_params_from_csv(PARAMS_CSV)

    print(f"exp1_parameters.csv not found — re-estimating unrestricted models ...")
    from data.data_loader import load_all_assets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assets = load_all_assets()

    params_dict = {}
    for asset_name, ret in assets.items():
        print(f"  Estimating {asset_name} ...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = estimate_alm_garch(ret.values, model='unrestricted',
                                     p=1.2, tau=1.0, n_starts=10, seed=42)
        params_dict[asset_name] = res.params
        print(f"loglik = {res.loglik:.4f}")

    return params_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 2: Foster-Lyapunov Stability Certificate")
    print("=" * 70)

    params_dict = load_or_estimate_params()
    asset_names = list(params_dict.keys())
    n_assets    = len(asset_names)

    print(f"\nComputing stability certificates for {n_assets} assets ...")

    cert_results = {}     # asset → raw dict from compute_stability_certificate
    table_rows   = []

    for asset_name, prm in params_dict.items():
        print(f"\n  {asset_name}")
        print(f"    xi+ = {prm.xi_plus:.6f}  xi- = {prm.xi_minus:.6f}")
        print(f"    gamma+ = {prm.gamma_plus:.4f}  gamma- = {prm.gamma_minus:.4f}")
        print(f"    p = {prm.p}  tau = {prm.tau}")
        print(f"    computing ...", end=" ", flush=True)

        cert = compute_stability_certificate(
            xi_plus=float(prm.xi_plus),
            xi_minus=float(prm.xi_minus),
            gamma_plus=float(prm.gamma_plus),
            gamma_minus=float(prm.gamma_minus),
            p=float(prm.p),
            tau=float(prm.tau),
        )

        classif = classify(cert['M_star'])
        print(
            f"M* = {cert['M_star']:+.6f}  "
            f"λ* = {cert['lambda_star']:.4f}  "
            f"c_argsup = {cert['c_argsup']:.4f}  "
            f"[{classif}]"
        )

        cert_results[asset_name] = cert
        table_rows.append({
            'asset':                 asset_name,
            'M_star':                cert['M_star'],
            'lambda_star':           cert['lambda_star'],
            'c_argsup':              cert['c_argsup'],
            'certificate_satisfied': cert['certificate_satisfied'],
            'classification':        classif,
        })

    # ── Classification table (console) ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Stability Classification Summary")
    print("=" * 70)
    print(f"  {'Asset':<14} {'M*':>10} {'λ*':>10} {'c_argsup':>10}  Classification")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}  {'-'*12}")
    for row in table_rows:
        print(
            f"  {row['asset']:<14} "
            f"{row['M_star']:>+10.6f} "
            f"{row['lambda_star']:>10.4f} "
            f"{row['c_argsup']:>10.4f}  "
            f"{row['classification']}"
        )
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────
    df_table = pd.DataFrame(table_rows).set_index('asset')
    path_csv = os.path.join(RESULTS_DIR, 'exp2_stability_table.csv')
    df_table.to_csv(path_csv)
    print(f"Saved: {path_csv}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_data = {}
    for asset_name, cert in cert_results.items():
        json_data[asset_name] = {
            'M_star':                float(cert['M_star']),
            'lambda_star':           float(cert['lambda_star']),
            'c_argsup':              float(cert['c_argsup']),
            'certificate_satisfied': bool(cert['certificate_satisfied']),
            'classification':        classify(cert['M_star']),
            'c_grid_min':            float(cert['c_grid'][0]),
            'c_grid_max':            float(cert['c_grid'][-1]),
            'n_c_grid':              int(len(cert['c_grid'])),
            # Store summary arrays (first/last 5 points to keep JSON small)
            'Lambda_grid_sample':    cert['Lambda_grid'][::50].tolist(),
            'S_grid_sample':         cert['S_grid'][::50].tolist(),
            'c_grid_sample':         cert['c_grid'][::50].tolist(),
        }

    path_json = os.path.join(RESULTS_DIR, 'exp2_stability.json')
    with open(path_json, 'w') as fh:
        json.dump(json_data, fh, indent=2)
    print(f"Saved: {path_json}")

    # ── Stability-curve plots ─────────────────────────────────────────────
    n_cols = 3
    n_rows = (n_assets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes_flat  = axes.flatten()

    for i, asset_name in enumerate(asset_names):
        cert      = cert_results[asset_name]
        c_grid    = cert['c_grid']
        Lam_grid  = cert['Lambda_grid']
        S_grid    = cert['S_grid']
        lam_star  = cert['lambda_star']
        M_star    = cert['M_star']
        classif   = classify(M_star)
        envelope  = Lam_grid + lam_star * S_grid

        ax = axes_flat[i]
        ax.plot(c_grid, Lam_grid,          color='steelblue', linewidth=1.4,
                label=r'$\Lambda_\delta(c)$')
        ax.plot(c_grid, lam_star * S_grid, color='orange',    linewidth=1.4,
                linestyle='--', label=r'$\lambda^* S_\delta(c)$')
        ax.plot(c_grid, envelope,          color='tomato',    linewidth=2.0,
                label='Envelope')
        ax.axhline(0.0, color='black',   linewidth=0.8, linestyle=':')
        ax.axhline(M_star, color='forestgreen', linewidth=1.5, linestyle='--',
                   label=f'$M^* = {M_star:+.5f}$')

        title_color = {
            'verified':    'forestgreen',
            'borderline':  'darkorange',
            'not_covered': 'crimson',
        }[classif]
        ax.set_title(f'{asset_name}  [{classif}]', fontsize=10, color=title_color)
        ax.set_xlabel('$c$', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    for j in range(n_assets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        r'ALM-GARCH Stability Certificates: $S_\delta(c)$, $\Lambda_\delta(c)$, and Envelope',
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path_fig = os.path.join(RESULTS_DIR, 'exp2_stability_curves.png')
    fig.savefig(path_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path_fig}")

    print("\nDone — Experiment 2 complete.")


if __name__ == '__main__':
    main()
