# ALM-GARCH: Asymmetric Long-Memory GARCH — Structural Asymmetry Inference

A Python implementation of the **Asymmetric Long-Memory GARCH (ALM-GARCH)** model,
covering estimation, hypothesis testing, Foster-Lyapunov stability certification,
robustness analysis, Monte Carlo simulation, and volatility forecasting across
six financial assets.

---

## Research Objectives

The ALM-GARCH model decomposes volatility asymmetry into two independent channels:

| Channel | Hypothesis | Parameters |
|---------|-----------|------------|
| **Level** | H₀: ξ₋ = ξ₊ | Scaling of squared returns |
| **Memory** | H₀: γ₋ = γ₊ | Persistence of shocks |
| **Joint** | H₀: ξ₋=ξ₊ and γ₋=γ₊ | Both channels simultaneously |

**Assets studied:** S&P 500, FTSE 100, DAX, Nikkei 225, KOSPI (equity indices,
open-to-close log returns), and Bitcoin (Bitstamp close-to-close log returns).

---

## Model Specification

Conditional variance: **σ²ₙ = μ + X_{n-1}**

Memory-decay function:
```
r(c) = (1 + τ/c)^{-1},   ρ(c) = r(c)^p      [τ=1, p=1.2 fixed]
```

Sign-dependent state recursion (s = '+' if rₙ ≥ 0, else '−'):
```
X_n = X_{n-1} ρ(c_{n-1}) + ξ_s · rₙ²
c_n = [X_{n-1} ρ + ξ_s z] / [(X_{n-1}/c_{n-1}) r(c)^{p+1} + (ξ_s/γ_s) z]
      where z = rₙ²
```

Estimation via Gaussian quasi-maximum likelihood (QML):
```
ℓₙ = -0.5 [log(2π) + log(σₙ²) + rₙ²/σₙ²]
```

---

## Repository Structure

```
project/
├── data/
│   ├── __init__.py
│   └── data_loader.py       # yfinance-based daily return & RV loader
├── src/
│   ├── __init__.py
│   ├── alm_garch.py         # Core ALM-GARCH filter, likelihood, estimation
│   ├── stability.py         # Foster-Lyapunov stability certificate (M*)
│   ├── bootstrap.py         # Gaussian & symmetrized-residual bootstrap
│   └── benchmarks.py        # HAR-RV, GARCH-family via arch package
├── exp/
│   ├── exp_1_estimation.py  # ALM-GARCH estimation + LR tests (Exp 1)
│   ├── exp_2_stability.py   # Foster-Lyapunov certificate (Exp 2)
│   ├── exp_3_robustness.py  # Robustness: p=2.0, constrained fits (Exp 3)
│   ├── exp_4_montecarlo.py  # Monte Carlo size/power evaluation (Exp 4)
│   ├── exp_5_forecasting.py # Out-of-sample QLIKE comparison (Exp 5)
│   └── exp_6_insample.py    # In-sample BIC comparison (Exp 6)
├── tests/
│   ├── test_alm_garch.py    # Unit & integration tests for core model
│   ├── test_benchmarks.py   # Tests for benchmark models
│   └── test_stability.py    # Tests for stability certificate
├── results/
│   └── RESULTS.md           # Summary of all experiment findings
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Experiments

| # | Title | Key Output |
|---|-------|-----------|
| 1 | ALM-GARCH Estimation & LR Tests | Parameter estimates, LR statistics, bootstrap p-values |
| 2 | Foster-Lyapunov Stability Certificate | M*, λ*, c_argsup per asset |
| 3 | Robustness (p=2.0, constrained) | Sensitivity to p choice, stability constraints |
| 4 | Monte Carlo Size & Power | LR test finite-sample properties |
| 5 | Out-of-sample QLIKE Forecasting | QLIKE means, Diebold-Mariano tests |
| 6 | In-sample BIC Comparison | BIC ranking across 6+ models |

---

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:** numpy, pandas, scipy, numba, arch, yfinance, statsmodels,
matplotlib, seaborn, pytest, joblib

---

## Data

Data is downloaded automatically from **Yahoo Finance** via `yfinance`:
- Equity indices: `^GSPC`, `^FTSE`, `^GDAXI`, `^N225`, `^KS11`
- Bitcoin: `BTC-USD`

Sample periods match the paper specifications (adjusted for data availability).
No data files need to be pre-downloaded; the loader fetches and caches automatically.

---

## Running Experiments

Run all experiments sequentially:
```bash
python exp/exp_1_estimation.py   # ~25 min (includes bootstrap B=19)
python exp/exp_2_stability.py    # ~2 min (reads exp1 results)
python exp/exp_3_robustness.py   # ~15 min
python exp/exp_4_montecarlo.py   # ~15 min (parallelized)
python exp/exp_5_forecasting.py  # ~15 min
python exp/exp_6_insample.py     # ~10 min
```

Run tests:
```bash
pytest tests/ -v
```

---

## Expected Empirical Findings

- **Joint symmetry** rejected at p ≈ 0.001 for **all six assets**
- **Level asymmetry** rejected for **all six assets** (ξ₋ > ξ₊ everywhere)
- **Memory asymmetry** rejected selectively: Nikkei 225, KOSPI, Bitcoin
- S&P 500 and FTSE 100 show nearly inactive positive branches (ξ₊ ≈ 0) → weak identification
- Stability certificate M* < 0 for most assets; Nikkei 225 not covered (M* > 0)

---

## Notes

- Bootstrap replications reduced to B=19 (paper uses B=999) for computational feasibility.
  Increase `B` in `exp/exp_1_estimation.py` for full replication.
- All core numba-JIT functions fall back gracefully to pure Python if numba is unavailable.
- Random seeds are stored in results metadata for reproducibility.
