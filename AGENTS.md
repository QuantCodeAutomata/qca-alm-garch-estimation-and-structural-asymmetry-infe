# ALM-GARCH Project — Repository Notes

## Project Structure

```
/workspace/project/
├── data/
│   ├── __init__.py
│   └── data_loader.py        # yfinance download + log-return computation
├── src/
│   ├── __init__.py
│   └── alm_garch.py          # Core ALM-GARCH model implementation
├── exp/                       # Experiment scripts (populated externally)
├── results/                   # Output artifacts
└── tests/                     # Test files
```

## Key Packages (confirmed installed)

- `yfinance==1.6.0`  — data download; returns MultiIndex DataFrame, use `.xs(ticker, axis=1, level=1)` to flatten
- `numba==0.67.0`    — JIT compilation for the filter loop
- `scipy==1.14.1`    — optimisation (`minimize`) and stats (`chi2.cdf`)
- `numpy`, `pandas`  — standard numerics

## data/data_loader.py

### Asset configuration (`ASSET_CONFIG`)
| Asset      | Ticker   | Start      | End (excl.) | Return type    |
|-----------|---------|-----------|------------|----------------|
| SP500      | ^GSPC    | 2000-01-03 | 2018-06-28  | open-to-close  |
| FTSE100    | ^FTSE    | 2000-01-04 | 2018-06-28  | open-to-close  |
| DAX        | ^GDAXI   | 2000-01-03 | 2018-06-28  | open-to-close  |
| Nikkei225  | ^N225    | 2000-02-02 | 2018-06-28  | open-to-close  |
| KOSPI      | ^KS11    | 2000-01-04 | 2018-06-28  | open-to-close  |
| Bitcoin    | BTC-USD  | 2012-01-02 | 2025-01-07  | close-to-close |

**yfinance end date is exclusive** — the date passed as `end` is NOT returned, so add one calendar day to include the desired last trading day.

### Return computation
- Equity indices: `r_t = log(Close_t) - log(Open_t)`
- Bitcoin: `r_t = log(Close_t) - log(Close_{t-1})`
- RV proxy: `rv_t = r_t^2` (daily squared returns; 5-min bars not available for full history)

### yfinance 1.6.0 column access
`yf.download()` returns a MultiIndex DataFrame with levels `(Price, Ticker)`.
Flatten with `df.xs(ticker, axis=1, level=1)` inside `_download_ohlcv`.

## src/alm_garch.py

### Model equations
```
sigma_n^2  = mu + X_{n-1}
r_n        = sigma_n * eps_n,      eps_n ~ N(0,1)
r(c)       = 1 / (1 + tau/c),      rho(c) = r(c)^p    [default: tau=1, p=1.2]

s = '+' if r_n >= 0, else '-'

X_n  = X_{n-1} * rho(c_{n-1}) + xi_s * r_n^2
c_n  = X_n / [ (X_{n-1}/c_{n-1}) * r(c)^{p+1} + (xi_s/gamma_s) * r_n^2 ]

X_0  = sample variance of returns
c_0  = max(gamma_+, gamma_-)
```

### Parameter space
All parameters (mu, xi_+, xi_-, gamma_+, gamma_-) must be strictly positive.
Optimisation uses log-space reparametrisation: `theta = log(params)`.

### Model variants
| Model              | theta size | Constraint         |
|-------------------|-----------|---------------------|
| unrestricted       | 5          | none                |
| level_restricted   | 4          | xi_+ = xi_-         |
| memory_restricted  | 4          | gamma_+ = gamma_-   |
| joint_symmetric    | 3          | xi_+ = xi_-, gamma_+ = gamma_- |

### Estimation
- `n_starts=10` random starts + 3 structured starts (total ≥ 13)
- Primary optimiser: L-BFGS-B; secondary: Nelder-Mead (adaptive)
- Final polish pass with tighter tolerances from best found theta
- Returns `ALMGARCHResult` NamedTuple

### Numba JIT filter
`_filter_core` is decorated with `@_njit(cache=True)`.
The public `alm_garch_filter` wraps it with a try/except that falls back to
`_filter_python` (identical pure-Python logic) if JIT compilation fails.

### Persistence
`compute_persistence(params)` returns:
- `rho_s = r(gamma_s)^p`  (equilibrium persistence per branch)
- `half_life_s = log(0.5) / log(rho_s)`
- `response_s[k] = xi_s * rho_s^k`  (impulse response at lag k)

### Numerical properties verified on synthetic data (N=500, std≈0.01)
- sigma2 mean ≈ 1e-4 ✓
- sigma2 always positive ✓
- standardised residuals std ≈ 1.0 ✓
- LR statistic / chi-squared p-value correct ✓


## src/stability.py — Foster-Lyapunov Certificate

Custom implementation; no standard package covers this.

### Certificate equations


### Gauss-Hermite quadrature (hermgauss convention)
- Nodes:   u_i = z_i^2          (NOT 2*z_i^2)
- Weights: w_i / sqrt(pi)        (renormalize from exp(-x^2) to unit-mass)
- E[f(U)] ~ sum_i (w_i/sqrt(pi)) * f(z_i^2)

### Optimization strategy (compute_stability_certificate)
- c-grid: n_c_grid=500 pts from gamma_min to c_max_factor*tau (=100 for defaults)
- Vectorized Lambda/S over grid using matrix outer-product broadcasts (no Python loop)
- Outer minimization over log(lambda) in [-8,2]: 200-pt scan then Brent refinement
- c_argsup refined by continuous minimize_scalar in +-5 grid-point window
- certificate_satisfied iff M* < 0

### Validation targets (from paper, depend on fitted model params)
- S&P 500:  lambda*~0.0450, M*~-0.011284, c_argsup~11.63
- FTSE 100: lambda*~0.0469, M*~-0.013400, c_argsup~11.28
- DAX:      lambda*~0.0341, M*~-0.010012, c_argsup~14.78
- Nikkei:   lambda*~0.0452, M*>0 (not covered)
- KOSPI:    lambda*~0.0312, M*~-0.000088 (borderline)
- Bitcoin:  lambda*~0.0924, M*~-0.016615, c_argsup~5.72

### stability_table
- Accepts {name: ALMGARCHParams} or {name: dict with xi_plus, xi_minus, gamma_plus, gamma_minus}
- Handles both NamedTuple (._asdict()) and dataclass (.as_dict()) ALMGARCHParams
- Returns pd.DataFrame indexed by asset: M_star, lambda_star, c_argsup, certificate_satisfied

## src/bootstrap.py — Bootstrap LR-test Calibration

### Two resampling schemes
1. gaussian: simulate_alm_garch(null_params, n, seed) -- exact model simulation
2. symmetrized: resample null-model std residuals with Rademacher sign-flip,
   replay ALM-GARCH recursion manually (full filter, not Gaussian draw)

### p-value formula: (1 + #{LR_b >= lr_observed}) / (B + 1)

### bootstrap_lr_pvalue(returns, restriction, lr_observed, B=999, scheme='gaussian',
###                     seed=42, n_starts=5, p=1.2, tau=1.0)
### Returns: {p_value, lr_bootstrap_distribution, null_params, converged_fraction}

- Null model fitted with max(n_starts+2, 7) starts for accuracy
- Bootstrap reps use n_starts=3 (lower for speed as specified)
- joblib.Parallel(n_jobs=-1, backend='loky') if available; sequential fallback
- restriction -> null model: level->level_restricted, memory->memory_restricted, joint->joint_symmetric

## Gauss-Hermite Quadrature Convention (IMPORTANT)

gauss_hermite_nodes(n) returns u_nodes = z_i^2 and weights = w_i / sqrt(pi).
The quadrature approximates integrals against (1/sqrt(pi)) * exp(-x^2), NOT chi-sq(1).
Correct moment values under this quadrature rule:
  E[z^2]  = sum_i (w_i/sqrt(pi)) * z_i^2 = 0.5  (NOT 1.0)
  E[z^4]  = 0.75  (NOT 3.0)
  Var[z^2]= 0.5   (NOT 2.0)
  sum(weights) = 1.0  ✓

The stability analysis uses this convention self-consistently throughout.

## exp/ — Experiment Scripts

### exp_4_montecarlo.py
- 5 DGPs: A_symmetric, B_level_asym, C_memory_asym, D_joint_asym, E_near_boundary
- M=200 reps × N=2000; joblib.Parallel(n_jobs=-1, backend='loky')
- Worker: _one_replication(rep_idx, dgp_params, N); seed=42+rep_idx
- LR_lev (df=1): unrestricted vs level_restricted
- LR_mem (df=1): unrestricted vs memory_restricted
- LR_sym (df=2): unrestricted vs joint_symmetric
- Saves: exp4_rejection_rates.csv, exp4_lr_distributions.png, exp4_score_correlation.csv

### exp_5_forecasting.py
- 70/30 train/test split, re-estimation every 250 trading days (expanding window)
- ALM-GARCH rolling forecast: _alm_one_step() helper updates filter state manually
  between re-estimations; forecast_t = mu + X_{t-1}
- HAR-RV: uses forecast_har_rv() on full RV series, then filters to eval period
- GARCH/GJR/EGARCH: uses forecast_garch_benchmark() (already expanding window)
- arch forecasts are in return^2 units (divided by 100^2); QLIKE directly comparable
- Saves: exp5_qlike_summary.csv, exp5_dm_tests.csv, exp5_forecasts_{asset}.csv,
         exp5_forecast_comparison.png

### exp_6_insample.py
- Full-sample BIC comparison for 6 models per asset
- BIC = -2*loglik + k*log(N)
- arch loglik in raw units: loglik_raw = arch_result.loglikelihood + N*log(100)
- ALM-GARCH k=5 (unrestricted), k=3 (joint_symmetric)
- arch k from len(res.params) (includes mean parameter)
- Saves: exp6_insample_comparison.csv, exp6_bic_ranking.png

## tests/ — Test Suite

### test_alm_garch.py (20 tests)
- Covers: filter, loglik, estimation, restricted models, LR test, persistence,
  simulation, edge cases (all-positive returns, constant returns)
- Uses SYM_PARAMS (symmetric, N=500) and ASYM_PARAMS as shared fixtures
- n_starts=2 for speed in estimation tests

### test_benchmarks.py (22 tests)
- Covers: HAR-RV fit/forecast, GARCH-family fit/forecast, QLIKE loss, DM test
- Tests QLIKE scale-invariance, DM antisymmetry, DM short-series NaN

### test_stability.py (17 tests)
- Covers: GH quadrature (normalization, nodes, moments), compute_m (bounds,
  monotonicity, limits), Lambda/S (stability), full certificate (stable,
  near-unit-root, return types, grid shapes, asymmetric, c_argsup range)

## exp/ — Experiment Scripts

All scripts require:
```python
sys.path.insert(0, '/workspace/project')
sys.path.insert(0, '/workspace/project/src')   # needed so bootstrap.py's bare `from alm_garch import ...` resolves
```

### exp/exp_1_estimation.py
- Loads all 6 assets via `load_all_assets()`
- Estimates 4 models: unrestricted, level_restricted, memory_restricted, joint_symmetric (p=1.2, tau=1.0)
- Computes LR_lev (df=1), LR_mem (df=1), LR_sym (df=2) via `lr_test`
- Bootstraps each test with B=199, both `scheme='gaussian'` and `scheme='symmetrized'`
- Persistence via `compute_persistence(params, n_lags=251)` — returns rho_plus, rho_minus, half-life, response arrays
- Response curves: `g_s(k) = xi_s * [gamma_s / (gamma_s + k*tau)]^p` for k=0..250  (NOT the geometric rho_s^k formula)
- Outputs: exp1_parameters.csv (all 4 models × 6 assets), exp1_lr_tests.csv, exp1_persistence.csv, exp1_response_curves.png

### exp/exp_2_stability.py
- Loads unrestricted params from exp1_parameters.csv (or re-estimates if missing)
- Calls `compute_stability_certificate(xi_plus, xi_minus, gamma_plus, gamma_minus, p, tau)` for each asset
- Classification: M* < -0.005 → 'verified'; -0.005 ≤ M* < 0 → 'borderline'; M* ≥ 0 → 'not_covered'
- Outputs: exp2_stability_table.csv, exp2_stability_curves.png (Lambda, lambda*S, envelope per asset), exp2_stability.json

### exp/exp_3_robustness.py
- Section 1: p=1.2 vs p=2.0. Loads full exp1 CSV for p=1.2 (all 4 model logliks); re-estimates everything for p=2.0
- Section 2: Stability-constrained estimation via `_make_stability_penalized_nll()` — penalty=1e6*(M*+1e-3)^2 when M*>-1e-3; uses n_c_grid=100, n_quad=32 during optimization; loads cached loglik/M* from exp1/exp2 CSV
- Section 3: Floor sensitivity — `xi_+` ≥ floor (floors=[0.0, 0.01, 0.02, 0.05]) for SP500 and FTSE100 only; modifies log-space lower bound `bounds[1]` for theta[1]=log(xi_plus)
- Helper `_estimate_unrestricted(r, bounds, p, tau, n_starts, seed, nll_fn)` accepts custom bounds and neg-loglik function
- `_unpack_theta(theta)` mirrors src.alm_garch's unrestricted case: e = exp(theta); return e[0..4]
- `_UNRES_BOUNDS` mirrors `_BOUNDS['unrestricted']` in alm_garch.py
- Outputs: exp3_fixed_p_sensitivity.csv, exp3_floor_sensitivity.csv, exp3_constrained_refits.csv, exp3_robustness_plot.png

## src/benchmarks.py — Benchmark Volatility Models

### HAR-RV (Corsi 2009)
- RV_t = b0 + b_d*RV_{t-1} + b_w*(1/5)*sum_{k=1}^5 RV_{t-k} + b_m*(1/22)*sum_{k=1}^22 RV_{t-k}
- fit_har_rv(rv): OLS with HAC(22) SE; dict keys: beta, rsquared, ols_result
- forecast_har_rv(rv, n_ahead=1): expanding window pd.Series, first forecast at obs 24

### GARCH family (arch 8.0.0)
- fit_garch_benchmark(returns, vol_model):
  - GARCH:     arch_model(r*100, vol='Garch', p=1, q=1)
  - GJR-GARCH: arch_model(r*100, vol='Garch', p=1, o=1, q=1)
  - EGARCH:    arch_model(r*100, vol='EGARCH', p=1, q=1)
  - FIGARCH:   arch_model(r*100, vol='FIGARCH', p=1, q=1) -- fallback to GARCH on error
- forecast_garch_benchmark(returns, vol_model, reestim_freq=250):
  - Expanding window; re-estimates every reestim_freq obs
  - Output in return^2 units (divided by 10000 from percent-squared)

### Loss and test
- qlike_loss(rv_actual, h_forecast): L_t = RV/h - log(RV/h) - 1 (element-wise)
- diebold_mariano_test(loss1, loss2, max_lag=None):
  - Newey-West Bartlett; default max_lag = floor(4*(n/100)^{2/9})
  - Returns {statistic, p_value, mean_diff}; statistic ~ N(0,1) under H0
