# Phase 2 Plan — Signal Generation & Execution

## Objective
Three modules: Kalman filter (prior/posterior), multi-criterion δ auto-selector, and the Numba state machine with position sizing + threshold rebalance. Called per fold by the Phase 4 orchestrator, using 1-min data from Phase 0 and pair metadata from Phase 1.

## Input
- `data/validated/1min_phase2.parquet` (Phase 0)
- Surviving pairs DataFrame: `[ticker_A, ticker_B, alpha_PCA, beta_PCA, half_life_days, R_measurement_noise]` (Phase 1)

## Pre-Execution: Per-Ticker Concentration Cap (Portfolio Construction Step)

Before passing pairs to the execution engine, apply a per-ticker cap to prevent concentration in folds with large pair counts (e.g. Fold 23: 6,008 pairs, Fold 40: 5,913 pairs).

**Rule:** For each ticker appearing in the surviving pairs, keep at most **5 pairs** ranked by `johansen_pval` ascending (most significant first). Drop the rest.

**Rationale:** With ~300 survivors, a single popular ticker can appear in 100+ pairs. Without this cap, `N_open_pairs_max=50` would sample heavily from one stock's pairs, creating hidden concentration. This is a portfolio construction constraint, not a discovery filter — Phase 1 output CSVs are unchanged.

**Implementation location:** `engine.py` as `apply_ticker_concentration_cap(pairs_df, max_pairs_per_ticker=5)` — called at the top of each fold's execution setup, after loading Phase 1 results.

```python
def apply_ticker_concentration_cap(pairs_df, max_pairs_per_ticker=5):
    # For each ticker (appearing as ticker_a OR ticker_b), keep top-5 by johansen_pval
    ranked = pairs_df.sort_values("johansen_pval")
    keep = set()
    ticker_count = {}
    for idx, row in ranked.iterrows():
        ca = ticker_count.get(row.ticker_a, 0)
        cb = ticker_count.get(row.ticker_b, 0)
        if ca < max_pairs_per_ticker and cb < max_pairs_per_ticker:
            keep.add(idx)
            ticker_count[row.ticker_a] = ca + 1
            ticker_count[row.ticker_b] = cb + 1
    return pairs_df.loc[sorted(keep)].reset_index(drop=True)
```

## Implementation Status
**Status: BUILT — awaiting Phase 2 hard-stop approval**
Files: `src/phase2_execution/kalman.py`, `delta_selector.py`, `engine.py`

## Spec Deviations (documented)

| Deviation | Spec Said | Implemented | Reason |
|---|---|---|---|
| Q formula | `Q = δ · I₂` | `Q = δ · R · I₂` | With R spanning 0.001–33 across pairs, Q=δ·I gives δ/R adaptation rates varying 5 orders of magnitude. Tight pairs (R≈0.001) get Kalman gain ~100× loose pairs (R≈33), making prior spreads white noise. Spec §2.1 explicitly states "dynamics controlled by δ/R ratio" — this enforces it uniformly. Validated: Fold 1 delta selection went from universal fail → δ=1e-7 selected, median HL=4.08d, ACF78=0.855. |
| Delta selector burn-in | Not specified | Discard first 500 bars before computing metrics | P₀ = R·I is a loose prior. At initialization the Kalman gain is elevated; first ~500 bars are transient as P converges. Without burn-in, tight pairs show HL≈0.1d (near-white-noise) from initialization artifact, not true spread behavior. 500 bars ≈ 6.4 trading days on 5-min data. |
| Concentration cap location | Not specified | Phase 2 `engine.py`, not Phase 1 | Keeps Phase 1 as pure statistical discovery. Cap is a portfolio construction decision, not a cointegration filter. |

## Reuse from Prior Weeks
| Code | Source | Adaptation needed |
|---|---|---|
| Kalman filter structure | `Week 2/week2_signal_engine/src/signals/kalman.py` | Change Q formula; change init state; expose prior vs posterior |
| Rolling Z-score + session warmup | `Week 2/week2_signal_engine/src/signals/zscore.py` | Direct reuse |
| Numba state machine | `Week 2/week2_signal_engine/src/signals/state_machine.py` | Add threshold rebalance hook |
| Period split guard | `Week 2/week2_signal_engine/src/utils/dates.py` | Direct reuse |

## New Code Required
| Component | Why not in prior weeks |
|---|---|
| Multi-criterion δ selector | Prior weeks used fixed δ=1e-5 |
| 2D Kalman state [α, β] with R explicit | Week 2 had 1D β with R estimated from first 500 bars |
| Prior vs posterior spread split | Week 2 used posterior (caused kurtosis 13.49 — untradeable) |
| Threshold rebalance with hysteresis | Not implemented in prior weeks |
| Borrow cost tracking | Not implemented in prior weeks |
| N_open_pairs_max cap | Not implemented in prior weeks |

## Files to Create
- `src/phase2_execution/kalman.py`
- `src/phase2_execution/delta_selector.py`
- `src/phase2_execution/engine.py`

---

## Module 2a — Kalman Filter (`kalman.py`)

### Key differences from Week 2
| Aspect | Week 2 | Week 4 |
|---|---|---|
| State vector | `[β]` (1D) | `[α, β]` (2D) — adds intercept |
| Q formula | `(δ/(1-δ)) × I` | `δ × I` |
| Init state | `[0, 1]` (zero intercept, unit beta) | `[α_PCA, β_PCA]` from Phase 1 |
| R (obs noise) | Estimated from first 500 bars | `var(residual)` from PCA fit in Phase 1 |
| P₀ | `I` | `R × I` (informative prior) |
| Output | posterior θ only | **Prior θ for signal; posterior θ for sizing** |

### Implementation
```python
def run_kalman(log_a, log_b, alpha_init, beta_init, R, delta):
    n = len(log_a)
    Q = delta * np.eye(2)
    theta = np.array([alpha_init, beta_init])
    P = R * np.eye(2)
    I2 = np.eye(2)

    # Outputs: prior and posterior states per bar
    alpha_prior  = np.full(n, np.nan)
    beta_prior   = np.full(n, np.nan)
    alpha_post   = np.full(n, np.nan)
    beta_post    = np.full(n, np.nan)
    spread_prior = np.full(n, np.nan)

    for i in range(n):
        # --- PREDICT (prior state) ---
        P_pred = P + Q
        theta_pred = theta   # random-walk process: E[θ_t|t-1] = θ_{t-1|t-1}

        # Prior spread (BEFORE incorporating A_t observation)
        alpha_prior[i] = theta_pred[0]
        beta_prior[i]  = theta_pred[1]
        spread_prior[i] = log_a[i] - theta_pred[0] - theta_pred[1] * log_b[i]

        # --- UPDATE (posterior state) ---
        H = np.array([1.0, log_b[i]])
        innov = log_a[i] - (H @ theta_pred)
        S = float(H @ P_pred @ H) + R
        K = (P_pred @ H) / S
        theta = theta_pred + K * innov
        P = (I2 - np.outer(K, H)) @ P_pred

        alpha_post[i] = theta[0]
        beta_post[i]  = theta[1]

    return spread_prior, alpha_prior, beta_prior, alpha_post, beta_post
```

---

## Module 2b — Delta Auto-Selector (`delta_selector.py`)

Run ONCE per fold on the formation window before trading begins.

```python
DELTA_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

def select_delta(pairs_df, formation_data, R_map):
    results = {}
    for delta in DELTA_GRID:
        kurt_list, hl_list, acf78_list = [], [], []

        for _, row in pairs_df.iterrows():
            log_a = formation_data[row.ticker_A]['log_close'].values
            log_b = formation_data[row.ticker_B]['log_close'].values
            R = R_map[(row.ticker_A, row.ticker_B)]

            spread_prior, *_ = run_kalman(log_a, log_b, row.alpha_PCA, row.beta_PCA, R, delta)

            # Three criteria
            kurt = scipy.stats.kurtosis(spread_prior, nan_policy='omit') + 3   # excess → total
            hl_days = compute_ou_half_life(spread_prior) / 78   # ÷78 for 5-min bars
            acf78 = abs(pd.Series(spread_prior).autocorr(lag=78))

            kurt_list.append(abs(kurt - 3))
            hl_list.append(hl_days)
            acf78_list.append(acf78)

        results[delta] = {
            'metric_kurt':   np.median(kurt_list),
            'median_HL':     np.median(hl_list),
            'median_ACF78':  np.median(acf78_list),
        }

    # Multi-criterion selection: argmin(kurtosis) subject to constraints
    feasible = {
        d: r for d, r in results.items()
        if 1.0 <= r['median_HL'] <= 10.0 and r['median_ACF78'] > 0.7
    }

    if not feasible:
        log("UNIVERSAL CONSTRAINT FAIL — no delta passes all 3 criteria")
        return None, results   # caller should skip fold

    optimal_delta = min(feasible, key=lambda d: feasible[d]['metric_kurt'])

    # Edge case flags
    if optimal_delta in (DELTA_GRID[0], DELTA_GRID[-1]):
        log("DELTA AT GRID BOUNDARY — expand grid next fold")

    return optimal_delta, results
```

---

## Module 2c — Execution Engine (`engine.py`)

Reuse Numba state machine from `Week 2/week2_signal_engine/src/signals/state_machine.py` as base.

### Z-Score Window (1-min bars)
```python
Z_window_bars = int(half_life_days * 390)   # 390 1-min bars/day
Z_window_bars = min(Z_window_bars, 2000)
burn_in = Z_window_bars // 2
```

### Session Warmup
First 30 bars of each session → force Z-score to NaN.
Reuse from `Week 2/week2_signal_engine/src/signals/zscore.py`.

### State Machine (Numba)
Reuse `Week 2/week2_signal_engine/src/signals/state_machine.py` base.
Add threshold rebalance event hook:
```python
@njit(cache=True, fastmath=False)
def _state_machine_with_rebalance(zscores, entry_z, beta_posterior, beta_ref,
                                   rebalance_threshold, short_notional,
                                   price_b_entry, price_b_current):
    n = len(zscores)
    positions = np.zeros(n, dtype=np.int8)
    rebalance_events = np.zeros(n, dtype=np.bool_)
    state = 0

    for i in range(n):
        z = zscores[i]

        # NaN handling: hold current state, no new orders
        if np.isnan(z):
            positions[i] = state
            continue

        # State transitions
        if state == 0:
            if z < -entry_z:
                state = 1
            elif z > entry_z:
                state = -1
        elif state != 0 and abs(z) < 1e-10:   # zero-crossing exit
            state = 0

        positions[i] = state

        # Threshold rebalance check (while in position)
        if state != 0:
            beta_drift = abs(beta_posterior[i] - beta_ref[i]) / abs(beta_ref[i])
            if beta_drift > rebalance_threshold:
                rebalance_events[i] = True

    return positions, rebalance_events
```

### Position Sizing (dollar-neutral)
```python
N_open_pairs_max = 50   # default
per_pair_dollar  = total_capital / N_open_pairs_max
long_notional    = per_pair_dollar * 0.5
short_notional   = per_pair_dollar * 0.5

shares_A = long_notional / price_A_at_entry
shares_B = (short_notional * beta_at_entry) / price_B_at_entry
```

### Threshold Rebalance with Hysteresis (default X=10%)
```python
beta_drift = (beta_posterior_t - beta_ref) / beta_ref
if abs(beta_drift) > X:   # default X=0.10 (10%)
    delta_beta      = beta_posterior_t - beta_ref
    delta_shares_B  = abs(delta_beta) * (short_notional / price_B_at_entry)
    rebalance_cost  = 0.003 * delta_shares_B * price_B_t   # 30bps one-side
    beta_ref = beta_posterior_t
    # Dead band: next trigger only if drift > X from NEW beta_ref
    # Ignore drift < X/2 from new beta_ref (hysteresis)
```

### Execution Lag (anti-lookahead)
```python
position_executed = np.roll(position_raw, 1)
position_executed[0] = 0   # flat at start
```

### Max Holding Period
Default: force flatten (state → 0) at 15:55 ET each session. Sensitivity: {EOS, 1d, 3d}.

## Skills to Invoke

### Before Implementing `kalman.py` → Invoke `kalman-filters` skill
The `.agents/skills/brainbytes-dev-everything-claude-trading-kalman-filters/SKILL.md` skill provides:
- **Implementation templates** for dynamic beta estimation (adapt 1D → 2D state [α, β])
- **Q/R calibration guidance** — maps to our multi-criterion δ grid search:
  - The skill's Grid Search method ↔ our δ ∈ {1e-7…1e-3} sweep
  - The skill's Q/R ratio invariance principle confirms why we select δ/R (not δ alone)
- **Diagnostic checklist** — maps to our prior-spread kurtosis and half-life checks:
  - "Innovations test: innovations should be white noise" ↔ our ACF78 constraint
  - "Residual normality" ↔ our kurtosis ≈ 3 constraint
- **Production quality gates** (10 checkpoints) — maps to our red flag triggers:
  - Gate 3: "No lookahead" ↔ our `exec_ts > signal_ts` assertion
  - Gate 7: "Parameter stability" ↔ our δ instability flag (>2 grid steps across folds)
  - Gate 9: "Out-of-sample performance" ↔ our NC bootstrap threshold

Read SKILL.md before writing `kalman.py` to align implementation with best practices. Do NOT use the Kalman Smoother (historical re-estimation — lookahead bias for trading).

### After All 3 Modules → `/simplify`
Focus on the Numba compilation path and the delta selection loop (hot paths).

### After Smoke Test → `/review`
Confirm vs pipeline spec §2.1–2.6 (Kalman setup, δ selection, spread construction, Z-score, state machine, sizing, rebalance).

### `linsdex` skill — Deferred to Week 5+
`.agents/skills/eddiecunningham-linsdex-linsdex/SKILL.md` provides a JAX-based parallel Kalman filter with O(log T) GPU complexity. Do NOT use in Week 4 (JAX dependency overhead, different API). Flag as potential Phase 2 performance upgrade if 45-fold runtime is too slow.

## Smoke Test (before hard stop)
1. Run `kalman.py` on 1 pair (CMS + DUK) for 2022-07 data
2. Assert `spread_prior` kurtosis is in [3, 6] — if > 8, wrong state used
3. Assert `spread_prior` half-life is in [1, 10] days
4. Run `delta_selector.py` on Fold 1 pairs — assert optimal δ is selected and logged
5. Verify δ is NOT at grid boundary (if it is, log warning and expand grid)
6. Run `engine.py` for 1 pair, 1 month — assert `exec_timestamp > signal_timestamp` for 100% of trades (anti-lookahead proof)
7. Spot-check that no position is open across session boundary (15:55 ET → next day 09:30 ET should be flat unless max-holding > EOS)

---
## ⛔ HARD STOP — Review Before Phase 3

**Before proceeding to Phase 3, verify:**
- [ ] Prior spread kurtosis is ~3–5 for test pair (NOT 13.49 — that would be posterior)
- [ ] Prior spread half-life is in [1, 10] days for test pair
- [ ] δ auto-selection runs without error; optimal δ logged with kurt/HL/ACF78 metrics
- [ ] State machine compiles under Numba without error (first run may take 2–3s)
- [ ] Execution lag assertion passes: `exec_ts > signal_ts` for 100% of trades
- [ ] Rebalance events logged with cost, separate from entry/exit commission
- [ ] `/simplify` run — hot paths (Numba, delta loop) reviewed
- [ ] `/review` run — Kalman, δ selection, state machine confirmed against §2.1–2.6

**Signal to proceed:** User explicitly types "Phase 2 approved, proceed to Phase 3"
