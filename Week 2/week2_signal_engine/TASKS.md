# Week 2 Tasks

## Scaffold/Setup
- [x] Initialize project structure
- [x] Set up virtual environment and install dependencies (`requirements.txt`)
- [x] Configure `params_example.yaml` for actual runs
  - Primary: CMS/DUK | Secondary: DOW/LYB | 6 alternatives | 1 benchmark (GOOG/GOOGL) | 2 negative controls

## Data Loading/Alignment
- [x] `src/data/loaders.py`: Load 1-min CSVs and align timestamps
  - DST-aware session filter (America/New_York, 09:30–16:00 ET) — fixed UTC-based approach that cut first hour in summer
  - Inner join on DatetimeIndex guarantees zero NaNs post-merge
- [x] `src/data/validation.py`: Data coverage and gap analysis
  - CMS: 99.06% coverage, DUK: 99.37% — both clean
  - Pre-market/after-hours bars present in raw CSVs; filtered before analysis
- [x] Split data into formation (Jan–Jun 2022) and trading (Jul–Dec 2022) periods
  - Guard: raises ValueError if formation_end >= trading_start (no lookahead invariant)

## Spread Characterization
- [x] `src/signals/spread.py`: OLS hedge ratio on formation, construct spread
  - log(A) = alpha + beta * log(B); estimated via np.linalg.lstsq on formation only
  - CMS/DUK: alpha=-0.6956, beta=1.0487
  - DOW/LYB: alpha=-0.8770, beta=1.0828
- [x] `src/analytics/characterize.py`: Half-life and Z-score window
  - CMS/DUK half-life: 679.7 bars — window clamped to max 240 (35% of HL)
  - DOW/LYB half-life: 777.8 bars — same cap applies
  - GOOG/GOOGL half-life: 19.6 bars — window=20, well-matched
- [x] `src/analytics/characterize.py`: Hurst exponent (variance-ratio estimator)
  - CMS/DUK H=0.452 (formation), 100% of rolling windows H<0.5 in trading
  - GOOG/GOOGL H=0.201 — strongest mean-reversion signal in the universe
- [x] Define fixed (2.0) and adaptive (formation abs-Z 95th pct) thresholds
  - CMS/DUK adaptive: 2.5675 | DOW/LYB adaptive: 2.694

## Signal Engine
- [x] `src/signals/zscore.py`: Vectorized rolling Z-score
  - Fully pandas .rolling() based — no Python loops
  - ddof=1, eps=1e-10 guard, min_periods=window//2
- [x] `src/signals/state_machine.py`: Stateful path-dependent position tracking
  - Numba @njit(cache=True) kernel — compiled to machine code at import
  - CRITICAL: fastmath=True removed — breaks NaN semantics (nnan LLVM flag causes
    np.isnan() to never fire, producing false long entries on burn-in bars)
  - NaN bars hold current state; mandatory zero-crossing exit before re-entry enforced

## Diagnostics
- [x] `src/analytics/diagnostics.py`: Tail behavior, skewness, kurtosis
  - CMS/DUK: std=1.3928 (excess kurtosis=0.057) — NOT genuine fat tails
  - Root cause: window-truncation artifact, not microstructure fat tails
- [x] `src/analytics/diagnostics.py`: Empirical coverage tables
  - CMS/DUK: +-2.0 coverage 86.05% vs 95.45% theory — gap +9.40%
  - Same ~9-11% gap across ALL 10 slow-HL pairs — confirms mechanical cause
  - GOOG/GOOGL gap only +2.94% — well-calibrated (window matches HL)
- [x] `src/analytics/diagnostics.py`: Quantile cross-validation (formation vs trading)
  - Zero regime shifts detected across all 11 pairs (all drift < 0.5)
- [x] `src/analytics/sensitivity.py`: Trade counts across thresholds [1.5, 2.0, adaptive, 2.5, 3.0]
  - CMS/DUK: 382 / 265 / 193 / 184 / 123 trades
  - Adaptive (2.5675) fires 30% fewer trades than fixed 2.0

## Visual Validation
- [x] `src/visuals/plots.py`: 3-panel signal validation chart (prices / spread / Z-score with trade markers)
  - Generated for all 11 pairs (33 PNG files total in outputs/figures/all_pairs/)
- [x] `src/visuals/plots.py`: Rolling Hurst regime monitor chart (all 11 pairs)
- [x] `src/visuals/plots.py`: QQ-plot for Z-score distribution (all 11 pairs)

## Pipeline Assembly
- [x] `src/pipeline/run_week2.py`: Full pipeline assembled
  - run_pipeline(): pure computation function — no I/O, safe for Week 3 unit tests
  - main(): loads config, runs primary + secondary, saves CSVs
  - Output columns: spread, rolling_mean, rolling_std, zscore, position,
    signal_valid (burn-in gate), position_executed (position.shift(1) — Week 3 PnL)
- [x] Signal CSVs saved: outputs/signals/signals_CMS_DUK.csv, signals_DOW_LYB.csv
  - CMS/DUK: 48,881 rows | DOW/LYB: 49,075 rows
- [x] Cross-pair comparison table (`run_all_pairs_diagnostics.py`) — all 11 pairs
- [x] Signal Logic Document — The Rules of Engagement  → `docs/Signal Logic Document.md`

## Final QA
- [x] Unit tests: 29 tests across test_spread.py, test_state_machine.py, test_zscore.py — all pass
- [x] Execution convention verified: signal_valid gates burn-in; position_executed encodes 1-bar shift
- [x] No-lookahead invariant verified end-to-end (formation_end < trading_start confirmed)
- [x] Document final results  → see `docs/Signal Logic Document.md`
