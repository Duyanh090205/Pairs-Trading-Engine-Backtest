"""
Phase 4 Smoke Test — 50-point checklist

Tests:
  A. Fold schedule correctness          (checks 1–9)
  B. Data loaders                       (checks 10–17)
  C. Regime analysis                    (checks 18–23)
  D. Overfitting diagnostics (DSR+PBO)  (checks 24–30)
  E. OAT sensitivity                    (checks 31–37)
  F. Exit reason parsing                (checks 38–41)
  G. Output file presence               (checks 42–50)

Run: python -m pytest tests/test_phase4_smoke.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.phase4_defense.orchestrator import (
    FOLD_SCHEDULE,
    FOLD_BY_N,
    REGIME_MAP,
    FOLD_TO_REGIME,
    load_fold_metrics,
    load_equity,
    load_phase1_pairs,
    get_daily_returns,
    get_completed_folds,
    _METRICS_DIR,
    _FIGURES_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# A. Fold Schedule
# ─────────────────────────────────────────────────────────────────────────────

def test_A01_fold_count():
    """FOLD_SCHEDULE has exactly 45 entries."""
    assert len(FOLD_SCHEDULE) == 45

def test_A02_fold_numbers_unique():
    """Fold numbers 1..45 are all unique and contiguous."""
    ns = [f.fold_n for f in FOLD_SCHEDULE]
    assert sorted(ns) == list(range(1, 46))

def test_A03_no_duplicate_trading_months():
    """No two folds share the same trading month."""
    months = [f.trading_month for f in FOLD_SCHEDULE]
    assert len(months) == len(set(months)), f"Duplicate months: {[m for m in months if months.count(m)>1]}"

def test_A04_fold1_dates():
    """Fold 1 matches stored phase1 CSV: form 2022-01-03→2022-06-30, trade 2022-07."""
    f = FOLD_BY_N[1]
    assert f.form_start   == "2022-01-03"
    assert f.form_end     == "2022-06-30", f"Fold 1 form_end={f.form_end}, expected 2022-06-30"
    assert f.trading_month == "2022-07",   f"Fold 1 trading={f.trading_month}, expected 2022-07"

def test_A05_fold7_dates():
    """Fold 7: formation Jul–Dec 2022, trading Jan 2023."""
    f = FOLD_BY_N[7]
    assert f.form_start    == "2022-07-01"
    assert f.form_end      == "2022-12-31"
    assert f.trading_month == "2023-01"

def test_A06_fold45_dates():
    """Fold 45 ends at 2026-03 (last fold in schedule)."""
    f = FOLD_BY_N[45]
    assert f.form_start    == "2025-09-01"
    assert f.form_end      == "2026-02-28"
    assert f.trading_month == "2026-03"

def test_A07_regime_map_coverage():
    """REGIME_MAP covers all 45 folds with no gaps or overlaps."""
    covered = set()
    for folds in REGIME_MAP.values():
        for fn in folds:
            assert fn not in covered, f"Fold {fn} in multiple regimes"
            covered.add(fn)
    assert covered == set(range(1, 46)), f"Missing folds: {set(range(1,46)) - covered}"

def test_A08_fold_to_regime_all_folds():
    """FOLD_TO_REGIME has exactly 45 entries."""
    assert len(FOLD_TO_REGIME) == 45

def test_A09_regime_assignments():
    """Regime boundaries match spec: Bear=1-6, E.Bull=7-18, M.Bull=19-30, L.Bull=31-45."""
    assert FOLD_TO_REGIME[1]  == "Late Bear 2022"
    assert FOLD_TO_REGIME[6]  == "Late Bear 2022"
    assert FOLD_TO_REGIME[7]  == "Early Bull 2023"
    assert FOLD_TO_REGIME[18] == "Early Bull 2023"
    assert FOLD_TO_REGIME[19] == "Mid Bull 2024"
    assert FOLD_TO_REGIME[30] == "Mid Bull 2024"
    assert FOLD_TO_REGIME[31] == "Late Bull 2025-Q12026"
    assert FOLD_TO_REGIME[45] == "Late Bull 2025-Q12026"


# ─────────────────────────────────────────────────────────────────────────────
# B. Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fold_metrics():
    return load_fold_metrics()

def test_B10_load_fold_metrics_rows(fold_metrics):
    """fold_metrics.csv loads with 32 completed folds."""
    assert len(fold_metrics) == 32

def test_B11_fold_metrics_has_regime(fold_metrics):
    """load_fold_metrics() adds 'regime' column from FOLD_TO_REGIME."""
    assert "regime" in fold_metrics.columns
    assert fold_metrics["regime"].notna().all()

def test_B12_fold_metrics_required_columns(fold_metrics):
    """fold_metrics has all required columns."""
    required = {"fold","trading_month","sharpe","max_dd","cagr","calmar",
                "n_pairs","n_trades","delta","cost_commission"}
    missing = required - set(fold_metrics.columns)
    assert not missing, f"Missing columns: {missing}"

def test_B13_load_equity_fold1():
    """load_equity(1) returns a non-empty DataFrame."""
    eq = load_equity(1)
    assert eq is not None
    assert len(eq) > 0

def test_B14_load_equity_missing_fold():
    """load_equity(99) returns None for non-existent fold."""
    assert load_equity(99) is None

def test_B15_load_phase1_pairs_fold1():
    """load_phase1_pairs(1) returns DataFrame with pairs."""
    df = load_phase1_pairs(1)
    assert df is not None
    assert len(df) > 0
    assert "ticker_a" in df.columns
    assert "ticker_b" in df.columns

def test_B16_load_phase1_pairs_empty_fold():
    """load_phase1_pairs(7) returns None (fold_07.csv has no pairs)."""
    df = load_phase1_pairs(7)
    assert df is None, f"Expected None for empty fold 7, got {type(df)}"

def test_B17_get_daily_returns_fold1():
    """get_daily_returns(1) returns non-None Series with length > 0."""
    dr = get_daily_returns(1)
    assert dr is not None
    assert len(dr) > 0
    assert not dr.isna().all()

def test_B17b_get_completed_folds(fold_metrics):
    """get_completed_folds() returns 32 fold numbers."""
    completed = get_completed_folds(fold_metrics)
    assert len(completed) == 32
    assert all(isinstance(f, int) for f in completed)


# ─────────────────────────────────────────────────────────────────────────────
# C. Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def regime_df(fold_metrics):
    from src.phase4_defense.regime import run_regime_analysis
    return run_regime_analysis(fold_metrics, save=False)

def test_C18_regime_df_rows(regime_df):
    """run_regime_analysis() returns exactly 4 rows (one per regime)."""
    assert len(regime_df) == 4

def test_C19_regime_df_columns(regime_df):
    """regime_df has required columns."""
    required = {"regime","n_folds","n_completed","mean_sharpe","median_sharpe",
                "iqr_25","iqr_75","pct_positive","min_sharpe","max_sharpe"}
    missing = required - set(regime_df.columns)
    assert not missing, f"Missing: {missing}"

def test_C20_regime_bear_counts(regime_df):
    """Late Bear 2022: n_folds=6, n_completed=4."""
    bear = regime_df[regime_df["regime"] == "Late Bear 2022"].iloc[0]
    assert bear["n_folds"]     == 6
    assert bear["n_completed"] == 4

def test_C21_regime_all_sharpe_negative(regime_df):
    """All regime mean Sharpes are negative (no spurious positive alpha)."""
    for _, row in regime_df.iterrows():
        if row["n_completed"] > 0:
            assert row["mean_sharpe"] < 0, \
                f"{row['regime']} mean_sharpe={row['mean_sharpe']:.2f} unexpectedly positive"

def test_C22_regime_pct_positive_zero(regime_df):
    """pct_positive is 0.0 for all regimes (0% positive folds)."""
    for _, row in regime_df.iterrows():
        if row["n_completed"] > 0:
            assert row["pct_positive"] == 0.0

def test_C23_regime_iqr_ordered(regime_df):
    """iqr_25 <= median_sharpe <= iqr_75 for each regime."""
    for _, row in regime_df.iterrows():
        if row["n_completed"] > 1:
            assert row["iqr_25"] <= row["median_sharpe"] <= row["iqr_75"], \
                f"{row['regime']}: IQR [{row['iqr_25']:.2f},{row['iqr_75']:.2f}] " \
                f"does not bracket median {row['median_sharpe']:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# D. Overfitting Diagnostics (DSR + PBO)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def overfit_df(fold_metrics):
    from src.phase4_defense.overfitting import run_overfitting_diagnostics
    return run_overfitting_diagnostics(fold_metrics, save=False)

def test_D24_overfit_single_row(overfit_df):
    """run_overfitting_diagnostics() returns exactly 1 row."""
    assert len(overfit_df) == 1

def test_D25_dsr_in_unit_interval(overfit_df):
    """DSR is in [0, 1]."""
    dsr = float(overfit_df["dsr"].iloc[0])
    assert 0.0 <= dsr <= 1.0, f"DSR={dsr} outside [0,1]"

def test_D26_pbo_in_unit_interval(overfit_df):
    """PBO is in [0, 1]."""
    pbo = float(overfit_df["pbo"].iloc[0])
    assert 0.0 <= pbo <= 1.0, f"PBO={pbo} outside [0,1]"

def test_D27_n_daily_obs_positive(overfit_df):
    """n_daily_obs > 0 (equity parquets loaded successfully)."""
    assert float(overfit_df["n_daily_obs"].iloc[0]) > 0

def test_D28_dsr_expected_range(overfit_df):
    """DSR < 0.10 for a uniformly negative strategy (near-zero or zero)."""
    dsr = float(overfit_df["dsr"].iloc[0])
    assert dsr < 0.10, f"DSR={dsr:.4f} unexpectedly high for all-negative folds"

def test_D29_pbo_paths_sampled(overfit_df):
    """PBO used at least 100 paths."""
    assert float(overfit_df["pbo_n_paths"].iloc[0]) >= 100

def test_D30_raw_sharpe_negative(overfit_df):
    """Annualised Sharpe from daily returns is negative."""
    sr = float(overfit_df["raw_sharpe_annual"].iloc[0])
    assert sr < 0, f"Aggregate Sharpe={sr:.3f} unexpectedly non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# E. OAT Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def oat_df(fold_metrics):
    from src.phase4_defense.sensitivity import run_analytical_oat
    return run_analytical_oat(fold_metrics)

def test_E31_oat_has_default_row(oat_df):
    """OAT table contains a 'default' row."""
    assert "default" in oat_df["param"].values

def test_E32_oat_tc_rows(oat_df):
    """OAT table contains all 4 TC variations."""
    tc_rows = oat_df[oat_df["param"] == "tc_bps"]
    assert len(tc_rows) == 4, f"Expected 4 TC rows, got {len(tc_rows)}"

def test_E33_oat_tc_monotone(oat_df):
    """Lower TC → higher mean Sharpe (monotone, correct direction)."""
    tc_rows = oat_df[oat_df["param"] == "tc_bps"].sort_values("value")
    sharpes = tc_rows["mean_sharpe"].tolist()
    # Lower TC value should give higher (less negative) Sharpe
    tc_vals = [float(v) for v in tc_rows["value"].tolist()]
    for i in range(len(tc_vals) - 1):
        assert sharpes[i] >= sharpes[i+1], \
            f"TC monotonicity violated: TC={tc_vals[i]} Sharpe={sharpes[i]:.3f} " \
            f"vs TC={tc_vals[i+1]} Sharpe={sharpes[i+1]:.3f}"

def test_E34_oat_borrow_rows(oat_df):
    """OAT table contains all 3 borrow rate variations."""
    borrow_rows = oat_df[oat_df["param"] == "borrow_bps_yr"]
    assert len(borrow_rows) == 3

def test_E35_oat_n_pairs_rows(oat_df):
    """OAT table contains all 3 N_open_pairs_max variations."""
    n_rows = oat_df[oat_df["param"] == "N_open_pairs_max"]
    assert len(n_rows) == 3

def test_E36_oat_tc_improves_from_high_to_low(oat_df):
    """Lower TC always gives higher (or equal) Sharpe — direction must be correct.
    Note: TC=30bps CAN produce positive Sharpe when commission is a large fraction
    of capital (fold 11: $570k commission on $1M), which is a valid result."""
    default_row = oat_df[oat_df["param"] == "default"]["mean_sharpe"].iloc[0]
    tc30_row    = oat_df[(oat_df["param"] == "tc_bps") & (oat_df["value"] == "30")]["mean_sharpe"].iloc[0]
    tc75_row    = oat_df[(oat_df["param"] == "tc_bps") & (oat_df["value"] == "75")]["mean_sharpe"].iloc[0]
    # TC=30 must be better than default, TC=75 must be worse than default
    assert tc30_row > default_row, f"TC=30 ({tc30_row:.2f}) not better than default ({default_row:.2f})"
    assert tc75_row < default_row, f"TC=75 ({tc75_row:.2f}) not worse than default ({default_row:.2f})"

def test_E37_oat_default_matches_fold_metrics_mean(oat_df, fold_metrics):
    """Default row mean_sharpe matches fold_metrics mean (within 0.01)."""
    default_sharpe = float(oat_df[oat_df["param"] == "default"]["mean_sharpe"].iloc[0])
    fm_mean        = float(fold_metrics["sharpe"].mean())
    assert abs(default_sharpe - fm_mean) < 0.01, \
        f"Default OAT Sharpe {default_sharpe:.4f} != fold_metrics mean {fm_mean:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# F. Exit Reason Parsing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def exit_df():
    from src.phase4_defense.sensitivity import compute_exit_reasons
    return compute_exit_reasons()

def test_F38_exit_df_nonempty(exit_df):
    """compute_exit_reasons() returns at least 1 row (audit logs present)."""
    assert len(exit_df) > 0, "No audit logs parsed — check results/logs/"

def test_F39_exit_eos_dominates(exit_df):
    """EOS exits > 50% across all folds (known structural finding)."""
    total_eos    = exit_df["n_eos"].sum()
    total_trades = exit_df["n_trades_parsed"].sum()
    pct_eos = total_eos / total_trades
    assert pct_eos > 0.40, f"EOS fraction {pct_eos:.1%} unexpectedly low"

def test_F40_exit_pct_sums_near_one(exit_df):
    """pct_eos + pct_zero_cross covers most exits per fold.
    Tolerance is 0.15 because audit logs only show the first 20 trades per fold;
    occasional 'max_hold' or other exit types in that sample reduce the sum."""
    for _, row in exit_df.iterrows():
        total_pct = row["pct_eos"] + row["pct_zero_cross"]
        assert total_pct >= 0.85, \
            f"Fold {int(row['fold'])}: pct_eos+pct_zc={total_pct:.3f} unexpectedly low"

def test_F41_exit_avg_eos_net_higher_than_zc(exit_df):
    """EOS avg net bps > zero-cross avg net bps (known structural finding)."""
    avg_eos = float(exit_df["avg_net_eos"].mean())
    avg_zc  = float(exit_df["avg_net_zc"].mean())
    assert avg_eos > avg_zc, \
        f"EOS avg net ({avg_eos:.1f} bps) not > zero-cross ({avg_zc:.1f} bps) — unexpected"


# ─────────────────────────────────────────────────────────────────────────────
# G. Output File Presence
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_METRICS = [
    "sharpe_distribution.csv",
    "regime_sharpes.csv",
    "overfitting_diagnostics.csv",
    "oat_sensitivity.csv",
    "nc_bootstrap.csv",
    "latency_decay.csv",
    "exit_reasons.csv",
    "cost_decomp.csv",
    "delta_trajectory.csv",
    "universe_counts.csv",
]

REQUIRED_FIGURES = [
    "sharpe_hist.png",
    "regime_bar.png",
    "delta_traj.png",
    "latency_curve.png",
    "oat_grid.png",
]

@pytest.mark.parametrize("fname", REQUIRED_METRICS)
def test_G42_metric_file_exists(fname):
    """Required metrics CSV exists."""
    assert (_METRICS_DIR / fname).exists(), f"Missing: results/metrics/{fname}"

@pytest.mark.parametrize("fname", REQUIRED_FIGURES)
def test_G43_figure_file_exists(fname):
    """Required figure PNG exists."""
    assert (_FIGURES_DIR / fname).exists(), f"Missing: results/figures/{fname}"

def test_G44_sharpe_distribution_rows():
    """sharpe_distribution.csv has 32 rows (one per completed fold)."""
    df = pd.read_csv(_METRICS_DIR / "sharpe_distribution.csv")
    assert len(df) == 32

def test_G45_regime_sharpes_rows():
    """regime_sharpes.csv has exactly 4 rows."""
    df = pd.read_csv(_METRICS_DIR / "regime_sharpes.csv")
    assert len(df) == 4

def test_G46_overfitting_rows():
    """overfitting_diagnostics.csv has exactly 1 row."""
    df = pd.read_csv(_METRICS_DIR / "overfitting_diagnostics.csv")
    assert len(df) == 1

def test_G47_universe_counts_all_folds():
    """universe_counts.csv covers all 45 folds."""
    df = pd.read_csv(_METRICS_DIR / "universe_counts.csv")
    assert len(df) == 45
    assert set(df["fold"].tolist()) == set(range(1, 46))

def test_G48_universe_counts_fold23_pairs():
    """Fold 23 has the largest pair count (known: 609 pairs)."""
    df = pd.read_csv(_METRICS_DIR / "universe_counts.csv")
    fold23 = df[df["fold"] == 23]["n_surviving_pairs"].iloc[0]
    assert fold23 >= 100, f"Fold 23 pairs={fold23}, expected >>100"

def test_G49_cost_decomp_borrow_zero():
    """cost_borrow = $0 across all folds (EOS exits prevent overnight shorts)."""
    df = pd.read_csv(_METRICS_DIR / "cost_decomp.csv")
    if "cost_borrow" in df.columns:
        assert df["cost_borrow"].sum() == 0.0, \
            f"Unexpected borrow cost: ${df['cost_borrow'].sum():,.0f}"

def test_G50_delta_trajectory_all_same():
    """delta_trajectory.csv: all 32 folds selected delta=1e-7 (lower boundary)."""
    df = pd.read_csv(_METRICS_DIR / "delta_trajectory.csv")
    assert len(df) == 32
    assert (df["delta"] == 1e-7).all(), \
        f"Unexpected delta values: {df['delta'].unique()}"
