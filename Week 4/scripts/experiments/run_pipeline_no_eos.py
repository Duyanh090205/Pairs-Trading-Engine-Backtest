"""
Full 45-Fold Pipeline — No-EOS Variant

Runs the same Phase 2+3 pipeline as run_full_pipeline.py but with the
end-of-session force-flatten removed (eos_flatten=False).  Outputs go
to a separate directory so the EOS-baseline results are never overwritten.

Usage:
    python run_pipeline_no_eos.py --max-holding 1d    # allow positions up to 1 trading day
    python run_pipeline_no_eos.py --max-holding 3d    # allow positions up to 3 trading days
    python run_pipeline_no_eos.py --max-holding none  # no time limit (zero-cross / signal only)

After the backtest completes, Phase 4 analytics run automatically on the
new results and a side-by-side comparison vs the EOS baseline is printed.
"""

import sys, os, argparse, logging, traceback, time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ---- Argument parsing ----
parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-holding", choices=["3d", "5d", "none"], default="5d",
    help="Maximum holding period per position (default: 5d)",
)
args = parser.parse_args()

MAX_HOLDING = args.max_holding

# Translate to engine parameters
EOS_FLATTEN = False  # always False for this script
if MAX_HOLDING == "3d":
    MAX_HOLDING_BARS = 390 * 3
elif MAX_HOLDING == "5d":
    MAX_HOLDING_BARS = 390 * 5   # ~1 full half-life for median 4.5d pair
else:  # "none"
    MAX_HOLDING_BARS = None

VARIANT_LABEL = f"no_eos_{MAX_HOLDING}"

print(f"=== No-EOS Variant: max_holding={MAX_HOLDING} ===\n")
print(f"  eos_flatten      = False")
print(f"  max_holding_bars = {MAX_HOLDING_BARS}")
print(f"  output suffix    = {VARIANT_LABEL}\n")

# Numba warmup
print("Warming up Numba JIT...")
from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution, _apply_max_holding
from src.phase2_execution.delta_selector import select_delta
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.phase3_backtest.neg_control import run_neg_control
from src.phase3_backtest.latency import run_latency_sweep
from src.phase3_backtest.audit_log import write_audit_log

# ---- Constants (same as baseline) ----
PHASE1_DIR       = "results/metrics/phase1_folds"
DATA_1MIN        = "data/validated/1min_phase2"
DATA_5MIN        = "data/validated/5min_phase1"
RESULTS_DIR      = f"results/metrics/{VARIANT_LABEL}"
LOG_DIR          = f"results/logs/{VARIANT_LABEL}"
BASELINE_METRICS = "results/metrics/fold_metrics.csv"

TOTAL_CAPITAL    = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
ENTRY_Z          = 2.0
TC_BPS           = 30.0
BORROW_BPS_YR    = 50.0

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---- Fold schedule (identical to baseline) ----
FOLD_SCHEDULE = [
    (1,  "2022-01-03", "2022-06-30", "2022-07"),
    (2,  "2022-02-01", "2022-07-31", "2022-08"),
    (3,  "2022-03-01", "2022-08-31", "2022-09"),
    (4,  "2022-04-01", "2022-09-30", "2022-10"),
    (5,  "2022-05-01", "2022-10-31", "2022-11"),
    (6,  "2022-06-01", "2022-11-30", "2022-12"),
    (7,  "2022-07-01", "2022-12-31", "2023-01"),
    (8,  "2022-08-01", "2023-01-31", "2023-02"),
    (9,  "2022-09-01", "2023-02-28", "2023-03"),
    (10, "2022-10-01", "2023-03-31", "2023-04"),
    (11, "2022-11-01", "2023-04-30", "2023-05"),
    (12, "2022-12-01", "2023-05-31", "2023-06"),
    (13, "2023-01-01", "2023-06-30", "2023-07"),
    (14, "2023-02-01", "2023-07-31", "2023-08"),
    (15, "2023-03-01", "2023-08-31", "2023-09"),
    (16, "2023-04-01", "2023-09-30", "2023-10"),
    (17, "2023-05-01", "2023-10-31", "2023-11"),
    (18, "2023-06-01", "2023-11-30", "2023-12"),
    (19, "2023-07-01", "2023-12-31", "2024-01"),
    (20, "2023-08-01", "2024-01-31", "2024-02"),
    (21, "2023-09-01", "2024-02-29", "2024-03"),
    (22, "2023-10-01", "2024-03-31", "2024-04"),
    (23, "2023-11-01", "2024-04-30", "2024-05"),
    (24, "2023-12-01", "2024-05-31", "2024-06"),
    (25, "2024-01-01", "2024-06-30", "2024-07"),
    (26, "2024-02-01", "2024-07-31", "2024-08"),
    (27, "2024-03-01", "2024-08-31", "2024-09"),
    (28, "2024-04-01", "2024-09-30", "2024-10"),
    (29, "2024-05-01", "2024-10-31", "2024-11"),
    (30, "2024-06-01", "2024-11-30", "2024-12"),
    (31, "2024-07-01", "2024-12-31", "2025-01"),
    (32, "2024-08-01", "2025-01-31", "2025-02"),
    (33, "2024-09-01", "2025-02-28", "2025-03"),
    (34, "2024-10-01", "2025-03-31", "2025-04"),
    (35, "2024-11-01", "2025-04-30", "2025-05"),
    (36, "2024-12-01", "2025-05-31", "2025-06"),
    (37, "2025-01-01", "2025-06-30", "2025-07"),
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
    (41, "2025-05-01", "2025-10-31", "2025-11"),
    (42, "2025-06-01", "2025-11-30", "2025-12"),
    (43, "2025-07-01", "2025-12-31", "2026-01"),
    (44, "2025-08-01", "2026-01-31", "2026-02"),
    (45, "2025-09-01", "2026-02-28", "2026-03"),
]


# ============================================================
# Data cache
# ============================================================

def _load_all_tickers(data_dir, compute_log_close):
    cache = {}
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    for fname in files:
        tk = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(data_dir, fname))
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache


def _slice_1min(cache, tickers, trading_month):
    year, month = int(trading_month[:4]), int(trading_month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == month)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_5min(cache, tickers, formation_start, formation_end):
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= formation_start) & (df.index <= formation_end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ============================================================
# Per-fold runner
# ============================================================

def run_fold(fold_n, formation_start, formation_end, trading_month, prev_delta,
             cache_1min, cache_5min):
    t0 = time.time()
    pairs_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(pairs_csv)
    except pd.errors.EmptyDataError:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 pairs (empty CSV) — skipped")
        return None, None

    if len(pairs_df) == 0:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 pairs — skipped")
        return None, None

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min   = _slice_1min(cache_1min, tickers, trading_month)
    formation_5min = _slice_5min(cache_5min, tickers, formation_start, formation_end)
    formation_1min = _slice_5min(cache_1min, tickers, formation_start, formation_end)

    if not trading_1min:
        print(f"  Fold {fold_n:02d} [{trading_month}]: no 1-min data — skipped")
        return None, None

    try:
        optimal_delta, delta_metrics = select_delta(pairs_df, formation_5min)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: select_delta FAILED: {e}")
        return None, None

    if optimal_delta is None:
        print(f"  Fold {fold_n:02d} [{trading_month}]: universal delta fail — skipped")
        return None, None

    try:
        fold_engine_results = run_fold_execution(
            pairs_df         = pairs_df,
            trading_1min     = trading_1min,
            delta            = optimal_delta,
            entry_z          = ENTRY_Z,
            eos_flatten      = EOS_FLATTEN,          # KEY CHANGE: False
            formation_5min   = formation_5min,
            formation_ref    = formation_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_execution FAILED: {e}")
        traceback.print_exc()
        return None, None

    # Apply max-holding cap post-execution if set
    if MAX_HOLDING_BARS is not None and fold_engine_results:
        fold_engine_results = {
            k: _apply_max_holding(df, MAX_HOLDING_BARS)
            for k, df in fold_engine_results.items()
            if not df.empty
        }

    if not fold_engine_results:
        print(f"  Fold {fold_n:02d} [{trading_month}]: engine returned 0 pairs")
        return None, None

    fold_config = {
        "fold": fold_n, "formation_start": formation_start,
        "formation_end": formation_end, "trading_month": trading_month,
        "delta": optimal_delta, "entry_z": ENTRY_Z,
        "tc_bps": int(TC_BPS), "borrow_bps_yr": int(BORROW_BPS_YR),
        "n_open_pairs_max": N_OPEN_PAIRS_MAX,
        "eos_flatten": EOS_FLATTEN, "max_holding": MAX_HOLDING,
    }

    try:
        fold_metrics = run_fold_pnl(
            fold_results     = fold_engine_results,
            trading_1min     = trading_1min,
            pairs_df         = pairs_df,
            delta            = optimal_delta,
            delta_metrics    = delta_metrics,
            total_capital    = TOTAL_CAPITAL,
            n_open_pairs_max = N_OPEN_PAIRS_MAX,
            tc_bps           = TC_BPS,
            borrow_bps_yr    = BORROW_BPS_YR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_pnl FAILED: {e}")
        traceback.print_exc()
        return None, None

    # Save per-pair trade metrics for volume stratification
    try:
        ppm = fold_metrics.get("per_pair_metrics", {})
        if ppm:
            rows = [
                {"fold": fold_n, "ticker_a": ta, "ticker_b": tb, **v}
                for (ta, tb), v in ppm.items()
            ]
            ppm_df = pd.DataFrame(rows)
            out_path = f"{RESULTS_DIR}/../pair_trade_metrics.csv"  # resolves to results/metrics/
            write_header = not os.path.exists(out_path)
            ppm_df.to_csv(out_path, mode="a", header=write_header, index=False)
    except Exception:
        pass

    nc = None
    try:
        nc = run_neg_control(
            trading_1min=trading_1min,
            delta=optimal_delta,
            primary_sharpe=fold_metrics["sharpe"],
            eos_flatten=EOS_FLATTEN,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_neg_control WARNING: {e}")

    lat = None
    try:
        lat = run_latency_sweep(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_latency_sweep WARNING: {e}")

    try:
        write_audit_log(
            fold_n=fold_n, fold_metrics=fold_metrics, nc_metrics=nc,
            latency_results=lat, delta=optimal_delta, delta_metrics=delta_metrics,
            config=fold_config, prev_delta=prev_delta, output_dir=LOG_DIR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: write_audit_log WARNING: {e}")

    eq = fold_metrics.get("bar_equity", pd.Series(dtype=float))
    if not eq.empty:
        eq.to_frame("equity").to_parquet(f"{RESULTS_DIR}/fold{fold_n:02d}_equity.parquet")

    elapsed  = time.time() - t0
    sharpe   = fold_metrics["sharpe"]
    n_trades = fold_metrics["n_trades"]
    nc_pass  = nc["nc_pass"] if nc else None
    t5       = lat["sharpe_by_lag"].get("t+5") if lat else None

    print(
        f"  Fold {fold_n:02d} [{trading_month}]  delta={optimal_delta:.0e}"
        f"  pairs={len(fold_engine_results)}  trades={n_trades}"
        f"  Sharpe={sharpe:+.3f}  MaxDD={fold_metrics['max_dd']:.3f}"
        f"  nc={'PASS' if nc_pass else 'FAIL' if nc_pass is not None else 'N/A'}"
        f"  t+5={f'{t5:+.3f}' if t5 is not None else 'N/A'}"
        f"  [{elapsed:.0f}s]"
    )

    summary_row = {
        "fold":            fold_n,
        "trading_month":   trading_month,
        "n_pairs":         len(fold_engine_results),
        "n_trades":        n_trades,
        "delta":           optimal_delta,
        "sharpe":          sharpe,
        "max_dd":          fold_metrics["max_dd"],
        "cagr":            fold_metrics["cagr"],
        "calmar":          fold_metrics["calmar"],
        "win_rate":        fold_metrics["win_rate"],
        "avg_hold_bars":   fold_metrics["avg_holding_bars"],
        "avg_net_bps":     fold_metrics["avg_net_bps"],
        "cost_commission": fold_metrics["cost_decomp"]["commission"],
        "cost_borrow":     fold_metrics["cost_decomp"]["borrow"],
        "cost_rebalance":  fold_metrics["cost_decomp"]["rebalance"],
        "nc_threshold":    nc["bootstrap_threshold"] if nc else None,
        "nc_pass":         nc["nc_pass"] if nc else None,
        "t1_sharpe":       lat["sharpe_by_lag"].get("t+1") if lat else None,
        "t5_sharpe":       lat["sharpe_by_lag"].get("t+5") if lat else None,
        "t10_sharpe":      lat["sharpe_by_lag"].get("t+10") if lat else None,
        "latency_pass":    lat["latency_pass"] if lat else None,
        "lookahead_ok":    fold_metrics["lookahead_ok"],
        "kalman_degen":    fold_metrics["kalman_degenerate"],
        "elapsed_s":       elapsed,
    }
    return summary_row, eq


# ============================================================
# Main
# ============================================================

print(f"Loading 1-min data cache from {DATA_1MIN}/ ...")
t_load = time.time()
cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
print(f"  {len(cache_1min)} tickers loaded in {time.time()-t_load:.1f}s")

print(f"Loading 5-min data cache from {DATA_5MIN}/ ...")
t_load = time.time()
cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_5min)} tickers loaded in {time.time()-t_load:.1f}s\n")

t_total    = time.time()
all_rows   = []
all_eq     = []
prev_delta = None

for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
    try:
        row, eq = run_fold(
            fold_n, formation_start, formation_end, trading_month,
            prev_delta, cache_1min, cache_5min,
        )
        if row is not None:
            all_rows.append(row)
            prev_delta = row["delta"]
            if eq is not None and not eq.empty:
                all_eq.append(eq)
        else:
            prev_delta = None
    except Exception as e:
        print(f"  Fold {fold_n:02d}: UNHANDLED ERROR: {e}")
        traceback.print_exc()
        prev_delta = None

# ---- Save outputs ----
summary_df = pd.DataFrame(all_rows)
out_csv = f"{RESULTS_DIR}/fold_metrics.csv"
summary_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

if all_eq:
    pd.concat(all_eq).sort_index().to_frame("equity").to_parquet(
        f"{RESULTS_DIR}/equity_full.parquet"
    )

total_min = (time.time() - t_total) / 60
print(f"\n=== Done in {total_min:.1f} min ===")
print(f"  Folds completed : {len(all_rows)} / 45")


# ============================================================
# Aggregate report + comparison vs EOS baseline
# ============================================================

def _regime_stats(df, lo, hi):
    sub = df[df["fold"].between(lo, hi)]
    if len(sub) == 0:
        return {"n": 0, "mean": float("nan"), "pct_pos": float("nan")}
    return {"n": len(sub), "mean": sub["sharpe"].mean(), "pct_pos": (sub["sharpe"] > 0).mean()}


if len(all_rows) > 0:
    df = summary_df

    print(f"\n{'='*60}")
    print(f"  NO-EOS VARIANT ({MAX_HOLDING})  —  {len(df)} completed folds")
    print(f"{'='*60}")
    print(f"  Sharpe : mean={df['sharpe'].mean():+.3f}  median={df['sharpe'].median():+.3f}"
          f"  std={df['sharpe'].std():.3f}  min={df['sharpe'].min():+.3f}  max={df['sharpe'].max():+.3f}")
    print(f"  % pos Sharpe : {(df['sharpe'] > 0).mean():.1%}")
    print(f"  MaxDD  : mean={df['max_dd'].mean():.4f}  worst={df['max_dd'].min():.4f}")
    print(f"  CAGR   : mean={df['cagr'].mean():+.4f}")
    print(f"  Win rate: mean={df['win_rate'].mean():.1%}")
    print(f"  Trades : total={df['n_trades'].sum():.0f}  mean/fold={df['n_trades'].mean():.0f}")
    print(f"  Avg hold (bars): {df['avg_hold_bars'].mean():.0f}")
    print(f"  Commission total: ${df['cost_commission'].sum():,.0f}")
    print(f"  Borrow total    : ${df['cost_borrow'].sum():,.0f}")
    print(f"  NC pass rate    : {df['nc_pass'].mean():.1%}")

    print(f"\n  --- Regime Breakdown ---")
    regimes = [
        ("Bear 2022        ", 1, 6),
        ("Early Bull 2023  ", 7, 18),
        ("Mid Bull 2024    ", 19, 30),
        ("Late Bull 2025-26", 31, 45),
    ]
    for name, lo, hi in regimes:
        s = _regime_stats(df, lo, hi)
        if s["n"] == 0:
            print(f"  {name}: no completed folds")
        else:
            print(f"  {name}  N={s['n']:2d}  Sharpe mean={s['mean']:+.3f}  pct_pos={s['pct_pos']:.0%}")

    # ---- Side-by-side comparison vs EOS baseline ----
    if os.path.exists(BASELINE_METRICS):
        print(f"\n{'='*60}")
        print(f"  COMPARISON: No-EOS ({MAX_HOLDING}) vs EOS baseline")
        print(f"{'='*60}")
        baseline = pd.read_csv(BASELINE_METRICS)
        merged = df.merge(baseline, on="fold", suffixes=("_noeos", "_eos"))

        def _cmp(col):
            noeos = merged[f"{col}_noeos"]
            eos   = merged[f"{col}_eos"]
            delta = (noeos - eos).mean()
            sign  = "+" if delta >= 0 else ""
            return f"{noeos.mean():+.3f} vs {eos.mean():+.3f}  (Δ={sign}{delta:.3f})"

        n_common = len(merged)
        print(f"  Matched folds   : {n_common}")
        print(f"  Sharpe          : {_cmp('sharpe')}")
        print(f"  MaxDD           : {_cmp('max_dd')}")
        print(f"  CAGR            : {_cmp('cagr')}")
        print(f"  Win rate        : {_cmp('win_rate')}")
        print(f"  Avg hold (bars) : {_cmp('avg_hold_bars')}")
        print(f"  Cost commission : {_cmp('cost_commission')}")
        print(f"  Cost borrow     : {_cmp('cost_borrow')}")

        print(f"\n  --- Per-fold Sharpe (no-EOS vs EOS) ---")
        for _, r in merged.sort_values("fold").iterrows():
            delta = r["sharpe_noeos"] - r["sharpe_eos"]
            sign  = "▲" if delta > 0.1 else ("▼" if delta < -0.1 else "≈")
            print(
                f"  Fold {int(r['fold']):02d} [{r['trading_month_noeos']}]"
                f"  no-eos={r['sharpe_noeos']:+6.3f}  eos={r['sharpe_eos']:+6.3f}"
                f"  Δ={delta:+.3f} {sign}"
            )
    else:
        print(f"\n  (No EOS baseline found at {BASELINE_METRICS} — skipping comparison)")

print(f"\nResults saved to: {RESULTS_DIR}/")
