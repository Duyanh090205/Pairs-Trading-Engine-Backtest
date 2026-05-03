"""
run_final_tc15.py — Option A + CORR25, TC=15 bps/leg (30 bps round-trip)

Identical to run_final_pipeline.py except:
  TC_BPS    30.0 -> 15.0   (halved: 15 bps entry + 15 bps exit = 30 bps round-trip)
  RESULTS_DIR -> results/metrics/final_tc15/
  LOG_DIR     -> results/logs/final_tc15/

Run with --skip-phase1 (Phase 1 folds already in results/metrics/phase1_folds/).
"""

import sys, os, logging, traceback, time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RUN_PHASE1 = "--skip-phase1" not in sys.argv

print("Warming up Numba JIT...")
from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.phase3_backtest.neg_control import run_neg_control
from src.phase3_backtest.latency import run_latency_sweep
from src.phase3_backtest.audit_log import write_audit_log

# ---- Constants ----
PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"
RESULTS_DIR = "results/metrics/final_tc15"
LOG_DIR     = "results/logs/final_tc15"

TOTAL_CAPITAL    = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
ENTRY_Z          = 3.0
TC_BPS           = 15.0        # halved: 15 bps/leg = 30 bps round-trip
BORROW_BPS_YR    = 50.0
FIXED_DELTA      = 1e-7
HL_MAX_DAYS      = 6.0
CORR25_THRESH    = 0.25
JOHANSEN_PVAL    = 0.05
MAX_PAIRS        = 500

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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


def _load_all_tickers(data_dir: str, compute_log_close: bool) -> dict:
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


def _slice_1min(cache: dict, tickers: set, trading_month: str) -> dict:
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


def _slice_range(cache: dict, tickers: set, start: str, end: str) -> dict:
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= start) & (df.index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _johansen_pval(log_a: np.ndarray, log_b: np.ndarray) -> float:
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


def apply_persistence_gate(pairs_df: pd.DataFrame, form_5min: dict, form_end: str) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame()
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= gate_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= gate_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        pv = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pv) and pv < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def apply_corr25_filter(pairs_df: pd.DataFrame, form_5min: dict,
                        form_start: str, form_end: str) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame()
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        ra = aln["log_close_a"].diff()
        rb = aln["log_close_b"].diff()
        valid = ra.notna() & rb.notna()
        if valid.sum() < 20:
            continue
        corr = float(ra[valid].corr(rb[valid]))
        if not np.isnan(corr) and corr >= CORR25_THRESH:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def run_fold(fold_n, formation_start, formation_end, trading_month, cache_1min, cache_5min):
    t0 = time.time()
    pairs_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(pairs_csv)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"  Fold {fold_n:02d} [{trading_month}]: no CSV — skip")
        return None, None

    if len(pairs_df) == 0:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 pairs — skip")
        return None, None

    if len(pairs_df) > MAX_PAIRS:
        n_before = len(pairs_df)
        pairs_df = pairs_df.nsmallest(MAX_PAIRS, "johansen_pval").reset_index(drop=True)
        print(f"  Fold {fold_n:02d}: spike cap {n_before} -> {MAX_PAIRS}")

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min   = _slice_1min(cache_1min, tickers, trading_month)
    formation_5min = _slice_range(cache_5min, tickers, formation_start, formation_end)
    formation_1min = _slice_range(cache_1min, tickers, formation_start, formation_end)

    if not trading_1min:
        print(f"  Fold {fold_n:02d} [{trading_month}]: no 1-min data — skip")
        return None, None

    pairs_df = apply_persistence_gate(pairs_df, formation_5min, formation_end)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after persistence gate — skip")
        return None, None

    pairs_df = pairs_df[pairs_df["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after HL cap — skip")
        return None, None

    pairs_df = apply_corr25_filter(pairs_df, formation_5min, formation_start, formation_end)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after CORR25 — skip")
        return None, None

    n_pairs_filtered = len(pairs_df)

    try:
        fold_engine_results = run_fold_execution(
            pairs_df=pairs_df,
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            entry_z=ENTRY_Z,
            eos_flatten=False,
            formation_5min=formation_5min,
            formation_ref=formation_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_execution FAILED: {e}")
        traceback.print_exc()
        return None, None

    if not fold_engine_results:
        print(f"  Fold {fold_n:02d} [{trading_month}]: engine 0 pairs")
        return None, None

    fold_config = {
        "fold": fold_n, "formation_start": formation_start,
        "formation_end": formation_end, "trading_month": trading_month,
        "delta": FIXED_DELTA, "entry_z": ENTRY_Z, "tc_bps": int(TC_BPS),
        "borrow_bps_yr": int(BORROW_BPS_YR), "n_open_pairs_max": N_OPEN_PAIRS_MAX,
        "eos_flatten": False, "hl_max_days": HL_MAX_DAYS, "corr25_thresh": CORR25_THRESH,
    }

    try:
        fold_metrics = run_fold_pnl(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
            pairs_df=pairs_df,
            delta=FIXED_DELTA,
            delta_metrics={},
            total_capital=TOTAL_CAPITAL,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_pnl FAILED: {e}")
        traceback.print_exc()
        return None, None

    # Save per-trade log
    try:
        tl = fold_metrics.get("trade_log", [])
        if tl:
            tl_df = pd.DataFrame(tl)
            tl_df.insert(0, "fold", fold_n)
            tl_path = f"{RESULTS_DIR}/trade_log.csv"
            write_header = not os.path.exists(tl_path)
            tl_df.to_csv(tl_path, mode="a", header=write_header, index=False)
    except Exception:
        pass

    nc = None
    try:
        nc = run_neg_control(
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            primary_sharpe=fold_metrics["sharpe"],
            eos_flatten=False,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_neg_control WARNING: {e}")

    lat = None
    try:
        lat = run_latency_sweep(fold_results=fold_engine_results, trading_1min=trading_1min)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_latency_sweep WARNING: {e}")

    try:
        write_audit_log(
            fold_n=fold_n, fold_metrics=fold_metrics, nc_metrics=nc, latency_results=lat,
            delta=FIXED_DELTA, delta_metrics={}, config=fold_config,
            prev_delta=FIXED_DELTA, output_dir=LOG_DIR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: write_audit_log WARNING: {e}")

    eq = fold_metrics.get("bar_equity", pd.Series(dtype=float))
    if not eq.empty:
        eq.to_frame("equity").to_parquet(f"{RESULTS_DIR}/fold{fold_n:02d}_equity.parquet")

    elapsed = time.time() - t0
    sharpe   = fold_metrics["sharpe"]
    n_trades = fold_metrics["n_trades"]
    nc_pass  = nc["nc_pass"] if nc else None
    t5       = lat["sharpe_by_lag"].get("t+5") if lat else None
    nc_str   = "PASS" if nc_pass else ("FAIL" if nc_pass is not None else "N/A")
    t5_str   = f"{t5:+.3f}" if t5 is not None else "N/A"

    print(
        f"  Fold {fold_n:02d} [{trading_month}]"
        f"  n_filt={n_pairs_filtered}  pairs={len(fold_engine_results)}  trades={n_trades}"
        f"  Sharpe={sharpe:+.3f}  MaxDD={fold_metrics['max_dd']:.3f}"
        f"  nc={nc_str}  t+5={t5_str}  [{elapsed:.0f}s]"
    )

    summary_row = {
        "fold": fold_n, "trading_month": trading_month,
        "n_pairs_filtered": n_pairs_filtered, "n_pairs_traded": len(fold_engine_results),
        "n_trades": n_trades, "delta": FIXED_DELTA, "sharpe": sharpe,
        "max_dd": fold_metrics["max_dd"], "cagr": fold_metrics["cagr"],
        "calmar": fold_metrics["calmar"], "win_rate": fold_metrics["win_rate"],
        "avg_hold_bars": fold_metrics["avg_holding_bars"],
        "avg_net_bps": fold_metrics["avg_net_bps"],
        "cost_commission": fold_metrics["cost_decomp"]["commission"],
        "cost_borrow": fold_metrics["cost_decomp"]["borrow"],
        "cost_rebalance": fold_metrics["cost_decomp"]["rebalance"],
        "nc_threshold": nc["bootstrap_threshold"] if nc else None,
        "nc_pass": nc["nc_pass"] if nc else None,
        "t1_sharpe": lat["sharpe_by_lag"].get("t+1") if lat else None,
        "t5_sharpe": lat["sharpe_by_lag"].get("t+5") if lat else None,
        "t10_sharpe": lat["sharpe_by_lag"].get("t+10") if lat else None,
        "latency_pass": lat["latency_pass"] if lat else None,
        "lookahead_ok": fold_metrics["lookahead_ok"],
        "kalman_degen": fold_metrics["kalman_degenerate"],
        "elapsed_s": elapsed,
    }
    return summary_row, eq


# ============================================================
print("=== TC=15 bps/leg Run — Option A + CORR25, halved TC ===")
print("  Config: no-EOS, Z=3.0, HL<=6d, delta=1e-7, CORR25>=0.25, TC=15bps/leg\n")

if RUN_PHASE1:
    from src.phase1_cointegration.discovery import run as run_phase1
    from src.utils.io import VALIDATED_DIR
    os.makedirs(PHASE1_DIR, exist_ok=True)
    print("=== Phase 1: Cointegration Discovery ===")
    t_p1 = time.time()
    for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
        out_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
        print(f"  Fold {fold_n:02d} [{trading_month}]: {formation_start} -> {formation_end} ...", end=" ", flush=True)
        try:
            pairs_df = run_phase1(formation_start, formation_end, VALIDATED_DIR)
            pairs_df.to_csv(out_csv, index=False)
            print(f"{len(pairs_df)} pairs")
        except Exception as e:
            print(f"ERROR: {e}")
            pd.DataFrame().to_csv(out_csv, index=False)
    print(f"Phase 1 complete: {time.time()-t_p1:.0f}s\n")

print(f"Loading 1-min cache from {DATA_1MIN}/ ...")
t_load = time.time()
cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
print(f"  {len(cache_1min)} tickers  [{time.time()-t_load:.1f}s]")

print(f"Loading 5-min cache from {DATA_5MIN}/ ...")
t_load = time.time()
cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_5min)} tickers  [{time.time()-t_load:.1f}s]\n")

t_total  = time.time()
all_rows = []
all_eq   = []

_ppm_path = f"{RESULTS_DIR}/pair_trade_metrics.csv"
if os.path.exists(_ppm_path):
    os.remove(_ppm_path)
_tl_path = f"{RESULTS_DIR}/trade_log.csv"
if os.path.exists(_tl_path):
    os.remove(_tl_path)

for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
    try:
        row, eq = run_fold(fold_n, formation_start, formation_end, trading_month,
                           cache_1min, cache_5min)
        if row is not None:
            all_rows.append(row)
            if eq is not None and not eq.empty:
                all_eq.append(eq)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: UNHANDLED ERROR: {e}")
        traceback.print_exc()

summary_df = pd.DataFrame(all_rows)
summary_df.to_csv(f"{RESULTS_DIR}/fold_metrics.csv", index=False)

if all_eq:
    pd.concat(all_eq).sort_index().to_frame("equity").to_parquet(
        f"{RESULTS_DIR}/equity_full.parquet"
    )

total_min = (time.time() - t_total) / 60
print(f"\n=== Done in {total_min:.1f} min ===")
print(f"  Folds completed : {len(all_rows)} / 45")
print(f"  Results CSV     : {RESULTS_DIR}/fold_metrics.csv")

if len(all_rows) > 0:
    df = summary_df
    print(f"\n{'='*60}")
    print(f"  TC=15bps/leg  ({len(df)} folds)   [vs TC=30bps/leg baseline in brackets]")
    print(f"{'='*60}")
    print(f"  Sharpe : mean={df['sharpe'].mean():+.3f}  median={df['sharpe'].median():+.3f}"
          f"  std={df['sharpe'].std():.3f}  min={df['sharpe'].min():+.3f}  max={df['sharpe'].max():+.3f}")
    print(f"  % pos  : {(df['sharpe'] > 0).mean():.1%}  [was 48%]")
    print(f"  MaxDD  : mean={df['max_dd'].mean():.4f}  worst={df['max_dd'].min():.4f}")
    print(f"  CAGR   : mean={df['cagr'].mean():+.4f}")
    print(f"  Win%   : mean={df['win_rate'].mean():.1%}")
    print(f"  Trades : total={df['n_trades'].sum():.0f}")
    print(f"  Comm $ : ${df['cost_commission'].sum():,.0f}  [was ~$55,627]")
    nc_valid = df['nc_pass'].dropna()
    print(f"  NC pass: {nc_valid.mean():.1%}  ({nc_valid.sum():.0f}/{len(nc_valid)})")

    print(f"\n  --- By Regime ---")
    regimes = [
        ("Bear 2022        ", df["fold"].between(1, 6)),
        ("Early Bull 2023  ", df["fold"].between(7, 18)),
        ("Mid Bull 2024    ", df["fold"].between(19, 30)),
        ("Late Bull 2025-26", df["fold"].between(31, 45)),
    ]
    for name, mask in regimes:
        sub = df[mask]
        if len(sub) == 0:
            print(f"  {name}: no completed folds")
            continue
        print(
            f"  {name}  N={len(sub):2d}"
            f"  Sharpe mean={sub['sharpe'].mean():+.3f}"
            f"  pct_pos={(sub['sharpe'] > 0).mean():.0%}"
            f"  trades={int(sub['n_trades'].sum()):5d}"
        )

    print(f"\n  --- Per-Fold (TC=15 vs TC=30) ---")
    print(f"  {'Fold':<5} {'Month':<9} {'Trades':>7} {'Sharpe@15':>10} {'MaxDD':>8} {'Win%':>6}")
    print("  " + "-" * 55)
    for _, r in df.iterrows():
        pos = " *" if r["sharpe"] > 0 else ""
        print(
            f"  {int(r['fold']):<5} {r['trading_month']:<9}"
            f" {int(r['n_trades']):>7}"
            f" {r['sharpe']:>+10.3f}{pos}"
            f" {r['max_dd']:>8.4f}"
            f" {r['win_rate']:>6.1%}"
        )
