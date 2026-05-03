"""
run_outlier_audit.py  —  Deep-dive into Folds 12 and 23 (Option A outliers).

Fold 12: 2023-06, SR=-7.38, 17 exec pairs, 2 trades
Fold 23: 2024-05, SR=-7.91, 54 exec pairs, 36 trades

For each fold, reports:
  - Per-pair gross PnL, net PnL, n_trades, max single-trade loss
  - Cost decomposition breakdown
  - Which pairs are the big losers
  - Borrow cost vs commission vs rebalance breakdown
"""

import sys, os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()

from src.phase2_execution.engine import run_fold_execution, _N_OPEN_PAIRS_MAX, _TOTAL_CAPITAL
from src.phase3_backtest.pnl import compute_pair_pnl

PHASE1_DIR = "results/metrics/phase1_folds"
DATA_1MIN  = "data/validated/1min_phase2"
DATA_5MIN  = "data/validated/5min_phase1"

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05
PER_PAIR_DOLLAR = _TOTAL_CAPITAL / _N_OPEN_PAIRS_MAX

AUDIT_FOLDS = [
    (12, "2022-12-01", "2023-05-31", "2023-06"),
    (23, "2023-11-01", "2024-04-30", "2024-05"),
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_cache(data_dir, compute_log_close=False):
    cache = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".parquet"):
            continue
        tk = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(data_dir, fname))
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache

def _slice(cache, tickers, start, end):
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        sliced = cache[tk][(cache[tk].index >= start) & (cache[tk].index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out

def _slice_month(cache, tickers, month):
    y, m = int(month[:4]), int(month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        sliced = cache[tk][(cache[tk].index.year == y) & (cache[tk].index.month == m)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out

def _johansen_pval(log_a, log_b):
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan

def apply_persistence_gate(pairs_df, form_5min, form_end):
    if pairs_df.empty:
        return pd.DataFrame()
    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        aln = (
            fa[(fa.index >= gate_start) & (fa.index <= gate_end)][["log_close"]]
            .join(
                fb[(fb.index >= gate_start) & (fb.index <= gate_end)][["log_close"]],
                lsuffix="_a", rsuffix="_b", how="inner",
            ).dropna()
        )
        if len(aln) < 20:
            continue
        pval = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pval) and pval < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------

print("Loading data caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")


# ---------------------------------------------------------------------------
# Per-fold audit
# ---------------------------------------------------------------------------

for fold_n, form_start, form_end, trading_month in AUDIT_FOLDS:
    print("=" * 80)
    print(f"FOLD {fold_n:02d}  Trading: {trading_month}  Formation: {form_start} to {form_end}")
    print("=" * 80)

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    pairs_df = pd.read_csv(csv)
    if len(pairs_df) > 500:
        pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)

    tickers  = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min    = _slice_month(cache_1min, tickers, trading_month)
    f1min    = _slice(cache_1min, tickers, form_start, form_end)
    f5min    = _slice(cache_5min, tickers, form_start, form_end)

    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    print(f"Pairs: {len(pairs_df)} base -> {len(pairs_gated)} after gate")

    exec_res = run_fold_execution(
        pairs_df=pairs_gated, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    print(f"Exec pairs: {len(exec_res)}\n")

    # Per-pair PnL breakdown
    pair_rows = []
    for (ta, tb), pair_df in exec_res.items():
        ca = t1min.get(ta, pd.DataFrame()).get("close", pd.Series(dtype=float))
        cb = t1min.get(tb, pd.DataFrame()).get("close", pd.Series(dtype=float))

        result = compute_pair_pnl(
            pair_df, ca, cb, PER_PAIR_DOLLAR,
            tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS,
        )

        bar_pnl     = result["bar_pnl"]
        bar_gross   = result["bar_gross"]
        trade_log   = result["trade_log"]
        n_trades    = len(trade_log)
        net_pnl     = float(bar_pnl.sum())
        gross_pnl   = float(bar_gross.sum())
        cost_comm   = float(result["cost_commission"])
        cost_borrow = float(result["cost_borrow"])
        cost_reb    = float(result["cost_rebalance"])

        # Worst single trade
        worst_trade = float("nan")
        if trade_log:
            worsts = [t.get("net_pnl", float("nan")) for t in trade_log]
            worst_trade = min(w for w in worsts if not np.isnan(w)) if worsts else float("nan")

        # Time-in-position (open bars)
        open_bars = int((pair_df["position"] != 0).sum())

        pair_rows.append({
            "pair":        f"{ta}/{tb}",
            "n_trades":    n_trades,
            "open_bars":   open_bars,
            "gross_$":     gross_pnl,
            "net_$":       net_pnl,
            "comm_$":      cost_comm,
            "borrow_$":    cost_borrow,
            "reb_$":       cost_reb,
            "worst_trade": worst_trade,
        })

    df_pairs = pd.DataFrame(pair_rows).sort_values("net_$")

    print(f"{'Pair':<20} {'Trd':>4} {'Bars':>5} {'Gross$':>8} {'Net$':>8} "
          f"{'Comm$':>7} {'Borrow$':>7} {'WorstTrd$':>10}")
    print("-" * 78)
    for _, r in df_pairs.iterrows():
        flag = " <<" if r["net_$"] < -2000 else ""
        print(f"{r['pair']:<20} {int(r['n_trades']):>4} {int(r['open_bars']):>5} "
              f"{r['gross_$']:>8.0f} {r['net_$']:>8.0f} "
              f"{r['comm_$']:>7.0f} {r['borrow_$']:>7.0f} {r['worst_trade']:>10.0f}{flag}")

    print("-" * 78)
    total_net   = df_pairs["net_$"].sum()
    total_gross = df_pairs["gross_$"].sum()
    total_comm  = df_pairs["comm_$"].sum()
    total_bor   = df_pairs["borrow_$"].sum()
    total_reb   = df_pairs["reb_$"].sum()
    print(f"{'TOTAL':<20} {int(df_pairs['n_trades'].sum()):>4} {'':>5} "
          f"{total_gross:>8.0f} {total_net:>8.0f} "
          f"{total_comm:>7.0f} {total_bor:>7.0f}")

    print(f"\nCost decomposition:")
    print(f"  Gross PnL:         ${total_gross:>8.0f}")
    print(f"  Commission:       -${total_comm:>8.0f}  ({total_comm/PER_PAIR_DOLLAR*10000/max(df_pairs['n_trades'].sum(),1):.0f} bps avg/trade)")
    print(f"  Borrow cost:      -${total_bor:>8.0f}")
    print(f"  Rebalance cost:   -${total_reb:>8.0f}")
    print(f"  Net PnL:           ${total_net:>8.0f}")

    print(f"\nTop 5 losers:")
    for _, r in df_pairs.head(5).iterrows():
        print(f"  {r['pair']:<20}  net=${r['net_$']:>8.0f}  trades={int(r['n_trades'])}  "
              f"open_bars={int(r['open_bars'])}  worst_trade=${r['worst_trade']:>8.0f}")

    print(f"\nTop 5 winners:")
    for _, r in df_pairs.tail(5).iloc[::-1].iterrows():
        print(f"  {r['pair']:<20}  net=${r['net_$']:>8.0f}  trades={int(r['n_trades'])}  "
              f"open_bars={int(r['open_bars'])}")

    print()
