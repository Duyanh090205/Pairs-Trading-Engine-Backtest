"""Offline discovery for live trading — pick pair list + factor model.

Reuses engine_daily.discovery_daily.run (the SAME function backtest uses).
Wraps it with live-specific I/O:
  - Reads live/state/daily_cache (Alpaca split-adjusted, NOT Polygon raw)
  - Slices last 252 trading days as formation
  - Saves pair list + factor_state to disk for live engine to load

Output:
  live/state/discovered_pairs.parquet — DataFrame from discovery
  live/state/factor_state.pkl         — factor loadings W, residual_log_prices,
                                          tickers list, diagnostics

Run once at start of paper run (offline). Re-run monthly if paper extends.

Usage:
  python scripts/run_live_discovery.py
  python scripts/run_live_discovery.py --formation-end 2026-04-30
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from engine_daily import discovery_daily
from engine_daily.portfolio import apply_ticker_concentration_cap

CACHE_DIR = ROOT / "live" / "state" / "daily_cache"
PAIRS_OUT = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_STATE_OUT = ROOT / "live" / "state" / "factor_state.pkl"
META_OUT = ROOT / "live" / "state" / "discovery_meta.json"

FORMATION_DAYS = 252


def load_cache_window(cache_dir: Path, formation_end: date,
                       formation_days: int = FORMATION_DAYS) -> dict[str, pd.DataFrame]:
    """Load formation-window slice for every ticker in cache."""
    cutoff = pd.Timestamp(formation_end)
    out: dict[str, pd.DataFrame] = {}
    for fp in sorted(cache_dir.iterdir()):
        if fp.suffix != ".parquet":
            continue
        df = pd.read_parquet(fp)
        sliced = df[df.index <= cutoff].tail(formation_days)
        if len(sliced) >= 100:    # discovery_daily.run requires >= 100 obs
            out[fp.stem] = sliced
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formation-end", default=None,
                        help="YYYY-MM-DD inclusive; default = today")
    parser.add_argument("--hl-min", type=float, default=5.0)
    parser.add_argument("--hl-max", type=float, default=30.0)
    parser.add_argument("--apply-ticker-cap", action="store_true",
                        help="Apply backtest's per-ticker concentration cap")
    parser.add_argument("--filter-tradable", default=None,
                        help="Path to JSON with 'tickers' list — keep only pairs where "
                             "BOTH legs are in this list (e.g. live/universe_top50.json)")
    parser.add_argument("--split-sample-gate", type=float, default=None,
                        help="Hedge against Johansen overfitting: for each surviving pair, "
                             "re-test cointegration on first vs second half of formation; "
                             "drop pair if either half p >= this threshold. "
                             "Recommended: 0.10 (loose). 0.05 = strict.")
    parser.add_argument("--no-ticker-reuse", action="store_true",
                        help="Enforce disjoint ticker sets across pairs: each ticker may "
                             "appear in at most ONE pair. Greedy selection by johansen_pval "
                             "(lowest p first). Eliminates Alpaca wash-trade rejections that "
                             "occur when the same symbol gets opposing orders from two pairs.")
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        print(f"ERROR: cache dir not found at {CACHE_DIR}\n"
              f"Run scripts/build_live_daily_cache.py first.", file=sys.stderr)
        return 1

    end = date.today() if args.formation_end is None else date.fromisoformat(args.formation_end)
    print(f"== Live discovery ==")
    print(f"  Cache:          {CACHE_DIR}")
    print(f"  Formation end:  {end}")
    print(f"  Formation days: {FORMATION_DAYS} (will use up to last {FORMATION_DAYS} bars per ticker)")
    print(f"  HL filter:      [{args.hl_min}, {args.hl_max}] days")
    print(f"  Ticker cap:     {args.apply_ticker_cap}")
    print()

    t0 = time.time()
    cache = load_cache_window(CACHE_DIR, end)
    print(f"Loaded {len(cache)} tickers (>=100 bars in formation window)")
    if len(cache) < 20:
        print(f"ERROR: only {len(cache)} tickers loaded — need at least 20 for PCA",
              file=sys.stderr)
        return 1

    # Diagnostic: actual formation date range
    sample = next(iter(cache.values()))
    print(f"  formation: {sample.index.min().date()} -> {sample.index.max().date()} "
          f"({len(sample)} bars)")

    print(f"\nRunning discovery_daily.run() — same function as V4 backtest...")
    t_disc = time.time()
    try:
        pairs_df, factor_state = discovery_daily.run(
            formation_data=cache,
            hl_min=args.hl_min,
            hl_max=args.hl_max,
        )
    except Exception as e:
        print(f"ERROR: discovery failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"Discovery elapsed: {time.time() - t_disc:.1f}s")

    if pairs_df.empty:
        print("WARNING: 0 pairs survived discovery filters", file=sys.stderr)
        return 1

    # Optional: split-sample stability gate (hedges Johansen overfitting).
    # IMPORTANT: must use RESIDUAL log-prices (post-PCA), not raw log-prices,
    # to match the input used by the original discovery Johansen test.
    if args.split_sample_gate is not None:
        from engine.phase1_cointegration.discovery import _johansen_pvalue
        import numpy as _np
        threshold = args.split_sample_gate
        before = len(pairs_df)
        resid_form = factor_state["residual_log_prices"]
        keep_mask = []
        for _, row in pairs_df.iterrows():
            ta, tb = row["ticker_a"], row["ticker_b"]
            if ta not in resid_form.columns or tb not in resid_form.columns:
                keep_mask.append(False)
                continue
            ra = resid_form[ta].dropna()
            rb = resid_form[tb].dropna()
            aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(aligned) < 200:
                keep_mask.append(False)
                continue
            half = len(aligned) // 2
            try:
                X1 = _np.column_stack([aligned.iloc[:half, 0], aligned.iloc[:half, 1]])
                X2 = _np.column_stack([aligned.iloc[half:, 0], aligned.iloc[half:, 1]])
                p1 = _johansen_pvalue(X1)
                p2 = _johansen_pvalue(X2)
            except Exception:
                p1, p2 = float("nan"), float("nan")
            keep_mask.append((p1 < threshold) and (p2 < threshold))
        pairs_df = pairs_df[keep_mask].copy()
        n_kept = len(pairs_df)
        print(f"Split-sample gate on RESIDUALS (p_thresh={threshold}): {before} -> {n_kept} pairs")

    # Optional: filter to tradable universe (both legs must be in the list)
    if args.filter_tradable:
        import json as _json
        tradable = set(_json.loads(Path(args.filter_tradable).read_text())["tickers"])
        before = len(pairs_df)
        mask = pairs_df["ticker_a"].isin(tradable) & pairs_df["ticker_b"].isin(tradable)
        pairs_df = pairs_df[mask].copy()
        print(f"Tradable filter ({len(tradable)} tickers): {before} -> {len(pairs_df)} pairs")

    # Optional: apply backtest's ticker concentration cap (max N pairs per ticker)
    if args.apply_ticker_cap:
        before = len(pairs_df)
        pairs_df = apply_ticker_concentration_cap(pairs_df)
        print(f"Ticker cap: {before} -> {len(pairs_df)} pairs")

    # Optional: no-ticker-reuse — each ticker appears in at most one pair.
    # Greedy by johansen_pval ascending (strongest cointegration kept).
    # Prevents Alpaca wash-trade rejections when same symbol gets opposing
    # orders from two pairs.
    if args.no_ticker_reuse:
        before = len(pairs_df)
        ordered = pairs_df.sort_values("johansen_pval", ascending=True)
        used: set[str] = set()
        keep_idx = []
        for idx, row in ordered.iterrows():
            ta, tb = row["ticker_a"], row["ticker_b"]
            if ta in used or tb in used:
                continue
            keep_idx.append(idx)
            used.add(ta)
            used.add(tb)
        pairs_df = pairs_df.loc[keep_idx].copy()
        print(f"No-ticker-reuse: {before} -> {len(pairs_df)} pairs "
              f"(dropped {before - len(pairs_df)} overlapping)")

    # Persist
    PAIRS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_parquet(PAIRS_OUT)
    with FACTOR_STATE_OUT.open("wb") as f:
        pickle.dump(factor_state, f)

    import json
    meta = {
        "decided_ts": datetime.now(timezone.utc).isoformat(),
        "formation_end": end.isoformat(),
        "formation_bars_per_ticker": int(len(sample)),
        "n_universe_tickers": len(cache),
        "tradable_universe_file": args.filter_tradable,
        "n_factor_tickers": len(factor_state["tickers"]),
        "n_factor_components": int(factor_state.get("n_components",
                                  factor_state.get("diagnostics", {}).get("n_components", "?"))),
        "cumulative_variance_explained": float(
            factor_state["diagnostics"]["cumulative_variance_explained"]
        ),
        "n_pairs": int(len(pairs_df)),
        "hl_min": args.hl_min,
        "hl_max": args.hl_max,
        "ticker_cap_applied": args.apply_ticker_cap,
        "split_sample_gate_p_thresh": args.split_sample_gate,
        "no_ticker_reuse": args.no_ticker_reuse,
        "total_elapsed_s": float(time.time() - t0),
    }
    META_OUT.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved:")
    print(f"  Pair list:    {PAIRS_OUT}  ({len(pairs_df)} pairs)")
    print(f"  Factor state: {FACTOR_STATE_OUT}")
    print(f"  Metadata:     {META_OUT}")
    print(f"\nTop 5 pairs by Johansen p-value:")
    if "johansen_pval" in pairs_df.columns:
        top5 = pairs_df.nsmallest(5, "johansen_pval")[
            ["ticker_a", "ticker_b", "beta_pca", "johansen_pval"]
        ]
        print(top5.to_string())
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
