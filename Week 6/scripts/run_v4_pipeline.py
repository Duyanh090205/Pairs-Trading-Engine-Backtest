"""
run_v4_pipeline.py — V4 daily entry point
==========================================

V4 vs V3.0:
    Daily bars (close-to-close, resampled from 5-min) for both formation and
    trading. 12-month formation + 1-month trading walk-forward (~39 folds).
    Single static-spread engine with rolling-60-day Z, alpha refit at trading
    window start (beta from formation), entry |Z|>=2.0, hard SL |Z|>=4.0.
    No A1 / A3 / EOS / vol-target intraday hacks.

CLI examples:
    python run_v4_pipeline.py --folds 1 --smoke
    python run_v4_pipeline.py --folds all
    python run_v4_pipeline.py --folds 1-5 --entry-z 2.0 --hard-sl-z 4.0
"""

from __future__ import annotations

import os as _os

# Single-thread BLAS (determinism + Johansen LAPACK noise — same rationale as V3)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_var] = "1"

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import numpy as np
import pandas as pd

WEEK6_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6_ROOT))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from engine_daily import discovery_daily
from engine_daily.engine_daily import run_fold_daily
from engine_daily.metrics_daily import aggregate_fold_metrics
from engine_daily.carry_forward import (
    GATE_PVAL_THRESHOLD,
    apply_gate,
    extract_open_at_eom,
    refund_eom_exit_cost,
)
from engine_daily.regime_detector import (
    build_features as regime_build_features,
    compute_stress_zscore,
    halt_for_fold,
    size_multiplier_for_fold,
)
from engine.phase1_cointegration.factor_residual import project_residual

# ---- Constants ----
VALIDATED_DIR = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated")
DATA_DAILY = VALIDATED_DIR / "daily_phase3"   # pre-built daily parquets
                                              # (see build_daily_parquets.py)

TOTAL_CAPITAL = 1_000_000.0
TC_BPS = 30.0
BORROW_BPS_YR = 50.0


# ============================================================
# Fold schedule — 39 folds, 12m formation, 1m trading
# ============================================================

def _build_fold_schedule_v4() -> list[tuple[int, str, str, str]]:
    """39 folds covering trading months Jan-2023 through Mar-2026.

    Each row is (fold_n, formation_start, formation_end, trading_month).
    Formation = the 12 months ending the day before `trading_month`.
    """
    schedule: list[tuple[int, str, str, str]] = []
    # Trading months: 2023-01 .. 2026-03 inclusive
    for fold_n, (year, month) in enumerate(
        [(y, m) for y in (2023, 2024, 2025) for m in range(1, 13)]
        + [(2026, 1), (2026, 2), (2026, 3)],
        start=1,
    ):
        trading_month = f"{year:04d}-{month:02d}"
        # Formation = 12 months before trading_month, ending the last day before it.
        trade_start = pd.Timestamp(f"{trading_month}-01")
        form_end = (trade_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        form_start = (trade_start - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        schedule.append((fold_n, form_start, form_end, trading_month))
    return schedule


FOLD_SCHEDULE = _build_fold_schedule_v4()


# ============================================================
# Data cache helpers
# ============================================================

def _load_all_daily(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load pre-built daily parquets (one per ticker) into memory.

    Each file is the output of `build_daily_parquets.py` and contains daily
    log_close + volume columns indexed by tz-naive Timestamp (midnight).
    """
    cache: dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data dir not found: {data_dir}\n"
            "Run `python build_daily_parquets.py` to generate it from 5-min source."
        )
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".parquet"):
            continue
        tk = fname[:-len(".parquet")]
        try:
            df = pd.read_parquet(data_dir / fname)
            cache[tk] = df
        except Exception:
            pass
    return cache


def _slice_daily(
    cache_daily: dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Slice each ticker's daily DataFrame to [start, end] (inclusive)."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out: dict[str, pd.DataFrame] = {}
    for tk in sorted(cache_daily.keys()):
        df = cache_daily[tk]
        sliced = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ============================================================
# Per-fold runner — discovery only (engine wires Day 2)
# ============================================================

def run_fold_v4(
    fold_n: int,
    formation_start: str,
    formation_end: str,
    trading_month: str,
    cache_daily: dict,
    out_dir: Path,
    entry_z: float = 2.0,
    z_window: int = 60,
    hard_sl_z: float = 4.0,
    hl_max: float = 30.0,
    hl_min: float = 5.0,
    cost_data=None,    # engine_daily.cost_engine.CostData | None
    carry_state_in: dict | None = None,   # E2: carry-forward state from prior fold
    fold_size_mult: float = 1.0,          # regime size dampener multiplier
) -> dict | None:
    """One V4 fold. Reads pre-built daily parquets (no 5-min resampling)."""
    t0 = time.time()
    log = lambda msg: print(f"  Fold {fold_n:02d} [{trading_month}]: {msg}", flush=True)

    # ---- 1. Slice formation window from pre-built daily cache ----
    log(f"slicing daily cache for formation {formation_start} .. {formation_end}")
    formation_daily = _slice_daily(cache_daily, formation_start, formation_end)
    if not formation_daily:
        log("no formation daily data -- skip")
        return None
    log(f"daily formation: {len(formation_daily)} tickers")

    # ---- 2. Discovery ----
    try:
        pairs_df, factor_state = discovery_daily.run(
            formation_data=formation_daily,
            hl_min=hl_min, hl_max=hl_max,
        )
    except Exception as e:
        log(f"discovery FAILED -- {e}")
        traceback.print_exc()
        return None

    if pairs_df.empty or not factor_state:
        log("0 pairs after discovery -- skip")
        return None

    log(f"discovery: {len(pairs_df)} pairs after beta>0 | "
        f"cum_var={factor_state['diagnostics']['cumulative_variance_explained']:.3f}")

    # ---- 3. Slice trading-month daily data ----
    trade_start = trading_month + "-01"
    trade_end = (pd.Timestamp(trade_start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    log(f"slicing daily cache for trading {trade_start} .. {trade_end}")
    trading_daily_raw = _slice_daily(cache_daily, trade_start, trade_end)
    if not trading_daily_raw:
        log("no trading daily data -- skip")
        return None
    log(f"daily trading: {len(trading_daily_raw)} tickers")

    # Project trading daily data through formation loadings W. Path A re-anchor:
    # shift each ticker's trading residual to continue from its formation last value.
    loadings_W = factor_state["loadings_W"]
    factor_tickers = factor_state["tickers"]
    resid_form_df = factor_state["residual_log_prices"]  # (T_form, n_tickers)

    # min_obs=10 because daily trading window has ~19-23 bars (Feb is shortest);
    # V3's default of 20 is for 5-min and doesn't apply here.
    resid_trade_dict = project_residual(
        trading_daily_raw, loadings_W, factor_tickers, min_obs=10,
    )
    # Path A re-anchor
    for tk in list(resid_trade_dict.keys()):
        if tk not in resid_form_df.columns:
            continue
        s_form_tk = resid_form_df[tk].dropna()
        if len(s_form_tk) == 0 or len(resid_trade_dict[tk]) == 0:
            continue
        shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
        resid_trade_dict[tk] = resid_trade_dict[tk] + shift

    resid_trade_df = pd.concat(resid_trade_dict, axis=1)
    log(f"trading residual: {resid_trade_df.shape[0]} bars x {resid_trade_df.shape[1]} tickers")

    # ---- 4. Engine ----
    # E2: carry-forward pairs from prior fold are injected by run_fold_daily.
    # They may need columns from resid_form / resid_trade that don't exist if
    # the ticker dropped from this fold's universe -- run_fold_daily handles
    # the skip cleanly (the pair won't be in results -> orchestrator treats
    # it as a "cut" and the prior fold's EOM exit cost stays).
    pair_results = run_fold_daily(
        pairs_df=pairs_df,
        resid_form=resid_form_df,
        resid_trade=resid_trade_df,
        alpha_lookback=60,
        entry_z=entry_z,
        z_window=z_window,
        hard_sl_z=hard_sl_z,
        cost_data=cost_data,
        carry_state_in=carry_state_in,
        current_fold_n=fold_n,
        fold_size_mult=fold_size_mult,
    )

    # ---- 5. Metrics ----
    fold_metrics = aggregate_fold_metrics(
        pair_results, total_capital=TOTAL_CAPITAL,
    )

    elapsed = time.time() - t0
    log(f"complete in {elapsed:.1f}s | n_trades={fold_metrics['n_trades']}, "
        f"sharpe={fold_metrics['sharpe']:+.3f}, "
        f"total_return={fold_metrics['total_return']:+.4f}, "
        f"exits z/sl/open={fold_metrics['exit_breakdown']['zero_cross']}/"
        f"{fold_metrics['exit_breakdown']['hard_sl']}/"
        f"{fold_metrics['exit_breakdown']['open_at_eom']}")

    return {
        "fold": fold_n,
        "trading_month": trading_month,
        "n_pairs": int(len(pairs_df)),
        "cum_var": float(factor_state["diagnostics"]["cumulative_variance_explained"]),
        "n_factor_tickers": len(factor_state["tickers"]),
        "elapsed_s": float(elapsed),
        "pairs_df": pairs_df,
        "factor_state": factor_state,
        "fold_metrics": fold_metrics,
        "pair_results": pair_results,
    }


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="all",
                        help="'all' or range '1-5' or comma-separated '1,2,3'")
    parser.add_argument("--out-dir", default=str(WEEK6_ROOT / "results" / "v4"))
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: write to results/v4/smoke/fold_NN/")
    parser.add_argument("--entry-z", type=float, default=2.0,
                        help="|Z| entry threshold (default 2.0 daily AL2010)")
    parser.add_argument("--z-window", type=int, default=60,
                        help="Rolling Z window in daily bars (default 60)")
    parser.add_argument("--hard-sl-z", type=float, default=4.0,
                        help="|Z| hard stop-loss threshold (default 4.0 daily)")
    parser.add_argument("--hl-max", type=float, default=30.0,
                        help="Max half-life days for cointegrated pairs (default 30)")
    parser.add_argument("--hl-min", type=float, default=5.0,
                        help="Min half-life days for cointegrated pairs (default 5)")
    parser.add_argument("--use-dynamic-cost", action="store_true",
                        help="Use Week 5 dynamic cost engine (per-ticker per-day "
                             "spread + market impact). Default OFF = flat 30 bps + "
                             "50 bps/yr borrow (legacy V4 baseline).")
    parser.add_argument("--no-dynamic-cost", dest="use_dynamic_cost",
                        action="store_false",
                        help="Explicitly use flat-cost baseline.")
    parser.set_defaults(use_dynamic_cost=False)
    parser.add_argument("--use-carry-forward", action="store_true",
                        help="E2: allow positions open at EOM to survive into "
                             "the next fold if pair passes Johansen p<0.05 on "
                             "next-fold's formation. Default OFF (V4 force-close).")
    parser.add_argument("--use-composite-filter", action="store_true",
                        help="Regime filter: skip fold if composite stress z-score "
                             "(vol + corr + dispersion) exceeds top tertile of "
                             "trailing 252d. Forward-looking. Default OFF.")
    parser.add_argument("--use-composite-sizer", action="store_true",
                        help="Regime size dampener: scale position size 100%/50%/0% "
                             "by composite stress z-score (q67/q85 trailing 252d). "
                             "Mutually exclusive with --use-composite-filter. "
                             "Forward-looking. Default OFF.")
    args = parser.parse_args()
    if args.use_composite_filter and args.use_composite_sizer:
        parser.error("--use-composite-filter and --use-composite-sizer are mutually exclusive")

    # Parse folds
    if args.folds == "all":
        fold_nums = [s[0] for s in FOLD_SCHEDULE]
    elif "-" in args.folds:
        lo, hi = args.folds.split("-")
        fold_nums = list(range(int(lo), int(hi) + 1))
    else:
        fold_nums = [int(x) for x in args.folds.split(",")]
    selected = [s for s in FOLD_SCHEDULE if s[0] in fold_nums]

    out_dir = Path(args.out_dir)
    if args.smoke and len(selected) == 1:
        out_dir = out_dir / "smoke" / f"fold_{selected[0][0]:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_mode = "DYNAMIC (Week 5)" if args.use_dynamic_cost else "FLAT 30bps + 50bps/yr (V4 baseline)"
    carry_mode = f"ON (gate p<{GATE_PVAL_THRESHOLD:.2f})" if args.use_carry_forward else "OFF (V4 force-close at EOM)"
    if args.use_composite_filter:
        regime_mode = "FILTER (binary halt if stress_z > q67 trailing 252d)"
    elif args.use_composite_sizer:
        regime_mode = "SIZER (100%/50%/0% by stress_z vs q67/q85 trailing)"
    else:
        regime_mode = "OFF"
    print(f"V4 daily pipeline | entry_z={args.entry_z}, z_window={args.z_window}, "
          f"hard_sl_z={args.hard_sl_z}, hl=[{args.hl_min}, {args.hl_max}]")
    print(f"Cost model: {cost_mode}")
    print(f"Carry-forward (E2): {carry_mode}")
    print(f"Regime mode (composite z-score): {regime_mode}")
    print(f"Folds: {len(selected)} | out_dir={out_dir}")
    print(f"Loading daily cache from {DATA_DAILY.name}/ ...")
    t_load = time.time()
    cache_daily = _load_all_daily(DATA_DAILY)
    print(f"  loaded {len(cache_daily)} tickers in {time.time()-t_load:.1f}s")

    cost_data = None
    if args.use_dynamic_cost:
        from engine_daily.cost_engine import load_cost_data
        print("Loading Week 5 spread cache + kappa map...")
        t_cd = time.time()
        cost_data = load_cost_data()
        print(f"  loaded {len(cost_data.daily_spread)} daily spread rows, "
              f"{len(cost_data.kappa_map)} tickers in kappa map "
              f"({time.time()-t_cd:.1f}s)\n")
    else:
        print()

    # Build a quick lookup so we know which fold is "next" for the gate test.
    sched_by_fold = {s[0]: s for s in FOLD_SCHEDULE}

    # Pre-compute regime features ONCE (used for halt or size-mult decisions).
    # Computed on full ticker cache; downstream calls slice forward-looking
    # at each fold's decision moment.
    regime_feats = None
    if args.use_composite_filter or args.use_composite_sizer:
        print("Building composite regime features (vol_60d + corr_60d + dispersion_20d)...")
        t_rf = time.time()
        regime_feats = regime_build_features(cache_daily)
        regime_feats = compute_stress_zscore(regime_feats)
        valid = regime_feats["stress_z"].dropna()
        print(f"  features built in {time.time()-t_rf:.1f}s; "
              f"valid stress_z from {valid.index[0].date() if len(valid) else 'n/a'}\n",
              flush=True)

    fold_rows = []
    carry_state: dict = {}             # E2: persists across folds
    selected_ids = [s[0] for s in selected]

    for fold_n, fs, fe, tm in selected:
        # ---- Regime filter (BINARY halt mode) ----
        if args.use_composite_filter and regime_feats is not None:
            trade_start_ts = pd.Timestamp(tm + "-01")
            halt, diag = halt_for_fold(regime_feats, trade_start_ts)
            if halt:
                print(f"  Fold {fold_n:02d} [{tm}]: REGIME HALT "
                      f"(stress_z={diag.get('stress_z', 'nan'):+.3f} > "
                      f"q67={diag.get('q67_trailing', 'nan'):+.3f}); skip\n",
                      flush=True)
                fold_rows.append({
                    "fold": fold_n, "trading_month": tm,
                    "n_pairs": 0, "n_trades": 0, "sharpe": 0.0,
                    "total_return": 0.0, "max_dd": 0.0,
                    "win_rate": 0.0, "avg_net_bps": 0.0,
                    "n_zero_cross": 0, "n_hard_sl": 0, "n_open_at_eom": 0,
                    "cum_var": 0.0, "n_factor_tickers": 0, "elapsed_s": 0.0,
                    "n_carry_in": 0, "n_carry_out": 0, "n_gate_fail": 0,
                    "regime_halt": True, "size_mult": 0.0,
                    "stress_z": diag.get("stress_z"),
                    "q67_trailing": diag.get("q67_trailing"),
                })
                continue

        # ---- Regime sizer (CONTINUOUS multiplier mode) ----
        fold_mult = 1.0
        regime_diag = {}
        if args.use_composite_sizer and regime_feats is not None:
            trade_start_ts = pd.Timestamp(tm + "-01")
            fold_mult, regime_diag = size_multiplier_for_fold(regime_feats, trade_start_ts)
            if fold_mult == 0.0:
                print(f"  Fold {fold_n:02d} [{tm}]: REGIME SIZER halt "
                      f"(stress_z={regime_diag.get('stress_z', 'nan'):+.3f} >= "
                      f"q85={regime_diag.get('q85_trailing', 'nan'):+.3f}); skip\n",
                      flush=True)
                fold_rows.append({
                    "fold": fold_n, "trading_month": tm,
                    "n_pairs": 0, "n_trades": 0, "sharpe": 0.0,
                    "total_return": 0.0, "max_dd": 0.0,
                    "win_rate": 0.0, "avg_net_bps": 0.0,
                    "n_zero_cross": 0, "n_hard_sl": 0, "n_open_at_eom": 0,
                    "cum_var": 0.0, "n_factor_tickers": 0, "elapsed_s": 0.0,
                    "n_carry_in": 0, "n_carry_out": 0, "n_gate_fail": 0,
                    "regime_halt": True, "size_mult": 0.0,
                    "stress_z": regime_diag.get("stress_z"),
                    "q67_trailing": regime_diag.get("q67_trailing"),
                    "q85_trailing": regime_diag.get("q85_trailing"),
                })
                continue
            elif fold_mult < 1.0:
                print(f"  Fold {fold_n:02d} [{tm}]: REGIME SIZER half-size "
                      f"(stress_z={regime_diag.get('stress_z', 'nan'):+.3f}, "
                      f"q67={regime_diag.get('q67_trailing', 'nan'):+.3f} <= sz < "
                      f"q85={regime_diag.get('q85_trailing', 'nan'):+.3f}); "
                      f"mult={fold_mult:.2f}", flush=True)

        n_carry_in = len(carry_state)
        res = run_fold_v4(
            fold_n=fold_n, formation_start=fs, formation_end=fe,
            trading_month=tm, cache_daily=cache_daily, out_dir=out_dir,
            entry_z=args.entry_z, z_window=args.z_window, hard_sl_z=args.hard_sl_z,
            hl_max=args.hl_max, hl_min=args.hl_min,
            cost_data=cost_data,
            carry_state_in=(carry_state if args.use_carry_forward else None),
            fold_size_mult=fold_mult,
        )
        if res is None:
            # Pair list was empty -> nothing to carry. Reset state if any
            # tickers were missing from this fold's universe.
            if args.use_carry_forward and carry_state:
                print(f"  Fold {fold_n:02d}: empty discovery -> dropping {len(carry_state)} carry positions", flush=True)
                carry_state = {}
            fold_rows.append({"fold": fold_n, "trading_month": tm,
                              "n_pairs": 0, "n_trades": 0, "sharpe": 0.0,
                              "total_return": 0.0, "cum_var": 0.0, "elapsed_s": 0.0,
                              "n_carry_in": n_carry_in, "n_carry_out": 0,
                              "n_gate_fail": n_carry_in})
            continue

        pair_results = res["pair_results"]
        pairs_df_used = res["pairs_df"]

        # ---- E2 gate test for next fold (only if --use-carry-forward) ----
        n_carry_out = 0
        n_gate_fail = 0
        if args.use_carry_forward:
            # Build beta lookup combining discovery + injected carry pairs.
            # BUG #1 FIX (2026-05-25): for any pair currently in carry_state,
            # ALWAYS override the discovery beta with carry.beta_original.
            # Reason: a carry pair may also be re-discovered by fold N's
            # own discovery (with a slightly different beta estimate due to
            # the 1-month-shifted formation). The frozen-beta invariant
            # requires the hedge ratio to stay at the value used when the
            # position was opened — anything else silently drifts hedging.
            beta_lookup_dict = {
                (r["ticker_a"], r["ticker_b"]): float(r["beta_pca"])
                for _, r in pairs_df_used.iterrows()
            }
            for key, state in carry_state.items():
                beta_lookup_dict[key] = float(state.beta_original)  # always override
            beta_lookup_df = pd.DataFrame([
                {"ticker_a": k[0], "ticker_b": k[1], "beta_pca": v}
                for k, v in beta_lookup_dict.items()
            ])
            candidates = extract_open_at_eom(pair_results, beta_lookup_df, fold_n)

            # Look up the NEXT fold for the gate's formation window
            next_fold_n = fold_n + 1
            next_sched = sched_by_fold.get(next_fold_n)
            if next_sched is None:
                # Last fold globally — nothing to carry into. Keep V4 force-close.
                print(f"  Fold {fold_n:02d}: last fold; carry-forward not applicable", flush=True)
                carry_state = {}
            else:
                from engine_daily.carry_forward import compute_residuals_for_gate
                _, fs_next, fe_next, _ = next_sched
                next_formation = _slice_daily(cache_daily, fs_next, fe_next)
                next_resid_df, _ = compute_residuals_for_gate(next_formation)

                passed, pvals = apply_gate(candidates, next_resid_df,
                                           gate_threshold=GATE_PVAL_THRESHOLD)
                n_carry_out = len(passed)
                n_gate_fail = len(candidates) - n_carry_out

                # Refund EOM exit cost for pairs that pass (so engine's force-close
                # cost is undone — true exit cost will be booked when position
                # actually closes in a later fold).
                if passed:
                    n_refunded = refund_eom_exit_cost(pair_results, passed)
                    print(f"  Fold {fold_n:02d}: carry-forward: "
                          f"{len(candidates)} candidates -> {n_carry_out} pass "
                          f"({n_gate_fail} cut); {n_refunded} EOM cost refunds applied",
                          flush=True)
                else:
                    print(f"  Fold {fold_n:02d}: carry-forward: "
                          f"{len(candidates)} candidates -> 0 pass "
                          f"({n_gate_fail} cut)", flush=True)

                carry_state = passed

        # ---- Recompute metrics AFTER any refund (so cum_pnl reflects credit) ----
        fold_metrics = aggregate_fold_metrics(pair_results, total_capital=TOTAL_CAPITAL)
        fm = fold_metrics
        eb = fm.get("exit_breakdown", {})
        fold_rows.append({
            "fold": res["fold"],
            "trading_month": res["trading_month"],
            "n_pairs": res["n_pairs"],
            "n_trades": fm.get("n_trades", 0),
            "sharpe": fm.get("sharpe", 0.0),
            "total_return": fm.get("total_return", 0.0),
            "max_dd": fm.get("max_dd", 0.0),
            "win_rate": fm.get("win_rate", 0.0),
            "avg_net_bps": fm.get("avg_net_bps", 0.0),
            "n_zero_cross": eb.get("zero_cross", 0),
            "n_hard_sl": eb.get("hard_sl", 0),
            "n_open_at_eom": eb.get("open_at_eom", 0),
            "cum_var": res["cum_var"],
            "n_factor_tickers": res["n_factor_tickers"],
            "elapsed_s": res["elapsed_s"],
            "n_carry_in": n_carry_in,
            "n_carry_out": n_carry_out,
            "n_gate_fail": n_gate_fail,
        })

    df = pd.DataFrame(fold_rows)
    out_path = out_dir / "fold_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")
    summary_cols = ["fold", "trading_month", "n_pairs", "n_trades", "sharpe",
                    "total_return", "avg_net_bps", "n_zero_cross", "n_hard_sl"]
    print(df[summary_cols].to_string(index=False))

    if len(df) > 1 and "sharpe" in df.columns:
        traded = df[df["n_trades"] > 0]
        if len(traded) > 0:
            mret = traded["total_return"]
            # Monthly-return annualized Sharpe — standard backtest figure
            if mret.std(ddof=1) > 1e-12:
                monthly_sh_ann = float(mret.mean() / mret.std(ddof=1) * np.sqrt(12))
            else:
                monthly_sh_ann = 0.0
            print(f"\n--- Aggregate across {len(traded)} traded folds ---")
            print(f"  Mean per-fold Sharpe   : {traded['sharpe'].mean():+.3f}")
            print(f"  Median per-fold Sharpe : {traded['sharpe'].median():+.3f}")
            print(f"  Monthly Sharpe (ann.)  : {monthly_sh_ann:+.3f}   <-- standard backtest figure")
            print(f"  Sum total return       : {mret.sum():+.4f}  ({100*mret.sum():.2f}%)")
            print(f"  Winning folds          : {(traded['sharpe']>0).sum()} / {len(traded)}")

            # Per-year regime check (descriptive, NOT for parameter selection)
            try:
                traded = traded.copy()
                traded["year"] = pd.to_datetime(traded["trading_month"]).dt.year
                yearly = traded.groupby("year").agg(
                    n_folds=("fold", "count"),
                    mean_sharpe=("sharpe", "mean"),
                    sum_return=("total_return", "sum"),
                )
                print(f"\n--- Per-year regime check (not used for parameter selection) ---")
                print(yearly.to_string(float_format=lambda x: f"{x:+.3f}"))
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
