"""
Multi-pair Chunk 4 Diagnostics
Runs the full diagnostic pipeline for all 11 pairs from params_example.yaml,
saves 3 plots per pair, and prints a comparative summary table.

Run from: week2_signal_engine/
  python run_all_pairs_diagnostics.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.utils.config import load_config
from src.data.loaders import load_pair
from src.utils.dates import split_periods
from src.signals.spread import estimate_hedge_ratio, compute_spread
from src.analytics.characterize import compute_half_life, hurst_exponent, window_from_half_life
from src.signals.zscore import compute_zscore
from src.signals.state_machine import generate_positions, count_trades
from src.analytics.diagnostics import empirical_coverage, tail_behavior, quantile_crossval
from src.visuals.plots import plot_signal_validation, plot_rolling_hurst, plot_qq_normal

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "outputs", "figures", "all_pairs")

# Week 1 Engle-Granger scan results — pairs were selected from this universe scan.
# Provides the formal cointegration test statistics that the Week 2 engine relies on.
WEEK1_SCAN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Week 1",
    "outputs", "pair_scan_results", "full_pair_scan_results.parquet"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_pairs(cfg):
    """Flatten config pairs into list of (category, ta, tb)."""
    pairs = []
    p = cfg["pairs"]
    pairs.append(("primary",      p["primary"][0],   p["primary"][1]))
    pairs.append(("secondary",    p["secondary"][0],  p["secondary"][1]))
    for alt in p.get("alternatives", []):
        pairs.append(("alternative", alt[0], alt[1]))
    for bm in p.get("benchmarks", []):
        pairs.append(("benchmark",   bm[0],  bm[1]))
    for nc in p.get("negative_controls", []):
        pairs.append(("neg_control", nc[0],  nc[1]))
    return pairs


def rolling_hurst_series(spread, rolling_window=5000, step=500, max_lag=100):
    vals = spread.dropna()
    rows, timestamps = [], []
    for start in range(0, len(vals) - rolling_window, step):
        end = start + rolling_window
        H   = hurst_exponent(vals.iloc[start:end], max_lag=max_lag)
        rows.append({"hurst": H})
        timestamps.append(vals.index[min(end - 1, len(vals) - 1)])
    if not rows:
        return pd.DataFrame(columns=["hurst"])
    df_h = pd.DataFrame(rows)
    df_h.index = pd.DatetimeIndex(timestamps[:len(df_h)])
    return df_h


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_pair(ta, tb, dates, label):
    result = {"pair": f"{ta}/{tb}", "label": label, "status": "OK", "error": None}
    try:
        merged, _ = load_pair(DATA_DIR, ta, tb)
        formation, trading = split_periods(merged, **dates)

        fa, fb       = formation["close_a"], formation["close_b"]
        alpha, beta  = estimate_hedge_ratio(fa, fb)
        spread_f     = compute_spread(fa, fb, alpha, beta)
        hl, _        = compute_half_life(spread_f)
        H_form       = hurst_exponent(spread_f)
        window, _    = window_from_half_life(hl)

        zdf_f    = compute_zscore(spread_f, window=window)
        z_f      = zdf_f["zscore"].dropna()
        adaptive = round(float(z_f.abs().quantile(0.95)), 4)

        spread_t = compute_spread(trading["close_a"], trading["close_b"], alpha, beta)
        zdf_t    = compute_zscore(spread_t, window=window)
        z_t      = zdf_t["zscore"]

        pos_fixed = generate_positions(z_t, entry_z=2.0,      exit_z=0.0)
        pos_adapt = generate_positions(z_t, entry_z=adaptive,  exit_z=0.0)
        n_fixed   = count_trades(pos_fixed)
        n_adapt   = count_trades(pos_adapt)

        cov_df  = empirical_coverage(z_t)
        cov_2s  = float(cov_df.loc[cov_df["threshold"] == "+-2.0", "empirical_pct"].values[0])
        gap_2s  = float(cov_df.loc[cov_df["threshold"] == "+-2.0", "gap_pct"].values[0])
        tb_stat = tail_behavior(z_t)
        xval    = quantile_crossval(z_f, z_t)

        hurst_df = rolling_hurst_series(spread_t)
        pct_mr   = (hurst_df["hurst"] < 0.5).mean() * 100 if len(hurst_df) > 0 else float("nan")

        # --- Plots ---
        pair_fig_dir = os.path.join(FIG_DIR, f"{ta}_{tb}")
        os.makedirs(pair_fig_dir, exist_ok=True)

        df_pipe = trading[["close_a", "close_b"]].copy()
        df_pipe["spread"]       = spread_t
        df_pipe["rolling_mean"] = zdf_t["rolling_mean"]
        df_pipe["rolling_std"]  = zdf_t["rolling_std"]
        df_pipe["zscore"]       = z_t
        df_pipe["position"]     = pos_fixed

        plot_errors = []
        for plot_fn, kwargs, fname in [
            (plot_signal_validation,
             dict(df=df_pipe, ticker_a=ta, ticker_b=tb, beta=beta, window=window, entry_z=2.0),
             "signal_validation.png"),
            (plot_rolling_hurst,
             dict(hurst_df=hurst_df, ticker_a=ta, ticker_b=tb) if len(hurst_df) > 0 else None,
             "rolling_hurst.png"),
            (plot_qq_normal,
             dict(zscore_series=z_t, ticker_a=ta, ticker_b=tb),
             "qq_plot.png"),
        ]:
            if kwargs is None:
                continue
            try:
                kwargs["save_path"] = os.path.join(pair_fig_dir, fname)
                plot_fn(**kwargs)
            except Exception as e:
                plot_errors.append(f"{fname}: {e}")

        hl_display = round(hl, 1) if (not pd.isna(hl) and hl != float("inf")) else str(hl)
        H_display  = round(H_form, 3) if not pd.isna(H_form) else float("nan")

        result.update({
            "beta":           round(beta, 4),
            "half_life":      hl_display,
            "H_form":         H_display,
            "window":         window,
            "adaptive_z":     adaptive,
            "std":            tb_stat["std"],
            "excess_kurt":    tb_stat["excess_kurtosis"],
            "skewness":       tb_stat["skewness"],
            "cov_2s_pct":     round(cov_2s, 2),
            "gap_2s_pct":     round(gap_2s, 2),
            "regime_shift":   xval["regime_shift"],
            "drift":          xval["abs_drift"],
            "n_fixed":        n_fixed,
            "n_adapt":        n_adapt,
            "pct_mr":         round(pct_mr, 1),
            "n_hurst_wins":   len(hurst_df),
            "plot_errors":    plot_errors,
        })
        if plot_errors:
            result["status"] = "WARN"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"]  = str(e)
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg   = load_config(CONFIG_PATH)
    dates = cfg["dates"]
    pairs = get_all_pairs(cfg)

    # Load Week 1 Engle-Granger scan results (pairs were selected from this scan)
    week1_scan = None
    if os.path.exists(WEEK1_SCAN_PATH):
        week1_scan = pd.read_parquet(WEEK1_SCAN_PATH)[
            ["ticker_a", "ticker_b", "coint_tstat", "raw_pval"]
        ]
    else:
        print(f"  [WARN] Week 1 scan not found at: {WEEK1_SCAN_PATH}")

    print("\n" + "=" * 70)
    print(f"MULTI-PAIR DIAGNOSTICS  ({len(pairs)} pairs)")
    print("=" * 70)

    results = []
    for label, ta, tb in pairs:
        print(f"  [{label.upper():12s}]  {ta}/{tb} ...", end="", flush=True)
        r = run_pair(ta, tb, dates, label)

        # Attach Week 1 EG cointegration evidence
        if week1_scan is not None:
            row = week1_scan[
                (week1_scan["ticker_a"] == ta) & (week1_scan["ticker_b"] == tb)
            ]
            if not row.empty:
                r["eg_tstat"] = round(float(row["coint_tstat"].iloc[0]), 4)
                r["eg_pval"]  = round(float(row["raw_pval"].iloc[0]),    6)
            else:
                r["eg_tstat"] = float("nan")
                r["eg_pval"]  = float("nan")

        results.append(r)
        if r["status"] == "OK":
            print(" OK")
        elif r["status"] == "WARN":
            print(f" WARN  (plot issues: {len(r['plot_errors'])})")
        else:
            print(f" ERROR: {r['error']}")

    # ── Comparative table ──────────────────────────────────────────────────
    print("\n\n" + "=" * 130)
    print("COMPARATIVE SUMMARY TABLE")
    print("=" * 130)

    cols = [
        ("Pair",       "pair",         12),
        ("Category",   "label",        12),
        ("EG_tstat",   "eg_tstat",      9),
        ("EG_pval",    "eg_pval",       9),
        ("Beta",       "beta",          7),
        ("HL(bars)",   "half_life",     9),
        ("H_form",     "H_form",        7),
        ("Window",     "window",        7),
        ("Adap_Z",     "adaptive_z",    7),
        ("Std",        "std",           6),
        ("Kurt",       "excess_kurt",   7),
        ("Skew",       "skewness",      7),
        ("Cov2s%",     "cov_2s_pct",    8),
        ("Gap2s%",     "gap_2s_pct",    8),
        ("Drift",      "drift",         7),
        ("Regime?",    "regime_shift",  8),
        ("N_fix",      "n_fixed",       7),
        ("N_adp",      "n_adapt",       7),
        ("%MR_win",    "pct_mr",        8),
    ]

    header = "  ".join(f"{title:<{w}}" for title, _, w in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        if r["status"] == "ERROR":
            print(f"  {r['pair']:<12}  {r['label']:<12}  ERROR: {r['error']}")
            continue
        row = "  ".join(f"{str(r.get(k, 'N/A')):<{w}}" for _, k, w in cols)
        print(row)

    # ── Interpretation notes ───────────────────────────────────────────────
    ok = [r for r in results if r["status"] != "ERROR"]
    mr_pairs     = [r for r in ok if isinstance(r.get("H_form"), float) and r["H_form"] < 0.5]
    regime_pairs = [r for r in ok if r.get("regime_shift")]
    kurt_pairs   = [r for r in ok if r.get("excess_kurt", 0) > 3]
    fat_tail_pairs = [r for r in ok if r.get("gap_2s_pct", 0) > 5]
    high_trade   = [r for r in ok if r.get("n_fixed", 0) > 300]

    print("\n\n" + "=" * 70)
    print("INTERPRETATION NOTES")
    print("=" * 70)
    print(f"  Mean-reverting formation spread (H < 0.5)   : {len(mr_pairs)}/{len(ok)} pairs")
    for r in mr_pairs:
        print(f"    {r['pair']:<14}  H={r['H_form']}")

    print(f"\n  Regime shift in volatility (drift > 0.5)    : {len(regime_pairs)}/{len(ok)} pairs")
    for r in regime_pairs:
        print(f"    {r['pair']:<14}  drift={r['drift']}")

    print(f"\n  Genuine fat tails (excess kurtosis > 3)     : {len(kurt_pairs)}/{len(ok)} pairs")
    for r in kurt_pairs:
        print(f"    {r['pair']:<14}  kurt={r['excess_kurt']}")

    print(f"\n  Significant coverage gap at +-2s (gap > 5%) : {len(fat_tail_pairs)}/{len(ok)} pairs")
    for r in fat_tail_pairs:
        print(f"    {r['pair']:<14}  cov={r['cov_2s_pct']:.1f}%  gap={r['gap_2s_pct']:+.2f}%")

    print(f"\n  High-frequency pairs (> 300 trades, fixed)  : {len(high_trade)}/{len(ok)} pairs")
    for r in high_trade:
        print(f"    {r['pair']:<14}  n_fixed={r['n_fixed']}")

    errors = [r for r in results if r["status"] == "ERROR"]
    if errors:
        print(f"\n  FAILED ({len(errors)}):")
        for r in errors:
            print(f"    {r['pair']}: {r['error']}")
    else:
        print("\n  All pairs completed without errors.")

    print(f"\n  Figures saved under: {FIG_DIR}")
    print()


if __name__ == "__main__":
    main()
