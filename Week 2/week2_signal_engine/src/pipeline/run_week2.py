"""
Week 2 Pipeline Orchestration

Entry point that stitches together data loading, spread characterisation,
Z-score computation, and position generation for the primary and secondary
pairs defined in the config file.

Usage (from week2_signal_engine/):
    python src/pipeline/run_week2.py
    python src/pipeline/run_week2.py configs/params_example.yaml

Outputs:
    outputs/signals/signals_{A}_{B}.csv  — full trading-period signal DataFrame

!! FLAG — run_pipeline vs main():
    run_pipeline() is a pure computation function. It accepts pre-fitted
    parameters (alpha, beta, window) estimated on the formation period and
    applies them to the supplied price series.  It never touches formation
    data and never reads from disk.  Keep it that way — Week 3 PnL code
    will call run_pipeline() directly in tests.

!! FLAG — burn-in NaNs in output:
    The first (window // 2) rows of the returned DataFrame will have NaN
    z-scores.  Positions during this period are 0 (flat) because the state
    machine holds its current state on NaN — which starts at 0.  Do NOT
    drop these rows before saving: Week 3 needs the full aligned index.
    Use the `signal_valid` column to gate PnL computation rather than
    relying on implicit .dropna() calls that can silently mis-align indices.
"""
import os
import sys

# Allow running directly: python src/pipeline/run_week2.py
_PIPELINE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_PIPELINE_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.signals.spread import compute_spread
from src.signals.zscore import compute_zscore, apply_session_warmup
from src.signals.state_machine import generate_positions, count_trades


# ---------------------------------------------------------------------------
# Core computation — pure function, no I/O
# ---------------------------------------------------------------------------

def run_pipeline(
    close_a,
    close_b,
    alpha: float | pd.Series,
    beta:  float | pd.Series,
    window: int,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    session_warmup: int = 0,
) -> pd.DataFrame:
    """Execute spread -> Z-score -> position pipeline on a price series.

    All parameters must be pre-estimated on the formation period. This
    function operates on the trading period only and never looks back at
    formation data.

    Args:
        close_a / close_b : pd.Series of closing prices (same DatetimeIndex).
        alpha             : Intercept — scalar (static OLS) or pd.Series (Kalman).
        beta              : Hedge ratio — scalar (static OLS) or pd.Series (Kalman).
                            Series must be indexed identically to close_a/close_b.
        window            : Rolling window in bars derived from formation half-life.
        entry_z           : Z-score threshold to enter a position (abs value).
        exit_z            : Z-score threshold to exit (0.0 = zero-crossing exit).
        session_warmup    : Bars to suppress Z-score after each session open.
                            0 = disabled.  30 = first 30 minutes flat each day.

    Returns:
        pd.DataFrame with columns:
            spread             : Log-price spread S(t) = log(A) - alpha - beta*log(B)
            rolling_mean       : Rolling mean of spread (window bars, ddof=1 std)
            rolling_std        : Rolling std of spread
            zscore             : (spread - rolling_mean) / rolling_std
            position           : int8 {-1, 0, +1} from the path-dependent state machine
            signal_valid       : bool — False during rolling burn-in (NaN z-score rows),
                                 True once the window has enough history.  Week 3 must
                                 gate all PnL computation on this column instead of using
                                 implicit .dropna() calls that can silently mis-align indices.
            position_executed  : int8 — position.shift(1), the position that was actually
                                 held at bar t (decided at bar t-1 close).  Week 3 PnL:
                                     return[t] = position_executed[t] * pct_change[t]
                                 Using position[t] instead would introduce 1-bar lookahead
                                 on every trade on 1-minute data.
    """
    spread = compute_spread(close_a, close_b, alpha, beta)
    zdf    = compute_zscore(spread, window=window)
    if session_warmup > 0:
        zdf = apply_session_warmup(zdf, session_warmup)
    pos    = generate_positions(zdf["zscore"], entry_z=entry_z, exit_z=exit_z)

    # signal_valid: True once the rolling window has enough history
    signal_valid = ~zdf["zscore"].isna()

    # position_executed: shift by 1 bar — the position held *during* bar t
    # was decided at bar t-1.  fillna(0) sets bar 0 to flat (no prior bar).
    position_executed = pos.shift(1).fillna(0).astype("int8")

    out = zdf.copy()
    out.insert(0, "spread", spread)
    out["position"]          = pos
    out["signal_valid"]      = signal_valid
    out["position_executed"] = position_executed
    return out[["spread", "rolling_mean", "rolling_std", "zscore",
                "position", "signal_valid", "position_executed"]]


# ---------------------------------------------------------------------------
# Orchestration — handles I/O, config, and pair loop
# ---------------------------------------------------------------------------

def _default_config_path():
    return os.path.join(_PROJECT_ROOT, "configs", "params_example.yaml")


def main(config_path: str | None = None) -> None:
    """Load config, run the full pipeline for primary and secondary pairs,
    save signal CSVs, and print a summary.

    Args:
        config_path : Path to YAML config.  Defaults to configs/params_example.yaml
                      (resolved relative to the project root).
    """
    from src.utils.config import load_config
    from src.data.loaders import load_pair
    from src.utils.dates import split_periods
    from src.signals.spread import estimate_hedge_ratio
    from src.analytics.characterize import compute_half_life, window_from_half_life
    from src.analytics.diagnostics import threshold_sensitivity_table

    cfg_path = config_path or _default_config_path()
    cfg      = load_config(cfg_path)
    dates    = cfg["dates"]
    engine   = cfg.get("engine", {})
    entry_z      = engine.get("entry_z_fixed", 2.0)
    exit_z       = engine.get("exit_z_fixed",  0.0)
    use_kalman   = bool(engine.get("use_kalman", False))
    kalman_delta = float(engine.get("kalman_delta", 1e-5))
    session_warmup = int(engine.get("session_open_warmup", 0))

    data_dir    = os.path.join(_PROJECT_ROOT, "data", "raw")
    signals_dir = os.path.join(_PROJECT_ROOT, "outputs", "signals")
    os.makedirs(signals_dir, exist_ok=True)

    pairs_to_run = [
        ("primary",   cfg["pairs"]["primary"]),
        ("secondary", cfg["pairs"]["secondary"]),
    ]

    mode_label = "KALMAN" if use_kalman else "STATIC OLS"
    print("\n" + "=" * 60)
    print(f"WEEK 2 SIGNAL ENGINE  [{mode_label}]")
    print("=" * 60)

    for category, (ta, tb) in pairs_to_run:
        print(f"\n--- {category.upper()}: {ta}/{tb} ---")

        merged, audit = load_pair(data_dir, ta, tb)
        formation, trading = split_periods(merged, **dates)

        # Static OLS — always estimated (used as reference even in Kalman mode)
        alpha_ols, beta_ols = estimate_hedge_ratio(
            formation["close_a"], formation["close_b"]
        )

        if use_kalman:
            # Run filter on FULL year — formation warms it up, trading continues
            from src.signals.kalman import kalman_hedge_ratio
            alpha_full, beta_full, kf_diag = kalman_hedge_ratio(
                merged["close_a"], merged["close_b"], delta=kalman_delta
            )
            # Slice trading period — no re-initialization at the boundary
            alpha_trade  = alpha_full.loc[trading.index]
            beta_trade   = beta_full.loc[trading.index]
            # Window estimation: use OLS formation spread, NOT the Kalman spread.
            # The Kalman posterior spread at bar t uses theta_t (state after
            # observing bar t) — this makes it approximately the scaled Kalman
            # innovation, which is white noise by construction (HL≈0.6 bars).
            # Feeding that into window_from_half_life() produces window=10 (min),
            # which generates ~1,300 trades/period at 10.7/day — not pairs trading.
            # Option A: anchor the Kalman window to the OLS formation half-life so
            # both methods operate at the same mean-reversion timescale.
            spread_f = compute_spread(
                formation["close_a"], formation["close_b"], alpha_ols, beta_ols
            )
            csv_suffix = "_kalman"
            print(f"  [KALMAN]  delta={kalman_delta}  "
                  f"final_beta={kf_diag['final_beta']:.6f}  "
                  f"(OLS beta={beta_ols:.6f}, "
                  f"drift={kf_diag['final_beta'] - beta_ols:+.6f})")
        else:
            alpha_trade = alpha_ols
            beta_trade  = beta_ols
            spread_f    = compute_spread(
                formation["close_a"], formation["close_b"], alpha_ols, beta_ols
            )
            csv_suffix = ""

        hl, _        = compute_half_life(spread_f)
        window, note = window_from_half_life(hl)

        # Formation adaptive threshold (95th pct of abs Z)
        zdf_f      = compute_zscore(spread_f, window=window)
        adaptive_z = round(float(zdf_f["zscore"].dropna().abs().quantile(0.95)), 4)

        # Trading-period signals (uses formation-derived parameters, no leakage)
        result_df = run_pipeline(
            trading["close_a"], trading["close_b"],
            alpha=alpha_trade, beta=beta_trade, window=window,
            entry_z=entry_z, exit_z=exit_z,
            session_warmup=session_warmup,
        )

        # Save signal CSV
        out_path = os.path.join(signals_dir, f"signals_{ta}_{tb}{csv_suffix}.csv")
        result_df.to_csv(out_path)

        # Summary
        n_trades     = count_trades(result_df["position"])
        trading_days = result_df.index.normalize().nunique()
        tpd          = n_trades / trading_days if trading_days else 0.0
        z_std        = round(float(result_df["zscore"].dropna().std(ddof=1)), 4)
        # Window-truncation correction: rolling std underestimates true variance
        # when window << half-life.  The true 2-sigma event in this Z-score
        # distribution sits at z = 2.0 * z_std.  Compare to adaptive_z.
        true_2sigma  = round(2.0 * z_std, 4)

        print(f"  alpha (OLS) = {alpha_ols:.6f}")
        print(f"  beta  (OLS) = {beta_ols:.6f}")
        print(f"  half-life   = {hl:.1f} bars  ({note})")
        print(f"  window      = {window} bars")
        print(f"  adaptive_z  = {adaptive_z}  (fixed entry_z = {entry_z})")
        print(f"  z_std       = {z_std}  (ideal=1.0; window-truncation inflates to {z_std})")
        print(f"  true_2sigma = {true_2sigma}  (2.0 * z_std — actual 2-sigma bar in this distribution)")
        print(f"  trades      = {n_trades}  over {trading_days} days  ({tpd:.2f}/day)")
        print(f"  signal CSV  -> {out_path}")

        # Threshold sensitivity table (formation Z-score, cost = 60 bps/round-trip)
        tst = threshold_sensitivity_table(
            zdf_f["zscore"].dropna(),
            thresholds=[1.5, 2.0, 2.5, adaptive_z],
        )
        print(f"\n  Threshold sensitivity (formation period, 60 bps/round-trip cost):")
        print(f"  {'entry_z':>8}  {'n_trades':>9}  {'avg_hold':>9}  {'total_cost_bps':>14}  {'breakeven_bps':>13}")
        for _, row in tst.iterrows():
            marker = " <-- fixed" if row["entry_z"] == entry_z else (
                     " <-- adaptive" if row["entry_z"] == adaptive_z else "")
            print(f"  {row['entry_z']:>8.4f}  {row['n_trades']:>9}  "
                  f"{row['avg_hold_bars']:>9.1f}  {row['total_cost_bps']:>14.0f}  "
                  f"{row['breakeven_per_trade']:>13.1f}{marker}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print()


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path=cfg_arg)
