"""
Distribution Diagnostics

Empirical coverage table, tail-behavior statistics, and quantile
cross-validation between formation and trading periods.
"""
import pandas as pd
from scipy.stats import norm


# Thresholds used throughout the research plan
_COVERAGE_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]


def empirical_coverage(zscore_series: pd.Series) -> pd.DataFrame:
    """Compare empirical Z-score coverage against normal-theory expectations.

    For each threshold t in [1.0, 1.5, 2.0, 2.5, 3.0], computes:
        - Normal theory: P(|Z| <= t) = 2*Phi(t) - 1
        - Empirical:     fraction of |z| <= t in the provided series
        - Gap:           theory - empirical (positive = fatter tails than normal)

    Args:
        zscore_series : Trading-period Z-scores (NaN will be dropped).

    Returns:
        DataFrame with columns [threshold, normal_pct, empirical_pct, gap_pct].

    !! FLAG — interpretation:
        A positive gap (normal > empirical) means more extreme values than a
        normal distribution predicts — fat tails.  For pairs-trading this is
        the key LTCM-relevant diagnostic: if ±2σ only covers 90% (vs 95.4%
        theory), the entry threshold fires more often than the model expects.
        Report the table as the centrepiece of the distribution analysis, not
        the normality p-values (which are near-guaranteed to reject on 48k bars).
    """
    z = zscore_series.dropna()
    rows = []
    for t in _COVERAGE_THRESHOLDS:
        theory_pct   = (2 * norm.cdf(t) - 1) * 100
        empirical_pct = (z.abs() <= t).mean() * 100
        rows.append({
            "threshold":     f"+-{t}",
            "normal_pct":    round(theory_pct,    2),
            "empirical_pct": round(empirical_pct, 2),
            "gap_pct":       round(theory_pct - empirical_pct, 2),
        })
    return pd.DataFrame(rows)


def tail_behavior(zscore_series: pd.Series) -> dict:
    """Compute tail-behaviour statistics for the Z-score distribution.

    Returns a dict with:
        n_obs        : Number of non-NaN observations
        mean         : Should be near 0 for a well-constructed spread
        std          : Should be near 1 (by construction, but rolling windows
                       introduce deviations)
        skewness     : Asymmetry; large magnitude signals directional drift
        excess_kurtosis : Kurtosis - 3; positive = fatter tails than normal
        pct_beyond_2 : Fraction of |z| > 2.0  (entry zone)
        pct_beyond_3 : Fraction of |z| > 3.0  (extreme events)
        q_025        : 2.5th percentile (lower tail)
        q_975        : 97.5th percentile (upper tail)
        abs_q95      : 95th percentile of |z|  (adaptive threshold basis)

    !! FLAG — excess_kurtosis on minute-bar data:
        Financial returns on 1-min data routinely show excess kurtosis > 5.
        High kurtosis does NOT mean the signal is broken — it reflects genuine
        fat-tail microstructure.  Report the number and connect it to the LTCM
        reading rather than treating it as a bug.
    """
    z = zscore_series.dropna()
    return {
        "n_obs":           int(len(z)),
        "mean":            round(float(z.mean()),     6),
        "std":             round(float(z.std(ddof=1)), 4),
        "skewness":        round(float(z.skew()),     4),
        "excess_kurtosis": round(float(z.kurtosis()), 4),   # pandas kurtosis is excess
        "pct_beyond_2":    round(float((z.abs() > 2.0).mean() * 100), 3),
        "pct_beyond_3":    round(float((z.abs() > 3.0).mean() * 100), 3),
        "q_025":           round(float(z.quantile(0.025)), 4),
        "q_975":           round(float(z.quantile(0.975)), 4),
        "abs_q95":         round(float(z.abs().quantile(0.95)), 4),
    }


def threshold_sensitivity_table(
    zscore: pd.Series,
    thresholds: list,
    cost_bps_rt: float = 60.0,
) -> pd.DataFrame:
    """Compare entry thresholds on a fixed Z-score series.

    For each entry_z in `thresholds`, runs the path-dependent state machine and
    reports trade count, average hold duration, and cost break-even.

    Args:
        zscore       : Z-score series (NaN rows hold position — same as state machine).
        thresholds   : List of entry_z values to compare, e.g. [1.5, 2.0, 2.5, 2.57].
        cost_bps_rt  : Cost per complete round-trip in basis points.
                       Default 60 bps = 4 one-way legs × 15 bps each.
                       A complete pairs trade involves 4 transactions:
                           open leg A, open leg B, close leg A, close leg B.

    Returns:
        DataFrame with columns:
            entry_z            : The threshold tested
            n_trades           : Number of round-trips entered
            avg_hold_bars      : Mean bars held per trade
            total_cost_bps     : n_trades × cost_bps_rt (total cost of running this threshold)
            breakeven_per_trade: Basis points of PnL needed per trade to cover costs (= cost_bps_rt)

    !! FLAG — interpretation:
        This table does NOT compute actual PnL (that is Week 3).  It shows the
        mechanical cost structure: how many trades, how long held, how much
        total cost.  The breakeven column is constant (= cost_bps_rt) because
        cost is per trade, not per bar.  The signal for choosing a threshold is:
        lower z fires more trades (higher total cost, lower average signal quality);
        higher z fires fewer trades (lower cost, only the clearest divergences).
    """
    from src.signals.state_machine import generate_positions, count_trades
    rows = []
    for z in thresholds:
        pos  = generate_positions(zscore, entry_z=float(z), exit_z=0.0)
        n    = count_trades(pos)
        hold = round((pos != 0).sum() / max(n, 1), 1)
        rows.append({
            "entry_z":             round(float(z), 4),
            "n_trades":            n,
            "avg_hold_bars":       hold,
            "total_cost_bps":      round(n * cost_bps_rt),
            "breakeven_per_trade": cost_bps_rt,
        })
    return pd.DataFrame(rows)


def quantile_crossval(
    z_formation: pd.Series,
    z_trading: pd.Series,
) -> dict:
    """Cross-validate the formation-derived threshold against the trading period.

    Computes the abs(Z) 95th percentile in both periods and flags any
    meaningful shift.  A large shift indicates the spread's volatility regime
    changed between formation and trading — the formation-derived adaptive
    threshold may be mis-calibrated.

    Returns:
        formation_q95  : abs(Z) 95th pct on formation period
        trading_q95    : abs(Z) 95th pct on trading period
        abs_drift      : |formation_q95 - trading_q95|
        regime_shift   : True if abs_drift > 0.5 (research plan threshold)

    !! FLAG — what a shift means:
        A formation_q95 of 2.56 and trading_q95 of 2.80 means the spread was
        more volatile in the trading period than expected from formation.
        The adaptive threshold of 2.56 (set from formation) will fire MORE
        trades in the trading period than intended.  Not a correctness error,
        but it must be stated in the Signal Logic Document.
    """
    q_f = float(z_formation.dropna().abs().quantile(0.95))
    q_t = float(z_trading.dropna().abs().quantile(0.95))
    drift = abs(q_f - q_t)
    return {
        "formation_q95": round(q_f, 4),
        "trading_q95":   round(q_t, 4),
        "abs_drift":     round(drift, 4),
        "regime_shift":  drift > 0.5,
    }
