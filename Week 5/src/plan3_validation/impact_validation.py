"""
4.4 — Impact prediction validation (DIAGNOSTIC ONLY).

Cross-sectional OLS: future_cost_proxy ~ spread_std_1d.
  future_cost_proxy = full_spread_l2_bps - full_spread_l1_bps  (depth decay).

Reports beta, R-squared, p-value. NO pass/fail gate per workflow.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPREADS = WEEK5_ROOT / "data" / "microstructure" / "spreads_1min.parquet"
DEFAULT_ROLLING = WEEK5_ROOT / "data" / "microstructure" / "spread_rolling.parquet"


def _ols_with_pvalue(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS y = a + b*x. Returns alpha, beta, r2, p_value (two-sided t-test on beta)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3 or np.std(x) < 1e-12:
        return {"alpha": float("nan"), "beta": float("nan"), "r2": float("nan"),
                "p_value": float("nan"), "n": n}
    x_mean, y_mean = x.mean(), y.mean()
    sx2 = ((x - x_mean) ** 2).sum()
    sxy = ((x - x_mean) * (y - y_mean)).sum()
    beta = sxy / sx2
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se_beta = np.sqrt(ss_res / (n - 2) / sx2) if n > 2 else float("nan")
    if not pd.isna(se_beta) and se_beta > 0:
        from scipy import stats
        t_stat = beta / se_beta
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    else:
        p = float("nan")
    return {"alpha": float(alpha), "beta": float(beta), "r2": float(r2),
            "p_value": float(p), "n": int(n)}


def validate_impact_prediction(
    spreads_path: Path | str = DEFAULT_SPREADS,
    rolling_path: Path | str = DEFAULT_ROLLING,
    max_rows: int = 2_000_000,
) -> dict:
    """
    Returns dict with keys: beta, r2, p_value, n, alpha, note.
    """
    needed_cols = ["timestamp_et", "ticker", "full_spread_l1_bps",
                   "full_spread_l2_bps", "spread_std_1d"]
    # Filter to traded tickers to avoid loading 195M rows
    kappa_path = Path(spreads_path).parent.parent / "kappa_per_fold.parquet"
    if kappa_path.exists():
        traded_tickers = pd.read_parquet(kappa_path)["ticker"].unique().tolist()
        ticker_filter = [("ticker", "in", traded_tickers)]
    else:
        ticker_filter = None
    try:
        spreads = pd.read_parquet(spreads_path, columns=needed_cols, filters=ticker_filter)
    except Exception:
        spreads = pd.read_parquet(spreads_path, filters=ticker_filter)
    if len(spreads) > max_rows:
        spreads = spreads.sample(n=max_rows, random_state=42)
    if "full_spread_l2_bps" not in spreads.columns:
        return {
            "beta": float("nan"), "r2": float("nan"), "p_value": float("nan"),
            "n": 0, "alpha": float("nan"),
            "note": "L2 spread column not present in microstructure data; "
                    "diagnostic skipped (synthetic data only has L1).",
        }

    if "spread_std_1d" not in spreads.columns:
        rolling = pd.read_parquet(rolling_path) if Path(rolling_path).exists() else None
        if rolling is not None and "spread_std_1d" in rolling.columns:
            spreads = spreads.merge(
                rolling[["timestamp_et", "ticker", "spread_std_1d"]],
                on=["timestamp_et", "ticker"], how="left",
            )
        else:
            spreads["spread_std_1d"] = float("nan")
    merged = spreads

    merged = merged.dropna(subset=["full_spread_l2_bps", "full_spread_l1_bps", "spread_std_1d"])
    proxy = (merged["full_spread_l2_bps"] - merged["full_spread_l1_bps"]).to_numpy()
    sigma = merged["spread_std_1d"].to_numpy()

    result = _ols_with_pvalue(sigma, proxy)
    result["note"] = "Diagnostic only. No pass/fail gate per workflow."
    return result
