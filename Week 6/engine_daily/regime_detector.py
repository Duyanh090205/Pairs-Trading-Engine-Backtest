"""
Regime detector — composite z-score halt filter (production module)
====================================================================

Computes a market-regime stress indicator from daily ticker returns and
exposes a per-fold halt decision used by `scripts/run_v4_pipeline.py` to
skip trading months where the regime indicator exceeds its trailing 252d
top tertile.

Pre-committed config (locked 2026-05-25 per HMM Regime Detection research,
`documents/HMM Regime Detection.md`):

    stress_z(t) = z(vol_60d)(t) - z(corr_60d)(t) + z(dispersion_20d)(t)
        where each feature is z-scored on trailing 252d (expanding window
        until 252d available, then strict rolling 252d).

    halt fold N iff:
        stress_z(t*) > q_67(stress_z over trailing 252d before t*)
        where t* = last trading day strictly before fold N's trading month.

All computations are FORWARD-LOOKING (only data up to and including t* is
used to decide trading on month containing t*+1 onwards). No production
look-ahead. See /deep-audit-bug verification 2026-05-25.

Public API:
    build_features(cache_daily)       -> pd.DataFrame
    compute_stress_zscore(feats)      -> pd.DataFrame (adds stress_z column)
    halt_for_fold(feats, trade_start) -> (bool, dict[diag])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Locked config — DO NOT tune without /deep-audit-bug review.
VOL_WINDOW = 60               # rolling 60d annualized vol of EW index
CORR_WINDOW = 60              # rolling 60d avg pairwise correlation
DISPERSION_WINDOW = 20        # 20d-avg of cross-sectional std
ZSCORE_WINDOW = 252           # trailing window for z-score normalization
MIN_BURNIN = 126              # expanding window until this many obs
THRESHOLD_QUANTILE = 0.67     # top tertile of trailing stress_z (halt for SIMPLE binary)
THRESHOLD_QUANTILE_EXTREME = 0.85  # top 15% — extreme stress (full halt in dampener mode)
CORR_SAMPLE_SIZE = 80         # number of tickers sampled for pairwise corr
                              # (full 528 is too slow at daily resolution;
                              # 80 alphabetical-first-complete is a stable proxy)

# Size dampener config (locked 2026-05-25 per HMM Regime Detection research §5.2):
#   stress_z < q67_trailing : multiplier = 1.0  (full size)
#   q67 <= stress_z < q85   : multiplier = 0.5  (half size — moderate stress)
#   stress_z >= q85         : multiplier = 0.0  (full halt — extreme stress)
SIZE_MULT_MODERATE = 0.5      # size scaling in moderate stress regime
SIZE_MULT_EXTREME = 0.0       # size scaling at extreme stress (= halt)


def build_features(cache_daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build daily regime features from the ticker daily cache.

    Output columns: vol_60d, corr_60d, dispersion_20d.
    Index: trading dates (tz-naive).
    """
    if not cache_daily:
        raise ValueError("cache_daily is empty")

    returns_dict = {tk: df["log_close"].diff() for tk, df in cache_daily.items()}
    R = pd.DataFrame(returns_dict).sort_index()

    # EW index returns
    eq_returns = R.mean(axis=1)
    vol_60d = eq_returns.rolling(VOL_WINDOW, min_periods=20).std() * np.sqrt(252)

    # Cross-sectional dispersion
    cs_std = R.std(axis=1)
    dispersion_20d = cs_std.rolling(DISPERSION_WINDOW, min_periods=5).mean()

    # Pairwise correlation on a sampled subset (full 528 too slow)
    R_clean = R.dropna(axis=1, thresh=int(0.95 * len(R)))
    sample_tickers = sorted(R_clean.columns.tolist())[:CORR_SAMPLE_SIZE]
    R_sample = R_clean[sample_tickers]

    corr_values: list[float] = []
    dates = R_sample.index
    for i in range(len(dates)):
        if i < CORR_WINDOW:
            corr_values.append(np.nan)
            continue
        window = R_sample.iloc[i - CORR_WINDOW:i].dropna(axis=1, thresh=50)
        if window.shape[1] < 10:
            corr_values.append(np.nan)
            continue
        C = window.corr().values
        n = C.shape[0]
        off_diag = C[np.triu_indices(n, k=1)]
        corr_values.append(float(np.nanmean(off_diag)))
    corr_60d = pd.Series(corr_values, index=dates)

    feats = pd.DataFrame({
        "vol_60d": vol_60d,
        "corr_60d": corr_60d,
        "dispersion_20d": dispersion_20d,
    })
    log.info("regime features built: %d days, %d/%d valid corr",
             len(feats), int(feats["corr_60d"].notna().sum()), len(feats))
    return feats


def _rolling_zscore_expanding(series: pd.Series) -> pd.Series:
    """Z-score with expanding window until MIN_BURNIN, then rolling ZSCORE_WINDOW.

    Strict forward-looking: z at index i uses values strictly before i
    (slice `:i` is exclusive of i).
    """
    z = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i < MIN_BURNIN:
            past = series.iloc[:i].dropna()
        else:
            past = series.iloc[max(0, i - ZSCORE_WINDOW):i].dropna()
        if len(past) < 20:
            z.iloc[i] = np.nan
            continue
        mu = past.mean()
        sigma = past.std(ddof=1)
        if sigma <= 1e-12:
            z.iloc[i] = 0.0
            continue
        z.iloc[i] = (series.iloc[i] - mu) / sigma
    return z


def compute_stress_zscore(feats: pd.DataFrame) -> pd.DataFrame:
    """Add z-scored feature columns + composite stress_z column to feats.

    Composite: stress_z = z_vol - z_corr + z_dispersion
        (high vol AND/OR low corr AND/OR high dispersion -> stress)
    """
    feats = feats.copy()
    feats["z_vol"] = _rolling_zscore_expanding(feats["vol_60d"])
    feats["z_corr"] = _rolling_zscore_expanding(feats["corr_60d"])
    feats["z_disp"] = _rolling_zscore_expanding(feats["dispersion_20d"])
    feats["stress_z"] = feats["z_vol"] - feats["z_corr"] + feats["z_disp"]
    return feats


def halt_for_fold(
    feats: pd.DataFrame,
    trade_start: pd.Timestamp,
) -> tuple[bool, dict]:
    """Decide whether to halt the trading month starting on `trade_start`.

    Forward-looking: uses ONLY data strictly before `trade_start`.

    Parameters
    ----------
    feats : output of compute_stress_zscore (must have `stress_z` column).
    trade_start : tz-naive Timestamp of first day of fold's trading month.

    Returns
    -------
    halt : True if regime indicator says skip this month.
    diag : dict with stress_z value, threshold, decision day, etc.
    """
    decision_dates = feats.index[feats.index < trade_start]
    if len(decision_dates) == 0:
        return False, {"reason": "no_data_before_trade_start"}
    t_star = decision_dates[-1]

    past_end = feats.index.get_loc(t_star)
    past_start = max(0, past_end - ZSCORE_WINDOW)
    past_window = feats.iloc[past_start:past_end]   # excludes t_star

    sz = feats.loc[t_star, "stress_z"]
    past_sz = past_window["stress_z"].dropna()
    if len(past_sz) < 30 or pd.isna(sz):
        return False, {
            "reason": "burnin_insufficient_history",
            "decision_day": str(t_star.date()),
            "stress_z": float(sz) if not pd.isna(sz) else None,
            "n_past": int(len(past_sz)),
        }

    q67 = float(past_sz.quantile(THRESHOLD_QUANTILE))
    halt = bool(sz > q67)
    return halt, {
        "decision_day": str(t_star.date()),
        "stress_z": float(sz),
        "q67_trailing": q67,
        "vol_60d": float(feats.loc[t_star, "vol_60d"]),
        "corr_60d": float(feats.loc[t_star, "corr_60d"]) if not pd.isna(feats.loc[t_star, "corr_60d"]) else None,
        "halt": halt,
    }


def size_multiplier_for_fold(
    feats: pd.DataFrame,
    trade_start: pd.Timestamp,
) -> tuple[float, dict]:
    """Decide position-size multiplier for the trading month starting at `trade_start`.

    3-step size dampener (per HMM Regime Detection research):
        stress_z < q67_trailing      -> mult = 1.0  (full size)
        q67 <= stress_z < q85        -> mult = 0.5  (half size)
        stress_z >= q85              -> mult = 0.0  (full halt)

    Forward-looking: uses ONLY data strictly before `trade_start`.

    Returns
    -------
    mult : multiplier in {0.0, 0.5, 1.0} applied to per-pair notional.
    diag : dict with stress_z value, both thresholds, regime label.
    """
    decision_dates = feats.index[feats.index < trade_start]
    if len(decision_dates) == 0:
        return 1.0, {"reason": "no_data_before_trade_start", "regime": "calm"}
    t_star = decision_dates[-1]

    past_end = feats.index.get_loc(t_star)
    past_start = max(0, past_end - ZSCORE_WINDOW)
    past_window = feats.iloc[past_start:past_end]

    sz = feats.loc[t_star, "stress_z"]
    past_sz = past_window["stress_z"].dropna()
    if len(past_sz) < 30 or pd.isna(sz):
        return 1.0, {
            "reason": "burnin_insufficient_history",
            "decision_day": str(t_star.date()),
            "regime": "calm",
        }

    q67 = float(past_sz.quantile(THRESHOLD_QUANTILE))
    q85 = float(past_sz.quantile(THRESHOLD_QUANTILE_EXTREME))

    if sz >= q85:
        mult = SIZE_MULT_EXTREME
        regime = "extreme_stress"
    elif sz >= q67:
        mult = SIZE_MULT_MODERATE
        regime = "moderate_stress"
    else:
        mult = 1.0
        regime = "calm"

    return mult, {
        "decision_day": str(t_star.date()),
        "stress_z": float(sz),
        "q67_trailing": q67,
        "q85_trailing": q85,
        "vol_60d": float(feats.loc[t_star, "vol_60d"]),
        "regime": regime,
        "multiplier": mult,
    }
