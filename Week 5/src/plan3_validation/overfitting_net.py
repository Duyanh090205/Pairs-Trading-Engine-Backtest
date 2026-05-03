"""
4.7 — DSR + PBO on net returns under each cost regime.

Formulas reproduced from Week 4 phase4_defense/overfitting.py:
  DSR  - Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio
  PBO  - Combinatorial cross-validation on per-fold Sharpes
"""
from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_LOG = WEEK5_ROOT / "data" / "cost_log.parquet"
DEFAULT_TRADE_LOG = WEEK5_ROOT / "data" / "week4_inputs" / "trade_log.csv"


def _expected_max_sharpe(n_trials: int) -> float:
    if n_trials <= 1:
        return 0.0
    p = 1.0 - 1.0 / n_trials
    return float(scipy_stats.norm.ppf(p))


def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt_excess: float = 0.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    if n_obs < 5 or n_trials < 1 or np.isnan(sharpe_obs):
        return float("nan")
    e_max_sr = _expected_max_sharpe(n_trials)
    sr_var = (
        1.0
        + 0.5 * sharpe_obs ** 2
        - skew * sharpe_obs
        + (kurt_excess / 4.0) * sharpe_obs ** 2
    ) / max(n_obs - 1, 1)
    sr_std = np.sqrt(max(sr_var, 1e-12))
    return float(scipy_stats.norm.cdf(
        (sharpe_obs - max(e_max_sr, sharpe_benchmark)) / sr_std
    ))


def _daily_returns_for_regime(
    cost_log: pd.DataFrame, trade_log: pd.DataFrame, cost_col: str
) -> pd.Series:
    df = cost_log.merge(
        trade_log[["trade_id", "exit_ts", "fold_id", "allocated_capital"]],
        on="trade_id",
    )
    df["exit_date"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("US/Eastern").dt.date
    if cost_col == "total_cost_gross":
        df["net_pnl"] = df["gross_pnl_dollars"]
    elif cost_col == "total_cost_static":
        df["net_pnl"] = df["gross_pnl_dollars"] - df["total_cost_static"]
    else:
        df["net_pnl"] = df["gross_pnl_dollars"] - df["total_cost_dollars"]
    aum = trade_log["allocated_capital"].sum()
    if aum <= 0:
        return pd.Series(dtype=float)
    return df.groupby("exit_date")["net_pnl"].sum().sort_index() / aum


def _per_fold_sharpe(
    cost_log: pd.DataFrame, trade_log: pd.DataFrame, cost_col: str
) -> dict[int, float]:
    df = cost_log.merge(
        trade_log[["trade_id", "exit_ts", "fold_id", "allocated_capital"]],
        on="trade_id",
    )
    df["exit_date"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("US/Eastern").dt.date
    if cost_col == "total_cost_gross":
        df["net_pnl"] = df["gross_pnl_dollars"]
    elif cost_col == "total_cost_static":
        df["net_pnl"] = df["gross_pnl_dollars"] - df["total_cost_static"]
    else:
        df["net_pnl"] = df["gross_pnl_dollars"] - df["total_cost_dollars"]

    out: dict[int, float] = {}
    for fold_id, sub in df.groupby("fold_id"):
        aum_fold = sub["allocated_capital"].sum()
        if aum_fold <= 0:
            out[int(fold_id)] = float("nan")
            continue
        daily = sub.groupby("exit_date")["net_pnl"].sum() / aum_fold
        if len(daily) < 2 or daily.std(ddof=1) < 1e-12:
            out[int(fold_id)] = float("nan")
            continue
        out[int(fold_id)] = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
    return out


def _compute_pbo(per_fold_sharpe: dict[int, float], n_splits: int = 4) -> float:
    valid = {f: s for f, s in per_fold_sharpe.items() if not np.isnan(s)}
    folds = list(valid.keys())
    n = len(folds)
    if n < 4:
        return float("nan")
    k = max(2, n // n_splits)
    rng = random.Random(42)
    combos = list(itertools.combinations(range(n), k))
    if len(combos) > 5_000:
        combos = rng.sample(combos, 5_000)
    n_overfit = 0
    n_total = 0
    for is_combo in combos:
        is_set = set(is_combo)
        oos_idx = [i for i in range(n) if i not in is_set]
        if not oos_idx:
            continue
        # Iterate is_combo (ordered tuple) consistently; is_best_idx must align with it.
        is_sharpes = [valid[folds[i]] for i in is_combo]
        oos_sharpes = [valid[folds[i]] for i in oos_idx]
        is_best_idx = int(np.argmax(is_sharpes))
        is_best_fold = folds[is_combo[is_best_idx]]
        is_best_sharpe = valid[is_best_fold]
        oos_median = float(np.median(oos_sharpes))
        if is_best_sharpe < oos_median:
            n_overfit += 1
        n_total += 1
    return n_overfit / n_total if n_total else float("nan")


def compute_overfitting_diagnostics(
    cost_log_path: Path | str = DEFAULT_COST_LOG,
    trade_log_path: Path | str = DEFAULT_TRADE_LOG,
) -> pd.DataFrame:
    """
    Returns DataFrame with rows ['Raw Sharpe', 'DSR p-value', 'PBO']
    and columns ['Gross', 'Static', 'Dynamic'].
    """
    cost_log = pd.read_parquet(cost_log_path)
    trade_log = pd.read_csv(trade_log_path)

    regimes = {
        "Gross": "total_cost_gross",
        "Static": "total_cost_static",
        "Dynamic": "total_cost_dollars",
    }
    out: dict[str, dict[str, float]] = {}
    for label, col in regimes.items():
        daily = _daily_returns_for_regime(cost_log, trade_log, col).to_numpy()
        if len(daily) < 2:
            out[label] = {"Raw Sharpe": float("nan"), "DSR p-value": float("nan"), "PBO": float("nan")}
            continue
        sd = daily.std(ddof=1)
        if sd < 1e-12:
            sharpe = float("nan")
        else:
            sharpe = float(daily.mean() / sd * np.sqrt(252))
        skew = float(scipy_stats.skew(daily))
        kurt = float(scipy_stats.kurtosis(daily, fisher=True))
        per_fold = _per_fold_sharpe(cost_log, trade_log, col)
        n_trials = max(1, sum(1 for s in per_fold.values() if not np.isnan(s)))
        dsr = deflated_sharpe_ratio(
            sharpe_obs=sharpe, n_obs=len(daily), n_trials=n_trials,
            skew=skew, kurt_excess=kurt,
        )
        pbo = _compute_pbo(per_fold)
        out[label] = {
            "Raw Sharpe": round(sharpe, 4) if not np.isnan(sharpe) else float("nan"),
            "DSR p-value": round(dsr, 4) if not np.isnan(dsr) else float("nan"),
            "PBO": round(pbo, 4) if not np.isnan(pbo) else float("nan"),
        }
    return pd.DataFrame(out)
