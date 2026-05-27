"""
Deep audit: WHY does V4 daily produce negative Sharpe? (10 tests)

Runs the winning combo (entry_z=2.5, hl_max=30) on all 39 folds, captures
per-trade + per-fold data, and answers 10 questions in a flow-driven order:

  STEP 1 (signal edge — answers "does the signal work?"):
    T1. Gross PnL per trade — any edge before costs?
    T2. Cost drag — what fraction of gross do costs eat?
    T7. Hit rate — % trades profitable (target: 50-60% per stat-arb lit)

  STEP 2 (only meaningful if T1 > 0 — "can we fix execution?"):
    T3. MFE vs MAE — do trades get favorable excursions?
    T4. Exit-type P&L — which exit reasons make money?
    T6. Time-to-revert — actual vs theoretical reversion time

  STEP 3 (only meaningful if T1 ≤ 0 — "why no signal?"):
    T8. Pair persistence — are cointegrated pairs stable across folds?
    T9. Post-hoc Johansen — do pairs stay cointegrated during trading?
    T10. Factor leak — does the strategy carry residual market beta?

  STEP 4 (CONTEXT ONLY — do NOT use to pick params, regime cherry-picking):
    T5. Regime split — when (which folds) does it work?

Outputs:
    results/v4/audit_trades.csv      per-trade rows across folds
    results/v4/audit_folds.csv       per-fold portfolio summary
    results/v4/audit_pair_history.csv pair-occurrence-per-fold
    results/v4/audit_findings.md      formatted summary

Notes:
- T9 (post-hoc Johansen) uses ~20 daily bars; chi2(8) p-values are noisy at
  this length. Treated as directional rather than precise.
- T10 uses S&P 500 equal-weight average return as market-factor proxy. Detects
  market-beta leak only (not Fama-French size/value/momentum — we don't have
  market caps or fundamentals on hand).
"""

from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_v] = "1"

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))
sys.path.insert(0, str(WEEK6 / "scripts"))   # for `from run_v4_pipeline import ...`
logging.basicConfig(level=logging.WARNING)

from run_v4_pipeline import (
    _load_all_daily, _slice_daily, FOLD_SCHEDULE,
    DATA_DAILY, TOTAL_CAPITAL,
)
from engine.phase1_cointegration.factor_residual import project_residual
from engine_daily import discovery_daily
from engine_daily.engine_daily import run_fold_daily

# Winning combo from grid sweep
ENTRY_Z = 2.5
HL_MAX  = 30.0
Z_WIN   = 60
HARD_SL = 4.0

OUT_DIR = WEEK6 / "results" / "v4"


# ---------------------------------------------------------------------------
# Per-trade extraction
# ---------------------------------------------------------------------------

def _extract_trades(pair_df: pd.DataFrame, ta: str, tb: str, fold_n: int) -> list[dict]:
    """Extract per-trade rows. Each trade row has gross/net pnl, costs,
    MFE, MAE, duration, exit_reason, entry Z."""
    pos = pair_df["position"].values
    spread = pair_df["spread"].values
    zscore = pair_df["zscore"].values
    pnl_gross = pair_df["daily_pnl_gross"].values
    pnl_net = pair_df["daily_pnl_net"].values
    cost_e = pair_df["cost_entry"].values
    cost_x = pair_df["cost_exit"].values
    borrow = pair_df["borrow_cost"].values
    exit_code = pair_df["exit_code"].values
    notional = pair_df.attrs.get("notional", 0.0)

    pos_prev = np.zeros_like(pos)
    pos_prev[1:] = pos[:-1]
    is_entry = (pos != 0) & (pos_prev == 0)
    is_exit_close = (pos == 0) & (pos_prev != 0)

    trades = []
    cur_entry = -1
    n = len(pos)
    for i in range(n):
        if is_entry[i]:
            cur_entry = i
        elif is_exit_close[i] and cur_entry >= 0:
            sl = slice(cur_entry, i + 1)
            entry_pos = pos[cur_entry]
            entry_z = zscore[cur_entry - 1] if cur_entry >= 1 else np.nan
            spread_change = spread[sl] - spread[cur_entry]
            signed_change = entry_pos * spread_change
            exit_code_val = exit_code[i - 1] if i >= 1 else 0
            trades.append({
                "fold": fold_n, "pair": f"{ta}/{tb}",
                "entry_idx": cur_entry, "exit_idx": i,
                "duration": i - cur_entry,
                "entry_pos": int(entry_pos),
                "entry_z": float(entry_z),
                "gross_pnl": float(pnl_gross[sl].sum()),
                "net_pnl":   float(pnl_net[sl].sum()),
                "total_cost": float(cost_e[sl].sum() + cost_x[sl].sum() + borrow[sl].sum()),
                "mfe": float(signed_change.max() * notional),
                "mae": float(signed_change.min() * notional),
                "exit_reason": "zero_cross" if exit_code_val == 1 else
                              ("hard_sl" if exit_code_val == 2 else "unknown"),
                "notional": notional,
            })
            cur_entry = -1
    if cur_entry >= 0:
        sl = slice(cur_entry, n)
        entry_pos = pos[cur_entry]
        entry_z = zscore[cur_entry - 1] if cur_entry >= 1 else np.nan
        spread_change = spread[sl] - spread[cur_entry]
        signed_change = entry_pos * spread_change
        trades.append({
            "fold": fold_n, "pair": f"{ta}/{tb}",
            "entry_idx": cur_entry, "exit_idx": None,
            "duration": n - cur_entry,
            "entry_pos": int(entry_pos),
            "entry_z": float(entry_z),
            "gross_pnl": float(pnl_gross[sl].sum()),
            "net_pnl":   float(pnl_net[sl].sum()),
            "total_cost": float(cost_e[sl].sum() + cost_x[sl].sum() + borrow[sl].sum()),
            "mfe": float(signed_change.max() * notional),
            "mae": float(signed_change.min() * notional),
            "exit_reason": "open_at_eom",
            "notional": notional,
        })
    return trades


def _johansen_pvalue(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """Johansen trace test p-value via chi2(8). Returns nan on failure."""
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"=== V4 negative-Sharpe deep audit (10 tests) ===")
    print(f"Combo: entry_z={ENTRY_Z}, hl_max={HL_MAX}, hard_sl_z={HARD_SL}, z_window={Z_WIN}")
    print(f"Folds: {len(FOLD_SCHEDULE)}\n")

    t0 = time.time()
    cache = _load_all_daily(DATA_DAILY)
    print(f"  cache loaded: {len(cache)} tickers ({time.time()-t0:.1f}s)\n")

    all_trades: list[dict] = []
    fold_metrics: list[dict] = []
    pairs_per_fold: dict[int, set[str]] = {}
    posthoc_johansen: list[dict] = []
    daily_returns_by_fold: dict[int, pd.Series] = {}      # for T10
    market_returns_by_fold: dict[int, pd.Series] = {}     # for T10

    for fold_n, fs, fe, tm in FOLD_SCHEDULE:
        t_fold = time.time()
        formation = _slice_daily(cache, fs, fe)
        if not formation:
            continue
        try:
            pairs_df, factor_state = discovery_daily.run(
                formation_data=formation, max_pairs=500,
                hl_min=5.0, hl_max=HL_MAX,
            )
        except Exception as e:
            print(f"  Fold {fold_n}: discovery failed: {e}")
            continue
        if pairs_df.empty or not factor_state:
            continue

        trade_start = tm + "-01"
        trade_end = (pd.Timestamp(trade_start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        trading = _slice_daily(cache, trade_start, trade_end)
        if not trading:
            continue

        W = factor_state["loadings_W"]
        factor_tickers = factor_state["tickers"]
        resid_form_df = factor_state["residual_log_prices"]
        resid_trade_dict = project_residual(trading, W, factor_tickers, min_obs=10)
        for tk in list(resid_trade_dict.keys()):
            if tk not in resid_form_df.columns:
                continue
            s_form_tk = resid_form_df[tk].dropna()
            if len(s_form_tk) == 0 or len(resid_trade_dict[tk]) == 0:
                continue
            shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
            resid_trade_dict[tk] = resid_trade_dict[tk] + shift
        resid_trade_df = pd.concat(resid_trade_dict, axis=1)

        pair_results = run_fold_daily(
            pairs_df=pairs_df, resid_form=resid_form_df, resid_trade=resid_trade_df,
            alpha_lookback=60, entry_z=ENTRY_Z, z_window=Z_WIN, hard_sl_z=HARD_SL,
        )

        # ---- T8 prep: pairs that entered this fold ----
        pairs_per_fold[fold_n] = {f"{ta}/{tb}" for (ta, tb) in pair_results.keys()}

        # ---- T9 prep: post-hoc Johansen on trading residuals for each pair ----
        for (ta, tb), df in pair_results.items():
            if ta in resid_trade_df.columns and tb in resid_trade_df.columns:
                ra = resid_trade_df[ta].dropna().values
                rb = resid_trade_df[tb].dropna().values
                n_min = min(len(ra), len(rb))
                if n_min >= 10:
                    p = _johansen_pvalue(ra[:n_min], rb[:n_min])
                    posthoc_johansen.append({
                        "fold": fold_n, "pair": f"{ta}/{tb}",
                        "pval_trading": p,
                        "pval_formation": float(pairs_df.set_index(["ticker_a","ticker_b"])
                                                .loc[(ta, tb), "johansen_pval"])
                                                if (ta, tb) in set(zip(pairs_df["ticker_a"], pairs_df["ticker_b"]))
                                                else np.nan,
                    })
            all_trades.extend(_extract_trades(df, ta, tb, fold_n))

        # ---- T10 prep: daily portfolio returns + market proxy ----
        if pair_results:
            per_pair_pnl = {f"{ta}/{tb}": df["daily_pnl_net"]
                            for (ta, tb), df in pair_results.items()}
            pnl_df = pd.DataFrame(per_pair_pnl).fillna(0.0)
            daily_portfolio_pnl = pnl_df.sum(axis=1)
            daily_returns_by_fold[fold_n] = daily_portfolio_pnl / TOTAL_CAPITAL

            # Market proxy: equal-weight daily return of all tickers in factor model
            log_trading = pd.concat(
                {tk: trading[tk]["log_close"] for tk in factor_tickers if tk in trading},
                axis=1, join="outer",
            ).ffill().dropna()
            if len(log_trading) > 1:
                market_ret = log_trading.diff().mean(axis=1).dropna()
                market_returns_by_fold[fold_n] = market_ret

        # Per-fold portfolio P&L
        fold_pnl = sum(df["daily_pnl_net"].sum() for df in pair_results.values())
        fold_gross = sum(df["daily_pnl_gross"].sum() for df in pair_results.values())
        fold_cost = sum((df["cost_entry"] + df["cost_exit"] + df["borrow_cost"]).sum()
                        for df in pair_results.values())
        fold_metrics.append({
            "fold": fold_n, "trading_month": tm,
            "gross_pnl": fold_gross, "cost": fold_cost, "net_pnl": fold_pnl,
            "n_pairs": len(pair_results),
        })
        print(f"  Fold {fold_n:02d} [{tm}]: pairs={len(pair_results)}, "
              f"gross=${fold_gross:>+8,.0f}, cost=${fold_cost:>8,.0f}, "
              f"net=${fold_pnl:>+8,.0f} ({time.time()-t_fold:.0f}s)")

    df_trades = pd.DataFrame(all_trades)
    df_folds = pd.DataFrame(fold_metrics)
    df_posthoc = pd.DataFrame(posthoc_johansen)
    df_trades.to_csv(OUT_DIR / "audit_trades.csv", index=False)
    df_folds.to_csv(OUT_DIR / "audit_folds.csv", index=False)
    df_posthoc.to_csv(OUT_DIR / "audit_posthoc_johansen.csv", index=False)

    n_trades = len(df_trades)
    print(f"\nCollected: {n_trades} trades across {len(fold_metrics)} folds "
          f"({time.time()-t0:.0f}s)\n")

    # =================================================================
    # STEP 1 — Does the signal have edge?
    # =================================================================
    print(f"{'#'*70}")
    print(f"# STEP 1 — Does the signal have ANY edge?")
    print(f"{'#'*70}\n")

    # T1
    print(f"--- T1: Gross PnL per trade (edge BEFORE costs) ---")
    g = df_trades["gross_pnl"]
    print(f"  Total trades:        {len(g)}")
    print(f"  Mean gross PnL:      ${g.mean():>+10.2f}")
    print(f"  Median gross PnL:    ${g.median():>+10.2f}")
    print(f"  Sum gross PnL:       ${g.sum():>+12,.0f}")
    t1_has_edge = g.mean() > 0
    print(f"  → T1 verdict: {'POSITIVE EDGE' if t1_has_edge else 'NO EDGE — signal wrong direction or noise'}\n")

    # T2
    print(f"--- T2: Cost drag ---")
    sum_gross = df_folds["gross_pnl"].sum()
    sum_cost = df_folds["cost"].sum()
    sum_net = df_folds["net_pnl"].sum()
    print(f"  Sum gross PnL:  ${sum_gross:>+12,.0f}")
    print(f"  Sum costs:      ${sum_cost:>12,.0f}")
    print(f"  Sum net PnL:    ${sum_net:>+12,.0f}")
    if sum_gross > 0:
        cost_pct_of_gross = 100 * sum_cost / sum_gross
        print(f"  Cost as % of gross: {cost_pct_of_gross:.1f}%")
    print(f"  Mean trade gross:  ${df_trades['gross_pnl'].mean():>+8.2f}")
    print(f"  Mean trade cost:   ${df_trades['total_cost'].mean():>8.2f}")
    if df_trades['gross_pnl'].mean() > 0:
        print(f"  → T2 verdict: gross POSITIVE; cost {'eats edge' if df_trades['total_cost'].mean() > df_trades['gross_pnl'].mean() else 'within profit margin'}")
    else:
        print(f"  → T2 verdict: gross NEGATIVE; cost is not the primary problem\n")

    # T7
    print(f"\n--- T7: Hit rate (% trades profitable) ---")
    hit_gross = (df_trades["gross_pnl"] > 0).mean()
    hit_net   = (df_trades["net_pnl"] > 0).mean()
    print(f"  Hit rate (gross PnL > 0): {100*hit_gross:.1f}%")
    print(f"  Hit rate (net PnL > 0):   {100*hit_net:.1f}%")
    print(f"  Stat-arb literature: 50-60% expected. <45% = signal probably wrong direction.")
    if hit_gross < 0.45:
        t7_verdict = "BELOW 45% — signal likely wrong direction"
    elif hit_gross < 0.50:
        t7_verdict = "BELOW LIT — weak signal"
    elif hit_gross < 0.60:
        t7_verdict = "IN LIT RANGE (50-60%)"
    else:
        t7_verdict = "STRONG (>60%)"
    print(f"  → T7 verdict: {t7_verdict}")

    # =================================================================
    # Branch — based on T1
    # =================================================================
    print(f"\n{'#'*70}")
    if t1_has_edge:
        print(f"# STEP 2 — Signal has edge; diagnose EXECUTION (T3, T4, T6)")
        print(f"{'#'*70}\n")
    else:
        print(f"# T1 negative — running both STEP 2 (exit logic) and STEP 3 (signal/factor)")
        print(f"{'#'*70}\n")

    # T3 (MFE/MAE)
    print(f"--- T3: MFE vs MAE per trade ---")
    print(f"  Median MFE: ${df_trades['mfe'].median():>+8.2f}  (best unrealized profit)")
    print(f"  Median MAE: ${df_trades['mae'].median():>+8.2f}  (worst unrealized loss)")
    n_mfe_dom = (df_trades["mfe"] > -df_trades["mae"]).sum()
    n_mae_dom = (df_trades["mfe"] < -df_trades["mae"]).sum()
    print(f"  Peak profit > peak loss: {n_mfe_dom} ({100*n_mfe_dom/len(df_trades):.1f}%)")
    print(f"  Peak loss > peak profit: {n_mae_dom} ({100*n_mae_dom/len(df_trades):.1f}%)")
    print(f"  → T3 verdict: {'SIGNAL OK at trade level' if n_mfe_dom > n_mae_dom else 'SIGNAL BAD — losses dominate during trade life'}")

    # T4 (exit-reason PnL)
    print(f"\n--- T4: P&L by exit reason ---")
    by_exit = df_trades.groupby("exit_reason").agg(
        n=("net_pnl", "count"),
        mean_gross=("gross_pnl", "mean"),
        mean_net=("net_pnl", "mean"),
        sum_net=("net_pnl", "sum"),
        median_duration=("duration", "median"),
    ).round(2)
    print(by_exit.to_string())

    # T6 (time-to-revert)
    print(f"\n--- T6: Time-to-revert (closed via zero_cross) ---")
    closed = df_trades[df_trades["exit_reason"] == "zero_cross"]
    if len(closed) > 0:
        print(f"  N zero_cross trades: {len(closed)}")
        print(f"  Median duration: {closed['duration'].median():.1f} days")
        print(f"  Mean duration:   {closed['duration'].mean():.1f} days")
        print(f"  Median entry |Z|: {closed['entry_z'].abs().median():.2f}")
        print(f"  Theoretical max: 1.44 × {ENTRY_Z} × {HL_MAX} = {1.44*ENTRY_Z*HL_MAX:.0f} days")
        open_eom = df_trades[df_trades["exit_reason"] == "open_at_eom"]
        if len(open_eom):
            print(f"  Closed (zero_cross) mean gross: ${closed['gross_pnl'].mean():>+8.2f}")
            print(f"  Open at EOM mean gross:        ${open_eom['gross_pnl'].mean():>+8.2f}")

    # =================================================================
    # STEP 3 — Signal / factor diagnostic (always run for completeness)
    # =================================================================
    print(f"\n{'#'*70}")
    print(f"# STEP 3 — Why no signal edge? (T8, T9, T10)")
    print(f"{'#'*70}\n")

    # T8 (pair persistence)
    print(f"--- T8: Pair persistence across folds ---")
    all_pairs = set()
    for fp in pairs_per_fold.values():
        all_pairs.update(fp)
    pair_fold_counts: dict[str, int] = {p: 0 for p in all_pairs}
    for fp in pairs_per_fold.values():
        for p in fp:
            pair_fold_counts[p] += 1
    fold_count_dist = pd.Series(list(pair_fold_counts.values())).value_counts().sort_index()
    n_pairs_total = len(all_pairs)
    print(f"  Unique pairs across all folds: {n_pairs_total}")
    print(f"  Distribution of fold-occurrences per pair:")
    for n_folds, count in fold_count_dist.items():
        pct = 100 * count / n_pairs_total
        bar = "#" * int(pct / 2)
        print(f"    appeared in {n_folds:>2d} fold(s): {count:>4d} pairs ({pct:>5.1f}%) {bar}")
    one_fold_pct = 100 * (pair_fold_counts and pd.Series(list(pair_fold_counts.values())).eq(1).sum() / n_pairs_total)
    print(f"  → T8 verdict: {'COINTEGRATION STABLE — many pairs reappear' if one_fold_pct < 50 else 'COINTEGRATION UNSTABLE — most pairs appear in only 1 fold (regime-dependent)'}")
    # Save pair history
    pd.DataFrame([{"pair": p, "fold_count": c}
                  for p, c in pair_fold_counts.items()]).to_csv(
        OUT_DIR / "audit_pair_history.csv", index=False)

    # T9 (post-hoc Johansen)
    print(f"\n--- T9: Post-hoc Johansen on TRADING window ---")
    if len(df_posthoc) > 0:
        valid = df_posthoc.dropna(subset=["pval_trading"])
        pct_significant = 100 * (valid["pval_trading"] < 0.05).mean()
        print(f"  N pairs tested:                {len(valid)}")
        print(f"  Still cointegrated (p<0.05):   {(valid['pval_trading']<0.05).sum()} ({pct_significant:.1f}%)")
        print(f"  Lost cointegration (p≥0.05):   {(valid['pval_trading']>=0.05).sum()} ({100-pct_significant:.1f}%)")
        print(f"  NOTE: trading window is ~20 daily bars; chi2(8) p-values are noisy at this length.")
        print(f"  → T9 verdict: {'Most pairs hold' if pct_significant > 50 else 'Most pairs LOST cointegration during trading — explains drift'}")

    # T10 (factor leak)
    print(f"\n--- T10: Market-beta leak (regress portfolio_ret ~ market_ret) ---")
    port_rets = []
    mkt_rets = []
    for f, pr in daily_returns_by_fold.items():
        if f not in market_returns_by_fold:
            continue
        mr = market_returns_by_fold[f]
        # Align indexes
        idx = pr.index.intersection(mr.index)
        if len(idx) > 1:
            port_rets.append(pr.loc[idx])
            mkt_rets.append(mr.loc[idx])
    if port_rets:
        port_ret = pd.concat(port_rets)
        mkt_ret = pd.concat(mkt_rets)
        # Drop any NaNs and align
        comb = pd.concat([port_ret.rename("port"), mkt_ret.rename("mkt")], axis=1).dropna()
        if len(comb) > 5:
            X = sm.add_constant(comb["mkt"])
            model = sm.OLS(comb["port"], X).fit()
            alpha = model.params.get("const", np.nan)
            beta = model.params.get("mkt", np.nan)
            t_alpha = model.tvalues.get("const", np.nan)
            t_beta = model.tvalues.get("mkt", np.nan)
            print(f"  N daily observations: {len(comb)}")
            print(f"  Alpha (intercept):    {alpha:+.6f}   (t={t_alpha:+.2f})")
            print(f"  Beta on market:       {beta:+.4f}    (t={t_beta:+.2f})")
            print(f"  R²:                   {model.rsquared:.4f}")
            leak = abs(t_beta) > 2.0
            print(f"  → T10 verdict: {'MARKET BETA LEAK (|t_beta|>2)' if leak else 'No significant market exposure'}")

    # =================================================================
    # STEP 4 — Regime context (CONTEXT ONLY, do NOT cherry-pick configs!)
    # =================================================================
    print(f"\n{'#'*70}")
    print(f"# STEP 4 — T5 Regime split (CONTEXT ONLY — not for config selection)")
    print(f"{'#'*70}\n")
    df_folds["bps_of_capital"] = 10000 * df_folds["net_pnl"] / TOTAL_CAPITAL
    df_folds_sorted = df_folds.sort_values("net_pnl", ascending=False)
    print("  Top 5 folds:")
    print(df_folds_sorted.head(5)[["fold","trading_month","gross_pnl","cost","net_pnl","bps_of_capital","n_pairs"]].to_string(index=False))
    print("\n  Bottom 5 folds:")
    print(df_folds_sorted.tail(5)[["fold","trading_month","gross_pnl","cost","net_pnl","bps_of_capital","n_pairs"]].to_string(index=False))
    print(f"\n  Folds positive: {(df_folds['net_pnl'] > 0).sum()} / {len(df_folds)}")
    print(f"  Sum across all: ${df_folds['net_pnl'].sum():>+12,.0f}")
    print(f"\n  ⚠️ Reminder: do NOT tune parameters per regime — that's cherry-picking.")
    print(f"     Use this section only to describe the failure mode in the report.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
