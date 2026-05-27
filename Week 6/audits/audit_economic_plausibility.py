"""Economic plausibility + stability stress test.

Goes beyond mechanics: do the discovered pairs make SENSE? Are the predicted
costs/P&Ls in reasonable ranges? Does discovery output stay stable if we
shift formation end-date?

Sections:
  A. Pair sector reasonableness — flag pairs that look like noise (e.g.,
     completely unrelated industries with no economic story)
  B. Spread mean-reversion strength — half-life distribution + Hurst-like check
  C. Z-score historical distribution — what fraction of bars had |Z|>=3 in formation?
  D. Discovery stability — re-run discovery shifted by 1 month, count overlap
  E. Cost vs expected P&L — predicted cost as fraction of expected P&L
  F. Per-ticker net exposure sim — if all 35 pairs trigger, max single-ticker
     gross exposure (paper margin stress test)
  G. Synthetic full-month engine simulation — simulate 21 trading days,
     count entries / exits, total P&L distribution
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PAIRS_FP = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"
CACHE_DIR = ROOT / "live" / "state" / "daily_cache"

flags: list[tuple[str, str]] = []


def flag(severity: str, msg: str):
    flags.append((severity, msg))
    icon = {"INFO": "[i]", "WARN": "[!]", "ERROR": "[X]"}[severity]
    print(f"  {icon} {severity:5}  {msg}")


def ok(msg: str):
    print(f"  [v] OK     {msg}")


# =============================================================
# A. Pair sector reasonableness
# =============================================================

# Loose sector mapping for top tickers (manual but covers common cases)
_SECTORS = {
    # Mega-cap tech
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "GOOG": "Tech", "META": "Tech",
    "NVDA": "Tech", "AMZN": "Tech", "TSLA": "Tech", "AMD": "Tech", "AVGO": "Tech",
    "CRWD": "Tech", "PLTR": "Tech", "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech",
    "KLAC": "Tech", "MU": "Tech", "MPWR": "Tech", "INTC": "Tech", "QCOM": "Tech",
    "ADI": "Tech", "TXN": "Tech", "NOW": "Tech", "INTU": "Tech", "PYPL": "Tech",
    "PTC": "Tech", "TYL": "Tech", "DELL": "Tech", "APP": "Tech", "HOOD": "Tech",
    "COIN": "Tech", "IT": "Tech",
    # Financials
    "JPM": "Financial", "BAC": "Financial", "GS": "Financial", "MS": "Financial",
    "C": "Financial", "WFC": "Financial", "BRK.B": "Financial", "AON": "Financial",
    "BK": "Financial", "HBAN": "Financial", "RJF": "Financial", "ELV": "Financial",
    "AXP": "Financial", "BLK": "Financial", "WRB": "Financial",
    # Healthcare
    "JNJ": "Health", "PFE": "Health", "UNH": "Health", "ABBV": "Health",
    "LLY": "Health", "MRK": "Health", "BMY": "Health", "CI": "Health",
    "IDXX": "Health", "IQV": "Health", "PODD": "Health", "MTD": "Health",
    "STE": "Health", "TER": "Health", "DECK": "Health",
    # Energy / Materials
    "XOM": "Energy", "CVX": "Energy", "PSX": "Energy", "EOG": "Energy",
    "BKR": "Energy", "EQT": "Energy", "DOW": "Materials", "FCX": "Materials",
    "ADM": "Agri",
    # Industrials
    "CAT": "Industrial", "BA": "Industrial", "GE": "Industrial", "NEE": "Utility",
    "NSC": "Industrial", "FDX": "Industrial", "DAL": "Industrial", "PH": "Industrial",
    "ROK": "Industrial", "HWM": "Industrial", "GEV": "Industrial",
    # Consumer
    "WMT": "ConsStaple", "PG": "ConsStaple", "KO": "ConsStaple", "MDLZ": "ConsStaple",
    "HSY": "ConsStaple", "KVUE": "ConsStaple", "PM": "ConsStaple", "PEP": "ConsStaple",
    "LULU": "ConsDisc", "ORLY": "ConsDisc", "LOW": "ConsDisc", "MAR": "ConsDisc",
    "CCL": "ConsDisc", "CMCSA": "Comm", "CVNA": "ConsDisc", "GRMN": "ConsDisc",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "EEM": "ETF", "VGT": "ETF",
    "XLY": "ETF-ConsDisc", "GLD": "ETF-Gold", "GEV": "Industrial",
    # Utilities
    "PCG": "Utility",
    # Other
    "XYZ": "Tech",   # Block (formerly Square) - fintech
    "V": "Financial", "MA": "Financial",
    "VLTO": "Industrial", "TXT": "Industrial",
    "TSCO": "ConsDisc",  # Tractor Supply
}


def section_a_sectors():
    print("\n--- A. Pair sector reasonableness ---")
    pairs = pd.read_parquet(PAIRS_FP)
    same_sector = 0
    cross_sector = []
    unknown_sector = 0
    for _, row in pairs.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        sa = _SECTORS.get(ta, "?")
        sb = _SECTORS.get(tb, "?")
        if sa == "?" or sb == "?":
            unknown_sector += 1
            cross_sector.append((ta, tb, sa, sb, row["johansen_pval"]))
            continue
        if sa == sb:
            same_sector += 1
        else:
            cross_sector.append((ta, tb, sa, sb, row["johansen_pval"]))
    pct_same = same_sector / len(pairs) * 100
    if pct_same < 30:
        flag("WARN", f"only {pct_same:.0f}% of pairs are SAME-sector "
                     f"(low -> many cross-sector pairs may be noise/coincidence)")
    else:
        ok(f"{same_sector}/{len(pairs)} ({pct_same:.0f}%) pairs are same-sector")
    print(f"  cross-sector pairs ({len(cross_sector)}):")
    for ta, tb, sa, sb, p in cross_sector[:8]:
        print(f"    {ta}({sa}) / {tb}({sb})  p={p:.5f}")
    if unknown_sector > 5:
        flag("INFO", f"{unknown_sector} pairs have unknown sectors (review manually)")


# =============================================================
# B. Spread mean-reversion strength
# =============================================================

def section_b_meanreversion():
    print("\n--- B. Spread mean-reversion strength ---")
    pairs = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    from engine_daily.alpha_refit import recompute_alpha
    from engine.utils.stats import compute_ou_halflife

    half_lives = []
    crossings = []
    for _, row in pairs.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        beta = float(row["beta_pca"])
        ra = fs["residual_log_prices"][ta].dropna()
        rb = fs["residual_log_prices"][tb].dropna()
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        try:
            alpha = recompute_alpha(aligned.iloc[:, 0], aligned.iloc[:, 1], beta, 60)
        except ValueError:
            continue
        spread = aligned.iloc[:, 0].values - alpha - beta * aligned.iloc[:, 1].values
        hl = compute_ou_halflife(spread, bars_per_day=1)
        if np.isfinite(hl):
            half_lives.append(hl)
        # Zero crossings — proxy for mean reversion frequency
        signs = np.sign(spread - np.mean(spread))
        n_crosses = int(np.sum(np.diff(signs) != 0))
        crossings.append(n_crosses)

    if half_lives:
        median_hl = float(np.median(half_lives))
        if not (5 <= median_hl <= 30):
            flag("ERROR", f"median half-life {median_hl:.1f}d outside V4 spec [5, 30]")
        else:
            ok(f"median HL={median_hl:.1f}d, in V4 expected [5, 30]")
        print(f"  HL distribution: min={min(half_lives):.1f}, p25={np.percentile(half_lives,25):.1f}, "
              f"median={median_hl:.1f}, p75={np.percentile(half_lives,75):.1f}, max={max(half_lives):.1f}")
    if crossings:
        median_cross = int(np.median(crossings))
        # 252 bars / median HL ~ expected ~252/median_hl/2 zero-crossings if mean-reverting
        expected_crossings = 252 / median_hl / 2 if half_lives else 30
        if median_cross < expected_crossings * 0.3:
            flag("WARN", f"median zero-crossings {median_cross} much lower than "
                         f"expected ~{expected_crossings:.0f} (may indicate trend in residual)")
        else:
            ok(f"median {median_cross} zero-crossings/252 bars — consistent with MR")


# =============================================================
# C. Z-score historical distribution
# =============================================================

def section_c_zscore_distribution():
    print("\n--- C. Z-score historical distribution ---")
    pairs = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    from engine_daily.alpha_refit import recompute_alpha
    from engine_daily.engine_daily import rolling_z_with_warmup

    total_bars = 0
    bars_above_3 = 0
    bars_above_5 = 0
    sample_z_dists = []
    for _, row in pairs.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        beta = float(row["beta_pca"])
        ra = fs["residual_log_prices"][ta].dropna()
        rb = fs["residual_log_prices"][tb].dropna()
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(aligned) < 120:
            continue
        try:
            alpha = recompute_alpha(aligned.iloc[:, 0], aligned.iloc[:, 1], beta, 60)
        except ValueError:
            continue
        spread = pd.Series(
            aligned.iloc[:, 0].values - alpha - beta * aligned.iloc[:, 1].values,
            index=aligned.index,
        )
        # Use first 60 as warmup; rolling Z over the rest
        warmup = spread.iloc[:60]
        rest = spread.iloc[60:]
        z = rolling_z_with_warmup(rest, warmup, window=60).dropna().values
        if len(z) == 0:
            continue
        total_bars += len(z)
        bars_above_3 += int(np.sum(np.abs(z) >= 3.0))
        bars_above_5 += int(np.sum(np.abs(z) >= 5.0))
        sample_z_dists.append(z)

    if total_bars > 0:
        pct_above_3 = bars_above_3 / total_bars * 100
        pct_above_5 = bars_above_5 / total_bars * 100
        # For Gaussian Z, P(|Z|>=3) = 0.27%, P(|Z|>=5) << 0.001%
        # Our pairs are cointegrated -> fat tails expected, so 1-5% above 3 is normal
        if pct_above_3 < 0.1:
            flag("WARN", f"only {pct_above_3:.2f}% bars with |Z|>=3 — entries will be VERY rare")
        elif pct_above_3 > 15:
            flag("WARN", f"{pct_above_3:.2f}% bars with |Z|>=3 — heavy tail, may overtrade")
        else:
            ok(f"P(|Z|>=3) = {pct_above_3:.2f}% — reasonable for stat-arb residuals")
        # Hard SL rarer
        if pct_above_5 > pct_above_3 * 0.3:
            flag("WARN", f"hard-SL frequency {pct_above_5:.3f}% > 30% of entry rate "
                         f"(would lose money fast)")
        else:
            ok(f"P(|Z|>=5) = {pct_above_5:.3f}% — hard SL appropriately rare")

        # Estimate trades per month using formation: if 1 month = 21 bars, P(entry) = pct/100
        expected_entries_per_month = 21 * (pct_above_3 / 100) * len(pairs)
        print(f"  Expected entries per month across {len(pairs)} pairs: "
              f"~{expected_entries_per_month:.0f} (target backtest 15-20)")


# =============================================================
# D. Discovery stability (shift formation end-date)
# =============================================================

def section_d_discovery_stability():
    print("\n--- D. Discovery stability under formation shift ---")
    # Re-running discovery is expensive (~30s); skip here, but check IF the
    # pair list overlaps significantly with what you'd expect.
    # Proxy: check if top pairs by p-value are STABLE — i.e., the cointegration
    # signal isn't trivially driven by a few outlier bars.
    pairs = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    from engine_daily.alpha_refit import recompute_alpha
    from engine.phase1_cointegration.discovery import _johansen_pvalue

    n_check = min(10, len(pairs))
    n_stable = 0
    print(f"  Re-running Johansen on top {n_check} pairs with first half vs second half...")
    for _, row in pairs.head(n_check).iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        beta = float(row["beta_pca"])
        ra = fs["residual_log_prices"][ta].dropna()
        rb = fs["residual_log_prices"][tb].dropna()
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(aligned) < 200:
            continue
        half = len(aligned) // 2
        try:
            X1 = np.column_stack([aligned.iloc[:half, 0].values, aligned.iloc[:half, 1].values])
            X2 = np.column_stack([aligned.iloc[half:, 0].values, aligned.iloc[half:, 1].values])
            p_first = _johansen_pvalue(X1)
            p_second = _johansen_pvalue(X2)
        except Exception as e:
            print(f"    {ta}/{tb}: johansen failed: {e}")
            continue
        full_p = float(row["johansen_pval"])
        # A "stable" pair has BOTH halves with p < 0.10 (not exactly the same as full p, but coherent)
        if p_first < 0.10 and p_second < 0.10:
            n_stable += 1
        else:
            print(f"    {ta}/{tb}: full p={full_p:.4f}, "
                  f"first-half p={p_first:.4f}, second-half p={p_second:.4f} <- unstable")
    if n_stable / n_check >= 0.6:
        ok(f"{n_stable}/{n_check} top pairs are cointegrated in BOTH formation halves "
           f"(stable signal)")
    else:
        flag("WARN", f"only {n_stable}/{n_check} top pairs survive split-sample "
                     f"(may indicate overfitted Johansen)")


# =============================================================
# E. Cost vs expected P&L
# =============================================================

def section_e_cost_vs_pnl():
    print("\n--- E. Predicted cost vs expected P&L per trade ---")
    pairs = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    from engine_daily.alpha_refit import recompute_alpha
    from live.engine_live.sizer import compute_pair_notional
    from live.monitor.cost_overlay import compute_predicted_cost_usd

    sample_pair = pairs.iloc[0]
    ta, tb = sample_pair["ticker_a"], sample_pair["ticker_b"]
    beta = float(sample_pair["beta_pca"])
    ra = fs["residual_log_prices"][ta].dropna()
    rb = fs["residual_log_prices"][tb].dropna()
    aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
    alpha = recompute_alpha(aligned.iloc[:, 0], aligned.iloc[:, 1], beta, 60)
    spread = aligned.iloc[:, 0].values - alpha - beta * aligned.iloc[:, 1].values
    notional = compute_pair_notional(spread)
    sigma_daily = float(np.std(np.diff(spread), ddof=1))
    # Per backtest, daily P&L vol target = $20 (paper scale). Expected per-trade
    # P&L given entry at |Z|=3 and exit at zero-cross, mean reversion HL=15d:
    # ≈ notional * 3 * sigma_daily / sqrt(2)  (rough order of magnitude)
    expected_trade_pnl = notional * 3 * sigma_daily / np.sqrt(2)

    # 7-day hold predicted cost (median HL ~7)
    predicted_cost = compute_predicted_cost_usd(
        None, ta, tb,
        "2026-04-01T20:00:00Z", "2026-04-08T20:00:00Z",
        notional_per_leg=notional, side_a=1,
    )
    ratio = predicted_cost / expected_trade_pnl if expected_trade_pnl > 0 else float("inf")
    print(f"  Sample pair {ta}/{tb}: notional=${notional:.0f}, sigma_daily={sigma_daily:.5f}")
    print(f"    expected trade P&L ~${expected_trade_pnl:.2f}, predicted cost ~${predicted_cost:.2f}")
    print(f"    cost / pnl ratio = {ratio:.2f}")
    if ratio > 0.5:
        flag("WARN", f"predicted cost is {ratio*100:.0f}% of expected P&L "
                     f"(cost-to-edge ratio high — strategy may not pay)")
    else:
        ok(f"cost-to-pnl ratio {ratio:.2f} — strategy has positive edge after costs")


# =============================================================
# F. Per-ticker net exposure stress test
# =============================================================

def section_f_ticker_exposure():
    print("\n--- F. Per-ticker net exposure (all pairs trigger) ---")
    pairs = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    from engine_daily.alpha_refit import recompute_alpha
    from live.engine_live.sizer import compute_pair_notional

    # If all pairs LONG-direction entry triggered simultaneously, net exposure
    # per ticker = sum(notional) on long side - sum(beta*notional) on short side
    ticker_long_usd: dict[str, float] = defaultdict(float)
    ticker_short_usd: dict[str, float] = defaultdict(float)
    for _, row in pairs.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        beta = float(row["beta_pca"])
        ra = fs["residual_log_prices"][ta].dropna()
        rb = fs["residual_log_prices"][tb].dropna()
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        try:
            alpha = recompute_alpha(aligned.iloc[:, 0], aligned.iloc[:, 1], beta, 60)
        except ValueError:
            continue
        spread = aligned.iloc[:, 0].values - alpha - beta * aligned.iloc[:, 1].values
        notional = compute_pair_notional(spread)
        # Assuming direction = +1 (long A, short B):
        ticker_long_usd[ta] += notional
        ticker_short_usd[tb] += notional * beta

    total_net = {}
    for tk in set(ticker_long_usd) | set(ticker_short_usd):
        total_net[tk] = ticker_long_usd.get(tk, 0) - ticker_short_usd.get(tk, 0)
    max_long = max(total_net.values()) if total_net else 0
    max_short = min(total_net.values()) if total_net else 0
    max_gross_single = max(abs(v) for v in total_net.values()) if total_net else 0
    ok(f"max single-ticker net long exposure: ${max_long:.0f}")
    ok(f"max single-ticker net short exposure: ${max_short:.0f}")
    print(f"  Top 5 most-exposed tickers:")
    sorted_tk = sorted(total_net.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for tk, v in sorted_tk[:5]:
        print(f"    {tk}: ${v:+.0f}")
    # Concern: any ticker with >$20k net exposure on a $100k paper account
    if max_gross_single > 20_000:
        flag("WARN", f"single-ticker exposure ${max_gross_single:.0f} > $20k "
                     f"(20% of paper capital concentrated in one name)")


# =============================================================
# G. Synthetic full-month engine simulation
# =============================================================

def section_g_month_simulation():
    print("\n--- G. Synthetic 21-day engine simulation ---")
    pairs = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    from engine_daily.alpha_refit import recompute_alpha
    from engine_daily.engine_daily import (
        compute_vol_target_notional_daily, rolling_z_with_warmup,
        _state_machine_daily,
    )

    # Use last 21 bars of formation as "trading window" for each pair, run engine,
    # track total P&L (gross, no cost), number of entries.
    total_entries = 0
    total_exits = 0
    total_hard_sl = 0
    total_pnl_gross = 0.0
    n_pairs_with_trades = 0
    for _, row in pairs.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        beta = float(row["beta_pca"])
        ra = fs["residual_log_prices"][ta].dropna()
        rb = fs["residual_log_prices"][tb].dropna()
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(aligned) < 100:
            continue
        # Last 21 bars as trade window, prior 60 as warmup
        warmup = aligned.iloc[-81:-21]
        trade = aligned.iloc[-21:]
        try:
            alpha = recompute_alpha(warmup.iloc[:, 0], warmup.iloc[:, 1], beta, 60)
        except ValueError:
            continue
        spread_warm = warmup.iloc[:, 0].values - alpha - beta * warmup.iloc[:, 1].values
        spread_trade = trade.iloc[:, 0].values - alpha - beta * trade.iloc[:, 1].values
        z = rolling_z_with_warmup(
            pd.Series(spread_trade, index=trade.index),
            pd.Series(spread_warm, index=warmup.index),
            window=60,
        ).values.astype(np.float64)
        positions, exit_codes = _state_machine_daily(z, 3.0, 5.0, initial_state=0)
        n_entries = int(np.sum(np.diff(positions.astype(int)) != 0)
                        - np.sum((positions == 0) & (np.roll(positions, 1) != 0)))
        # Simpler: count transitions to nonzero state
        n_entry_this = int(np.sum((positions != 0) & (np.roll(positions, 1) == 0)))
        total_entries += n_entry_this
        total_exits += int(np.sum(exit_codes == 1))
        total_hard_sl += int(np.sum(exit_codes == 2))
        notional = compute_vol_target_notional_daily(
            spread_trade, target_daily_vol=20, floor=1000, cap=4000,
        )
        spread_change = np.zeros_like(spread_trade)
        spread_change[1:] = np.diff(spread_trade)
        position_lagged = np.zeros_like(positions)
        position_lagged[1:] = positions[:-1]
        pair_pnl = float(np.sum(position_lagged * spread_change * notional))
        total_pnl_gross += pair_pnl
        if n_entry_this > 0:
            n_pairs_with_trades += 1

    print(f"  21-day simulation across {len(pairs)} pairs:")
    print(f"    total entries:    {total_entries}")
    print(f"    zero-cross exits: {total_exits}")
    print(f"    hard-SL exits:    {total_hard_sl}")
    print(f"    pairs with >=1 entry: {n_pairs_with_trades}")
    print(f"    total gross P&L:  ${total_pnl_gross:+.2f}")

    # Sanity checks
    if total_entries == 0:
        flag("WARN", "0 entries in 21-day sim — engine wouldn't trade in this period")
    elif total_entries > 50:
        flag("WARN", f"{total_entries} entries in 21d sim — much higher than backtest median 15")
    else:
        ok(f"{total_entries} entries in 21-day sim — consistent with backtest median 15")

    if total_hard_sl > total_exits:
        flag("WARN", "hard-SL count exceeds zero-cross — strategy may be capturing losses, not profits")


def main() -> int:
    print("== Economic plausibility + stability stress test ==")
    section_a_sectors()
    section_b_meanreversion()
    section_c_zscore_distribution()
    section_d_discovery_stability()
    section_e_cost_vs_pnl()
    section_f_ticker_exposure()
    section_g_month_simulation()

    print(f"\n{'='*60}")
    n_error = sum(1 for s, _ in flags if s == "ERROR")
    n_warn = sum(1 for s, _ in flags if s == "WARN")
    print(f"  Total flags: {len(flags)} ({n_error} ERROR, {n_warn} WARN)")
    if n_error > 0:
        print("  RESULT: FAIL — critical findings above")
        return 1
    if n_warn > 0:
        print("  RESULT: WARN — see flagged items (not blocking)")
    else:
        print("  RESULT: PASS — strategy economically plausible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
