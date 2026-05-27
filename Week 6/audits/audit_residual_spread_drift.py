"""
Residual-routing diagnostic — hypothesis 3 from the V3.0 stuck-strategy context.

Question: do trading-time residuals (1-min, outer-join + ffill via project_residual)
drift from formation-time residuals (5-min, inner-join via fit_factor_model) such
that alpha/beta fit on the former no longer give a mean-reverting spread on the
latter?

Test plan (fold 1, formation 2022-01-03 -> 2022-06-30, trading 2022-07):

  1. ALGORITHMIC EQUIVALENCE: feed the SAME 5-min formation data to BOTH
     fit_factor_model and project_residual. They should produce identical
     residuals (dense data, full universe). If not, the two functions are
     not algorithmically equivalent even under matched inputs.

  2. ANCHOR / LEVEL DRIFT: for each top-N pair, compute
        spread_form  = resid_a_form  - alpha - beta * resid_b_form
        spread_trade = resid_a_trade - alpha - beta * resid_b_trade
     and compare:
       - mean shift in units of formation sigma  (anchor mismatch indicator)
       - std ratio                                (volatility regime change)
       - lag-1 autocorr of spread_trade            (mean-reversion proxy:
                                                    << 1 = mean-reverting,
                                                    ~ 1 = random walk)

  3. FRACTION OF FORMATION TICKERS MISSING AT TRADING TIME: every missing
     ticker shrinks W_avail and biases factor returns at trading time.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6_ROOT))

from engine.phase1_cointegration.factor_residual import (
    fit_factor_model, project_residual,
)
from engine.phase1_cointegration import discovery as v3_discovery

VALIDATED_DIR = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated")
DATA_1MIN = VALIDATED_DIR / "1min_phase2"
DATA_5MIN = VALIDATED_DIR / "5min_phase1"

FORMATION_START = "2022-01-03"
FORMATION_END   = "2022-06-30"
TRADING_MONTH   = "2022-07"
N_TOP_PAIRS     = 8


def _load_all(data_dir: Path, compute_log_close: bool):
    cache = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".parquet"):
            continue
        tk = fname[:-len(".parquet")]
        try:
            df = pd.read_parquet(data_dir / fname)
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache


def _slice_range(cache, tickers, start, end):
    out = {}
    for tk in sorted(tickers):
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= start) & (df.index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_month(cache, tickers, month):
    year, m = int(month[:4]), int(month[5:7])
    out = {}
    for tk in sorted(tickers):
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == m)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def main() -> int:
    print("=== Residual-routing diagnostic, fold 1 ===\n")

    t0 = time.time()
    print("Loading 5-min and 1-min caches...")
    cache_5min = _load_all(DATA_5MIN, compute_log_close=False)
    cache_1min = _load_all(DATA_1MIN, compute_log_close=True)
    print(f"  5-min: {len(cache_5min)} tickers | 1-min: {len(cache_1min)} tickers"
          f" ({time.time()-t0:.0f}s)\n")

    # ---------- Discovery on fold 1 ----------
    print(f"Running discovery on formation {FORMATION_START} -> {FORMATION_END}...")
    t1 = time.time()
    formation_data_full = _slice_range(
        cache_5min, set(cache_5min.keys()), FORMATION_START, FORMATION_END,
    )
    pairs_df, factor_state = v3_discovery.run(
        formation_start=FORMATION_START,
        formation_end=FORMATION_END,
        validated_dir=VALIDATED_DIR,
        max_pairs=500,
        formation_data=formation_data_full,
    )
    if pairs_df.empty or not factor_state:
        print("Discovery returned no pairs; aborting.")
        return 1
    print(f"  Discovery done in {time.time()-t1:.0f}s | {len(pairs_df)} pairs survive\n")

    W = factor_state["loadings_W"]
    factor_tickers = factor_state["tickers"]
    resid_form = factor_state["residual_log_prices"]  # 5-min, inner-join
    print(f"Factor model: W shape {W.shape}, {len(factor_tickers)} formation tickers")
    print(f"  resid_form shape: {resid_form.shape} (5-min, inner-join)")
    print(f"  cum_var: {factor_state['diagnostics']['cumulative_variance_explained']:.3f}\n")

    # Restrict formation_data_full to the tickers fit_factor_model kept
    # (factor_tickers comes from aligned.columns after inner-join).
    formation_data_factor = {tk: formation_data_full[tk] for tk in factor_tickers
                             if tk in formation_data_full}

    # ---------- Test 1: algorithmic equivalence on matched 5-min data ----------
    print("[Test 1] project_residual vs fit_factor_model on SAME 5-min formation data")
    resid_form_via_proj_dict = project_residual(
        formation_data_factor, W, factor_tickers,
    )
    resid_form_via_proj = pd.concat(resid_form_via_proj_dict, axis=1)

    # Align indexes/columns
    shared_idx = resid_form.index.intersection(resid_form_via_proj.index)
    shared_cols = [c for c in resid_form.columns if c in resid_form_via_proj.columns]
    A = resid_form.loc[shared_idx, shared_cols]
    B = resid_form_via_proj.loc[shared_idx, shared_cols]
    diff = (A - B).abs()
    max_diff = float(diff.max().max())
    mean_diff = float(diff.mean().mean())
    print(f"  shared shape: {A.shape}")
    print(f"  max abs diff: {max_diff:.6e}")
    print(f"  mean abs diff: {mean_diff:.6e}")
    if max_diff < 1e-8:
        print("  => EQUIVALENT on dense formation data (no routing bug under matched inputs)\n")
    elif max_diff < 1e-3:
        print("  => MINOR drift, likely from outer-join+ffill ordering on near-dense data\n")
    else:
        print("  => SIGNIFICANT drift even on formation data — routing bug confirmed\n")

    # ---------- Test 2: trading-time projection ----------
    print(f"Loading trading 1-min data for {TRADING_MONTH}...")
    trading_data = _slice_month(cache_1min, set(factor_tickers), TRADING_MONTH)
    print(f"  trading_data: {len(trading_data)} tickers (of {len(factor_tickers)} formation)")

    missing_trade = set(factor_tickers) - set(trading_data.keys())
    print(f"  formation tickers MISSING at trading time: {len(missing_trade)} "
          f"({100*len(missing_trade)/len(factor_tickers):.1f}%)")
    if missing_trade and len(missing_trade) <= 10:
        print(f"    missing: {sorted(missing_trade)}")

    resid_trade_dict = project_residual(trading_data, W, factor_tickers)
    print(f"  resid_trade: {len(resid_trade_dict)} ticker series\n")

    # ---------- Path A: per-ticker re-anchoring of trading residuals ----------
    # The trading residual restarts cumsum from trading_window[0]'s log_close.
    # Re-anchor so trading_residual[0] == formation_residual[tk].iloc[-1], i.e.
    # the trading series is the continuation of the formation series.
    # Each ticker gets its own additive shift; relative dynamics (cumsum of returns)
    # are unchanged.
    print("[Path A] Re-anchoring trading residuals to continue from formation last value")
    resid_trade_anchored = {}
    shifts = []
    skipped = 0
    for tk, s_trade in resid_trade_dict.items():
        if tk not in resid_form.columns:
            skipped += 1
            continue
        s_form_tk = resid_form[tk].dropna()
        if len(s_form_tk) == 0 or len(s_trade) == 0:
            skipped += 1
            continue
        shift = float(s_form_tk.iloc[-1]) - float(s_trade.iloc[0])
        resid_trade_anchored[tk] = s_trade + shift
        shifts.append(shift)
    shifts_arr = np.array(shifts)
    print(f"  re-anchored: {len(resid_trade_anchored)} tickers (skipped {skipped})")
    print(f"  per-ticker shift: mean={shifts_arr.mean():+.4f}, "
          f"median={np.median(shifts_arr):+.4f}, "
          f"std={shifts_arr.std():.4f}, "
          f"|shift| max={np.abs(shifts_arr).max():.4f}\n")

    # ---------- Test 3: spread drift across top pairs ----------
    print(f"[Test 3] Top {N_TOP_PAIRS} pairs by Johansen p-value:\n")
    top = pairs_df.sort_values("johansen_pval").head(N_TOP_PAIRS)

    header = (
        f"  {'pair':>13s} {'beta':>5s} | "
        f"{'sh/sig':>7s} {'std_r':>6s} {'ac1':>7s} | "
        f"{'sh/sig*':>7s} {'std_r*':>7s} {'ac1*':>7s}"
    )
    print(header + "    (* = after Path A re-anchor)")
    print("  " + "-" * (len(header) - 2))

    def _pair_stats(df_a, df_b, alpha, beta, ref_std):
        """Return (mean, std, shift/sigma, std_ratio, lag1_ac) for the spread."""
        df = pd.concat([df_a, df_b], axis=1, join="inner").dropna()
        df.columns = ["a", "b"]
        if len(df) < 30:
            return None
        spread = df["a"].values - alpha - beta * df["b"].values
        m = float(spread.mean())
        s = float(spread.std(ddof=1))
        shift_sig = (m - 0.0) / ref_std if ref_std > 1e-12 else np.nan
        std_r = s / ref_std if ref_std > 1e-12 else np.nan
        sp = spread - m
        if len(sp) > 2 and sp.std(ddof=1) > 1e-12:
            ac1 = float(np.corrcoef(sp[:-1], sp[1:])[0, 1])
        else:
            ac1 = np.nan
        return m, s, shift_sig, std_r, ac1

    summary_rows = []
    for _, row in top.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        alpha = float(row["alpha_pca"])
        beta = float(row["beta_pca"])

        if ta not in resid_form.columns or tb not in resid_form.columns:
            continue
        # Formation spread (the reference)
        form = _pair_stats(resid_form[ta], resid_form[tb], alpha, beta, ref_std=1.0)
        if form is None:
            continue
        s_form_mean, s_form_std, _, _, _ = form
        # Re-compute with the form_std as reference now
        form = _pair_stats(resid_form[ta], resid_form[tb], alpha, beta, ref_std=s_form_std)
        _, _, _, _, ac1_form = form  # (mean ~ 0, std_ratio = 1, shift/sig ~ 0)

        # Trading spread — un-anchored (production)
        if ta not in resid_trade_dict or tb not in resid_trade_dict:
            continue
        trade_un = _pair_stats(
            resid_trade_dict[ta], resid_trade_dict[tb], alpha, beta, ref_std=s_form_std,
        )
        # Trading spread — re-anchored (Path A)
        if ta not in resid_trade_anchored or tb not in resid_trade_anchored:
            continue
        trade_an = _pair_stats(
            resid_trade_anchored[ta], resid_trade_anchored[tb], alpha, beta, ref_std=s_form_std,
        )
        if trade_un is None or trade_an is None:
            continue

        _, _, sh_un, sr_un, ac1_un = trade_un
        _, _, sh_an, sr_an, ac1_an = trade_an

        print(
            f"  {ta+'/'+tb:>13s} {beta:>5.2f} | "
            f"{sh_un:>+7.2f} {sr_un:>6.2f} {ac1_un:>7.4f} | "
            f"{sh_an:>+7.2f} {sr_an:>7.2f} {ac1_an:>7.4f}"
        )
        summary_rows.append({
            "pair": f"{ta}/{tb}", "alpha": alpha, "beta": beta,
            "s_form_std": s_form_std,
            "shift_unanchored": sh_un, "std_ratio_unanchored": sr_un, "ac1_unanchored": ac1_un,
            "shift_anchored":   sh_an, "std_ratio_anchored":   sr_an, "ac1_anchored":   ac1_an,
        })

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        print()
        print("=== Aggregate (across top pairs) ===")
        print(f"{'metric':>30s} | {'production':>12s}  {'Path A':>12s}")
        print(f"{'-'*30} | {'-'*12}  {'-'*12}")
        print(f"{'median |shift / form-sigma|':>30s} | "
              f"{df_sum['shift_unanchored'].abs().median():>12.2f}  "
              f"{df_sum['shift_anchored'].abs().median():>12.2f}")
        print(f"{'max    |shift / form-sigma|':>30s} | "
              f"{df_sum['shift_unanchored'].abs().max():>12.2f}  "
              f"{df_sum['shift_anchored'].abs().max():>12.2f}")
        print(f"{'median std_ratio (trade/form)':>30s} | "
              f"{df_sum['std_ratio_unanchored'].median():>12.2f}  "
              f"{df_sum['std_ratio_anchored'].median():>12.2f}")
        print(f"{'median lag-1 autocorr':>30s} | "
              f"{df_sum['ac1_unanchored'].median():>12.4f}  "
              f"{df_sum['ac1_anchored'].median():>12.4f}")

        out_path = WEEK6_ROOT / "audit_residual_spread_drift.csv"
        df_sum.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

    print("\nInterpretation guide:")
    print("  - |shift/sigma| > 3 => formation alpha is far off-center for trade window")
    print("                          (anchor mismatch / regime drift in residual returns)")
    print("  - std_ratio outside [0.3, 3] => vol regime shifted; rolling Z compensates,")
    print("                                    but hard-SL band |Z|>=5.5 fires at a")
    print("                                    different excursion size than formation")
    print("  - lag-1 autocorr > 0.999 at 1-min => essentially random walk; no reversion")
    print("                                       to trade. < 0.995 => mean-reverting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
