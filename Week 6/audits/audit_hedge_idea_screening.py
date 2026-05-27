"""Screen current 35 pairs against 4 potential hedge ideas.

For each idea, compute how many of the 35 pairs survive the additional filter.
Helps decide which hedges are practical (don't eliminate all signal).
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine.phase1_cointegration.discovery import _johansen_pvalue
from engine.utils.stats import compute_ou_halflife
from engine_daily.alpha_refit import recompute_alpha

PAIRS_FP = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"

# Loose sector mapping
_SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "GOOG": "Tech", "META": "Tech",
    "NVDA": "Tech", "AMZN": "Tech", "TSLA": "Tech", "AMD": "Tech", "AVGO": "Tech",
    "CRWD": "Tech", "PLTR": "Tech", "KLAC": "Tech", "MU": "Tech", "MPWR": "Tech",
    "INTC": "Tech", "QCOM": "Tech", "PYPL": "Tech", "PTC": "Tech", "TYL": "Tech",
    "DELL": "Tech", "APP": "Tech", "HOOD": "Tech", "COIN": "Tech", "IT": "Tech",
    "VGT": "Tech",   # tech ETF
    "JPM": "Financial", "BAC": "Financial", "GS": "Financial", "MS": "Financial",
    "BRK.B": "Financial", "AON": "Financial", "BK": "Financial", "HBAN": "Financial",
    "RJF": "Financial", "ELV": "Health", "WRB": "Financial",
    "JNJ": "Health", "PFE": "Health", "UNH": "Health", "ABBV": "Health",
    "BMY": "Health", "CI": "Health", "IDXX": "Health", "IQV": "Health",
    "PODD": "Health", "MTD": "Health", "STE": "Health", "TER": "Health",
    "DECK": "ConsDisc",
    "XOM": "Energy", "CVX": "Energy", "PSX": "Energy", "EOG": "Energy",
    "BKR": "Energy", "EQT": "Energy", "DOW": "Materials", "FCX": "Materials",
    "CAT": "Industrial", "BA": "Industrial", "GE": "Industrial",
    "NSC": "Industrial", "FDX": "Industrial", "DAL": "Industrial", "PH": "Industrial",
    "ROK": "Industrial", "HWM": "Industrial", "GEV": "Industrial",
    "NEE": "Utility", "PCG": "Utility",
    "PG": "ConsStaple", "MDLZ": "ConsStaple", "HSY": "ConsStaple", "KVUE": "ConsStaple",
    "PM": "ConsStaple", "PEP": "ConsStaple",
    "LULU": "ConsDisc", "ORLY": "ConsDisc", "LOW": "ConsDisc", "MAR": "ConsDisc",
    "CCL": "ConsDisc", "CVNA": "ConsDisc", "GRMN": "ConsDisc", "TSCO": "ConsDisc",
    "CMCSA": "Comm",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "EEM": "ETF",
    "XLY": "ETF",   # treat sector ETF as its own bucket
    "GLD": "ETF",
    "ADM": "Agri", "XYZ": "Tech", "V": "Financial", "MA": "Financial",
    "VLTO": "Industrial", "TXT": "Industrial",
}


def main() -> int:
    pairs = pd.read_parquet(PAIRS_FP)
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    resid = fs["residual_log_prices"]

    n_total = len(pairs)
    print(f"== Hedge idea screening on {n_total} pairs ==\n")

    # Compute per-pair diagnostics
    diag = []
    for _, row in pairs.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        beta = float(row["beta_pca"])
        ra = resid[ta].dropna()
        rb = resid[tb].dropna()
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(aligned) < 200:
            diag.append(None)
            continue
        alpha = recompute_alpha(aligned.iloc[:, 0], aligned.iloc[:, 1], beta, 60)
        spread = aligned.iloc[:, 0].values - alpha - beta * aligned.iloc[:, 1].values
        # Half-life
        hl = compute_ou_halflife(spread, bars_per_day=1)
        # Split-sample Johansen
        half = len(aligned) // 2
        try:
            X1 = np.column_stack([aligned.iloc[:half, 0], aligned.iloc[:half, 1]])
            X2 = np.column_stack([aligned.iloc[half:, 0], aligned.iloc[half:, 1]])
            p1 = _johansen_pvalue(X1)
            p2 = _johansen_pvalue(X2)
        except Exception:
            p1, p2 = float("nan"), float("nan")
        # Raw correlation of log_close (on residual log-prices proxy)
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        # Sector match
        sa = _SECTORS.get(ta, "?")
        sb = _SECTORS.get(tb, "?")
        same_sector = (sa == sb) and (sa != "?")
        diag.append({
            "pair": f"{ta}/{tb}",
            "ta": ta, "tb": tb,
            "p_full": float(row["johansen_pval"]),
            "p_half1": p1,
            "p_half2": p2,
            "hl": float(hl) if np.isfinite(hl) else float("nan"),
            "corr": corr,
            "same_sector": same_sector,
        })
    diag = [d for d in diag if d is not None]
    df = pd.DataFrame(diag)
    print(f"Diagnostics computed for {len(df)} pairs\n")

    # ===== Idea 1: Split-sample stability gate =====
    # Both halves p < 0.05 (strict) or p < 0.10 (loose)
    print("--- Idea 1: Split-sample stability gate ---")
    strict = df[(df["p_half1"] < 0.05) & (df["p_half2"] < 0.05)]
    loose = df[(df["p_half1"] < 0.10) & (df["p_half2"] < 0.10)]
    print(f"  STRICT (both halves p<0.05): {len(strict)}/{len(df)} pairs survive")
    print(f"  LOOSE  (both halves p<0.10): {len(loose)}/{len(df)} pairs survive")
    if len(loose) > 0:
        print(f"  Sample surviving pairs (loose):")
        for _, r in loose.head(5).iterrows():
            print(f"    {r['pair']}: p_full={r['p_full']:.4f} p_h1={r['p_half1']:.4f} p_h2={r['p_half2']:.4f}")

    # ===== Idea 2: Same-sector filter =====
    print("\n--- Idea 2: Same-sector filter ---")
    same = df[df["same_sector"]]
    print(f"  Same-sector only: {len(same)}/{len(df)} pairs survive")
    print(f"  Surviving pairs:")
    for _, r in same.iterrows():
        sa = _SECTORS.get(r["ta"])
        print(f"    {r['pair']} ({sa}): p_full={r['p_full']:.4f}")

    # ===== Idea 3: Tighter HL filter [5, 15] =====
    print("\n--- Idea 3: Tighter half-life [5, 15] days ---")
    tight_hl = df[(df["hl"] >= 5) & (df["hl"] <= 15)]
    print(f"  HL in [5,15]: {len(tight_hl)}/{len(df)} pairs survive")
    print(f"  Current HL distribution: median={df['hl'].median():.1f}, "
          f"min={df['hl'].min():.1f}, max={df['hl'].max():.1f}")

    # ===== Idea 4: Correlation pre-filter (residuals corr > 0.3) =====
    # Note: residuals already have factors removed, so high corr is hard
    print("\n--- Idea 4: Residual correlation pre-filter ---")
    corr_03 = df[df["corr"].abs() > 0.3]
    corr_05 = df[df["corr"].abs() > 0.5]
    print(f"  |residual corr| > 0.3: {len(corr_03)}/{len(df)} pairs")
    print(f"  |residual corr| > 0.5: {len(corr_05)}/{len(df)} pairs")
    print(f"  corr distribution: median={df['corr'].abs().median():.3f}, "
          f"min={df['corr'].abs().min():.3f}, max={df['corr'].abs().max():.3f}")

    # ===== Combinations =====
    print("\n--- Combined filters ---")
    # Idea 1 (loose) + Idea 3 (tight HL)
    combo_1_3 = df[(df["p_half1"] < 0.10) & (df["p_half2"] < 0.10)
                    & (df["hl"] >= 5) & (df["hl"] <= 15)]
    print(f"  Idea 1 (loose) + Idea 3 (HL[5,15]): {len(combo_1_3)}/{len(df)}")
    # Idea 1 (strict) + Idea 3
    combo_1s_3 = df[(df["p_half1"] < 0.05) & (df["p_half2"] < 0.05)
                     & (df["hl"] >= 5) & (df["hl"] <= 15)]
    print(f"  Idea 1 (strict) + Idea 3 (HL[5,15]): {len(combo_1s_3)}/{len(df)}")
    # All 3 ideas
    combo_all = df[(df["p_half1"] < 0.10) & (df["p_half2"] < 0.10)
                    & (df["hl"] >= 5) & (df["hl"] <= 15)
                    & df["same_sector"]]
    print(f"  Ideas 1+2+3: {len(combo_all)}/{len(df)}")

    print("\n=== Summary table ===")
    print(f"  Baseline (current):                {n_total} pairs")
    print(f"  + Idea 1 strict (both halves<0.05): {len(strict)} pairs")
    print(f"  + Idea 1 loose  (both halves<0.10): {len(loose)} pairs")
    print(f"  + Idea 2 (same-sector only):        {len(same)} pairs")
    print(f"  + Idea 3 (HL [5,15]):               {len(tight_hl)} pairs")
    print(f"  + Idea 1 loose + Idea 3 combined:   {len(combo_1_3)} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
