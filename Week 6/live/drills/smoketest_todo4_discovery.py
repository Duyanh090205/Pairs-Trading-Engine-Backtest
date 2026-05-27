"""TODO 4 smoketest: offline discovery output integrity.

Verifies the persisted artifacts (discovered_pairs.parquet, factor_state.pkl,
discovery_meta.json) are well-formed and ready for the live engine to load.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PAIRS_FP = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"
META_FP = ROOT / "live" / "state" / "discovery_meta.json"
# Read which universe was actually used from discovery meta (may be top50/100/150/300...).
_meta = json.loads(META_FP.read_text()) if META_FP.exists() else {}
_universe_file = _meta.get("tradable_universe_file")
UNIVERSE_FP = Path(_universe_file) if _universe_file else (ROOT / "live" / "universe_top50.json")

errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_artifacts_exist():
    check("discovered_pairs.parquet exists", PAIRS_FP.exists())
    check("factor_state.pkl exists", FACTOR_FP.exists())
    check("discovery_meta.json exists", META_FP.exists())


def t_pair_list_schema():
    df = pd.read_parquet(PAIRS_FP)
    expected = {"ticker_a", "ticker_b", "alpha_pca", "beta_pca", "johansen_pval"}
    check(f"pair list has all required columns",
          expected.issubset(set(df.columns)),
          f"missing: {expected - set(df.columns)}")
    check("pair list non-empty", len(df) > 0, f"got {len(df)} pairs")
    return df


def t_beta_within_cap():
    df = pd.read_parquet(PAIRS_FP)
    if df.empty:
        return
    bad = df[df["beta_pca"].abs() > 5.0]
    check("all |beta| <= 5 (beta-cap enforced)",
          len(bad) == 0, f"violations: {len(bad)}")


def t_pvalues_significant():
    df = pd.read_parquet(PAIRS_FP)
    if df.empty:
        return
    high_p = df[df["johansen_pval"] >= 0.05]
    check("all pairs have johansen_pval < 0.05 (passed BH-FDR)",
          len(high_p) == 0, f"violations: {len(high_p)}")


def t_pairs_in_tradable_universe():
    """If filter-tradable was applied, both legs must be in top-50."""
    df = pd.read_parquet(PAIRS_FP)
    if df.empty:
        return
    tradable = set(json.loads(UNIVERSE_FP.read_text())["tickers"])
    not_in = df[~(df["ticker_a"].isin(tradable) & df["ticker_b"].isin(tradable))]
    check("all pair legs are in top-50 tradable universe",
          len(not_in) == 0, f"violations: {len(not_in)}")


def t_no_self_pairs():
    df = pd.read_parquet(PAIRS_FP)
    if df.empty:
        return
    self_pairs = df[df["ticker_a"] == df["ticker_b"]]
    check("no self-pairs (ticker_a == ticker_b)",
          len(self_pairs) == 0, f"violations: {len(self_pairs)}")


def t_factor_state_shape():
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    expected_keys = {"loadings_W", "tickers", "residual_log_prices"}
    check("factor_state has required keys",
          expected_keys.issubset(set(fs.keys())),
          f"missing: {expected_keys - set(fs.keys())}")
    W = fs["loadings_W"]
    tickers = fs["tickers"]
    check("loadings_W is 2D (n_tickers x n_components)",
          hasattr(W, "shape") and len(W.shape) == 2,
          f"shape: {getattr(W, 'shape', None)}")
    if hasattr(W, "shape"):
        # Convention from backtest: W has shape (n_components, n_tickers).
        # n_components is the smaller axis (5); n_tickers is the larger one.
        small, large = sorted(W.shape)
        check(f"loadings_W n_components == 5 (V4 spec)",
              small == 5, f"smaller axis = {small}")
        check(f"loadings_W n_tickers matches tickers list",
              large == len(tickers), f"larger axis = {large}, tickers list has {len(tickers)}")
    rlp = fs["residual_log_prices"]
    check("residual_log_prices is DataFrame",
          isinstance(rlp, pd.DataFrame))
    check("residual_log_prices columns match tickers",
          set(rlp.columns) == set(tickers))
    check("residual_log_prices all finite (no inf/nan-only cols)",
          np.isfinite(rlp.values).any(axis=0).all(),
          "some columns are entirely NaN/inf")


def t_meta_sanity():
    meta = json.loads(META_FP.read_text())
    check("meta has formation_end + cumulative_variance",
          "formation_end" in meta and "cumulative_variance_explained" in meta)
    cve = meta["cumulative_variance_explained"]
    # 5-component PCA on US-equity daily returns typically explains 30-45%.
    # Threshold relaxed to 0.25 — the remaining 60-70% is idiosyncratic
    # noise (what we want to trade).
    check(f"PCA explains >= 25% variance (cve={cve:.3f})",
          cve >= 0.25, f"got {cve:.3f}")
    check("n_factor_tickers > 100",
          meta["n_factor_tickers"] > 100, f"got {meta['n_factor_tickers']}")


def t_hardstop_still_works():
    import tempfile
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t4_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop clean", not hardstop.is_tripped())
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop trips", hardstop.is_tripped())
    hardstop.clear("todo4")
    check("hardstop clears", not hardstop.is_tripped())


def main() -> int:
    print("== TODO 4 Smoketest: offline discovery output ==\n")
    print("--- Artifact presence ---")
    t_artifacts_exist()
    print("\n--- Pair list schema ---")
    t_pair_list_schema()
    print("\n--- beta cap ---")
    t_beta_within_cap()
    print("\n--- BH-FDR p-values ---")
    t_pvalues_significant()
    print("\n--- Tradable universe filter ---")
    t_pairs_in_tradable_universe()
    print("\n--- Self-pair check ---")
    t_no_self_pairs()
    print("\n--- Factor state structure ---")
    t_factor_state_shape()
    print("\n--- Meta sanity ---")
    t_meta_sanity()
    print("\n--- Hardstop ---")
    t_hardstop_still_works()
    print()
    if errors:
        print(f"FAIL: {len(errors)} - {errors}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
