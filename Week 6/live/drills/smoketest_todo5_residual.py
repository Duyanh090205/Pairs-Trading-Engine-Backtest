"""TODO 5 smoketest: residual projector + Path A re-anchor.

Verifies the live wrapper produces SAME residuals as backtest's run_fold_v4
inline code path for the same inputs.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine.phase1_cointegration.factor_residual import project_residual

FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"
CACHE_DIR = ROOT / "live" / "state" / "daily_cache"

errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def _load_trading_data(n_recent: int = 5) -> dict[str, pd.DataFrame]:
    """Build trading_data dict from recent N bars of the cache."""
    out = {}
    for fp in sorted(CACHE_DIR.glob("*.parquet")):
        df = pd.read_parquet(fp)
        if len(df) >= n_recent + 5:
            out[fp.stem] = df.tail(n_recent).copy()
    return out


def t_construction():
    from live.engine_live.residual_projector import ResidualProjector
    proj = ResidualProjector(FACTOR_FP)
    check("loaded loadings_W", proj.loadings_W is not None)
    check("loaded tickers list", len(proj.tickers) > 100,
          f"got {len(proj.tickers)}")
    check("formation_residuals is DataFrame",
          isinstance(proj.formation_residuals, pd.DataFrame))
    check("formation_last has per-ticker values",
          len(proj.formation_last) >= 100,
          f"got {len(proj.formation_last)}")


def t_compute_basic():
    from live.engine_live.residual_projector import ResidualProjector
    proj = ResidualProjector(FACTOR_FP)
    trading = _load_trading_data(n_recent=10)
    check("trading data has reasonable size",
          len(trading) > 100, f"got {len(trading)}")
    residuals = proj.compute_residuals(trading)
    check("residuals returned dict",
          isinstance(residuals, dict))
    check("returned series for most tickers",
          len(residuals) > 100, f"got {len(residuals)}")
    sample_tk = next(iter(residuals.keys()))
    series = residuals[sample_tk]
    check(f"{sample_tk}: residual is finite",
          np.all(np.isfinite(series.dropna())))


def t_reanchor_path_a():
    """Path A invariant: residual_trading[0] == formation_residuals.iloc[-1] for the ticker."""
    from live.engine_live.residual_projector import ResidualProjector
    proj = ResidualProjector(FACTOR_FP)
    trading = _load_trading_data(n_recent=15)
    residuals = proj.compute_residuals(trading)
    # Pick a ticker known to be in formation
    sample_tk = None
    for tk in residuals:
        if tk in proj.formation_last and not residuals[tk].dropna().empty:
            sample_tk = tk
            break
    assert sample_tk is not None, "no eligible ticker for re-anchor test"
    first_trading_resid = float(residuals[sample_tk].dropna().iloc[0])
    formation_last = proj.formation_last[sample_tk]
    diff = abs(first_trading_resid - formation_last)
    check(f"{sample_tk} re-anchor: trading[0] == formation_last (diff < 1e-9)",
          diff < 1e-9, f"diff = {diff:.6e}")


def t_anchor_caching():
    """Calling compute_residuals twice should use cached anchor (not recompute)."""
    from live.engine_live.residual_projector import ResidualProjector
    proj = ResidualProjector(FACTOR_FP)
    trading_1 = _load_trading_data(n_recent=15)
    proj.compute_residuals(trading_1)
    anchors_after_first = dict(proj._anchor_shifts)
    # Second call with DIFFERENT trading data — anchors should NOT change
    trading_2 = _load_trading_data(n_recent=20)   # different size, same window edge
    proj.compute_residuals(trading_2)
    anchors_after_second = proj._anchor_shifts
    check("anchor shifts cached (not recomputed on 2nd call)",
          all(anchors_after_first.get(tk) == anchors_after_second.get(tk)
              for tk in anchors_after_first),
          "anchors changed between calls")


def t_reset_anchors():
    from live.engine_live.residual_projector import ResidualProjector
    proj = ResidualProjector(FACTOR_FP)
    proj.compute_residuals(_load_trading_data(n_recent=15))
    check("anchors populated after compute", len(proj._anchor_shifts) > 0)
    proj.reset_anchors()
    check("anchors empty after reset", len(proj._anchor_shifts) == 0)


def t_cross_path_vs_backtest():
    """Live ResidualProjector output must match the inline path in run_v4_pipeline.run_fold_v4.

    Backtest code (run_v4_pipeline.py:217-228):
        resid_trade_dict = project_residual(trading_daily_raw, loadings_W,
                                            factor_tickers, min_obs=10)
        # Path A re-anchor
        for tk in list(resid_trade_dict.keys()):
            if tk not in resid_form_df.columns: continue
            s_form_tk = resid_form_df[tk].dropna()
            if len(s_form_tk) == 0 or len(resid_trade_dict[tk]) == 0: continue
            shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
            resid_trade_dict[tk] = resid_trade_dict[tk] + shift
    """
    from live.engine_live.residual_projector import ResidualProjector
    proj = ResidualProjector(FACTOR_FP)
    trading = _load_trading_data(n_recent=10)

    # Live path
    live_residuals = proj.compute_residuals(trading)

    # Backtest inline path (reproduce exactly):
    with FACTOR_FP.open("rb") as f:
        fs = pickle.load(f)
    bt_raw = project_residual(
        trading, fs["loadings_W"], fs["tickers"], min_obs=10,
    )
    resid_form_df = fs["residual_log_prices"]
    for tk in list(bt_raw.keys()):
        if tk not in resid_form_df.columns:
            continue
        s_form_tk = resid_form_df[tk].dropna()
        if len(s_form_tk) == 0 or len(bt_raw[tk]) == 0:
            continue
        shift = float(s_form_tk.iloc[-1]) - float(bt_raw[tk].iloc[0])
        bt_raw[tk] = bt_raw[tk] + shift

    # Compare: keys identical, values bit-identical
    common = set(live_residuals.keys()) & set(bt_raw.keys())
    check(f"key sets identical ({len(common)} common tickers)",
          set(live_residuals.keys()) == set(bt_raw.keys()))

    max_diff = 0.0
    n_compared = 0
    for tk in common:
        ls = live_residuals[tk].dropna()
        bs = bt_raw[tk].dropna()
        if ls.empty or bs.empty:
            continue
        aligned = pd.concat([ls.rename("live"), bs.rename("bt")], axis=1).dropna()
        if aligned.empty:
            continue
        diff = (aligned["live"] - aligned["bt"]).abs().max()
        max_diff = max(max_diff, float(diff))
        n_compared += 1
    check(f"cross-path bit-identical (n={n_compared} tickers, max diff={max_diff:.2e})",
          max_diff < 1e-9, f"max diff = {max_diff:.6e}")


def t_hardstop_still_works():
    import tempfile
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t5_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop clean", not hardstop.is_tripped())
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop trips", hardstop.is_tripped())
    hardstop.clear("todo5")
    check("hardstop clears", not hardstop.is_tripped())


def main() -> int:
    print("== TODO 5 Smoketest: residual projector + re-anchor ==\n")
    print("--- Construction ---")
    t_construction()
    print("\n--- Basic compute ---")
    t_compute_basic()
    print("\n--- Path A re-anchor invariant ---")
    t_reanchor_path_a()
    print("\n--- Anchor caching ---")
    t_anchor_caching()
    print("\n--- Reset anchors ---")
    t_reset_anchors()
    print("\n--- Cross-path vs backtest inline code ---")
    t_cross_path_vs_backtest()
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
