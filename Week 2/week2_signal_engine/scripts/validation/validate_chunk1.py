"""
Chunk 1 Validation Script
Run from: week2_signal_engine/
  python validate_chunk1.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.utils.config import load_config
from src.data.loaders import _load_ticker, load_pair
from src.data.validation import audit_data
from src.utils.dates import split_periods

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "params_example.yaml")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition

all_passed = True

# ── 1. Config loading ─────────────────────────────────────────────────────────
print("\n=== 1. Config Loading ===")
cfg = load_config(CONFIG_PATH)
all_passed &= check("load_config returns dict", isinstance(cfg, dict))
all_passed &= check("'pairs' key present", "pairs" in cfg)
all_passed &= check("'dates' key present", "dates" in cfg)
all_passed &= check("'engine' key present", "engine" in cfg)
all_passed &= check("primary pair is CMS/DUK", cfg["pairs"]["primary"] == ["CMS", "DUK"])

# ── 2. Per-ticker load (CMS) ──────────────────────────────────────────────────
print("\n=== 2. Single-Ticker Load (CMS) ===")
df_cms = _load_ticker(DATA_DIR, "CMS")
all_passed &= check("CMS has UTC DatetimeIndex", str(df_cms.index.dtype) == "datetime64[ns, UTC]")
all_passed &= check("CMS only has 'close' column", list(df_cms.columns) == ["close"])
all_passed &= check("CMS index is monotonic", df_cms.index.is_monotonic_increasing)
all_passed &= check("CMS no NaNs", not df_cms.isna().any().any())
all_passed &= check("CMS > 90k bars", len(df_cms) > 90_000, f"{len(df_cms):,} bars")

# ── 3. Audit (CMS) ────────────────────────────────────────────────────────────
print("\n=== 3. Audit (CMS) ===")
audit_cms = audit_data(df_cms, ticker="CMS")
all_passed &= check("coverage_pct > 99%", audit_cms["coverage_pct"] > 99.0,
                    f"{audit_cms['coverage_pct']}%")
all_passed &= check("median_gap == 1 min", audit_cms["median_gap"] == pd.Timedelta("1min"),
                    str(audit_cms["median_gap"]))
all_passed &= check("trading_days ~251", 245 <= audit_cms["trading_days"] <= 255,
                    f"{audit_cms['trading_days']} days")
print(f"    n_missing_bars: {audit_cms['n_missing_bars']}")
print(f"    largest_intraday_gap: {audit_cms['largest_intraday_gap']}")

# ── 4. Audit (DUK) ────────────────────────────────────────────────────────────
print("\n=== 4. Audit (DUK) ===")
df_duk = _load_ticker(DATA_DIR, "DUK")
audit_duk = audit_data(df_duk, ticker="DUK")
all_passed &= check("DUK coverage_pct > 99%", audit_duk["coverage_pct"] > 99.0,
                    f"{audit_duk['coverage_pct']}%")
all_passed &= check("DUK median_gap == 1 min", audit_duk["median_gap"] == pd.Timedelta("1min"),
                    str(audit_duk["median_gap"]))

# ── 5. load_pair (CMS/DUK) ────────────────────────────────────────────────────
print("\n=== 5. load_pair (CMS / DUK) ===")
merged, alignment_audit = load_pair(DATA_DIR, "CMS", "DUK")
all_passed &= check("merged columns", list(merged.columns) == ["close_a", "close_b"])
all_passed &= check("merged no NaNs", not merged.isna().any().any())
all_passed &= check("alignment_rate_a > 0.99", alignment_audit["alignment_rate_a"] > 0.99,
                    f"{alignment_audit['alignment_rate_a']:.4f}")
all_passed &= check("alignment_rate_b > 0.99", alignment_audit["alignment_rate_b"] > 0.99,
                    f"{alignment_audit['alignment_rate_b']:.4f}")
print(f"    bars: A={alignment_audit['bars_a_pre_join']:,}, B={alignment_audit['bars_b_pre_join']:,}, joined={alignment_audit['bars_after_join']:,}")
print(f"    range: {alignment_audit['first_timestamp']} to {alignment_audit['last_timestamp']}")

# ── 6. split_periods ─────────────────────────────────────────────────────────
print("\n=== 6. split_periods (CMS / DUK) ===")
dates = cfg["dates"]
formation, trading = split_periods(merged, **dates)

all_passed &= check("formation non-empty", len(formation) > 0, f"{len(formation):,} bars")
all_passed &= check("trading non-empty", len(trading) > 0, f"{len(trading):,} bars")
all_passed &= check("no overlap (no lookahead)", formation.index.max() < trading.index.min(),
                    f"formation ends {formation.index.max()}, trading starts {trading.index.min()}")
# The sum may be slightly < merged because bars on the boundary day (June 30)
# fall between formation_end midnight UTC and trading_start midnight UTC.
# The critical invariant is no overlap — the small gap is expected.
boundary_gap = len(merged) - len(formation) - len(trading)
all_passed &= check("boundary gap < 1 full day (< 400 bars)", boundary_gap < 400,
                    f"{boundary_gap} bars excluded at period boundary")
all_passed &= check("formation no NaNs", not formation.isna().any().any())
all_passed &= check("trading no NaNs", not trading.isna().any().any())
print(f"    Formation: {formation.shape} | {formation.index.min()} to {formation.index.max()}")
print(f"    Trading:   {trading.shape}   | {trading.index.min()} to {trading.index.max()}")

# ── 7. Secondary pair: DOW / LYB ─────────────────────────────────────────────
print("\n=== 7. Secondary Pair (DOW / LYB) ===")
merged2, audit2 = load_pair(DATA_DIR, "DOW", "LYB")
formation2, trading2 = split_periods(merged2, **dates)
all_passed &= check("DOW/LYB merged no NaNs", not merged2.isna().any().any())
all_passed &= check("DOW/LYB no lookahead", formation2.index.max() < trading2.index.min())
all_passed &= check("DOW/LYB alignment_rate_a > 0.99", audit2["alignment_rate_a"] > 0.99,
                    f"{audit2['alignment_rate_a']:.4f}")
print(f"    bars: A={audit2['bars_a_pre_join']:,}, B={audit2['bars_b_pre_join']:,}, joined={audit2['bars_after_join']:,}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
if all_passed:
    print("\033[92mALL CHECKS PASSED -- Chunk 1 complete.\033[0m")
else:
    print("\033[91mSOME CHECKS FAILED -- review output above.\033[0m")
    sys.exit(1)
