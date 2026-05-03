"""
Deliverable 1 - Creating "Bad Data"
CP1: Verify clean data | CP3: Generate 20 flawed datasets (4 methods x 5 k-values)

Run: python scripts/deliverable1.py
"""

import os
import sys
import time

# Ensure scripts/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    load_and_verify_clean_data,
    inject_h1_future_close,
    inject_h2_timestamp_backdating,
    inject_h3_spread_level,
    inject_h4_normalization_leak,
)

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FLAWED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "flawed")

K_VALUES = [5, 10, 20, 30, 40, 50]
METHODS  = {
    "h1": inject_h1_future_close,
    "h2": inject_h2_timestamp_backdating,
    "h3": inject_h3_spread_level,
    "h4": inject_h4_normalization_leak,
}


def main():
    os.makedirs(FLAWED_DIR, exist_ok=True)

    print("\n" + "=" * 58)
    print("  DELIVERABLE 1 - Creating Bad Data")
    print("=" * 58)

    # ── CP1: Load and verify clean data ──────────────────────────
    print("\n--- CP1: Loading and verifying clean data ---")
    t0 = time.time()
    df_clean, summary = load_and_verify_clean_data(RAW_DIR)
    print(f"    Load time: {time.time()-t0:.1f}s")

    # Confirm all assertions passed
    assertions = summary["assertions"]
    failed = []
    for k, v in assertions.items():
        if k == "A6_bars_per_day":
            if not v["pass"]:
                failed.append(k)
        else:
            if not v:
                failed.append(k)
    if failed:
        raise RuntimeError(f"CP1 assertions FAILED: {failed}")
    print("    CP1: All 6 assertions PASSED")

    # ── CP3: Generate 20 flawed datasets ─────────────────────────
    print("\n--- CP3: Generating 20 flawed datasets ---")
    results_log = []

    for method_name, inject_fn in METHODS.items():
        for k in K_VALUES:
            fname = f"flawed_{method_name}_k{k}.csv"
            fpath = os.path.join(FLAWED_DIR, fname)

            print(f"\n  [{method_name.upper()} k={k}%] -> {fname}")
            t1 = time.time()
            df_flawed = inject_fn(df_clean, k_percent=k, seed=42)
            elapsed = time.time() - t1

            # Save with ts_utc as index
            df_flawed.to_csv(fpath)

            n_rows    = len(df_flawed)
            assert n_rows == len(df_clean), \
                f"{fname}: row count {n_rows:,} != clean {len(df_clean):,}"
            n_cols    = len(df_flawed.columns)
            fsize_mb  = os.path.getsize(fpath) / 1e6
            results_log.append({
                "file": fname,
                "rows": n_rows,
                "cols": n_cols,
                "size_mb": fsize_mb,
                "time_s":  elapsed,
            })
            print(f"    Saved: {fname}  rows={n_rows:,}  cols={n_cols}  "
                  f"size={fsize_mb:.1f}MB  time={elapsed:.1f}s")

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"  SUMMARY: {len(results_log)} Flawed Datasets Created")
    print("=" * 58)
    print(f"  {'File':<25s}  {'Rows':>8s}  {'Cols':>4s}  {'MB':>6s}")
    print("  " + "-" * 50)
    for r in results_log:
        print(f"  {r['file']:<25s}  {r['rows']:>8,}  {r['cols']:>4d}  {r['size_mb']:>6.1f}")
    n_expected = len(METHODS) * len(K_VALUES)
    print("=" * 58)
    print(f"  Total files: {len(results_log)} / {n_expected}")
    print(f"  Output dir : {os.path.abspath(FLAWED_DIR)}")
    print("=" * 58 + "\n")

    # Verify all 20 files exist
    missing = []
    for method_name in METHODS:
        for k in K_VALUES:
            fname = f"flawed_{method_name}_k{k}.csv"
            if not os.path.exists(os.path.join(FLAWED_DIR, fname)):
                missing.append(fname)
    if missing:
        raise RuntimeError(f"Missing flawed files: {missing}")
    print("  All 20 flawed datasets confirmed present.")


if __name__ == "__main__":
    main()
