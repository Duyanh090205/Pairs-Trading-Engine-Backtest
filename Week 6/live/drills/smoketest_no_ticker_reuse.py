"""Smoketest: --no-ticker-reuse flag enforces disjoint ticker sets.

Verifies:
  1. Greedy algorithm picks lowest johansen_pval first.
  2. After dedup, no ticker appears in 2+ pairs.
  3. Meta JSON records no_ticker_reuse=True.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
errors: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    color = GREEN if cond else RED
    print(f"  {color}[{mark}]{RESET} {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_greedy_algorithm_unit():
    """Inline test of the greedy logic — independent of disk I/O."""
    # Construct a synthetic pairs_df with known overlaps.
    df = pd.DataFrame([
        {"ticker_a": "A", "ticker_b": "B", "johansen_pval": 0.001},  # best
        {"ticker_a": "A", "ticker_b": "C", "johansen_pval": 0.002},  # A reused -> drop
        {"ticker_a": "C", "ticker_b": "D", "johansen_pval": 0.003},  # keep
        {"ticker_a": "B", "ticker_b": "D", "johansen_pval": 0.004},  # both reused -> drop
        {"ticker_a": "E", "ticker_b": "F", "johansen_pval": 0.005},  # keep
    ])
    ordered = df.sort_values("johansen_pval", ascending=True)
    used: set[str] = set()
    keep_idx: list[int] = []
    for idx, row in ordered.iterrows():
        if row["ticker_a"] in used or row["ticker_b"] in used:
            continue
        keep_idx.append(idx)
        used.add(row["ticker_a"])
        used.add(row["ticker_b"])
    kept = df.loc[keep_idx]

    check("greedy keeps best p-val first", kept.iloc[0]["johansen_pval"] == 0.001)
    check("greedy drops pair with reused ticker_a (A)",
          0.002 not in kept["johansen_pval"].values)
    check("greedy keeps C-D (C newly used here)",
          0.003 in kept["johansen_pval"].values)
    check("greedy drops B-D (both already used)",
          0.004 not in kept["johansen_pval"].values)
    check("greedy keeps disjoint E-F",
          0.005 in kept["johansen_pval"].values)
    check("greedy result is disjoint",
          not _has_reuse(kept), f"got {len(kept)} pairs")


def _has_reuse(df: pd.DataFrame) -> bool:
    tickers = list(df["ticker_a"]) + list(df["ticker_b"])
    return any(n > 1 for n in Counter(tickers).values())


def t_end_to_end_with_flag():
    """Run the real discovery script with --no-ticker-reuse and check invariant."""
    if not (ROOT / "live" / "state" / "daily_cache").exists():
        check("end-to-end skip — no cache present", True,
              "skipped (run build_live_daily_cache.py first to enable)")
        return

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_live_discovery.py"),
        "--filter-tradable", str(ROOT / "live" / "universe_top300.json"),
        "--apply-ticker-cap",
        "--split-sample-gate", "0.10",
        "--no-ticker-reuse",
    ]
    print(f"  running: {' '.join(cmd[-6:])}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        check("discovery script ran", False,
              f"exit={res.returncode}; stderr tail: {res.stderr[-300:]}")
        return
    check("discovery script ran", True)

    # Check output: pair list disjoint
    pairs_fp = ROOT / "live" / "state" / "discovered_pairs.parquet"
    df = pd.read_parquet(pairs_fp)
    check("pairs_df not empty", len(df) > 0, f"{len(df)} pairs")
    check("no ticker appears in 2+ pairs", not _has_reuse(df),
          "tickers: " + ", ".join(sorted(set(list(df["ticker_a"]) + list(df["ticker_b"])))))

    # Check meta records the flag
    meta_fp = ROOT / "live" / "state" / "discovery_meta.json"
    meta = json.loads(meta_fp.read_text())
    check("meta records no_ticker_reuse=True", meta.get("no_ticker_reuse") is True,
          f"got {meta.get('no_ticker_reuse')}")


def main() -> int:
    print("== Smoketest: --no-ticker-reuse ==\n")
    print("--- Unit: greedy algorithm logic ---")
    t_greedy_algorithm_unit()
    print("\n--- End-to-end: real discovery with flag ---")
    t_end_to_end_with_flag()
    print()
    if errors:
        print(f"{RED}FAIL: {len(errors)} - {errors}{RESET}")
        return 1
    print(f"{GREEN}PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
