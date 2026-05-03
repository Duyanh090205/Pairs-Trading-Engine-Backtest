"""
Plan 0 real-data runner.

Two-pass strategy (optimised for 16-core / 3.9 GB free RAM):

Pass 1 — sequential single-scan (2-3 min)
  Read each of 204 row groups once.
  Apply session filter + microstructure features.
  Flush per-ticker interim parquets every FLUSH_EVERY row groups.
  Peak RAM: ~1.5 GB (one accumulation batch).

Pass 2 — parallel per-ticker compute (3-5 min)
  ProcessPoolExecutor(MAX_WORKERS).
  Each worker: load ticker's interim files, compute seasonality +
  rolling, write output temp parquet.
  Peak RAM per worker: ~100 MB.

Assembly — concat all output temps into 4 final parquets (~1 min).

Total expected: 7-10 min vs ~44 min for 526 per-ticker full-file scans.
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

WEEK5_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK5_ROOT))

from src.plan0_gateway import (
    compute_intraday_seasonality,
    compute_microstructure_features,
    compute_rolling_instability,
    process_orderbook,
)

# ── paths ──────────────────────────────────────────────────────────────────
ORDERBOOK  = WEEK5_ROOT / "data" / "orderbook.parquet"
MICRO_DIR  = WEEK5_ROOT / "data" / "microstructure"
TMP_TICKER = MICRO_DIR / "_tmp_ticker"   # per-ticker interim parquets
TMP_OUT    = MICRO_DIR / "_tmp_out"      # per-ticker output parquets before assembly

# ── config ─────────────────────────────────────────────────────────────────
FLUSH_EVERY  = 50        # row groups accumulated before flushing to disk
MAX_WORKERS  = 4         # parallel workers for Pass 2

# Formation end = fold 1 formation window end (no lookahead)
FORMATION_END = pd.Timestamp("2022-06-30 23:59:59", tz="UTC")

READ_COLS = [
    "timestamp", "ticker",
    "l1_bid_px", "l1_ask_px",
    "l2_bid_px", "l2_ask_px",
    "l3_bid_px", "l3_ask_px",
    "l1_bid_sz",
]

INTERIM_COLS = [
    "timestamp_et", "ticker", "is_valid",
    "full_spread_l1_bps", "half_spread_l1_bps",
    "full_spread_l2_bps", "full_spread_l3_bps",
    "liquidity_l1",
]

OUT_COLS = [
    "timestamp_et", "ticker", "is_valid",
    "full_spread_l1_bps", "half_spread_l1_bps",
    "full_spread_l2_bps", "full_spread_l3_bps",
    "liquidity_l1", "spread_std_1d", "raw_spread_mean_1d",
]


# ── Pass 1 helpers ──────────────────────────────────────────────────────────

def _flush_ticker_buffers(
    buffers: dict[str, list[pd.DataFrame]],
    batch_idx: int,
) -> int:
    """Write each ticker's accumulated rows to its own batch parquet. Returns row count."""
    total = 0
    for ticker, chunks in buffers.items():
        if not chunks:
            continue
        ticker_dir = TMP_TICKER / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(ticker_dir / f"batch_{batch_idx:03d}.parquet",
                      compression="snappy", index=False)
        total += len(df)
    return total


def run_pass1(pf: pq.ParquetFile) -> list[str]:
    """
    Single-scan over all row groups. Returns sorted list of tickers found.
    """
    n_groups = pf.metadata.num_row_groups
    TMP_TICKER.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list[pd.DataFrame]] = defaultdict(list)
    all_tickers: set[str] = set()
    total_rows_written = 0
    batch_idx = 0
    t_start = time.time()

    for i in range(n_groups):
        # Read one row group
        rg_df = pf.read_row_group(i, columns=READ_COLS).to_pandas()

        # Ingest + features (session filter, spread bps)
        df_clean = process_orderbook(rg_df)
        del rg_df
        df_feat = compute_microstructure_features(df_clean)
        del df_clean
        df_interim = df_feat[INTERIM_COLS].copy()
        del df_feat

        # Accumulate per ticker
        for ticker, grp in df_interim.groupby("ticker", sort=False):
            buffers[ticker].append(grp.reset_index(drop=True))
            all_tickers.add(ticker)
        del df_interim

        # Flush every FLUSH_EVERY row groups
        if (i + 1) % FLUSH_EVERY == 0:
            rows = _flush_ticker_buffers(buffers, batch_idx)
            total_rows_written += rows
            buffers.clear()
            gc.collect()
            batch_idx += 1
            elapsed = time.time() - t_start
            pct = (i + 1) / n_groups * 100
            eta = elapsed / (i + 1) * (n_groups - i - 1)
            print(f"  Pass 1 | RG {i+1:3d}/{n_groups} ({pct:.0f}%) | "
                  f"batch {batch_idx} flushed | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

    # Final flush
    if any(buffers.values()):
        rows = _flush_ticker_buffers(buffers, batch_idx)
        total_rows_written += rows
        buffers.clear()
        gc.collect()

    elapsed = time.time() - t_start
    print(f"  Pass 1 done: {total_rows_written:,} interim rows, "
          f"{len(all_tickers)} tickers, {elapsed:.1f}s")
    return sorted(all_tickers)


# ── Pass 2 worker ───────────────────────────────────────────────────────────

def _process_ticker_worker(ticker: str) -> dict:
    """
    Worker: load all interim batches for one ticker → seasonality → rolling
    → write output parquet. Returns stats dict.
    """
    # Resolve paths fresh (worker is a spawned process on Windows)
    week5 = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(week5))

    tmp_ticker = week5 / "data" / "microstructure" / "_tmp_ticker"
    tmp_out    = week5 / "data" / "microstructure" / "_tmp_out"
    tmp_out.mkdir(parents=True, exist_ok=True)

    form_end = pd.Timestamp("2022-06-30 23:59:59", tz="UTC")
    out_cols = [
        "timestamp_et", "ticker", "is_valid",
        "full_spread_l1_bps", "half_spread_l1_bps",
        "full_spread_l2_bps", "full_spread_l3_bps",
        "liquidity_l1", "spread_std_1d", "raw_spread_mean_1d",
    ]

    try:
        ticker_dir = tmp_ticker / ticker
        batch_files = sorted(ticker_dir.glob("batch_*.parquet"))
        if not batch_files:
            return {"ticker": ticker, "ok": False, "error": "no interim files"}

        df = pd.concat(
            [pd.read_parquet(f) for f in batch_files],
            ignore_index=True,
        ).sort_values("timestamp_et")

        # Seasonality from formation window only
        form_mask = df["timestamp_et"].dt.tz_convert("UTC") <= form_end
        df_seas = compute_intraday_seasonality(df[form_mask])

        # Rolling instability (full timeline)
        df_roll = compute_rolling_instability(df, df_seas)
        del df, df_seas

        df_out = df_roll[out_cols].copy()
        del df_roll

        out_path = tmp_out / f"{ticker}.parquet"
        df_out.to_parquet(out_path, compression="snappy", index=False)

        stats = {
            "ticker": ticker,
            "ok": True,
            "n_rows": int(len(df_out)),
            "median_l1_bps": float(
                df_out.loc[df_out["is_valid"], "full_spread_l1_bps"].median()
            ),
            "pct_std_valid": float(df_out["spread_std_1d"].notna().mean() * 100),
        }
        del df_out
        gc.collect()
        return stats

    except Exception as exc:
        return {"ticker": ticker, "ok": False, "error": str(exc)}


def run_pass2(tickers: list[str]) -> list[dict]:
    """Parallel per-ticker compute. Returns list of stats dicts."""
    TMP_OUT.mkdir(parents=True, exist_ok=True)
    results = []
    done = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_ticker_worker, t): t for t in tickers}
        for future in as_completed(futures):
            stats = future.result()
            results.append(stats)
            done += 1
            if not stats["ok"]:
                print(f"  [WARN] {stats['ticker']}: {stats.get('error','unknown error')}")
            if done % 50 == 0 or done == len(tickers):
                elapsed = time.time() - t_start
                eta = elapsed / done * (len(tickers) - done)
                print(f"  Pass 2 | {done}/{len(tickers)} tickers | "
                      f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

    return results


# ── Assembly ────────────────────────────────────────────────────────────────

def assemble_outputs() -> dict:
    """
    Streams per-ticker output parquets one at a time into 4 final parquets.
    Uses pyarrow ParquetWriter to avoid loading all 526 tickers into RAM at once.
    Peak memory: ~1 ticker at a time (~14 MB) instead of all 526 (~7.5 GB).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq_writer

    MICRO_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    out_files = sorted(TMP_OUT.glob("*.parquet"))
    if not out_files:
        raise RuntimeError("No output temp files found — Pass 2 may have failed.")

    print(f"  Assembly: streaming {len(out_files)} ticker files -> 4 parquets ...")

    roll_cols = ["timestamp_et", "ticker", "spread_std_1d", "raw_spread_mean_1d"]

    # Read schema from first file
    schema_full  = pq_writer.read_schema(out_files[0])
    schema_roll  = pa.schema([schema_full.field(c) for c in roll_cols])

    p_full = MICRO_DIR / "spreads_1min.parquet"
    p_roll = MICRO_DIR / "spread_rolling.parquet"

    total_rows = 0
    n_tickers  = 0
    seas_chunks: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    with pq_writer.ParquetWriter(str(p_full), schema_full, compression="snappy") as w_full, \
         pq_writer.ParquetWriter(str(p_roll), schema_roll, compression="snappy") as w_roll:

        for fp in out_files:
            df = pd.read_parquet(fp)
            total_rows += len(df)
            n_tickers  += 1

            # spreads_1min + spread_rolling (streamed, no global sort needed —
            # Plan 2 hooks set_index and sort per-ticker at load time)
            tbl = pa.Table.from_pandas(df, schema=schema_full, preserve_index=False)
            w_full.write_table(tbl)
            w_roll.write_table(tbl.select(roll_cols))

            # seasonality: accumulate formation-window valid rows (small per ticker)
            form_valid = df[
                df["is_valid"]
                & (df["timestamp_et"].dt.tz_convert("UTC") <= FORMATION_END)
            ]
            if not form_valid.empty:
                seas_chunks.append(form_valid[["ticker", "timestamp_et",
                                               "full_spread_l1_bps", "is_valid"]].copy())

            # summary stats per ticker
            valid = df[df["is_valid"]]["full_spread_l1_bps"]
            if not valid.empty:
                summary_rows.append({
                    "ticker":     df["ticker"].iloc[0],
                    "n_obs":      int(len(valid)),
                    "mean_bps":   float(valid.mean()),
                    "median_bps": float(valid.median()),
                    "p95_bps":    float(valid.quantile(0.95)),
                    "p99_bps":    float(valid.quantile(0.99)),
                    "std_bps":    float(valid.std()),
                })

            del df, tbl
            if n_tickers % 100 == 0:
                gc.collect()
                print(f"    {n_tickers}/{len(out_files)} tickers streamed ...")

    print(f"  Wrote {p_full.name}: {os.path.getsize(p_full)/1e6:.0f} MB")
    print(f"  Wrote {p_roll.name}: {os.path.getsize(p_roll)/1e6:.0f} MB")

    # ── spread_seasonality ────────────────────────────────────────────────
    gc.collect()
    df_form = pd.concat(seas_chunks, ignore_index=True)
    del seas_chunks
    df_seas = compute_intraday_seasonality(df_form)
    del df_form
    p = MICRO_DIR / "spread_seasonality.parquet"
    df_seas.to_parquet(p, compression="snappy", index=False)
    print(f"  Wrote {p.name}: {os.path.getsize(p)/1e6:.1f} MB ({len(df_seas):,} rows)")

    # ── spread_summary ────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    p = MICRO_DIR / "spread_summary.parquet"
    df_summary.to_parquet(p, compression="snappy", index=False)
    print(f"  Wrote {p.name}: {os.path.getsize(p)/1e6:.1f} MB ({len(df_summary)} tickers)")

    elapsed = time.time() - t_start
    print(f"  Assembly done in {elapsed:.1f}s")

    return {
        "total_rows": total_rows,
        "n_tickers":  n_tickers,
        "summary":    df_summary,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def run_plan0_real(
    cleanup_tmp: bool = True,
    max_workers: int = MAX_WORKERS,
) -> dict:
    global MAX_WORKERS
    MAX_WORKERS = max_workers

    t_wall = time.time()
    print("=" * 65)
    print("Plan 0 — Real Data Runner")
    print(f"  Orderbook: {ORDERBOOK} ({ORDERBOOK.stat().st_size/1e9:.2f} GB)")
    print(f"  Workers (Pass 2): {MAX_WORKERS}")
    print("=" * 65)

    # Clean stale temp dirs from prior runs
    for d in (TMP_TICKER, TMP_OUT):
        if d.exists():
            shutil.rmtree(d)

    # ── Pass 1 ──────────────────────────────────────────────────────────
    print("\n[Pass 1] Single-scan: session filter + features + per-ticker flush")
    pf = pq.ParquetFile(str(ORDERBOOK))
    tickers = run_pass1(pf)
    del pf

    # ── Pass 2 ──────────────────────────────────────────────────────────
    print(f"\n[Pass 2] Parallel per-ticker: seasonality + rolling ({len(tickers)} tickers)")
    stats_list = run_pass2(tickers)

    failed = [s for s in stats_list if not s["ok"]]
    ok_stats = [s for s in stats_list if s["ok"]]
    if failed:
        print(f"\n  WARNING: {len(failed)} tickers failed:")
        for s in failed[:10]:
            print(f"    {s['ticker']}: {s.get('error')}")

    # ── Assembly ────────────────────────────────────────────────────────
    print("\n[Assembly] Concatenating outputs -> 4 final parquets")
    summary = assemble_outputs()

    # ── Cleanup ─────────────────────────────────────────────────────────
    if cleanup_tmp:
        shutil.rmtree(TMP_TICKER, ignore_errors=True)
        shutil.rmtree(TMP_OUT,    ignore_errors=True)
        print("  Temp dirs cleaned up.")

    # ── Final report ────────────────────────────────────────────────────
    total_elapsed = time.time() - t_wall
    spread_df = summary["summary"].sort_values("median_bps")

    print("\n" + "=" * 65)
    print(f"Plan 0 Complete — {total_elapsed/60:.1f} min total")
    print(f"  Total rows in spreads_1min.parquet: {summary['total_rows']:,}")
    print(f"  Tickers processed: {summary['n_tickers']} ok / {len(failed)} failed")
    print()
    print("── Spread Summary (tightest / widest 10) ──")
    pd.set_option("display.float_format", "{:.2f}".format)
    combined = pd.concat([
        spread_df.head(10),
        pd.DataFrame([{"ticker": "...", "n_obs": "...", "mean_bps": "...",
                       "median_bps": "...", "p95_bps": "...", "p99_bps": "...",
                       "std_bps": "..."}]),
        spread_df.tail(10),
    ], ignore_index=True)
    print(combined.to_string(index=False))
    print("=" * 65)

    return {
        "tickers": tickers,
        "ok_count": len(ok_stats),
        "failed_count": len(failed),
        "failed_tickers": [s["ticker"] for s in failed],
        "total_rows": summary["total_rows"],
        "elapsed_s": total_elapsed,
        "per_ticker_stats": {s["ticker"]: s for s in ok_stats},
    }


if __name__ == "__main__":
    result = run_plan0_real()
