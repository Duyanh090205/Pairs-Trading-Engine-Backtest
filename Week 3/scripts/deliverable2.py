"""
Deliverable 2 - Verified Backtest Engine
CP4: Data pipeline | CP5: Signal engine + sizing (Version A + B)
CP6a-e: Benchmarking and sensitivity sweeps | CP7: Final log

All cost scenarios run in a single pass:
  signals + sizings are computed ONCE (cost-independent phase)
  PnL is computed per cost scenario (cheap inner loop)
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from math import sqrt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_and_verify_clean_data
from engine import (compute_spread, generate_signals,
                    sizing_version_a, sizing_version_b, compute_daily_pnl)

_W2_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Week 2", "week2_signal_engine")
)
if _W2_ROOT not in sys.path:
    sys.path.insert(0, _W2_ROOT)

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FLAWED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "flawed")
LOG_DIR    = os.path.join(os.path.dirname(__file__), "..", "logs")

EASTERN = "America/New_York"

# ---------------------------------------------------------------------------
# Verified parameters (from verify_params.py - ALL CHECKS PASSED)
# ---------------------------------------------------------------------------
CMS_DUK_ALPHA     = -0.695816
CMS_DUK_BETA      =  1.048794
CMS_DUK_WINDOW    =  677

DOW_LYB_ALPHA     = -0.905720
DOW_LYB_BETA      =  1.088981
DOW_LYB_WINDOW    =  774

GOOG_GOOGL_ALPHA  =  0.102777
GOOG_GOOGL_BETA   =  0.987187
GOOG_GOOGL_WINDOW =  19

MONTHLY_BETA_MAP = {
     7: 0.764627,   # Jun-end  -> Jul trades
     8: 0.773766,   # Jul-end  -> Aug trades
     9: 0.754115,   # Aug-end  -> Sep trades
    10: 0.757557,   # Sep-end  -> Oct trades
    11: 0.787362,   # Oct-end  -> Nov trades
    12: 0.778809,   # Nov-end  -> Dec trades
}

PAIR_CONFIG = {
    "cms_duk": {
        "leg1": "CMS", "leg2": "DUK",
        "alpha": CMS_DUK_ALPHA, "beta": CMS_DUK_BETA, "window": CMS_DUK_WINDOW,
    },
    "dow_lyb": {
        "leg1": "DOW", "leg2": "LYB",
        "alpha": DOW_LYB_ALPHA, "beta": DOW_LYB_BETA, "window": DOW_LYB_WINDOW,
    },
    "goog_googl": {
        "leg1": "GOOG", "leg2": "GOOGL",
        "alpha": GOOG_GOOGL_ALPHA, "beta": GOOG_GOOGL_BETA, "window": GOOG_GOOGL_WINDOW,
    },
}

K_VALUES      = [5, 10, 20, 30, 40, 50]
METHODS       = ["h1", "h2", "h3", "h4"]
FORMATION_END = "2022-06-30"
TRADING_START = "2022-07-01"
TRADING_END   = "2022-12-30"

# ---------------------------------------------------------------------------
# Cost scenarios — add/remove rows freely
# (entry_bps, exit_bps, label)
# ---------------------------------------------------------------------------
COST_SCENARIOS = [
    (30, 30, "30+30 bps  conservative"),
    (10, 10, "10+10 bps  realistic   "),
    ( 2,  2, " 2+2  bps  low-cost    "),
    ( 0,  0, " 0+0  bps  no-cost     "),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_pair(df_all, t1, t2, start=None, end=None):
    d1 = df_all[df_all["ticker"] == t1][["close"]]
    d2 = df_all[df_all["ticker"] == t2][["close"]]
    if start:
        d1 = d1.loc[start:]
        d2 = d2.loc[start:]
    if end:
        d1 = d1.loc[:end]
        d2 = d2.loc[:end]
    return d1, d2


def precompute(leg1, leg2, cfg, use_lag=True, z_threshold=2.0,
               monthly_beta_map=None, spread_input=None):
    """Signal + sizing — cost-independent. Run once per (data, params) combination."""
    if leg1.index.duplicated().any():
        leg1 = leg1.groupby(level=0).last()
    if leg2.index.duplicated().any():
        leg2 = leg2.groupby(level=0).last()
    raw = (spread_input if spread_input is not None
           else compute_spread(leg1, leg2, cfg["alpha"], cfg["beta"]))
    sig = generate_signals(raw, window=cfg["window"],
                           z_threshold=z_threshold, warmup=30, use_lag=use_lag)
    sig["close_a"] = leg1["close"].reindex(sig.index)
    sig["close_b"] = leg2["close"].reindex(sig.index)
    sized_A = sizing_version_a(sig.copy(), beta=cfg["beta"])
    sized_B = (sizing_version_b(sig.copy(), monthly_beta_map)
               if monthly_beta_map is not None else None)
    return sized_A, sized_B


def pnl_metrics(sized, entry_bps, exit_bps):
    """PnL + metrics — cheap, runs once per cost scenario."""
    daily, trades = compute_daily_pnl(sized, entry_bps=entry_bps, exit_bps=exit_bps)
    return daily, trades, compute_metrics(daily, trades)


def compute_metrics(daily, trades):
    mu    = daily["daily_return"].mean()
    sigma = daily["daily_return"].std(ddof=1)
    sr    = (mu / sigma * sqrt(252)) if sigma > 0 else 0.0

    equity      = daily["equity"]
    rolling_max = equity.cummax()
    max_dd      = (equity / rolling_max - 1).min()
    n_days      = len(daily)
    last_eq     = equity.iloc[-1]
    cagr        = (last_eq ** (252.0 / n_days) - 1
                   if n_days > 0 and last_eq > 0 else 0.0)
    calmar      = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    if len(trades) > 0:
        win_rate      = float((trades["net_pnl_bps"] > 0).mean())
        avg_gross_bps = float(trades["gross_pnl_bps"].mean())
        n_trades      = len(trades)
    else:
        win_rate = avg_gross_bps = 0.0
        n_trades = 0

    return dict(sharpe=sr, max_dd=max_dd, cagr=cagr, calmar=calmar,
                win_rate=win_rate, avg_gross_bps=avg_gross_bps, n_trades=n_trades)


def ols_fit(log_a, log_b):
    X = np.column_stack([np.ones(len(log_b)), log_b])
    coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return float(coef[0]), float(coef[1])


def _half_life_ar1(spread):
    """OU half-life via AR(1): ΔS = a + λ*S_{t-1} + ε  →  hl = -ln(2)/λ."""
    ds    = spread.diff().dropna()
    s_lag = spread.shift(1).dropna()
    ds, s_lag = ds.align(s_lag, join="inner")
    X = np.column_stack([np.ones(len(s_lag)), s_lag.values])
    coef, *_ = np.linalg.lstsq(X, ds.values, rcond=None)
    lam = float(coef[1])
    return float(-np.log(2) / lam) if lam < 0 else np.inf


def _window_from_half_life(hl, min_w=10, max_w=2000, default_w=60):
    """Convert half-life (bars) to rolling window with guardrails."""
    if np.isnan(hl) or np.isinf(hl) or hl <= 0:
        return default_w
    return int(np.clip(round(hl), min_w, max_w))


def load_flawed(fname):
    fpath = os.path.join(FLAWED_DIR, fname)
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ---------------------------------------------------------------------------
# CP6f helpers - data sanity checks (detect flawed files from the data itself)
# ---------------------------------------------------------------------------

EXPECTED_COLS = {"ticker", "window_start", "open", "close", "high", "low",
                 "volume", "transactions"}

def _check_ohlc(df):
    v = ((df["high"] < df["close"]) | (df["high"] < df["open"])
         | (df["low"]  > df["close"]) | (df["low"]  > df["open"]))
    n = int(v.sum())
    return (n == 0), n

def _check_ts_duplicates(df):
    n = int(df.reset_index().duplicated(subset=["ticker", "ts_utc"]).sum())
    return (n == 0), n

def _check_ts_session(df):
    ts_et = df.index.tz_convert(EASTERN)
    mins = ts_et.hour * 60 + ts_et.minute
    n = int(((mins < 570) | (mins > 959)).sum())
    return (n == 0), n

def _check_price_plausible(df):
    n_nonpos = int((df["close"] <= 0).sum())
    if n_nonpos > 0:
        return False, n_nonpos
    med = df.groupby("ticker")["close"].transform("median")
    ratio = df["close"] / med
    n = int(((ratio < 0.1) | (ratio > 10.0)).sum())
    return (n == 0), n

def _check_bar_cadence(df):
    s = df.reset_index()[["ticker", "ts_utc"]].copy()
    s["day"] = s["ts_utc"].dt.tz_convert(EASTERN).dt.date
    s = s.sort_values(["ticker", "ts_utc"])
    gap = s.groupby(["ticker", "day"])["ts_utc"].diff().dt.total_seconds()
    n = int(((gap.notna()) & (gap != 60.0)).sum())
    return (n == 0), n

def _check_schema(df):
    cols = set(df.reset_index().columns) - {"ts_utc", "ts_et"}
    extra   = cols - EXPECTED_COLS
    missing = EXPECTED_COLS - cols
    return (not extra and not missing), (sorted(extra), sorted(missing))

def run_data_sanity_audit(clean_df, flawed_dir):
    """Run 4 sanity checks on clean data + all flawed files.
    Returns a list of per-file result dicts. Does not halt or modify state.
    Checks: OHLC consistency, timestamp no-duplicates, session conformity, schema.
    (Price/Cadence excluded — fail on clean data due to real market events.)"""
    results = []
    files = [("clean", clean_df)]
    for method in METHODS:
        for k in K_VALUES:
            fname = f"flawed_{method}_k{k}.csv"
            try:
                df_f = pd.read_csv(os.path.join(flawed_dir, fname),
                                   index_col=0, parse_dates=True)
                if df_f.index.tz is None:
                    df_f.index = df_f.index.tz_localize("UTC")
                df_f.index.name = "ts_utc"
                files.append((fname.replace(".csv", ""), df_f))
            except Exception as exc:
                results.append({"file": fname.replace(".csv", ""),
                                "error": str(exc)})

    for label, df in files:
        row = {"file": label}
        try:
            row["ohlc"],    row["ohlc_n"]     = _check_ohlc(df)
            row["ts_dup"],  row["ts_dup_n"]   = _check_ts_duplicates(df)
            row["ts_sess"], row["ts_sess_n"]  = _check_ts_session(df)
            row["schema"],  row["schema_det"] = _check_schema(df)
            row["all_pass"] = all(row[k] for k in
                ["ohlc", "ts_dup", "ts_sess", "schema"])
        except Exception as exc:
            row["error"] = str(exc)
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_lines = []

    def P(line=""):
        print(line)
        log_lines.append(str(line))

    t0 = time.time()
    P("=" * 70)
    P("  DELIVERABLE 2 - VERIFIED BACKTEST ENGINE")
    P(f"  Cost scenarios : {len(COST_SCENARIOS)} "
      f"(signals precomputed once, PnL looped per scenario)")
    P("=" * 70)

    # ── CP4: Load clean data ──────────────────────────────────────
    P("\n--- CP4: Data Pipeline ---")
    df_clean, summary = load_and_verify_clean_data(RAW_DIR)
    P(f"  Tickers     : {summary['tickers']}")
    P(f"  Total bars  : {summary['total_bars']:,}")
    P(f"  Date range  : {summary['date_range'][0]} to {summary['date_range'][1]}")

    trading = {}
    for pname, cfg in PAIR_CONFIG.items():
        l1, l2 = extract_pair(df_clean, cfg["leg1"], cfg["leg2"],
                               start=TRADING_START, end=TRADING_END)
        trading[pname] = (l1, l2)

    # ── CP5: Clean backtest Version A + B (CMS/DUK) ──────────────
    P("\n--- CP5: Clean Backtest (CMS/DUK) ---")
    cfg_cd       = PAIR_CONFIG["cms_duk"]
    l1_cd, l2_cd = trading["cms_duk"]

    sized_A_cd, sized_B_cd = precompute(l1_cd, l2_cd, cfg_cd,
                                         monthly_beta_map=MONTHLY_BETA_MAP)

    # Timestamp check is cost-independent — use first scenario
    _, trades_ref, _ = pnl_metrics(sized_A_cd, *COST_SCENARIOS[0][:2])
    ts_violations = sum(
        1 for _, row in trades_ref.iterrows()
        if pd.notna(row.get("signal_ts")) and pd.notna(row.get("exec_ts"))
        and row["exec_ts"] <= row["signal_ts"]
    )
    ts_pass = ts_violations == 0
    P(f"  Timestamp check: {'PASS' if ts_pass else 'FAIL'}"
      f"  ({len(trades_ref)} trades verified, {ts_violations} violations)")

    P(f"\n  {'Cost scenario':<26} {'RT':>4}  "
      f"{'Sharpe A':>10} {'Sharpe B':>10} {'Trades':>7} "
      f"{'MaxDD A':>8} {'WinRate A':>9}")
    P("  " + "-" * 80)

    # cp5_results[(entry_bps, exit_bps)] = (mA, mB, trades_A, daily_A)
    cp5_results = {}
    for entry_bps, exit_bps, label in COST_SCENARIOS:
        rt = entry_bps + exit_bps
        daily_A, trades_A, mA = pnl_metrics(sized_A_cd, entry_bps, exit_bps)
        daily_B, trades_B, mB = pnl_metrics(sized_B_cd, entry_bps, exit_bps)
        P(f"  {label:<26} {rt:>4}  "
          f"{mA['sharpe']:>+10.4f} {mB['sharpe']:>+10.4f} {mA['n_trades']:>7}"
          f"  {mA['max_dd']:>7.4f} {mA['win_rate']:>8.1%}")
        cp5_results[(entry_bps, exit_bps)] = (mA, mB, trades_A, daily_A)

    # ── CP6a: Sharpe sensitivity sweep ───────────────────────────
    P("\n--- CP6a: Sharpe Sensitivity Sweep ---")
    # sharpe_tbls[(entry_bps, exit_bps)][pname][(method, k)] = sharpe
    sharpe_tbls = {(e, x): {pname: {} for pname in PAIR_CONFIG}
                   for e, x, _ in COST_SCENARIOS}

    # CMS/DUK clean baselines reused from CP5
    for entry_bps, exit_bps, _ in COST_SCENARIOS:
        sharpe_tbls[(entry_bps, exit_bps)]["cms_duk"][("clean", 0)] = \
            cp5_results[(entry_bps, exit_bps)][0]["sharpe"]

    # Other pairs: precompute signals once, loop costs
    for pname, cfg in PAIR_CONFIG.items():
        if pname == "cms_duk":
            continue
        l1, l2 = trading[pname]
        sized_clean, _ = precompute(l1, l2, cfg)
        for entry_bps, exit_bps, _ in COST_SCENARIOS:
            _, _, m = pnl_metrics(sized_clean, entry_bps, exit_bps)
            sharpe_tbls[(entry_bps, exit_bps)][pname][("clean", 0)] = m["sharpe"]

    for entry_bps, exit_bps, label in COST_SCENARIOS:
        tbl = sharpe_tbls[(entry_bps, exit_bps)]
        P(f"  {label.strip()} clean baselines:"
          f"  cms_duk={tbl['cms_duk'][('clean',0)]:+.4f}"
          f"  dow_lyb={tbl['dow_lyb'][('clean',0)]:+.4f}"
          f"  goog_googl={tbl['goog_googl'][('clean',0)]:+.4f}")

    n_done, n_total = 0, len(METHODS) * len(K_VALUES)
    for method in METHODS:
        for k in K_VALUES:
            fname = f"flawed_{method}_k{k}.csv"
            df_f  = load_flawed(fname)
            for pname, cfg in PAIR_CONFIG.items():
                leg1 = df_f[df_f["ticker"] == cfg["leg1"]][["close"]].loc[TRADING_START:TRADING_END]
                leg2 = df_f[df_f["ticker"] == cfg["leg2"]][["close"]].loc[TRADING_START:TRADING_END]
                sp_in = None
                if method == "h3" and pname == "cms_duk" and "spread_biased" in df_f.columns:
                    sp_in = df_f[df_f["ticker"] == "CMS"][["spread_biased"]].loc[TRADING_START:TRADING_END]

                try:
                    sized_A, _ = precompute(leg1, leg2, cfg, spread_input=sp_in)
                    for entry_bps, exit_bps, _ in COST_SCENARIOS:
                        _, _, m = pnl_metrics(sized_A, entry_bps, exit_bps)
                        sharpe_tbls[(entry_bps, exit_bps)][pname][(method, k)] = m["sharpe"]
                except Exception as exc:
                    for entry_bps, exit_bps, _ in COST_SCENARIOS:
                        sharpe_tbls[(entry_bps, exit_bps)][pname][(method, k)] = float("nan")
                    P(f"  [WARN] {pname} {method} k={k}: {exc}")

            n_done += 1
            if n_done % 5 == 0:
                P(f"  Progress: {n_done}/{n_total} file-method combinations")

    P("")
    for entry_bps, exit_bps, cost_label in COST_SCENARIOS:
        rt  = entry_bps + exit_bps
        tbl = sharpe_tbls[(entry_bps, exit_bps)]
        P(f"  === {cost_label.strip()} ({rt}bps RT) ===")
        for pname, cfg in PAIR_CONFIG.items():
            clean_sr = tbl[pname][("clean", 0)]
            P(f"  {pname.upper()} ({cfg['leg1']}/{cfg['leg2']}) -- Clean: {clean_sr:+.4f}")
            hdr = "  " + f"{'Method':<6}" + "".join(f" {'k='+str(k)+'%':>8}" for k in K_VALUES)
            P(hdr)
            P("  " + "-" * (6 + 9 * len(K_VALUES)))
            for method in METHODS:
                vals = [tbl[pname].get((method, k), float("nan")) for k in K_VALUES]
                row  = f"  {method.upper():<6}"
                for v in vals:
                    row += f" {v:>+8.3f}"
                P(row)
            P("")

    # ── CP6b: Engine-level bias comparison ───────────────────────
    P("--- CP6b: Execution Lag Comparison (CMS/DUK) ---")
    variants = [
        ("Clean engine + clean data",  True,  None, None),
        ("Clean engine + H1 k=50%",    True,  "h1", 50),
        ("Clean engine + H3 k=50%",    True,  "h3", 50),
        ("Biased engine + clean data", False, None, None),
        ("Biased engine + H1 k=50%",   False, "h1", 50),
    ]
    cost_hdrs = "  ".join(f"{'Shrp('+lbl.strip()+')':>20}" for _, _, lbl in COST_SCENARIOS)
    P(f"  {'lag':>3}  {'Variant':<35}  {cost_hdrs}  {'Trades':>6}")
    P("  " + "-" * (45 + 22 * len(COST_SCENARIOS)))

    cp6b_rows = []  # (label, use_lag, {(e,x): (sr, n_tr)})
    for label, use_lag, method, k in variants:
        if method is None:
            vl1, vl2, sp_in = l1_cd, l2_cd, None
        else:
            df_f = load_flawed(f"flawed_{method}_k{k}.csv")
            vl1  = df_f[df_f["ticker"] == "CMS"][["close"]].loc[TRADING_START:TRADING_END]
            vl2  = df_f[df_f["ticker"] == "DUK"][["close"]].loc[TRADING_START:TRADING_END]
            sp_in = (df_f[df_f["ticker"] == "CMS"][["spread_biased"]].loc[TRADING_START:TRADING_END]
                     if method == "h3" and "spread_biased" in df_f.columns else None)

        sized_v, _ = precompute(vl1, vl2, cfg_cd, use_lag=use_lag, spread_input=sp_in)
        cost_res = {}
        sharpe_strs = []
        n_tr_ref = 0
        for entry_bps, exit_bps, _ in COST_SCENARIOS:
            _, tr_v, mv = pnl_metrics(sized_v, entry_bps, exit_bps)
            cost_res[(entry_bps, exit_bps)] = (mv["sharpe"], len(tr_v))
            sharpe_strs.append(f"{mv['sharpe']:>+20.4f}")
            n_tr_ref = len(tr_v)
        lag_s = "Y" if use_lag else "N"
        P(f"  {lag_s:>3}  {label:<35}  {'  '.join(sharpe_strs)}  {n_tr_ref:>6}")
        cp6b_rows.append((label, use_lag, cost_res))

    # ── CP6c: Version A vs B ──────────────────────────────────────
    P("\n--- CP6c: Version A vs Version B (CMS/DUK clean) ---")
    metric_labels = [
        ("sharpe",        "Sharpe"),
        ("max_dd",        "Max Drawdown"),
        ("cagr",          "CAGR"),
        ("calmar",        "Calmar"),
        ("win_rate",      "Win Rate"),
        ("avg_gross_bps", "Avg Gross (bps)"),
        ("n_trades",      "N Trades"),
    ]
    cp6c_results = {}  # {(e,x): [(label, va, vb, delta)]}
    for entry_bps, exit_bps, cost_label in COST_SCENARIOS:
        rt  = entry_bps + exit_bps
        mA_ = cp5_results[(entry_bps, exit_bps)][0]
        mB_ = cp5_results[(entry_bps, exit_bps)][1]
        P(f"\n  [{cost_label.strip()} | {rt}bps RT]")
        P(f"  {'Metric':<22} {'Ver A':>10} {'Ver B':>10} {'Delta':>10}")
        P("  " + "-" * 54)
        rows = []
        for key, label in metric_labels:
            va, vb = mA_[key], mB_[key]
            delta  = vb - va
            if isinstance(va, int):
                P(f"  {label:<22} {va:>10d} {vb:>10d} {delta:>+10d}")
            else:
                P(f"  {label:<22} {va:>10.4f} {vb:>10.4f} {delta:>+10.4f}")
            rows.append((label, va, vb, delta))
        cp6c_results[(entry_bps, exit_bps)] = rows

    # ── CP6d: Z-threshold sweep ───────────────────────────────────
    P("\n--- CP6d: Z-threshold Sensitivity (CMS/DUK clean) ---")
    cost_hdrs = "  ".join(f"{'Shrp('+lbl.strip()+')':>22}" for _, _, lbl in COST_SCENARIOS)
    P(f"  {'Z':>6}  {'Trades':>6}  {cost_hdrs}")
    P("  " + "-" * (16 + 24 * len(COST_SCENARIOS)))

    cp6d_rows = []
    for z in [2.0, 2.57]:
        sized_z, _ = precompute(l1_cd, l2_cd, cfg_cd, z_threshold=z)
        cost_z, sharpe_strs, n_tr_z = {}, [], 0
        for entry_bps, exit_bps, _ in COST_SCENARIOS:
            _, tr_z, mz = pnl_metrics(sized_z, entry_bps, exit_bps)
            cost_z[(entry_bps, exit_bps)] = (mz["sharpe"], len(tr_z))
            sharpe_strs.append(f"{mz['sharpe']:>+22.4f}")
            n_tr_z = len(tr_z)
        P(f"  {z:>6.2f}  {n_tr_z:>6}  {'  '.join(sharpe_strs)}")
        cp6d_rows.append((z, cost_z))

    # ── CP6e: Negative controls ───────────────────────────────────
    P("\n--- CP6e: Negative Controls (clean data) ---")
    cost_hdrs = "  ".join(f"{'Shrp('+lbl.strip()+')':>22}" for _, _, lbl in COST_SCENARIOS)
    P(f"  {'Pair':<12} {'alpha':>8} {'beta':>8} {'win':>6}  {cost_hdrs}  {'Trades':>6}")
    P("  " + "-" * (38 + 24 * len(COST_SCENARIOS)))

    cp6e_rows = []
    for t1, t2 in [("CVNA", "ISRG"), ("INTC", "JPM")]:
        la_f, lb_f = extract_pair(df_clean, t1, t2, end=FORMATION_END)
        aligned    = la_f.join(lb_f, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aligned) > 10:
            log_a = np.log(np.abs(aligned["close_a"].values) + 1e-9)
            log_b = np.log(np.abs(aligned["close_b"].values) + 1e-9)
            alpha_nc, beta_nc = ols_fit(log_a, log_b)
            spread_f = (np.log(np.abs(aligned["close_a"]) + 1e-9)
                        - alpha_nc
                        - beta_nc * np.log(np.abs(aligned["close_b"]) + 1e-9))
            window_nc = _window_from_half_life(_half_life_ar1(spread_f))
        else:
            alpha_nc, beta_nc, window_nc = 0.0, 1.0, 680
        la_t, lb_t  = extract_pair(df_clean, t1, t2, start=TRADING_START, end=TRADING_END)
        cfg_nc      = {"leg1": t1, "leg2": t2, "alpha": alpha_nc,
                       "beta": beta_nc, "window": window_nc}
        try:
            sized_nc, _ = precompute(la_t, lb_t, cfg_nc)
            cost_nc, sharpe_strs, n_tr_nc = {}, [], 0
            for entry_bps, exit_bps, _ in COST_SCENARIOS:
                _, tr_nc, mnc = pnl_metrics(sized_nc, entry_bps, exit_bps)
                cost_nc[(entry_bps, exit_bps)] = (mnc["sharpe"], len(tr_nc))
                sharpe_strs.append(f"{mnc['sharpe']:>+22.4f}")
                n_tr_nc = len(tr_nc)
            P(f"  {t1}/{t2:<9} {alpha_nc:>8.4f} {beta_nc:>8.4f} w={window_nc:<4d}  "
              f"{'  '.join(sharpe_strs)}  {n_tr_nc:>6}")
            cp6e_rows.append((f"{t1}/{t2}", alpha_nc, beta_nc, window_nc, cost_nc))
        except Exception as exc:
            P(f"  {t1}/{t2}  ERROR: {exc}")
            cp6e_rows.append((f"{t1}/{t2}", alpha_nc, beta_nc, window_nc, {}))

    # ── CP6f: Data Sanity Audit ───────────────────────────────────
    P("\n--- CP6f: Data Sanity Audit (assertion-level detection) ---")
    t_sanity = time.time()
    sanity_results = run_data_sanity_audit(df_clean, FLAWED_DIR)
    P(f"  (scan time: {time.time()-t_sanity:.1f}s)")

    hdr = (f"  {'File':<22} {'OHLC':>6} {'TSdup':>6} {'TSsess':>7} "
           f"{'Schema':>7}   Verdict")
    P(hdr)
    P("  " + "-" * (len(hdr) - 2))
    n_flawed = 0
    n_caught = 0
    fmt_cell = lambda b: "  ok  " if b else " FAIL "
    for r in sanity_results:
        if "error" in r:
            P(f"  {r['file']:<22}  ERROR: {r['error']}")
            continue
        if r["file"] == "clean":
            verdict = "[clean]"
        elif not r["all_pass"]:
            verdict = "FLAGGED"
        else:
            verdict = "!! MISSED !!"
        P(f"  {r['file']:<22} {fmt_cell(r['ohlc']):>6}"
          f" {fmt_cell(r['ts_dup']):>6}"
          f" {fmt_cell(r['ts_sess']):>7}"
          f" {fmt_cell(r['schema']):>7}   {verdict}")
        if r["file"] != "clean":
            n_flawed += 1
            if not r["all_pass"]:
                n_caught += 1

    P(f"\n  Detection coverage: {n_caught}/{n_flawed} flawed files caught "
      f"by at least one check.")
    P("  Note: H3 detection relies on the extra 'spread_biased' column (schema "
      "check). A smarter")
    P("        in-place H3 injection could evade this check — raw OHLC stays "
      "valid in that case.")

    # ── Red flag summary ──────────────────────────────────────────
    P("\n--- Red Flag Summary ---")
    red_flags = []
    mA_cons = cp5_results[COST_SCENARIOS[0][:2]][0]
    mB_cons = cp5_results[COST_SCENARIOS[0][:2]][1]
    if mA_cons["sharpe"] > 5.0:
        red_flags.append(f"Version A Sharpe={mA_cons['sharpe']:.3f} > 5.0 on clean data")
    if mB_cons["sharpe"] < mA_cons["sharpe"]:
        red_flags.append(f"Version B ({mB_cons['sharpe']:.3f}) underperforms "
                         f"Version A ({mA_cons['sharpe']:.3f})")
    if not ts_pass:
        red_flags.append(f"{ts_violations} trades with exec_ts <= signal_ts")
    e0, x0, _ = COST_SCENARIOS[0]
    dl_clean = sharpe_tbls[(e0, x0)]["dow_lyb"][("clean", 0)]
    for k in K_VALUES:
        dl_h3 = sharpe_tbls[(e0, x0)]["dow_lyb"].get(("h3", k), float("nan"))
        if not np.isnan(dl_h3) and abs(dl_h3 - dl_clean) > 0.15:
            red_flags.append(f"DOW/LYB H3 k={k}% Sharpe={dl_h3:.3f} != clean={dl_clean:.3f}")

    # Sanity-layer red flags
    if any(r.get("file") == "clean" and not r.get("all_pass", True)
           for r in sanity_results):
        red_flags.append("Clean data failed a sanity check (ohlc/ts_dup/ts_sess/schema)")
    missed = [r["file"] for r in sanity_results
              if r.get("file") != "clean" and "error" not in r and r.get("all_pass")]
    if missed:
        red_flags.append(f"Sanity checks missed {len(missed)} flawed files: "
                         f"{missed[:3]}...")

    if red_flags:
        for rf in red_flags:
            P(f"  !! RED FLAG: {rf}")
    else:
        P("  All red flag checks passed.")

    # ── CP7: Write log ────────────────────────────────────────────
    log_fname = "verified_backtest_log_multi_cost.txt"
    P(f"\n--- CP7: Writing {log_fname} ---")
    log_path = os.path.join(LOG_DIR, log_fname)
    elapsed  = time.time() - t0
    _write_log(log_path, elapsed, summary, trades_ref, sharpe_tbls,
               cp6b_rows, cp6c_results, cp6d_rows, cp6e_rows,
               ts_pass, ts_violations, red_flags, sanity_results)
    P(f"  Written: {log_path}")
    P(f"\n  Total runtime: {elapsed:.1f}s")
    P("=" * 70)
    P("  DELIVERABLE 2 COMPLETE")
    P("=" * 70)


# ---------------------------------------------------------------------------
# CP7 log writer
# ---------------------------------------------------------------------------

def _write_log(path, elapsed, summary, trades_ref, sharpe_tbls,
               cp6b_rows, cp6c_results, cp6d_rows, cp6e_rows,
               ts_pass, ts_violations, red_flags, sanity_results=None):
    import datetime
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    W = lines.append
    def rule(c="=", n=70): W(c * n)
    def blank(): W("")

    rule()
    W("  VERIFIED BACKTEST LOG - Week 3 (Multi-Cost)")
    W(f"  Run: {run_ts}  |  Runtime: {elapsed:.1f}s")
    rule()
    blank()

    # Section 1: Header
    W("SECTION 1: HEADER")
    rule("-", 70)
    W(f"  Formation period : 2022-01-03 to {FORMATION_END}")
    W(f"  Trading period   : {TRADING_START} to {TRADING_END}")
    W(f"  Primary pair     : CMS / DUK")
    W(f"  OLS alpha        : {CMS_DUK_ALPHA}")
    W(f"  OLS beta         : {CMS_DUK_BETA}")
    W(f"  Rolling window   : {CMS_DUK_WINDOW} bars")
    W(f"  Z threshold      : 2.0 (default), 2.57 (sweep)")
    W(f"  Session warmup   : 30 bars/day")
    W(f"  Burn-in          : {CMS_DUK_WINDOW // 2} bars (window//2)")
    W(f"  Execution lag    : 1 bar (position_executed[t] = position[t-1])")
    W(f"  Cost scenarios:")
    for e, x, lbl in COST_SCENARIOS:
        W(f"    {lbl.strip():<26} entry={e}bps  exit={x}bps  RT={e+x}bps")
    W(f"  Kalman monthly beta map:")
    W(f"    Jul: {MONTHLY_BETA_MAP[7]:.6f}  Aug: {MONTHLY_BETA_MAP[8]:.6f}"
      f"  Sep: {MONTHLY_BETA_MAP[9]:.6f}")
    W(f"    Oct: {MONTHLY_BETA_MAP[10]:.6f}  Nov: {MONTHLY_BETA_MAP[11]:.6f}"
      f"  Dec: {MONTHLY_BETA_MAP[12]:.6f}")
    W(f"  Secondary pairs:")
    W(f"    DOW/LYB     alpha={DOW_LYB_ALPHA}  beta={DOW_LYB_BETA}  window={DOW_LYB_WINDOW}")
    W(f"    GOOG/GOOGL  alpha={GOOG_GOOGL_ALPHA}  beta={GOOG_GOOGL_BETA}  window={GOOG_GOOGL_WINDOW}")
    W(f"  Total bars loaded: {summary['total_bars']:,}")
    blank()

    # Section 2: Timestamp Verification
    W("SECTION 2: TIMESTAMP VERIFICATION")
    rule("-", 70)
    W(f"  Total trades verified : {len(trades_ref)}")
    W(f"  Violations (exec_ts <= signal_ts) : {ts_violations}")
    W(f"  Status : {'PASS' if ts_pass else 'FAIL'}")
    W(f"  Note: position_executed[t] = position[t-1] (1-bar lag enforced throughout)")
    blank()

    # Section 3: Trade Log Sample (conservative cost reference)
    W("SECTION 3: TRADE LOG SAMPLE (first 30 trades, Version A CMS/DUK, 30+30bps)")
    rule("-", 70)
    sample = trades_ref.head(30)
    W(f"  {'signal_ts':<25} {'exec_ts':<25} {'side':<6}"
      f" {'gross':>8} {'entry_c':>8} {'exit_c':>8} {'net':>8}")
    W("  " + "-" * 95)
    for _, row in sample.iterrows():
        W(f"  {str(row.get('signal_ts',''))[:25]:<25} "
          f"{str(row.get('exec_ts',''))[:25]:<25} "
          f"{str(row.get('side',''))[:5]:<6}"
          f" {row.get('gross_pnl_bps',0.0):>+8.1f}"
          f" {row.get('entry_cost_bps',0.0):>8.1f}"
          f" {row.get('exit_cost_bps',0.0):>8.1f}"
          f" {row.get('net_pnl_bps',0.0):>+8.1f}")
    if len(trades_ref) > 30:
        W(f"  ... ({len(trades_ref) - 30} more trades not shown)")
    blank()

    # Section 4: Sharpe Sensitivity Tables (one block per cost scenario)
    W("SECTION 4: SHARPE SENSITIVITY TABLE")
    rule("-", 70)
    for entry_bps, exit_bps, cost_label in COST_SCENARIOS:
        rt  = entry_bps + exit_bps
        tbl = sharpe_tbls[(entry_bps, exit_bps)]
        W(f"  --- {cost_label.strip()} ({rt}bps RT) ---")
        for pname, cfg in PAIR_CONFIG.items():
            clean_sr = tbl[pname][("clean", 0)]
            W(f"  {pname.upper()} ({cfg['leg1']}/{cfg['leg2']}) Clean={clean_sr:+.4f}")
            hdr_w = "  " + f"{'Method':<6}" + "".join(f" {'k='+str(k)+'%':>9}" for k in K_VALUES)
            W(hdr_w)
            W("  " + "-" * (6 + 10 * len(K_VALUES)))
            for method in METHODS:
                vals = [tbl[pname].get((method, k), float("nan")) for k in K_VALUES]
                row  = f"  {method.upper():<6}"
                for v in vals:
                    row += f" {v:>+9.3f}"
                W(row)
            blank()
    W("  Note: H3 for DOW/LYB and GOOG/GOOGL is flat — spread_biased column only")
    W("        exists for CMS/DUK rows (pair-specific bias by design).")
    blank()

    # Section 5: Version A vs B per cost scenario
    W("SECTION 5: VERSION A vs VERSION B — CMS/DUK clean, all cost scenarios")
    rule("-", 70)
    for entry_bps, exit_bps, cost_label in COST_SCENARIOS:
        rt = entry_bps + exit_bps
        W(f"  [{cost_label.strip()} | {rt}bps RT]")
        W(f"  {'Metric':<22} {'Version A':>12} {'Version B':>12} {'Delta':>12}")
        W("  " + "-" * 60)
        for label, va, vb, delta in cp6c_results[(entry_bps, exit_bps)]:
            if isinstance(va, int):
                W(f"  {label:<22} {va:>12d} {vb:>12d} {int(delta):>+12d}")
            else:
                W(f"  {label:<22} {va:>12.4f} {vb:>12.4f} {delta:>+12.4f}")
        blank()

    # Section 6: Comparative Analysis
    W("SECTION 6: COMPARATIVE ANALYSIS")
    rule("-", 70)
    blank()

    # 6b
    W("  6b. Engine-level Execution Lag (CMS/DUK):")
    cost_hdrs = "  ".join(f"{'Sharpe('+lbl.strip()+')':>20}" for _, _, lbl in COST_SCENARIOS)
    W(f"  {'lag':>3}  {'Variant':<35}  {cost_hdrs}  {'Trades':>6}")
    W("  " + "-" * (42 + 22 * len(COST_SCENARIOS)))
    for label, use_lag, cost_res in cp6b_rows:
        sharpe_strs = [f"{cost_res.get((e,x),(float('nan'),0))[0]:>+20.4f}"
                       for e, x, _ in COST_SCENARIOS]
        n_tr = next(iter(cost_res.values()))[1] if cost_res else 0
        W(f"  {'Y' if use_lag else 'N':>3}  {label:<35}  {'  '.join(sharpe_strs)}  {n_tr:>6}")
    blank()

    # 6d
    W("  6d. Z-threshold Sensitivity (CMS/DUK clean):")
    cost_hdrs = "  ".join(f"{'Sharpe('+lbl.strip()+')':>22}" for _, _, lbl in COST_SCENARIOS)
    W(f"  {'Z':>6}  {'Trades':>6}  {cost_hdrs}")
    W("  " + "-" * (14 + 24 * len(COST_SCENARIOS)))
    for z, cost_z in cp6d_rows:
        sharpe_strs = [f"{cost_z.get((e,x),(float('nan'),0))[0]:>+22.4f}"
                       for e, x, _ in COST_SCENARIOS]
        n_tr = next(iter(cost_z.values()))[1] if cost_z else 0
        W(f"  {z:>6.2f}  {n_tr:>6}  {'  '.join(sharpe_strs)}")
    blank()

    # 6e
    W("  6e. Negative Controls (clean data, formation OLS):")
    cost_hdrs = "  ".join(f"{'Sharpe('+lbl.strip()+')':>22}" for _, _, lbl in COST_SCENARIOS)
    W(f"  {'Pair':<12} {'alpha':>8} {'beta':>8} {'win':>6}  {cost_hdrs}  {'Trades':>6}")
    W("  " + "-" * (36 + 24 * len(COST_SCENARIOS)))
    for pair_label, alpha_nc, beta_nc, window_nc, cost_nc in cp6e_rows:
        sharpe_strs = [f"{cost_nc.get((e,x),(float('nan'),0))[0]:>+22.4f}"
                       for e, x, _ in COST_SCENARIOS]
        n_tr = next(iter(cost_nc.values()))[1] if cost_nc else 0
        W(f"  {pair_label:<12} {alpha_nc:>8.4f} {beta_nc:>8.4f} {window_nc:>6d}  "
          f"{'  '.join(sharpe_strs)}  {n_tr:>6}")
    blank()

    # Section 6.5: Data Sanity Audit (CP6f)
    W("SECTION 6.5: DATA SANITY AUDIT (CP6f - assertion-level detection)")
    rule("-", 70)
    W("  Detects flawed files from the data itself (no backtest needed).")
    W("  Checks: OHLC consistency | ts duplicates | ts session | strict column schema")
    blank()
    if sanity_results:
        W(f"  {'File':<22} {'OHLC':>6} {'TSdup':>6} {'TSsess':>7} "
          f"{'Schema':>7}   Verdict")
        W("  " + "-" * 64)
        n_flawed = n_caught = 0
        for r in sanity_results:
            if "error" in r:
                W(f"  {r['file']:<22}  ERROR: {r['error']}")
                continue
            if r["file"] == "clean":
                verdict = "[clean]"
            elif not r["all_pass"]:
                verdict = "FLAGGED"
            else:
                verdict = "!! MISSED !!"
            cell = lambda b: "  ok  " if b else " FAIL "
            W(f"  {r['file']:<22} {cell(r['ohlc']):>6}"
              f" {cell(r['ts_dup']):>6}"
              f" {cell(r['ts_sess']):>7}"
              f" {cell(r['schema']):>7}   {verdict}")
            if r["file"] != "clean":
                n_flawed += 1
                if not r["all_pass"]:
                    n_caught += 1
        blank()
        W(f"  Detection coverage: {n_caught}/{n_flawed} flawed files caught "
          f"by at least one check.")
        W("  Per-check failing row counts:")
        for r in sanity_results[:30]:
            if "error" in r or r.get("file") == "clean" or r.get("all_pass", True):
                continue
            fails = []
            for chk in ["ohlc", "ts_dup", "ts_sess"]:
                if not r[chk]:
                    fails.append(f"{chk}={r[chk+'_n']}")
            if not r["schema"]:
                extra, missing = r["schema_det"]
                fails.append(f"schema(extra={extra},missing={missing})")
            W(f"    {r['file']:<22} " + " | ".join(fails))
        blank()
        W("  Caveat: H3 detection depends on the extra 'spread_biased' column")
        W("          (schema check). An in-place H3 injection would evade all 4")
        W("          checks because raw OHLC stays valid. Real H3-type bias")
        W("          (leaky features in derived data) needs re-computation, not")
        W("          data assertions.")
    else:
        W("  (sanity_results not provided)")
    blank()

    # Section 7: Audit Trail
    W("SECTION 7: AUDIT TRAIL")
    rule("-", 70)
    W(f"  Parameter source   : scripts/verify_params.py  (ALL CHECKS PASSED)")
    W(f"  Data source        : data/raw/  ({summary['total_files']} files, "
      f"{summary['total_bars']:,} session-filtered bars)")
    W(f"  Flawed datasets    : data/flawed/  (20 files, generated by deliverable1.py)")
    W(f"  Signal engine      : scripts/engine.py")
    W(f"  Week 2 imports     : src.signals.zscore, state_machine, kalman")
    W(f"  Execution lag      : position_executed[t] = position[t-1] (use_lag=True)")
    W(f"  Exit cost fix      : exit notional = prev_notional (shift(1)) at exit bar")
    W(f"  PnL formula        : bar_pnl = shares_a*close_a.diff() + shares_b*close_b.diff()")
    W(f"  Ref notional       : mean(|shares_a|*pa + |shares_b|*pb) at entry bars")
    W(f"  Sharpe formula     : mean(daily_return) / std(daily_return, ddof=1) * sqrt(252)")
    W(f"  Cost independence  : signals/sizings precomputed once; PnL looped per scenario")
    blank()
    if red_flags:
        W("  RED FLAGS RAISED:")
        for rf in red_flags:
            W(f"    !! {rf}")
    else:
        W("  No red flags raised on any dataset.")
    blank()
    rule()
    W("  END OF VERIFIED BACKTEST LOG")
    rule()

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
