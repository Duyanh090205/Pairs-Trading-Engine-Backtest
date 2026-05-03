"""
run_temporal_gate_test.py

Implements and validates a "temporal stability gate":

Split the 6-month formation window into 3 consecutive 2-month blocks.
For each pair, compute per-block:
  1. Beta drift     -- PCA beta in each block vs overall beta
  2. Spread mean shift -- mean of spread in each block vs overall mean
  3. OU half-life   -- half-life in each block (require all in [1,10]d)
  4. Coint passes   -- how many of 3 blocks pass Johansen (require >=2)

STEP 1: Compute stability scores for all gated pairs, join to trading outcomes,
        run discriminator test (same methodology as gate_filter_validation.py).

STEP 2: If discriminator passes (AUROC > 0.60, p < 0.10), sweep thresholds
        and find best gate config.

STEP 3: Compare on 10-fold ablation: Option A baseline vs Option A + temporal gate.
"""

import sys, os, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

from src.phase4_defense.orchestrator import FOLD_SCHEDULE
from src.utils.stats import compute_ou_halflife
from src.phase2_execution.kalman import warmup_kalman
from src.phase2_execution.engine import run_fold_execution, _N_OPEN_PAIRS_MAX, _TOTAL_CAPITAL
from src.phase3_backtest.metrics_runner import run_fold_pnl

PHASE1_DIR    = "results/metrics/phase1_folds"
DATA_1MIN     = "data/validated/1min_phase2"
DATA_5MIN     = "data/validated/5min_phase1"
JOHANSEN_PVAL = 0.05
FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0

TEST_FOLDS = [
    (1,  "2022-01-03", "2022-06-30", "2022-07", "Bear"),
    (2,  "2022-02-01", "2022-07-31", "2022-08", "Bear"),
    (5,  "2022-05-01", "2022-10-31", "2022-11", "Bear"),
    (6,  "2022-06-01", "2022-11-30", "2022-12", "Bear"),
    (8,  "2022-08-01", "2023-01-31", "2023-02", "Bull"),
    (10, "2022-10-01", "2023-03-31", "2023-04", "Bull"),
    (11, "2022-11-01", "2023-04-30", "2023-05", "Bull"),
    (12, "2022-12-01", "2023-05-31", "2023-06", "Bull"),
    (13, "2023-01-01", "2023-06-30", "2023-07", "Bull"),
    (17, "2023-05-01", "2023-10-31", "2023-11", "Bull"),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_cache(data_dir, compute_log_close=False):
    cache = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".parquet"):
            continue
        tk = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(data_dir, fname))
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache

def _slice(cache, tickers, start, end):
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        sliced = cache[tk][(cache[tk].index >= start) & (cache[tk].index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out

def _slice_month(cache, tickers, month):
    y, m = int(month[:4]), int(month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        sliced = cache[tk][(cache[tk].index.year == y) & (cache[tk].index.month == m)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ---------------------------------------------------------------------------
# Core block-metrics computation
# ---------------------------------------------------------------------------

def _johansen_pval(log_a, log_b):
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan

def _pca_beta(log_a, log_b):
    """PCA secondary eigenvector beta on centered log-prices."""
    X = np.column_stack([log_a - log_a.mean(), log_b - log_b.mean()])
    cov = X.T @ X / max(len(X) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    v2 = vecs[:, idx[1]]   # secondary eigenvector
    if abs(v2[1]) < 1e-10:
        return np.nan
    return -v2[0] / v2[1]

def compute_block_metrics(fa_lc, fb_lc, alpha_pca, beta_pca, form_start, form_end):
    """
    Split [form_start, form_end] into 3 equal 2-month blocks.
    Returns dict with per-block and summary metrics, or None if data insufficient.

    fa_lc, fb_lc: Series of log_close indexed by timestamp.
    alpha_pca, beta_pca: Phase 1 frozen estimates (used for spread).
    """
    ts = pd.Timestamp(form_start)
    te = pd.Timestamp(form_end)
    # 3 blocks of 2 months each
    b1_end = ts + pd.DateOffset(months=2) - pd.Timedelta(days=1)
    b2_end = ts + pd.DateOffset(months=4) - pd.Timedelta(days=1)
    blocks = [
        (str(ts.date()), str(b1_end.date())),
        (str((b1_end + pd.Timedelta(days=1)).date()), str(b2_end.date())),
        (str((b2_end + pd.Timedelta(days=1)).date()), str(te.date())),
    ]

    betas, hl_days, coint_pass, spread_means = [], [], [], []
    overall_spread = []

    for b_start, b_end in blocks:
        a_b = fa_lc[(fa_lc.index >= b_start) & (fa_lc.index <= b_end)].values
        b_b = fb_lc[(fb_lc.index >= b_start) & (fb_lc.index <= b_end)].values
        # align by index
        fa_b = fa_lc[(fa_lc.index >= b_start) & (fa_lc.index <= b_end)]
        fb_b = fb_lc[(fb_lc.index >= b_start) & (fb_lc.index <= b_end)]
        aln = fa_b.rename("a").to_frame().join(fb_b.rename("b"), how="inner").dropna()
        if len(aln) < 40:   # need ~2 weeks minimum
            return None
        la, lb = aln["a"].values, aln["b"].values

        # PCA beta for this block
        beta_b = _pca_beta(la, lb)
        betas.append(beta_b)

        # Spread using FROZEN Phase 1 beta (not block beta)
        spread = la - alpha_pca - beta_pca * lb
        spread_means.append(float(np.mean(spread)))
        overall_spread.extend(spread.tolist())

        # OU half-life on block spread
        try:
            hl = compute_ou_halflife(spread)
        except Exception:
            hl = np.nan
        hl_days.append(hl)

        # Johansen on this block
        pval = _johansen_pval(la, lb)
        coint_pass.append(1 if (not np.isnan(pval) and pval < JOHANSEN_PVAL) else 0)

    if any(np.isnan(b) for b in betas):
        return None

    overall_std = np.std(overall_spread, ddof=1) if len(overall_spread) > 1 else 1.0
    if overall_std < 1e-10:
        return None

    # --- Summary metrics ---
    # 1. Beta max relative drift across 3 blocks vs overall Phase 1 beta
    beta_rel_drifts = [abs(b - beta_pca) / (abs(beta_pca) + 1e-10) for b in betas]
    beta_max_drift = float(max(beta_rel_drifts))
    beta_cv = float(np.std(betas, ddof=1) / (abs(np.mean(betas)) + 1e-10))

    # 2. Spread mean max shift (in sigma) between consecutive blocks
    shifts = [abs(spread_means[i+1] - spread_means[i]) / overall_std for i in range(2)]
    mean_shift_max = float(max(shifts))
    # Also total range (max - min) of block means
    mean_range = float((max(spread_means) - min(spread_means)) / overall_std)

    # 3. OU half-life stats
    valid_hl = [h for h in hl_days if not np.isnan(h) and h > 0]
    hl_all_valid = len(valid_hl) == 3
    hl_all_in_range = all(1.0 <= h <= 10.0 for h in valid_hl) if len(valid_hl) == 3 else False
    hl_cv = float(np.std(valid_hl, ddof=1) / (np.mean(valid_hl) + 1e-10)) if len(valid_hl) >= 2 else np.nan
    hl_max = float(max(valid_hl)) if valid_hl else np.nan

    # 4. Cointegration persistence
    n_coint_pass = int(sum(coint_pass))

    return {
        "betas":           betas,
        "hl_days":         hl_days,
        "coint_pass":      coint_pass,
        "n_coint_pass":    n_coint_pass,
        "beta_max_drift":  beta_max_drift,   # max relative drift of block beta vs Phase 1 beta
        "beta_cv":         beta_cv,          # CV of 3 block betas
        "mean_shift_max":  mean_shift_max,   # max consecutive block mean shift (sigma)
        "mean_range":      mean_range,       # total range of block means (sigma)
        "hl_all_valid":    hl_all_valid,
        "hl_all_in_range": hl_all_in_range,
        "hl_cv":           hl_cv,            # CV of half-lives across 3 blocks
        "hl_max":          hl_max,           # worst (longest) half-life across blocks
    }


# ---------------------------------------------------------------------------
# Persistence gate (original Johansen on last month)
# ---------------------------------------------------------------------------

def apply_persistence_gate(pairs_df, form_5min, form_end):
    if pairs_df.empty:
        return pd.DataFrame()
    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        aln = (
            fa[(fa.index >= gate_start) & (fa.index <= gate_end)][["log_close"]]
            .join(fb[(fb.index >= gate_start) & (fb.index <= gate_end)][["log_close"]],
                  lsuffix="_a", rsuffix="_b", how="inner").dropna()
        )
        if len(aln) < 20:
            continue
        pval = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pval) and pval < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def apply_temporal_gate(pairs_df, form_5min, form_start, form_end,
                        max_beta_drift=0.30,
                        max_mean_shift=1.00,
                        require_coint_blocks=2,
                        require_hl_in_range=True):
    """
    Temporal stability gate: pair must show consistent behaviour across
    all three 2-month blocks of the formation window.

    Rejects if ANY of:
      - beta_max_drift > max_beta_drift       (hedge ratio unstable)
      - mean_shift_max > max_mean_shift       (spread mean drifts)
      - n_coint_pass < require_coint_blocks   (not cointegrated consistently)
      - hl_all_in_range == False              (half-life out of range in some block)
        (only if require_hl_in_range=True)
    """
    if pairs_df.empty:
        return pd.DataFrame(), {}

    survivors = []
    reject_reason = {"beta": 0, "shift": 0, "coint": 0, "hl": 0, "data": 0}

    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = form_5min.get(ta)
        fb = form_5min.get(tb)
        if fa is None or fb is None:
            reject_reason["data"] += 1
            continue

        fa_lc = fa["log_close"] if "log_close" in fa.columns else np.log(fa["close"].clip(lower=1e-10))
        fb_lc = fb["log_close"] if "log_close" in fb.columns else np.log(fb["close"].clip(lower=1e-10))

        m = compute_block_metrics(fa_lc, fb_lc, row["alpha_pca"], row["beta_pca"], form_start, form_end)
        if m is None:
            reject_reason["data"] += 1
            continue

        if m["beta_max_drift"] > max_beta_drift:
            reject_reason["beta"] += 1
            continue
        if m["mean_shift_max"] > max_mean_shift:
            reject_reason["shift"] += 1
            continue
        if m["n_coint_pass"] < require_coint_blocks:
            reject_reason["coint"] += 1
            continue
        if require_hl_in_range and not m["hl_all_in_range"]:
            reject_reason["hl"] += 1
            continue

        survivors.append(row)

    return pd.DataFrame(survivors).reset_index(drop=True), reject_reason


# ===========================================================================
# STEP 1: Discriminator validation
# ===========================================================================

print("Loading caches...")
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_5min)} (5-min)\n")

print("Loading trading outcomes...")
trades_df = pd.read_csv("results/metrics/diagnostic/divergence_trades.csv")
pair_out_rows = []
for (fold, pair), g in trades_df.groupby(["fold", "pair"]):
    pair_out_rows.append({
        "fold": fold, "pair": pair,
        "n_trades": len(g),
        "any_never_rev": int(g["never_rev"].any()),
        "bad": int(g["never_rev"].any() or g["max_adv_pct"].max() > 0.5),
    })
pair_outcomes = pd.DataFrame(pair_out_rows)
print(f"  {len(pair_outcomes)} pair-fold trading outcomes\n")

print("Computing temporal stability scores for all gated pairs...")
score_rows = []

for spec in FOLD_SCHEDULE:
    fold_n        = spec.fold_n
    form_start    = spec.form_start
    form_end      = spec.form_end
    trading_month = spec.trading_month

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if len(pairs_df) > 500:
            pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        continue
    if pairs_df.empty:
        continue

    tickers  = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    form_5min = _slice(cache_5min, tickers, form_start, form_end)

    # Only score pairs that survived the Johansen gate (Option A's existing gate)
    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")

    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = form_5min.get(ta)
        fb = form_5min.get(tb)
        if fa is None or fb is None:
            continue

        if "log_close" not in fa.columns:
            fa = fa.assign(log_close=np.log(fa["close"].clip(lower=1e-10)))
        if "log_close" not in fb.columns:
            fb = fb.assign(log_close=np.log(fb["close"].clip(lower=1e-10)))

        # Check Johansen gate passes
        aln = (
            fa[(fa.index >= gate_start) & (fa.index <= gate_end)][["log_close"]]
            .join(fb[(fb.index >= gate_start) & (fb.index <= gate_end)][["log_close"]],
                  lsuffix="_a", rsuffix="_b", how="inner").dropna()
        )
        if len(aln) < 20:
            continue
        pval = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if np.isnan(pval) or pval >= JOHANSEN_PVAL:
            continue

        fa_lc = fa["log_close"]
        fb_lc = fb["log_close"]
        m = compute_block_metrics(fa_lc, fb_lc, row["alpha_pca"], row["beta_pca"], form_start, form_end)
        if m is None:
            continue

        score_rows.append({
            "fold":           fold_n,
            "trading_month":  trading_month,
            "pair":           f"{ta}/{tb}",
            "regime":         "Bear" if trading_month <= "2022-12" else "Bull",
            "beta_max_drift": m["beta_max_drift"],
            "beta_cv":        m["beta_cv"],
            "mean_shift_max": m["mean_shift_max"],
            "mean_range":     m["mean_range"],
            "n_coint_pass":   m["n_coint_pass"],
            "hl_all_in_range":int(m["hl_all_in_range"]),
            "hl_cv":          m["hl_cv"],
            "hl_max":         m["hl_max"],
        })

scores_df = pd.DataFrame(score_rows)
print(f"  {len(scores_df)} gated pairs scored\n")

# Join to trading outcomes
merged = pd.merge(scores_df, pair_outcomes[["fold", "pair", "n_trades", "any_never_rev", "bad"]],
                  on=["fold", "pair"], how="left")
merged["traded"]      = merged["n_trades"].notna() & (merged["n_trades"] > 0)
merged["any_never_rev"] = merged["any_never_rev"].fillna(0).astype(int)
merged["bad"]         = merged["bad"].fillna(0).astype(int)
traded = merged[merged["traded"]].copy()
print(f"Traded pairs: {len(traded)}  |  Never-rev: {traded['any_never_rev'].sum()} ({traded['any_never_rev'].mean():.1%})")
print()

print("=" * 90)
print("STEP 1: DISCRIMINATOR VALIDATION")
print("=" * 90)
print(f"Base rate (good pairs): {(traded['any_never_rev']==0).mean():.1%}")
print()

metrics_to_test = [
    ("beta_max_drift", "Beta max relative drift (higher=worse)", "above"),
    ("beta_cv",        "Beta CV across 3 blocks (higher=worse)", "above"),
    ("mean_shift_max", "Max consecutive spread mean shift (sigma, higher=worse)", "above"),
    ("mean_range",     "Total spread mean range across blocks (sigma, higher=worse)", "above"),
    ("n_coint_pass",   "Cointegration blocks passing (lower=worse)", "below"),
    ("hl_cv",          "OU half-life CV across blocks (higher=worse)", "above"),
    ("hl_max",         "Worst OU half-life across blocks (higher=worse)", "above"),
    ("hl_all_in_range","All 3 blocks HL in [1,10] (0=bad)", "below"),
]

results_disc = []
for col, label, expected_dir in metrics_to_test:
    valid = traded[traded[col].notna()].copy()
    if len(valid) < 10:
        continue
    good = valid[valid["any_never_rev"] == 0][col]
    bad  = valid[valid["any_never_rev"] == 1][col]
    if len(bad) < 3:
        continue

    u_stat, p_mw = stats.mannwhitneyu(good, bad, alternative="two-sided")
    auroc_raw = u_stat / (len(good) * len(bad))
    # Correct direction: if bad scores higher (expected_dir=above), auroc should be < 0.5
    auroc_corrected = 1 - auroc_raw if auroc_raw < 0.5 else auroc_raw

    signal_ok = auroc_corrected >= 0.60 and p_mw < 0.10
    flag = " <<< PASS" if signal_ok else ""

    print(f"{col:<20}: good={good.median():.3f}  bad={bad.median():.3f}  "
          f"AUROC={auroc_corrected:.3f}  p={p_mw:.4f}{flag}")
    results_disc.append({"metric": col, "auroc": auroc_corrected, "p": p_mw, "pass": signal_ok})

results_disc = pd.DataFrame(results_disc)
passing = results_disc[results_disc["pass"]]
print(f"\nFilters passing discriminator (AUROC>0.60, p<0.10): {len(passing)}/{len(results_disc)}")
if not passing.empty:
    print(passing[["metric","auroc","p"]].to_string(index=False))


# ===========================================================================
# STEP 2: Threshold sweep on passing metrics
# ===========================================================================

print("\n" + "=" * 90)
print("STEP 2: THRESHOLD SWEEP — best combination gate")
print("=" * 90)

# Test a grid of threshold combinations
# Use the 3 most intuitive controls: beta_max_drift, mean_shift_max, n_coint_pass
# Sweep these on all gated pairs (not just traded) to see pair survival rates

combos = []
for beta_t in [0.20, 0.30, 0.50, 9.9]:
    for shift_t in [0.75, 1.00, 1.50, 9.9]:
        for coint_t in [3, 2, 1]:
            for hl_t in [True, False]:
                mask = (
                    (merged["beta_max_drift"] <= beta_t) &
                    (merged["mean_shift_max"] <= shift_t) &
                    (merged["n_coint_pass"]   >= coint_t) &
                    (merged["hl_all_in_range"] >= (1 if hl_t else 0))
                )
                n_kept  = mask.sum()
                n_total = len(merged)
                # Among traded pairs
                t_mask  = mask & merged["traded"]
                t_kept  = t_mask.sum()
                t_good  = int((t_mask & (merged["any_never_rev"]==0)).sum())
                t_bad   = int((t_mask & (merged["any_never_rev"]==1)).sum())
                t_bad_total = int((merged["traded"] & (merged["any_never_rev"]==1)).sum())
                t_good_total = int((merged["traded"] & (merged["any_never_rev"]==0)).sum())

                recall_bad  = t_bad  / max(t_bad_total,  1)   # fraction of bad pairs kept
                keep_good   = t_good / max(t_good_total, 1)   # fraction of good pairs kept
                surv_rate   = n_kept / max(n_total, 1)

                combos.append({
                    "beta_t": beta_t, "shift_t": shift_t, "coint_t": coint_t, "hl_t": hl_t,
                    "surv%": surv_rate, "keep_good%": keep_good, "recall_bad%": recall_bad,
                    "bad_eliminated%": 1 - recall_bad,
                    "score": keep_good - recall_bad,   # maximize good kept minus bad kept
                })

combos_df = pd.DataFrame(combos).sort_values("score", ascending=False)

print(f"{'Beta':>6} {'Shift':>6} {'Coint':>6} {'HL':>5}  {'Surv%':>7} {'KeepGood%':>11} {'RecallBad%':>12} {'BadElim%':>10} {'Score':>7}")
print("-" * 80)
for _, r in combos_df.head(15).iterrows():
    print(f"{r['beta_t']:>6.2f} {r['shift_t']:>6.2f} {int(r['coint_t']):>6}  "
          f"{'Y' if r['hl_t'] else 'N':>5}  "
          f"{r['surv%']:>6.0%} {r['keep_good%']:>10.0%} {r['recall_bad%']:>11.0%} "
          f"{r['bad_eliminated%']:>9.0%} {r['score']:>7.3f}")

best = combos_df.iloc[0]
print(f"\nBest config: beta<={best['beta_t']}, shift<={best['shift_t']}, "
      f"coint>={int(best['coint_t'])}, hl_in_range={'Yes' if best['hl_t'] else 'No'}")
print(f"  Survival: {best['surv%']:.0%} of gated pairs  |  "
      f"Keeps {best['keep_good%']:.0%} of good pairs  |  "
      f"Eliminates {best['bad_eliminated%']:.0%} of bad pairs")


# ===========================================================================
# STEP 3: Ablation on 10 folds — Option A vs Option A + temporal gate
# ===========================================================================

print("\n" + "=" * 90)
print("STEP 3: ABLATION — Option A vs Option A + Temporal Gate (10 folds)")
print("=" * 90)

warmup_kalman()
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min_fresh = _load_cache(DATA_5MIN, compute_log_close=False)

def run_config(pairs_df, t1min, f1min):
    if pairs_df is None or pairs_df.empty:
        return float("nan"), 0, 0
    exec_res = run_fold_execution(
        pairs_df=pairs_df, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    if not exec_res:
        return float("nan"), 0, 0
    m = run_fold_pnl(
        exec_res, t1min, pairs_df, FIXED_DELTA, {},
        total_capital=_TOTAL_CAPITAL, n_open_pairs_max=_N_OPEN_PAIRS_MAX,
        tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS,
    )
    return float(m.get("sharpe", float("nan"))), len(exec_res), int(m.get("n_trades", 0))

# Use best threshold from sweep
BETA_T  = float(best["beta_t"])
SHIFT_T = float(best["shift_t"])
COINT_T = int(best["coint_t"])
HL_T    = bool(best["hl_t"])

rows = []
for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if len(pairs_df) > 500:
            pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        continue
    if pairs_df.empty:
        continue

    tickers   = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min     = _slice_month(cache_1min, tickers, trading_month)
    f1min     = _slice(cache_1min, tickers, form_start, form_end)
    f5min     = _slice(cache_5min_fresh, tickers, form_start, form_end)
    n_base    = len(pairs_df)

    print(f"Fold {fold_n:02d} [{trading_month}] {regime} - {n_base} base pairs", flush=True)

    # Option A: Johansen gate only
    t = time.time()
    pairs_A = apply_persistence_gate(pairs_df, f5min, form_end)
    s_A, ne_A, nt_A = run_config(pairs_A, t1min, f1min)
    print(f"  Option A  ({len(pairs_A):>3} pairs): SR={s_A:+.2f}  exec={ne_A}  trades={nt_A}  ({time.time()-t:.0f}s)")

    # Option A+T: Johansen gate + temporal stability gate
    t = time.time()
    pairs_AT, reasons = apply_temporal_gate(
        pairs_A, f5min, form_start, form_end,
        max_beta_drift=BETA_T, max_mean_shift=SHIFT_T,
        require_coint_blocks=COINT_T, require_hl_in_range=HL_T,
    )
    s_AT, ne_AT, nt_AT = run_config(pairs_AT, t1min, f1min)
    rej_str = f"beta={reasons['beta']} shift={reasons['shift']} coint={reasons['coint']} hl={reasons['hl']}"
    print(f"  Option A+T({len(pairs_AT):>3} pairs): SR={s_AT:+.2f}  exec={ne_AT}  trades={nt_AT}  ({time.time()-t:.0f}s)  rejected: [{rej_str}]")

    rows.append({
        "fold": fold_n, "month": trading_month, "regime": regime,
        "n_A": len(pairs_A), "n_AT": len(pairs_AT),
        "sr_A": s_A, "sr_AT": s_AT,
        "trades_A": nt_A, "trades_AT": nt_AT,
    })
    print()

df = pd.DataFrame(rows)
print("=" * 90)
print(f"RESULT: Option A vs Option A + Temporal Gate")
print(f"Gate params: beta<={BETA_T}, shift<={SHIFT_T}, coint>={COINT_T}, hl_in_range={HL_T}")
print("=" * 90)
print(f"{'Fold':<6} {'Mo':<9} {'Reg':<5}  {'A pairs':>8} {'A+T pairs':>10}  {'A SR':>7} {'Trd':>5}  {'A+T SR':>7} {'Trd':>5}  {'Delta':>7}")
print("-" * 90)
for _, r in df.iterrows():
    d = (r['sr_AT'] - r['sr_A']) if not (np.isnan(r['sr_AT']) or np.isnan(r['sr_A'])) else float('nan')
    print(f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5}  {int(r['n_A']):>8} {int(r['n_AT']):>10}  "
          f"{r['sr_A']:>+7.2f} {int(r['trades_A']):>5}  "
          f"{r['sr_AT']:>+7.2f} {int(r['trades_AT']):>5}  "
          f"{d:>+7.2f}")
print("-" * 90)
sA = df["sr_A"].dropna(); sAT = df["sr_AT"].dropna()
print(f"{'MEAN':<6} {'':9} {'':5}  {df['n_A'].mean():>8.0f} {df['n_AT'].mean():>10.0f}  "
      f"{sA.mean():>+7.2f} {df['trades_A'].sum():>5.0f}  "
      f"{sAT.mean():>+7.2f} {df['trades_AT'].sum():>5.0f}  "
      f"{sAT.mean()-sA.mean():>+7.2f}")
print(f"{'POS%':<6} {'':9}  {'':5}  {'':>8} {'':>10}  "
      f"{(sA>0).mean()*100:>6.0f}%  {'':>5}  {(sAT>0).mean()*100:>6.0f}%")

print("\nPer regime:")
for reg in ["Bear", "Bull"]:
    sub = df[df["regime"] == reg]
    a = sub["sr_A"].dropna(); at = sub["sr_AT"].dropna()
    print(f"  {reg}: A={a.mean():+.2f}  A+T={at.mean():+.2f}  delta={at.mean()-a.mean():+.2f}  (n={len(sub)})")
