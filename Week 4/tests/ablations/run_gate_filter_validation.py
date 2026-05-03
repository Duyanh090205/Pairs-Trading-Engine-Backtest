"""
run_gate_filter_validation.py

For every pair that survived the persistence gate AND traded in Option A,
compute three candidate filter scores using month-6 5-min data:

  Filter A — OU half-life on month-6 spread
  Filter B — Linear drift slope (normalised)
  Filter C — Mean shift between first and second half of month-6

Then join to trading outcomes (from divergence_trades.csv) and test
whether each score discriminates good outcomes from bad ones.

A filter is worth implementing only if the score distributions of
"good" vs "bad" pairs are clearly separated.
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

from src.phase4_defense.orchestrator import FOLD_SCHEDULE
from src.utils.stats import compute_ou_halflife

PHASE1_DIR = "results/metrics/phase1_folds"
DATA_5MIN  = "data/validated/5min_phase1"
JOHANSEN_PVAL = 0.05


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_5min_cache():
    cache = {}
    for fname in os.listdir(DATA_5MIN):
        if not fname.endswith(".parquet"):
            continue
        tk = fname.replace(".parquet", "")
        try:
            cache[tk] = pd.read_parquet(os.path.join(DATA_5MIN, fname))
        except Exception:
            pass
    return cache

def _johansen_pval(log_a, log_b):
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Filter score functions (all operate on month-6 5-min spread)
# ---------------------------------------------------------------------------

def compute_spread(fa, fb, alpha_pca, beta_pca, start, end):
    """Compute static spread S = log(A) - alpha - beta*log(B) on [start, end]."""
    fa_s = fa[(fa.index >= start) & (fa.index <= end)][["log_close"]] if "log_close" in fa.columns \
           else fa[(fa.index >= start) & (fa.index <= end)][["close"]].assign(log_close=lambda d: np.log(d["close"].clip(lower=1e-10)))[["log_close"]]
    fb_s = fb[(fb.index >= start) & (fb.index <= end)][["log_close"]] if "log_close" in fb.columns \
           else fb[(fb.index >= start) & (fb.index <= end)][["close"]].assign(log_close=lambda d: np.log(d["close"].clip(lower=1e-10)))[["log_close"]]
    aln = fa_s.join(fb_s, lsuffix="_a", rsuffix="_b", how="inner").dropna()
    if len(aln) < 20:
        return None
    spread = aln["log_close_a"].values - alpha_pca - beta_pca * aln["log_close_b"].values
    return spread


def filter_A_ou_halflife(spread):
    """OU half-life (days) on month-6 spread. 78 bars/day on 5-min."""
    try:
        hl = compute_ou_halflife(spread)   # returns days
        return float(hl)
    except Exception:
        return np.nan


def filter_B_drift_slope(spread):
    """Normalised linear drift: |b * T| / std(S)."""
    if len(spread) < 20:
        return np.nan
    t = np.arange(len(spread), dtype=float)
    slope, _, _, _, _ = stats.linregress(t, spread)
    std_s = np.std(spread, ddof=1)
    if std_s < 1e-10:
        return np.nan
    return abs(slope * len(spread)) / std_s


def filter_C_mean_shift(spread):
    """Absolute mean shift between first and second half of month-6, in sigma."""
    if len(spread) < 40:
        return np.nan
    mid = len(spread) // 2
    s1, s2 = spread[:mid], spread[mid:]
    std_s = np.std(spread, ddof=1)
    if std_s < 1e-10:
        return np.nan
    return abs(np.mean(s1) - np.mean(s2)) / std_s


# ---------------------------------------------------------------------------
# Load 5-min cache and trade outcomes
# ---------------------------------------------------------------------------

print("Loading 5-min cache...")
cache_5min = _load_5min_cache()
print(f"  {len(cache_5min)} tickers\n")

print("Loading divergence audit outcomes...")
trades_df = pd.read_csv("results/metrics/diagnostic/divergence_trades.csv")

# Aggregate per (fold, pair): is any trade a never-reverter? avg net loss?
def agg_outcome(g):
    return pd.Series({
        "n_trades":       len(g),
        "pct_never_rev":  g["never_rev"].mean(),
        "pct_reverted":   g["reverted"].mean(),
        "med_adv_pct":    g["max_adv_pct"].median(),
        "any_never_rev":  int(g["never_rev"].any()),
        "bad":            int(g["never_rev"].any() or g["max_adv_pct"].max() > 0.5),
    })

pair_outcomes = (
    trades_df.groupby(["fold", "pair"])
    .apply(agg_outcome)
    .reset_index()
)
print(f"  {len(pair_outcomes)} pair-fold trading outcomes loaded\n")


# ---------------------------------------------------------------------------
# Compute filter scores for every gated pair in every fold
# ---------------------------------------------------------------------------

rows = []
print("Computing filter scores for all gated pairs...")

for spec in FOLD_SCHEDULE:
    fold_n        = spec.fold_n
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

    # Month-6 window
    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")

    # Only process pairs that survived the Johansen gate
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = cache_5min.get(ta)
        fb = cache_5min.get(tb)
        if fa is None or fb is None:
            continue

        # Recompute log_close if needed
        if "log_close" not in fa.columns:
            fa = fa.assign(log_close=np.log(fa["close"].clip(lower=1e-10)))
        if "log_close" not in fb.columns:
            fb = fb.assign(log_close=np.log(fb["close"].clip(lower=1e-10)))

        # Johansen gate (same as Option A)
        aln = (
            fa[(fa.index >= gate_start) & (fa.index <= gate_end)][["log_close"]]
            .join(fb[(fb.index >= gate_start) & (fb.index <= gate_end)][["log_close"]],
                  lsuffix="_a", rsuffix="_b", how="inner").dropna()
        )
        if len(aln) < 20:
            continue
        pval = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if np.isnan(pval) or pval >= JOHANSEN_PVAL:
            continue  # didn't survive gate — skip

        # Compute spread on month-6
        spread = compute_spread(fa, fb, row["alpha_pca"], row["beta_pca"], gate_start, gate_end)
        if spread is None:
            continue

        hl    = filter_A_ou_halflife(spread)
        drift = filter_B_drift_slope(spread)
        shift = filter_C_mean_shift(spread)
        pair_key = f"{ta}/{tb}"

        rows.append({
            "fold":           fold_n,
            "trading_month":  trading_month,
            "pair":           pair_key,
            "regime":         "Bear" if trading_month <= "2022-12" else "Bull",
            "phase1_hl":      float(row["half_life_days"]),
            "A_ou_hl":        hl,
            "B_drift":        drift,
            "C_shift":        shift,
        })

scores_df = pd.DataFrame(rows)
print(f"  {len(scores_df)} gated pairs scored\n")


# ---------------------------------------------------------------------------
# Join filter scores to trading outcomes
# ---------------------------------------------------------------------------

merged = pd.merge(
    scores_df,
    pair_outcomes[["fold", "pair", "n_trades", "pct_never_rev",
                   "pct_reverted", "med_adv_pct", "any_never_rev", "bad"]],
    on=["fold", "pair"], how="left"
)

# Pairs that never traded have NaN outcomes — mark them separately
# (Z=3.0 was too selective; they survived gate but never got an entry signal)
merged["traded"]    = merged["n_trades"].notna() & (merged["n_trades"] > 0)
merged["never_rev"] = merged["any_never_rev"].fillna(0).astype(int)
merged["bad"]       = merged["bad"].fillna(0).astype(int)
merged["n_trades"]  = merged["n_trades"].fillna(0)

print(f"Pairs that traded:    {merged['traded'].sum()}")
print(f"Pairs that never traded (no entry signal): {(~merged['traded']).sum()}")
print()


# ---------------------------------------------------------------------------
# Discriminator analysis — traded pairs only
# ---------------------------------------------------------------------------

traded = merged[merged["traded"]].copy()
print(f"Discriminator analysis on {len(traded)} traded pairs")
print("=" * 80)

for fname, col, label in [
    ("A", "A_ou_hl",  "OU half-life (days) in month-6"),
    ("B", "B_drift",  "Drift slope (normalised |b*T|/std)"),
    ("C", "C_shift",  "Mean shift between halves (sigma)"),
]:
    valid = traded[traded[col].notna()]
    good  = valid[valid["never_rev"] == 0][col]
    bad   = valid[valid["never_rev"] == 1][col]

    if len(bad) < 3:
        print(f"\nFilter {fname} ({label}): too few never-rev pairs to evaluate (n={len(bad)})")
        continue

    # Mann-Whitney U test (non-parametric, doesn't assume normality)
    u_stat, p_mw = stats.mannwhitneyu(good, bad, alternative="two-sided")
    auroc = u_stat / (len(good) * len(bad))   # AUROC from U stat

    print(f"\nFilter {fname}: {label}")
    print(f"  Good (never_rev=0): n={len(good):>4}  median={good.median():.3f}  p25={good.quantile(0.25):.3f}  p75={good.quantile(0.75):.3f}")
    print(f"  Bad  (never_rev=1): n={len(bad):>4}  median={bad.median():.3f}  p25={bad.quantile(0.25):.3f}  p75={bad.quantile(0.75):.3f}")
    print(f"  Mann-Whitney p={p_mw:.4f}   AUROC={auroc:.3f}  (0.5=random, 1.0=perfect)")

    # Also check vs "bad" outcome (any never_rev OR >50% adverse)
    good2 = valid[valid["bad"] == 0][col]
    bad2  = valid[valid["bad"] == 1][col]
    if len(bad2) >= 3:
        u2, p2 = stats.mannwhitneyu(good2, bad2, alternative="two-sided")
        auroc2 = u2 / (len(good2) * len(bad2))
        print(f"  vs broad 'bad' (n={len(bad2)}): Mann-Whitney p={p2:.4f}   AUROC={auroc2:.3f}")

    # Best threshold (maximise accuracy)
    thresholds = np.nanpercentile(valid[col].values, np.arange(10, 91, 5))
    best_acc = 0; best_thr = None; best_dir = None
    for thr in thresholds:
        for direction in ["above", "below"]:
            if direction == "above":
                pred_bad = valid[col] > thr
            else:
                pred_bad = valid[col] < thr
            tp = int(((pred_bad) & (valid["never_rev"] == 1)).sum())
            tn = int(((~pred_bad) & (valid["never_rev"] == 0)).sum())
            acc = (tp + tn) / len(valid)
            if acc > best_acc:
                best_acc = acc; best_thr = thr; best_dir = direction
    print(f"  Best threshold: {col} {best_dir} {best_thr:.3f} -> accuracy={best_acc:.0%}")

# ---------------------------------------------------------------------------
# Correlation between filter scores and adverse excursion
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Correlation: filter score vs max adverse excursion (all traded pairs)")
print("=" * 80)
for fname, col in [("A", "A_ou_hl"), ("B", "B_drift"), ("C", "C_shift")]:
    sub = traded[traded[col].notna() & traded["med_adv_pct"].notna()]
    if len(sub) < 5:
        continue
    r, p = stats.spearmanr(sub[col], sub["med_adv_pct"])
    print(f"  Filter {fname} vs med_adv_pct: Spearman r={r:+.3f}  p={p:.4f}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print("A filter is worth implementing if AUROC > 0.60 and Mann-Whitney p < 0.10.")
print("Below that, the score does not reliably separate good from bad pairs.")
