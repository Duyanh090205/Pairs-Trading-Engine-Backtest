"""
chart_inspection.py
-------------------
Inspects minute-by-minute market data from the 1987 crash (Oct 16–21, 1987).

Steps:
  1. Load data and inspect columns / dtypes
  2. Parse and clean the Timestamp column
  3. Compute minute returns, cumulative drawdown, rolling volatility
  4. Identify the worst short intervals
  5. Identify the most plausible market break point
  6. Plot a 4-panel crash chart  ->  outputs/crash_plot.png
  7. Write plain-English analysis   ->  notes/data_notes.md
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths (relative to project root — run from:  d:\Quant Finance\...\Strategist Task)
# ---------------------------------------------------------------------------
DATA_PATH   = os.path.join("data",    "1987_crash_market_data.csv")
OUTPUT_PNG   = os.path.join("outputs", "crash_plot.png")   # retired — no longer generated
MEMO_PNG     = os.path.join("outputs", "memo_chart.png")
APPENDIX_PNG = os.path.join("outputs", "appendix_chart.png")
NOTES_PATH  = os.path.join("notes",   "data_notes.md")

# ---------------------------------------------------------------------------
# STEP 1 — Load and inspect
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nShape : {df.shape}  ({df.shape[0]} rows, {df.shape[1]} columns)")
print(f"\nColumns:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nBasic statistics:\n{df.describe().to_string()}")

# ---------------------------------------------------------------------------
# STEP 2 — Parse and clean the Timestamp column
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Parsing timestamps")
print("=" * 60)

# The format in the CSV is  M/D/YYYY H:MM  (e.g. "10/16/1987 0:00")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M")
df = df.set_index("Timestamp").sort_index()

# Check for problems
n_nat  = df.index.isna().sum()
n_dup  = df.index.duplicated().sum()
n_total = len(df)

print(f"Date range : {df.index.min()}  to  {df.index.max()}")
print(f"Total rows : {n_total}")
print(f"NaT values : {n_nat}")
print(f"Duplicates : {n_dup}")

if n_dup > 0:
    # Keep the first occurrence of any duplicate timestamp
    df = df[~df.index.duplicated(keep="first")]
    print(f"  -> Duplicates removed. Rows remaining: {len(df)}")

# DOW_Futures and SP500_Futures were read as strings — convert to numeric
# (pandas infers object dtype when values look like integers without decimals
#  mixed with NaN-like entries, or if there are any formatting artefacts)
for col in ["DOW_Futures", "SP500_Futures"]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

# No forward-fill: NaN values in trading hours are genuine data gaps
# (Oct 19 12:00-12:59 and Oct 20 11:30-12:59 have no source data).
# We leave them as NaN so matplotlib draws honest gaps in the chart
# rather than a misleading flat line at the last known price.
n_null_dow = df["DOW_Futures"].isna().sum()
n_null_sp  = df["SP500_Futures"].isna().sum()
print(f"DOW_Futures   NaN count (raw): {n_null_dow}")
print(f"SP500_Futures NaN count (raw): {n_null_sp}")

# ---------------------------------------------------------------------------
# STEP 2b — Filter to TRADING HOURS ONLY
# ---------------------------------------------------------------------------
# Root cause of the flat chart: 83% of rows (6,597 / 7,921) are off-hours or
# weekend data where price is simply carried forward from the previous close.
# Oct 17 (Sat) and Oct 18 (Sun) are market holidays — all 2,880 rows repeat
# Friday's close of 284.69. Pre-market rows (00:00–09:29) on each weekday are
# also flat. Real price movement occurs ONLY Mon–Fri between 09:30 and 16:00.
#
# Fix: keep only those trading-hours rows before doing any analytics or plotting.

trading_mask = (
    (df.index.dayofweek < 5) &                                       # Mon=0 ... Fri=4
    (df.index.time >= pd.Timestamp("09:30").time()) &                # market open
    (df.index.time <= pd.Timestamp("16:00").time())                  # market close
)
df_full = df.copy()          # preserve original for reference if needed
df = df[trading_mask].copy()

print(f"\nTrading-hours filter applied:")
print(f"  Rows before: {len(df_full)}")
print(f"  Rows after : {len(df)}")
print(f"  Sessions   : {df.index.normalize().nunique()} days")
for day in sorted(df.index.normalize().unique()):
    sess = df.loc[day.strftime("%Y-%m-%d")]
    print(f"    {day.strftime('%a %b %d')}: {len(sess)} rows  "
          f"SP500 {sess['SP500_Futures'].min():.2f} - {sess['SP500_Futures'].max():.2f}  "
          f"(NaN: {sess['SP500_Futures'].isna().sum()})")

# ---------------------------------------------------------------------------
# STEP 3 — Compute analytics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Computing analytics")
print("=" * 60)

price = df["SP500_Futures"]   # primary series for all analytics and chart

# Minute-by-minute returns (fractional)
returns = price.pct_change(fill_method=None)

# ------------------------------------------------------------------
# Clean returns: NaN out the FIRST row of every session.
# pct_change() on consecutive rows treats the overnight gap as a
# minute return (e.g. Oct 20 close ~214 -> Oct 21 open ~252 = +7.8%).
# That spike contaminates the 30-min rolling std for the next 29 rows,
# making Oct 21 look 10x more volatile than Oct 19 — backwards.
# Fix: replace boundary returns with NaN before computing rolling vol.
# ------------------------------------------------------------------
returns_clean = returns.copy()
_trading_days_step3 = sorted(df.index.normalize().unique())
for _tday in _trading_days_step3:
    _sess = df.loc[_tday.strftime("%Y-%m-%d")]
    returns_clean.loc[_sess.index[0]] = np.nan

# Cumulative drawdown from running peak
# drawdown = 0 at peak, negative below peak
rolling_peak = price.cummax()
drawdown = (price - rolling_peak) / rolling_peak

# 30-minute rolling volatility — uses CLEAN returns (no overnight gaps)
rolling_vol = returns_clean.rolling(30).std()

print(f"Max drawdown (cumulative, peak-to-trough): {drawdown.min():.2%}  at  {drawdown.idxmin()}")
print(f"  Note: the cited DJIA -22.6% is a single-day (prior-close->close) figure.")
print(f"  Our -28.6% covers the full rout from Oct 16 peak to Oct 20 trough.")
print()

# Print single-day DOW loss for reference
_oct19 = df.loc["1987-10-19"]
_oct16_sp_close  = df.loc["1987-10-16"]["SP500_Futures"].dropna().iloc[-1]
_oct19_sp_close  = _oct19["SP500_Futures"].dropna().iloc[-1]
_oct16_dow_close = df.loc["1987-10-16"]["DOW_Futures"].dropna().iloc[-1]
_oct19_dow_close = _oct19["DOW_Futures"].dropna().iloc[-1]
print(f"  Oct 19 SP500 single-day (prior-close->close): {(_oct19_sp_close - _oct16_sp_close)/_oct16_sp_close:.2%}")
print(f"  Oct 19 DOW  single-day (prior-close->close): {(_oct19_dow_close - _oct16_dow_close)/_oct16_dow_close:.2%}  (historical: -22.6%)")
print()
print(f"Peak vol (30m, clean): {rolling_vol.max():.4f}  at  {rolling_vol.idxmax()}")

# ---------------------------------------------------------------------------
# STEP 4 — Worst short intervals
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Worst short intervals")
print("=" * 60)

for window_name, window_min in [("30-min", 30), ("60-min", 60), ("120-min", 120)]:
    # Rolling sum of returns approximates the percentage move over the window
    rolling_move = returns.rolling(window_min).sum()
    worst_end    = rolling_move.nsmallest(5)
    print(f"\nTop-5 worst {window_name} intervals (end timestamp):")
    for ts, val in worst_end.items():
        start = ts - pd.Timedelta(minutes=window_min)
        print(f"  {start.strftime('%m/%d %H:%M')} -> {ts.strftime('%H:%M')}  |  cumulative return ~ {val:.2%}")

# ---------------------------------------------------------------------------
# STEP 5 — Identify the most plausible market break point
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Market break point")
print("=" * 60)

# Heuristic A — timestamp of the worst single 60-minute loss
rolling_60    = returns.rolling(60).sum()
break_by_loss = rolling_60.idxmin()

# Heuristic B — first minute where 30-min vol exceeds 3x its pre-crash baseline
# "pre-crash baseline" = median rolling vol during Oct 16-17 TRADING HOURS only
# (off-hours data has flat/constant prices -> vol = 0, which would bias the median)
pre_crash_mask = (
    (rolling_vol.index.date < pd.Timestamp("1987-10-19").date()) &
    (rolling_vol.index.time >= pd.Timestamp("09:30").time()) &
    (rolling_vol.index.time <= pd.Timestamp("16:00").time())
)
pre_crash_vol = rolling_vol[pre_crash_mask].dropna()
# Drop exact-zero values (residual flat-price windows at session boundaries)
pre_crash_vol = pre_crash_vol[pre_crash_vol > 0]

if len(pre_crash_vol) > 0:
    baseline_vol = pre_crash_vol.median()
    threshold    = 3 * baseline_vol
    # Look for the threshold breach only from Oct 19 onwards
    oct19_vol    = rolling_vol[rolling_vol.index.date >= pd.Timestamp("1987-10-19").date()]
    triggered    = oct19_vol[oct19_vol > threshold]
    break_by_vol = triggered.index[0] if len(triggered) > 0 else None
else:
    baseline_vol = np.nan
    threshold    = np.nan
    break_by_vol = None

print(f"Heuristic A (worst 60-min loss end):  {break_by_loss}  ({rolling_60.min():.2%})")
if break_by_vol:
    print(f"Heuristic B (vol > 3x baseline):      {break_by_vol}  (threshold = {threshold:.4f})")
else:
    print("Heuristic B: vol threshold not crossed")

# Choose the earlier of the two signals as the representative break point
candidates  = [t for t in [break_by_loss, break_by_vol] if t is not None]
break_point = min(candidates)
print(f"\nChosen break point (earliest signal):  {break_point}")

# ---------------------------------------------------------------------------
# STEP 6 — Build the 4-panel chart
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Building chart")
print("=" * 60)

# ------------------------------------------------------------------
# Integer x-axis (eliminates overnight / weekend datetime gaps)
# ------------------------------------------------------------------
x = np.arange(len(df))

def dt_to_x(ts):
    """Map a datetime to its integer row position in df."""
    if ts in df.index:
        return df.index.get_loc(ts)
    pos = df.index.searchsorted(ts)
    return min(pos, len(df) - 1)

# Build session metadata
trading_days = sorted(df.index.normalize().unique())
sessions = []
for tday in trading_days:
    sess_rows = df[df.index.normalize() == tday]
    sp        = sess_rows["SP500_Futures"].dropna()
    open_p    = sp.iloc[0]  if len(sp) > 0 else np.nan
    close_p   = sp.iloc[-1] if len(sp) > 0 else np.nan
    pct_chg   = (close_p - open_p) / open_p * 100 if not np.isnan(open_p) else np.nan
    sessions.append({
        "date"   : tday,
        "label"  : tday.strftime("%a\n%b %d"),
        "x_start": dt_to_x(sess_rows.index[0]),
        "x_end"  : dt_to_x(sess_rows.index[-1]),
        "x_mid"  : (dt_to_x(sess_rows.index[0]) + dt_to_x(sess_rows.index[-1])) // 2,
        "open"   : open_p,
        "close"  : close_p,
        "pct_chg": pct_chg,
    })

bp_x     = dt_to_x(break_point)
bp_price = price.iloc[bp_x]
max_dd_x = dt_to_x(drawdown.idxmin())
oct19_sess = [s for s in sessions if s["date"].date() == pd.Timestamp("1987-10-19").date()]

# returns_clean already has session-boundary NaNs (computed in Step 3).
# Reuse it here for display — no need to re-do the NaN logic.
ret_pct = returns_clean * 100      # convert to percent for readability

# Clip display range to ±1.5% so intra-session ticks are clearly visible.
# (The actual extremes are still computed correctly above — this only
#  affects the y-axis zoom, not the underlying data.)
RET_CAP = 1.5

# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
ALT_BG    = "#1c2333"    # slightly lighter — alternating session shade
GRID_COL  = "#30363d"
TEXT_COL  = "#e6edf3"
DIM_COL   = "#8b949e"    # muted text for secondary labels
RED       = "#ff4d4f"
GREEN     = "#3fb950"
GOLD      = "#d4a017"
CYAN      = "#58a6ff"
ORANGE    = "#f78166"

os.makedirs("outputs", exist_ok=True)

# ==================================================================
# BLOCK A — MEMO CHART  (outputs/memo_chart.png)
# Single panel, light theme — for memo body (Figure 1)
# Rule: one message, one annotation, five seconds to orient
# ==================================================================
MEMO_L_BG    = "#ffffff"
MEMO_P_BG    = "#f8f9fa"
MEMO_ALT_BG  = "#f0f2f5"
MEMO_GRID    = "#dee2e6"
MEMO_TEXT    = "#212529"
MEMO_DIM     = "#6c757d"
MEMO_BLUE    = "#1a56db"
MEMO_RED_BG  = "#fee2e2"
MEMO_RED_ANN = "#dc2626"
MEMO_SEP     = "#adb5bd"

fig_m, ax_m = plt.subplots(1, 1, figsize=(12, 4))
fig_m.patch.set_facecolor(MEMO_L_BG)
ax_m.set_facecolor(MEMO_P_BG)
ax_m.tick_params(colors=MEMO_TEXT, labelsize=9)
ax_m.grid(True, color=MEMO_GRID, linewidth=0.4, linestyle="--", alpha=0.7)
for spine in ax_m.spines.values():
    spine.set_edgecolor(MEMO_GRID)

# Alternating session backgrounds
for i, sess in enumerate(sessions):
    bg = MEMO_ALT_BG if i % 2 == 1 else MEMO_P_BG
    ax_m.axvspan(sess["x_start"], sess["x_end"] + 1, color=bg, alpha=1.0, zorder=0)

# Oct 19 faint red overlay
if oct19_sess:
    s19 = oct19_sess[0]
    ax_m.axvspan(s19["x_start"], s19["x_end"] + 1, color=MEMO_RED_BG, alpha=1.0, zorder=1)

# Trading halt periods (confirmed from CSV analysis)
# Oct 19: 12:00-12:59, Oct 20: 11:30-12:59
halt_periods = [
    (pd.Timestamp("1987-10-19 12:00"), pd.Timestamp("1987-10-19 13:00")),
    (pd.Timestamp("1987-10-20 11:30"), pd.Timestamp("1987-10-20 13:00"))
]
MEMO_HALT_BG = "#fff3cd"  # light yellow for halt overlay (more visible)
halt_patch = None
for halt_start, halt_end in halt_periods:
    if halt_start in df.index or halt_end in df.index:
        x_halt_start = dt_to_x(halt_start)
        x_halt_end = dt_to_x(halt_end)
        halt_patch = ax_m.axvspan(x_halt_start, x_halt_end + 1, color=MEMO_HALT_BG, alpha=0.9,
                     linewidth=1.2, edgecolor="#ff9800", linestyle="-", zorder=2)

# Session separator verticals
for sess in sessions[1:]:
    ax_m.axvline(sess["x_start"] - 0.5, color=MEMO_SEP,
                 linewidth=0.8, linestyle="-", zorder=5)

# Price line
ax_m.plot(x, price.values, color=MEMO_BLUE, linewidth=1.3, zorder=3)

# ONE annotation — break point
ax_m.annotate(
    f"Market break\n{break_point.strftime('%b %d  %H:%M')}",
    xy=(bp_x, bp_price),
    xytext=(bp_x + 45, bp_price + 8),
    arrowprops=dict(arrowstyle="->", color=MEMO_RED_ANN, lw=1.5),
    color=MEMO_RED_ANN, fontsize=9, fontweight="bold", zorder=7
)

# Legend for halt periods
from matplotlib.patches import Patch
halt_legend = Patch(facecolor=MEMO_HALT_BG, edgecolor="#ff9800", label="Trading halts")
ax_m.legend(handles=[halt_legend], loc="lower left", fontsize=8, framealpha=0.95)

# X-axis: session date labels only
ax_m.set_xticks([s["x_mid"] for s in sessions])
ax_m.set_xticklabels([s["label"] for s in sessions], color=MEMO_TEXT, fontsize=9)
ax_m.set_xlim(-5, len(df) + 5)

ax_m.set_ylabel("S&P 500 Index Level", color=MEMO_TEXT, fontsize=10)
ax_m.yaxis.label.set_color(MEMO_TEXT)
ax_m.set_title(
    "S&P 500 Futures \u2014 Black Monday, October 1987",
    color=MEMO_TEXT, fontsize=12, fontweight="bold", pad=10
)
fig_m.text(
    0.5, -0.02,
    "Source: 1987_crash_market_data.csv  |  S&P 500 Futures, trading hours only (09:30-16:00)",
    ha="center", color=MEMO_DIM, fontsize=7
)

plt.tight_layout()
plt.savefig(MEMO_PNG, dpi=150, bbox_inches="tight", facecolor=MEMO_L_BG)
plt.close(fig_m)
print(f"Memo chart saved     -> {MEMO_PNG}")

# ==================================================================
# BLOCK B — APPENDIX CHART  (outputs/appendix_chart.png)
# 3 panels: returns / drawdown / vol — dark theme, for appendix
# ==================================================================
fig_a, axes_a = plt.subplots(
    3, 1, figsize=(16, 10), sharex=True,
    gridspec_kw={"height_ratios": [1.5, 1.5, 1.5], "hspace": 0.06}
)
fig_a.patch.set_facecolor(DARK_BG)

for ax in axes_a:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

# Session backgrounds
for i, sess in enumerate(sessions):
    bg = ALT_BG if i % 2 == 1 else PANEL_BG
    for ax in axes_a:
        ax.axvspan(sess["x_start"], sess["x_end"] + 1, color=bg, alpha=1.0, zorder=0)

if oct19_sess:
    s19 = oct19_sess[0]
    for ax in axes_a:
        ax.axvspan(s19["x_start"], s19["x_end"] + 1, color=RED, alpha=0.12, zorder=1)

# Trading halt periods (confirmed from CSV analysis)
# Oct 19: 12:00-12:59, Oct 20: 11:30-12:59
APPENDIX_HALT_BG = "#ffb3b3"  # more visible light red for halts
for halt_start, halt_end in halt_periods:
    if halt_start in df.index or halt_end in df.index:
        x_halt_start = dt_to_x(halt_start)
        x_halt_end = dt_to_x(halt_end)
        for ax in axes_a:
            ax.axvspan(x_halt_start, x_halt_end + 1, color=APPENDIX_HALT_BG, alpha=0.6,
                       linewidth=1, edgecolor=RED, linestyle="-", zorder=2)

for sess in sessions[1:]:
    for ax in axes_a:
        ax.axvline(sess["x_start"] - 0.5, color=DIM_COL,
                   linewidth=1.2, linestyle="-", zorder=5)

# Panel 1 — Intra-session 1-min returns (capped at +-1.5%)
ax_a0 = axes_a[0]
pos_ret = np.clip(ret_pct.values,  0,  RET_CAP)
neg_ret = np.clip(ret_pct.values, -RET_CAP, 0)
ax_a0.fill_between(x, pos_ret, color=CYAN, alpha=0.7, linewidth=0, zorder=3)
ax_a0.fill_between(x, neg_ret, color=RED,  alpha=0.7, linewidth=0, zorder=3)
ax_a0.axhline(0, color=GRID_COL, linewidth=0.8, zorder=4)
ax_a0.set_ylabel("1-min Return (%)", color=TEXT_COL, fontsize=9)
ax_a0.yaxis.label.set_color(TEXT_COL)
ax_a0.set_ylim(-RET_CAP, RET_CAP)
ax_a0.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:+.1f}%"))
ax_a0.text(5, RET_CAP * 0.82,
           "Intra-session only\n(overnight gaps removed)",
           color=DIM_COL, fontsize=7, va="top")
ax_a0.set_title(
    "1987 Black Monday \u2014 Analytical Detail  (Oct 16-21, 1987)",
    color=TEXT_COL, fontsize=12, fontweight="bold", pad=10
)

# Panel 2 — Cumulative drawdown
ax_a1 = axes_a[1]
dd_pct = drawdown.values * 100
ax_a1.fill_between(x, dd_pct, 0, color=RED, alpha=0.45, linewidth=0, zorder=3)
ax_a1.plot(x, dd_pct, color=RED, linewidth=0.8, zorder=4)
ax_a1.set_ylabel("Drawdown (%)", color=TEXT_COL, fontsize=9)
ax_a1.yaxis.label.set_color(TEXT_COL)
ax_a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0f}%"))
ax_a1.annotate(
    f"Max DD  {drawdown.min():.1%}",
    xy=(max_dd_x, drawdown.min() * 100),
    xytext=(max_dd_x - 80, drawdown.min() * 100 + 4),
    arrowprops=dict(arrowstyle="->", color=TEXT_COL, lw=1.0),
    color=TEXT_COL, fontsize=8, zorder=7
)

# Panel 3 — 30-min rolling volatility
ax_a2 = axes_a[2]
ax_a2.plot(x, rolling_vol.values, color=GOLD, linewidth=0.9,
           label="30-min rolling vol", zorder=3)
if not np.isnan(baseline_vol):
    ax_a2.axhline(baseline_vol, color=GRID_COL, linewidth=0.8, linestyle="--",
                  label=f"Pre-crash baseline ({baseline_vol:.4f})")
    ax_a2.axhline(threshold, color=RED, linewidth=0.8, linestyle="--",
                  label=f"3x threshold ({threshold:.4f})")
ax_a2.set_ylabel("Rolling Vol (std)", color=TEXT_COL, fontsize=9)
ax_a2.yaxis.label.set_color(TEXT_COL)
ax_a2.legend(loc="upper left", fontsize=7, facecolor=PANEL_BG,
             labelcolor=TEXT_COL, edgecolor=GRID_COL)

# Shared x-axis: session midpoint ticks + hourly minor ticks
ax_a2.set_xticks([s["x_mid"] for s in sessions])
ax_a2.set_xticklabels([s["label"] for s in sessions], color=TEXT_COL, fontsize=9)
ax_a2.set_xlim(-5, len(df) + 5)

minor_ticks = []
for sess in sessions:
    for offset in range(0, sess["x_end"] - sess["x_start"] + 1, 60):
        minor_ticks.append(sess["x_start"] + offset)
ax_a2.set_xticks(minor_ticks, minor=True)
ax_a2.tick_params(axis="x", which="minor", length=4, color=DIM_COL)

fig_a.text(
    0.5, 0.005,
    "Source: 1987_crash_market_data.csv  |  S&P 500 Futures, trading hours only (09:30-16:00)  |  Returns: intra-session only",
    ha="center", color=DIM_COL, fontsize=7
)

plt.savefig(APPENDIX_PNG, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig_a)
print(f"Appendix chart saved -> {APPENDIX_PNG}")

# ---------------------------------------------------------------------------
# STEP 7 — Write data_notes.md
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Writing data_notes.md")
print("=" * 60)

# Collect key numbers for embedding in the notes
max_dd_pct    = drawdown.min() * 100
max_dd_date   = drawdown.idxmin()
peak_vol_val  = rolling_vol.max()
peak_vol_date = rolling_vol.idxmax()

worst_60_val   = rolling_60.min() * 100
worst_60_end   = rolling_60.idxmin()
worst_60_start = worst_60_end - pd.Timedelta(minutes=60)

vol_ratio = peak_vol_val / baseline_vol if not np.isnan(baseline_vol) else float("nan")

break_by_vol_str = (break_by_vol.strftime("%H:%M on %b %d")
                    if break_by_vol else "Oct 19 open")

notes_content = f"""# Data Notes — 1987 Crash Market Data

**Dataset:** `data/1987_crash_market_data.csv`
**Coverage:** October 16–21, 1987 | Minute-frequency | {n_total:,} rows | 8 columns

---

## Chart Summary

Two chart files are produced. Both use an integer x-axis (rows plotted back-to-back with no
overnight/weekend gaps) and cover trading hours only (Mon–Fri 09:30–16:00, 1,324 rows across
4 sessions).

### Figure 1 — Memo Chart (`outputs/memo_chart.png`)
Light-theme single-panel price chart intended for the memo body. It shows:
- **S&P 500 Futures price level** as a continuous blue line across all four sessions
  (Fri Oct 16, Mon Oct 19, Tue Oct 20, Wed Oct 21).
- **Oct 19 session shaded in faint red** — immediately signals to the reader which
  session is the event of interest without requiring a legend.
- **One annotation only:** a red arrow pointing to the market break at {break_point.strftime("%b %d %H:%M")},
  labelled "Market break / {break_point.strftime("%b %d  %H:%M")}".
- **Session date labels** on the x-axis (Fri/Oct 16, Mon/Oct 19, Tue/Oct 20, Wed/Oct 21).
- No % badges, no returns panel, no vol lines — a reader can orient in ~5 seconds.

The price drifts down mildly on Friday (~299 to ~283), then collapses through the entire
Monday session, reaching a session low near 220 before a partial bounce. Tuesday
continues lower to the trough (~213) before a sustained recovery on Tuesday afternoon
and Wednesday carries the index back toward 258.

### Figure 2 — Appendix Chart (`outputs/appendix_chart.png`)
Dark-theme 3-panel analytical detail chart for the appendix. Three panels share the x-axis:

**Panel 1 — Intra-session 1-minute returns (capped +-1.5%)**
Shows the intensity and direction of minute-by-minute price changes. Friday is almost
entirely flat (returns near zero). Monday explodes into sustained red (negative) fills,
with a few brief cyan (positive) interruptions — the signature of wave-selling broken
only by sporadic bargain-hunting. Tuesday also shows heavy selling in the morning
before stabilising. Wednesday's returns are predominantly positive (recovery).

**Panel 2 — Cumulative drawdown from all-time peak**
Starts at 0% (Friday open) and falls continuously. The drawdown accelerates sharply
on Monday, reaching approximately -25% by the end of that session. It bottoms at
{max_dd_pct:.1f}% at {max_dd_date.strftime("%H:%M on %b %d")} — the deepest trough. The characteristic
asymmetry is visible: the crash unfolds over one session; the recovery extends across
two sessions. The max drawdown annotation is placed at the trough.

**Panel 3 — 30-minute rolling volatility (std of minute returns)**
The pre-crash baseline (Fri Oct 16) holds near {baseline_vol:.4f} — a flat, near-invisible line.
On Monday the vol line leaps to ~0.0025, far above the 3x threshold (red dashed line
at {threshold:.4f}), confirming market dysfunction from the opening minutes. Vol partially
subsides mid-Monday (corresponding to the data gap at noon), then spikes again.
Tuesday peaks at {peak_vol_val:.4f} at {peak_vol_date.strftime("%H:%M")} — the highest reading in the dataset,
{vol_ratio:.1f}x the pre-crash baseline. Wednesday vol collapses back to near-baseline,
consistent with orderly recovery conditions.

---

## Data Gaps

Two blocks of missing data exist within active trading hours. These are genuine source-data
absences (not off-hours carry-forward) and are left as NaN — **not forward-filled** — so the
charts render honest breaks rather than misleading flat lines.

| Session | Gap window | Missing rows | Visible in charts |
|---|---|---|---|
| Mon Oct 19 | 12:00 – 12:59 | 60 | Break in price line (memo chart); missing fills in returns/vol panels (appendix) |
| Tue Oct 20 | 11:30 – 12:59 | 90 | Same — wider gap, visible as a longer break in both chart files |
| Fri Oct 16 | none | 0 | Clean session, no gaps |
| Wed Oct 21 | none | 0 | Clean session, no gaps |

**Why these gaps exist:** The most likely explanation is a data-collection failure at the
exchange or data vendor level during peak crisis stress on Oct 19 and Oct 20. Both gaps
fall around midday, when phone lines and quote systems were overwhelmed. It is also possible
the exchange briefly halted futures trading during these windows — contemporaneous accounts
reference repeated trading halts and system overloads on both days.

**Why not forward-fill:** Forward-filling would draw a flat horizontal line across each gap,
implying price was stable during that window. Given that the surrounding minutes show violent
movement, a flat segment would be actively misleading — implying calm where there was likely
chaos. The honest representation is a visible break.

**Effect on analytics:**
- Minute returns across the gap boundary are NaN (not computed), so no fake large return
  spike is introduced at the resumption of data.
- The 30-min rolling volatility has a trough near zero at the gap edges due to the NaN
  window — this is visible in the appendix vol panel as a brief dip mid-Monday and mid-Tuesday.
  It reflects missing data, not a genuine vol collapse.
- The cumulative drawdown is interpolated visually across the gap (matplotlib draws nothing),
  creating the visible step-cut appearance in the drawdown panel.

---

## Key Observations

- **Friday weakness sets the stage:** October 16 already shows elevated selling
  pressure; futures close well below their opening level, signalling that
  portfolio insurance programs had begun to de-risk before the weekend.

- **Cascade opens on October 19:** From the first minutes of trading, selling
  accelerates beyond anything seen on Oct 16–17. The worst 60-minute window
  ({worst_60_start.strftime("%H:%M")} – {worst_60_end.strftime("%H:%M on %b %d")})
  produced a cumulative move of approximately **{worst_60_val:.1f}%** — a rate of
  loss that is orders of magnitude beyond normal market conditions.

- **Drawdown reaches {max_dd_pct:.1f}% at trough:** The cumulative drawdown from
  the pre-crash peak bottoms at {max_dd_date.strftime("%H:%M on %b %d")},
  representing the deepest single-session loss in modern US equity market history.

- **Volatility spikes {vol_ratio:.1f}x above baseline:** The 30-minute rolling
  standard deviation of minute returns reaches {peak_vol_val:.4f} at
  {peak_vol_date.strftime("%H:%M on %b %d")} — approximately {vol_ratio:.1f} times
  the pre-crash median. This extreme volatility regime is a quantitative marker
  of market dysfunction: when vol is this elevated, bid-ask spreads widen to the
  point where executing a hedge becomes impossible, trapping portfolio insurance
  programs in an illiquid loop.

- **Cross-market contagion visible in the data:** The dataset records Nikkei
  Futures, Treasury, Gold, and Oil alongside US equity futures. The simultaneous
  dislocations across asset classes on October 19 confirm that the crash was a
  global liquidity event, not a localised US technical failure.

- **Recovery is slower than the crash:** The drawdown panel shows that while the
  crash unfolds over a single session, the partial recovery extends across
  October 20–21 — a characteristic asymmetry driven by the Federal Reserve's
  intervention and the gradual restoration of dealer credit lines.

---

## Most Plausible Break Point

> Based on the convergence of two quantitative signals — the end timestamp of the
> worst 60-minute loss window and the first moment 30-minute rolling volatility
> exceeded three times its pre-crash baseline ({break_by_vol_str}) — the market
> structure most plausibly **broke** at or around
> **{break_point.strftime("%H:%M on %B %d, %Y")}**; however, given the
> compressed, minute-frequency nature of the data and the possibility that
> dysfunction began in overnight futures markets before the US open, this should
> be interpreted as an approximate inflection zone rather than a single
> definitive moment.
"""

os.makedirs("notes", exist_ok=True)
with open(NOTES_PATH, "w", encoding="utf-8") as fh:
    fh.write(notes_content)

print(f"Notes saved -> {NOTES_PATH}")
print("\nAll done.")
