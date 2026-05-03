"""
Plotting Utilities

Three diagnostic charts required for the Week 2 Signal Logic Document:
  1. plot_signal_validation  — 3-panel: prices / spread / Z-score with trade markers
  2. plot_rolling_hurst      — regime monitor over the trading period
  3. plot_qq_normal          — QQ-plot of Z-score vs standard normal

All functions save to a path and return the Figure for optional display.
Non-interactive backend (Agg) is set at import so plots work headlessly.
"""
import matplotlib
matplotlib.use("Agg")   # headless — no display required

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import probplot


# ---------------------------------------------------------------------------
# 1. Three-panel signal validation chart
# ---------------------------------------------------------------------------

def plot_signal_validation(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    window: int,
    entry_z: float = 2.0,
    save_path: str = "outputs/figures/signal_validation.png",
    sample_week: str | None = None,
) -> plt.Figure:
    """Three-panel chart: prices, spread ±2σ band, Z-score with trade markers.

    Args:
        df          : Full pipeline output DataFrame containing columns:
                        close_a, close_b, spread, rolling_mean, rolling_std,
                        zscore, position.
        ticker_a/b  : Ticker labels for the legend.
        beta        : Hedge ratio (displayed in spread panel title).
        window      : Rolling window in bars (displayed in spread panel title).
        entry_z     : Entry threshold — drawn as horizontal lines on Z panel.
        save_path   : Output file path (PNG).
        sample_week : ISO date string for the start of the sample week to plot,
                      e.g. "2022-09-12".  If None, uses a week from ~2/3 through
                      the trading period (roughly November for Jul–Dec data).

    Panel layout
    ------------
    Panel 1 — Prices: close_a and close_b (normalised to 100 at window start
              so they fit on the same axis without the price-level difference
              dominating the visual).
    Panel 2 — Spread with rolling mean and ±entry_z*std band.
    Panel 3 — Z-score with threshold lines and entry/exit markers.
              Entry long  : green triangle up
              Entry short : red triangle down
              Exit        : black cross

    !! FLAG — sample_week selection:
        The chart covers a 5-day window.  Weeks with zero trades (all flat)
        are uninformative.  The auto-selection logic picks a week that contains
        at least one trade entry; if none is found in the default search window
        it falls back to the first week of the trading period.  Always visually
        inspect the saved chart — a bad week choice produces a flat Z-score
        panel with no markers, which looks like the engine is broken even when
        it is not.

    !! FLAG — normalised prices in Panel 1:
        Normalising to 100 removes the price-level difference between the two
        legs.  This is purely cosmetic for the validation chart.  It does NOT
        affect the spread or Z-score computation, which use raw log prices.
    """
    # -- Find a sample week that contains at least one trade ------------------
    if sample_week is None:
        sample_week = _find_active_week(df)

    week_end = pd.Timestamp(sample_week, tz=df.index.tz) + pd.Timedelta(days=7)
    week = df.loc[sample_week:week_end.strftime("%Y-%m-%d")]

    if week.empty or "position" not in week.columns:
        raise ValueError(
            f"sample_week '{sample_week}' produced an empty slice or df is missing "
            f"required columns. Columns present: {list(df.columns)}"
        )

    # -- Normalise prices to 100 at week start --------------------------------
    price_a = week["close_a"] / week["close_a"].iloc[0] * 100
    price_b = week["close_b"] / week["close_b"].iloc[0] * 100

    # -- Spread band ----------------------------------------------------------
    upper = week["rolling_mean"] + entry_z * week["rolling_std"]
    lower = week["rolling_mean"] - entry_z * week["rolling_std"]

    # -- Trade markers (entry/exit transitions) --------------------------------
    pos      = week["position"]
    prev_pos = pos.shift(1).fillna(0).astype("int8")
    entries_long  = week[(pos == 1)  & (prev_pos == 0)]
    entries_short = week[(pos == -1) & (prev_pos == 0)]
    exits         = week[(pos == 0)  & (prev_pos != 0)]

    # -- Build figure ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{ticker_a} / {ticker_b}  —  Signal Validation  "
        f"({sample_week})",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Prices
    ax0 = axes[0]
    ax0.plot(price_a.index, price_a.values, label=ticker_a, lw=1.2)
    ax0.plot(price_b.index, price_b.values, label=ticker_b, lw=1.2, alpha=0.85)
    ax0.set_ylabel("Price (normalised to 100)")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # Panel 2: Spread + band
    ax1 = axes[1]
    ax1.plot(week.index, week["spread"].values, label="Spread", lw=1.0, alpha=0.8)
    ax1.plot(week.index, week["rolling_mean"].values,
             color="black", lw=1.0, ls="--", label="Rolling mean")
    ax1.fill_between(week.index, lower.values, upper.values,
                     alpha=0.15, color="steelblue", label=f"+-{entry_z}*std band")
    ax1.set_ylabel("Log-price spread")
    ax1.set_title(
        f"Spread  (beta={beta:.4f}, window={window} bars)", fontsize=10
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 3: Z-score + markers
    ax2 = axes[2]
    ax2.plot(week.index, week["zscore"].values, lw=1.0, alpha=0.8, label="Z-score")
    ax2.axhline( entry_z, color="red",   ls="--", lw=1.0, label=f"+{entry_z}")
    ax2.axhline(-entry_z, color="green", ls="--", lw=1.0, label=f"-{entry_z}")
    ax2.axhline(0,        color="black", ls=":",  lw=0.8, alpha=0.5)

    if not entries_long.empty:
        ax2.scatter(entries_long.index, entries_long["zscore"].values,
                    marker="^", c="green", s=60, zorder=5, label="Entry long")
    if not entries_short.empty:
        ax2.scatter(entries_short.index, entries_short["zscore"].values,
                    marker="v", c="red", s=60, zorder=5, label="Entry short")
    if not exits.empty:
        ax2.scatter(exits.index, exits["zscore"].values,
                    marker="x", c="black", s=40, zorder=5, label="Exit")

    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper left", fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # X-axis formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Rolling Hurst regime monitor
# ---------------------------------------------------------------------------

def plot_rolling_hurst(
    hurst_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    save_path: str = "outputs/figures/rolling_hurst.png",
) -> plt.Figure:
    """Rolling Hurst exponent over the trading period.

    Args:
        hurst_df : DataFrame with a DatetimeIndex and a 'hurst' column,
                   as returned by the rolling_hurst() helper in the pipeline.
        ticker_a/b : Ticker labels for the title.
        save_path  : Output file path.

    !! FLAG — estimator noise:
        On rolling sub-windows (typically 5000 bars = ~13 sessions) the
        Hurst estimator is noisy.  Values jumping between 0.3 and 0.7 within
        a few windows are normal and do not mean the spread is rapidly
        switching regimes.  Smooth interpretation: if the rolling H stays
        consistently above 0.5 for an extended stretch (say, a full month),
        that is a regime-shift signal worth noting.  Single-window spikes
        above 0.5 are noise.
    """
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(hurst_df.index, hurst_df["hurst"].values,
            lw=1.2, color="steelblue", label="Rolling H")
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="H = 0.5 (random walk)")
    ax.fill_between(hurst_df.index, hurst_df["hurst"].values, 0.5,
                    where=(hurst_df["hurst"].values < 0.5),
                    alpha=0.15, color="green", label="Mean-reverting (H < 0.5)")
    ax.fill_between(hurst_df.index, hurst_df["hurst"].values, 0.5,
                    where=(hurst_df["hurst"].values > 0.5),
                    alpha=0.15, color="red", label="Trending (H > 0.5)")

    pct_mr = (hurst_df["hurst"] < 0.5).mean() * 100
    ax.set_title(
        f"{ticker_a}/{ticker_b}  —  Rolling Hurst  "
        f"({pct_mr:.1f}% of windows H < 0.5)",
        fontsize=12,
    )
    ax.set_ylabel("Hurst exponent H")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. QQ-plot vs standard normal
# ---------------------------------------------------------------------------

def plot_qq_normal(
    zscore_series: pd.Series,
    ticker_a: str,
    ticker_b: str,
    save_path: str = "outputs/figures/qq_plot.png",
) -> plt.Figure:
    """QQ-plot of trading-period Z-scores against the standard normal.

    Deviations from the diagonal reveal fat tails (S-shape) or skewness
    (asymmetric deviation).  This is the visual companion to the empirical
    coverage table.

    !! FLAG — sample size:
        scipy.stats.probplot with 48k+ points is slow and produces an
        over-plotted chart.  A random sample of 5000 points is used for
        the visual without materially changing the diagnostic picture.
        The full series is still used for all numerical metrics.
    """
    z = zscore_series.dropna()

    # Subsample for visual clarity
    rng = np.random.default_rng(42)
    n_plot = min(5000, len(z))
    z_sample = pd.Series(
        rng.choice(z.values, size=n_plot, replace=False)
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    (osm, osr), (slope, intercept, _) = probplot(z_sample.values, dist="norm")
    ax.scatter(osm, osr, s=6, alpha=0.4, color="steelblue", label=f"Z-score (n={n_plot})")
    ax.plot(osm, slope * np.array(osm) + intercept,
            color="red", lw=1.5, ls="--", label="Normal reference line")

    ax.set_xlabel("Theoretical quantiles (standard normal)")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(
        f"{ticker_a}/{ticker_b}  —  QQ-Plot: Trading Period Z-Score vs Normal",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Kalman filter beta drift chart
# ---------------------------------------------------------------------------

def plot_kalman_beta(
    beta_series: pd.Series,
    ticker_a: str,
    ticker_b: str,
    formation_end: pd.Timestamp,
    static_beta: float,
    save_path: str = "outputs/figures/kalman_beta.png",
) -> plt.Figure:
    """Plot time-varying Kalman hedge ratio beta(t) over the full year.

    Args:
        beta_series   : Full-year beta series from kalman_hedge_ratio().
        ticker_a/b    : Ticker labels for title.
        formation_end : Last timestamp of the formation period — drawn as a
                        vertical boundary between formation and trading zones.
        static_beta   : Scalar OLS beta — drawn as a horizontal reference line.
        save_path     : Output file path.

    !! FLAG — interpretation:
        If the Kalman beta is flat and hugs the static OLS line, the relationship
        was stable in 2022 and the static approach was adequate.  If the beta
        drifts materially during the trading period, static OLS was introducing
        phantom signal — the Kalman spread will show better stationarity.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(beta_series.index, beta_series.values,
            lw=1.2, color="steelblue", label="Kalman beta(t)")
    ax.axhline(static_beta, color="red", ls="--", lw=1.2,
               label=f"Static OLS beta = {static_beta:.4f}")
    ax.axvline(formation_end, color="black", ls=":", lw=1.2, alpha=0.7,
               label="Formation | Trading boundary")

    # Shade the two periods
    x_min = beta_series.index[0]
    x_max = beta_series.index[-1]
    ax.axvspan(x_min, formation_end, alpha=0.05, color="blue",  label="Formation period")
    ax.axvspan(formation_end, x_max, alpha=0.05, color="green", label="Trading period")

    final_beta = float(beta_series.iloc[-1])
    drift      = round(final_beta - static_beta, 4)
    ax.set_title(
        f"{ticker_a}/{ticker_b}  —  Kalman Dynamic Beta"
        f"  (final={final_beta:.4f}, drift vs OLS={drift:+.4f})",
        fontsize=11,
    )
    ax.set_ylabel("Hedge ratio beta(t)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_active_week(df: pd.DataFrame) -> str:
    """Return the start date of a week in the middle third of df that has trades."""
    if "position" not in df.columns:
        # Fall back to first week
        return df.index[0].strftime("%Y-%m-%d")

    n = len(df)
    search_start = n // 3
    search_end   = (2 * n) // 3

    idx = df.index[search_start:search_end]
    pos = df["position"].iloc[search_start:search_end]
    prev = pos.shift(1).fillna(0)
    entries = idx[(pos != 0) & (prev == 0)]

    if len(entries) == 0:
        return df.index[0].strftime("%Y-%m-%d")

    # Pick the Monday (or first bar of that week) around the first entry found
    entry_ts = entries[0]
    week_start = entry_ts - pd.Timedelta(days=entry_ts.weekday())
    return week_start.strftime("%Y-%m-%d")


def _save(fig: plt.Figure, path: str) -> None:
    """Create parent directories if needed and save the figure."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


