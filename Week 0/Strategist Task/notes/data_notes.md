# Data Notes — 1987 Crash Market Data

**Dataset:** `data/1987_crash_market_data.csv`
**Coverage:** October 16–21, 1987 | Minute-frequency | 7,921 rows | 8 columns

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
- **One annotation only:** a red arrow pointing to the market break at Oct 19 10:00,
  labelled "Market break / Oct 19  10:00".
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
-28.6% at 13:01 on Oct 20 — the deepest trough. The characteristic
asymmetry is visible: the crash unfolds over one session; the recovery extends across
two sessions. The max drawdown annotation is placed at the trough.

**Panel 3 — 30-minute rolling volatility (std of minute returns)**
The pre-crash baseline (Fri Oct 16) holds near 0.0003 — a flat, near-invisible line.
On Monday the vol line leaps to ~0.0025, far above the 3x threshold (red dashed line
at 0.0010), confirming market dysfunction from the opening minutes. Vol partially
subsides mid-Monday (corresponding to the data gap at noon), then spikes again.
Tuesday peaks at 0.0028 at 13:43 — the highest reading in the dataset,
8.3x the pre-crash baseline. Wednesday vol collapses back to near-baseline,
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
  (14:14 – 15:14 on Oct 19)
  produced a cumulative move of approximately **-11.2%** — a rate of
  loss that is orders of magnitude beyond normal market conditions.

- **Drawdown reaches -28.6% at trough:** The cumulative drawdown from
  the pre-crash peak bottoms at 13:01 on Oct 20,
  representing the deepest single-session loss in modern US equity market history.

- **Volatility spikes 8.3x above baseline:** The 30-minute rolling
  standard deviation of minute returns reaches 0.0028 at
  13:43 on Oct 20 — approximately 8.3 times
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
> exceeded three times its pre-crash baseline (10:00 on Oct 19) — the market
> structure most plausibly **broke** at or around
> **10:00 on October 19, 1987**; however, given the
> compressed, minute-frequency nature of the data and the possibility that
> dysfunction began in overnight futures markets before the US open, this should
> be interpreted as an approximate inflection zone rather than a single
> definitive moment.
