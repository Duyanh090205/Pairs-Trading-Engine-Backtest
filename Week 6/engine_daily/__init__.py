"""V4 Daily Strategy (AL2010-standard) — module init.

See ~/.claude/plans/v4-daily-strategy.md for the 4-day ship plan.

Pipeline at glance:
    1. data_daily.resample_5min_to_daily — close-to-close daily bars from 5-min
    2. discovery_daily.run               — PCA + Johansen + HL filter on daily
    3. alpha_refit.recompute_alpha       — Q2: alpha = mean(resid_a - beta*resid_b)
                                            on last 60 daily bars of formation
    4. engine_daily.run_pair             — daily state machine (60-bar rolling Z)
    5. metrics_daily                     — Sharpe / drawdown / exit_breakdown
"""
