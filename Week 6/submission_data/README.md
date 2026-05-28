# Submission Data — Week 6 Live Paper Trading

Reference snapshots for the Week 6 submission. Each file captures live API
state at a moment in time; filenames include UTC timestamp.

## Snapshot schedule

| When | Files captured | Purpose |
|---|---|---|
| `entry_<ts>` (Thu 2026-05-28 ~14:13Z) | orders, positions, pnl, engine_info, log, status | Post-entry-fill baseline |
| `mid_<ts>` (Fri ~14:00Z, optional) | same set | Mid-hold floating P&L |
| `exit_<ts>` (Fri ~20:00Z, post-EOM) | same set | After EOM flatten — final realized data |

## Entry snapshot summary (Thu 2026-05-28 14:13Z)

**Engine state:**
- 19 pairs loaded, 3 open positions (TDG_UPS, DELL_PYPL, AMT_ON)
- session_start_equity = $100,000
- kill_switch cleared (was tripped 13:37Z, fixed via recovery endpoint 17:29Z)

**Open positions (notional, both legs):**
- AMT_ON: long AMT $926 / short ON $1,595 (gross $2,521)
- DELL_PYPL: short DELL $956 / long PYPL $523 (gross $1,479)
- TDG_UPS: short TDG $1,242 / long UPS $940 (gross $2,182)
- Total gross: $6,182 (6.2% of $100k account)
- Net short bias: ~−$1,404 (1.4% — close to market-neutral)

**Realized entry slippage (decision→fill, 6 legs):**
- Aggregate: −20.42 bps (favorable, below backtest assumed 22 bps)
- Range: −436 bps (DELL gapped up, good for short) to +177 bps (ON gapped down, bad for short)
- N=6 legs — not statistically meaningful; expect mostly overnight gap, not pure execution friction

## Bugs found and fixed during live deploy

| Bug | Commit | Pushed |
|---|---|---|
| Wash trade rejection (KLAC in 2 pairs) | `578224b` | 2026-05-27 |
| `formation_end=today` → min_obs crash | `a5eec28` | 2026-05-27 |
| Dedupe blocked canceled orders | `dc55e44` | 2026-05-27 |
| `OrderStatus.FILLED` vs `filled` → positions not populated → kill_switch trip | `e0c3818` | 2026-05-28 |
| `entry_z=0` hardcoded (TODO never finished) | `d527dee` | 2026-05-28 |

## Next analysis (Fri post-EOM)

After Friday 19:55Z EOM flatten runs, run:
- Exit-side slippage compute
- Realized P&L per pair
- Cost drift table populated (backtest predicted vs live realized)
- Hold duration per pair (~30h expected)
- Submission write-up
