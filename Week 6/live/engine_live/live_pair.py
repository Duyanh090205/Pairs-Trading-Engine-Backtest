"""One-step state machine for a live pair.

Mirrors engine_daily.engine_daily._state_machine_daily exactly, but evaluates
a single bar (not a full series). Pure function — no I/O, no side effects.

State convention (same as backtest):
  state =  0 → flat
  state = +1 → long-A short-B  (spread is BELOW mean, expect mean-revert UP)
  state = -1 → short-A long-B  (spread is ABOVE mean, expect mean-revert DOWN)

Action:
  'hold'        — no change
  'enter_long'  — flat → +1
  'enter_short' — flat → -1
  'exit_zero'   — natural exit at zero-cross
  'exit_hard'   — hard SL (catastrophic widen)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Decision:
    action: str          # 'hold'|'enter_long'|'enter_short'|'exit_zero'|'exit_hard'
    new_state: int       # -1, 0, +1


def decide(state: int, z: float, entry_z: float = 3.0,
           hard_sl_z: float = 5.0) -> Decision:
    """Single-bar decision (no execution lag here; lag is applied by caller).

    Matches engine_daily._state_machine_daily one-iteration logic exactly:
      1. NaN z → hold
      2. Hard SL: long & z<=-hard_sl, OR short & z>=hard_sl
      3. Zero-cross: long & z>=0, OR short & z<=0
      4. Entry (only if flat): z<-entry → long, z>+entry → short
    """
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return Decision("hold", state)

    if state == 1 and z <= -hard_sl_z:
        return Decision("exit_hard", 0)
    if state == -1 and z >= hard_sl_z:
        return Decision("exit_hard", 0)

    if state == 1 and z >= 0.0:
        return Decision("exit_zero", 0)
    if state == -1 and z <= 0.0:
        return Decision("exit_zero", 0)

    if state == 0:
        if z < -entry_z:
            return Decision("enter_long", 1)
        if z > entry_z:
            return Decision("enter_short", -1)

    return Decision("hold", state)
