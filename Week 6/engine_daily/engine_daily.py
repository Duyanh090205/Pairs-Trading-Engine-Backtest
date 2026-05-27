"""
Phase 2 — Daily Execution Engine (V4)
=======================================

Per pair, per fold:
    1. Build daily residual spread:  spread = resid_a - alpha - beta * resid_b
    2. 60-day rolling Z (seeded with last 60 formation bars so day 0 has full window)
    3. State machine: entry |Z|>=entry_z, exit on zero-cross, hard SL |Z|>=hard_sl_z
    4. 1-bar execution lag: position[t] = signal[t-1]
    5. Vol-target sizing per pair: notional = $200/day / spread_daily_std (clipped)
    6. Daily P&L = position[t-1] * spread_change[t] * notional

Bar frequency = 1 day. No A1 (kill-zone), A3 (Z-velocity), or EOS flatten — those
are intraday concepts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VOL_TARGET_DAILY = 200.0       # $ daily P&L vol target per pair (same as V3)
_VOL_TARGET_FLOOR = 10_000.0    # min notional per pair
_VOL_TARGET_CAP = 40_000.0      # max notional per pair
_TC_BPS = 30.0                  # commission, per side
_BORROW_BPS_YR = 50.0           # short borrow, annualized
_TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Rolling Z with formation warmup
# ---------------------------------------------------------------------------

def rolling_z_with_warmup(
    spread_trading: pd.Series,
    spread_formation_tail: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling Z-score on trading-window spread, seeded with the last `window`
    bars of formation-window spread so day 0 of trading has a full window.

    Returns a Series indexed by `spread_trading.index` with finite Z values from
    bar 0 (assuming `spread_formation_tail` has at least `window` finite values).
    """
    if len(spread_formation_tail) < window:
        log.warning("rolling_z_with_warmup: formation tail %d < window %d -- using all",
                    len(spread_formation_tail), window)
        seed = spread_formation_tail
    else:
        seed = spread_formation_tail.tail(window)

    combined = pd.concat([seed, spread_trading])
    # Rolling mean / std on the combined series; index is preserved.
    roll_mean = combined.rolling(window, min_periods=window).mean()
    roll_std = combined.rolling(window, min_periods=window).std(ddof=1).clip(lower=1e-10)
    z_full = (combined - roll_mean) / roll_std

    # Restrict to trading-window index (the seed bars are warmup).
    z_trading = z_full.loc[spread_trading.index].copy()
    return z_trading


# ---------------------------------------------------------------------------
# Vol-target sizing on daily spread
# ---------------------------------------------------------------------------

def compute_vol_target_notional_daily(
    spread: np.ndarray,
    target_daily_vol: float = _VOL_TARGET_DAILY,
    floor: float = _VOL_TARGET_FLOOR,
    cap: float = _VOL_TARGET_CAP,
) -> float:
    """
    Per-pair notional s.t. expected daily $ P&L vol = target_daily_vol.

    notional = target_daily_vol / spread_daily_std (clipped to [floor, cap]).
    """
    if len(spread) < 10:
        return floor
    spread_clean = spread[np.isfinite(spread)]
    if len(spread_clean) < 10:
        return floor
    daily_ret = np.diff(spread_clean)
    if len(daily_ret) < 2:
        return floor
    sigma_daily = float(np.std(daily_ret, ddof=1))
    if sigma_daily < 1e-12:
        return cap
    notional = target_daily_vol / sigma_daily
    return float(max(floor, min(cap, notional)))


# ---------------------------------------------------------------------------
# Daily state machine (numba)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _state_machine_daily(
    zscores: np.ndarray,
    entry_z: float,
    hard_sl_z: float,
    initial_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Daily V4 state machine.

    Args:
        zscores:        per-day Z-score (np.nan during warmup / missing data)
        entry_z:        |Z| threshold for entry (e.g. 2.0)
        hard_sl_z:      |Z| threshold for hard SL (catastrophic-widen exit)
        initial_state:  +1/-1 if continuing a carried position; 0 = flat (V4 default)

    Returns (positions, exit_codes) where:
        exit_code = 1 -> zero-cross natural exit
        exit_code = 2 -> hard stop-loss
    """
    n = len(zscores)
    positions = np.zeros(n, dtype=np.int8)
    exit_codes = np.zeros(n, dtype=np.int8)
    state = np.int8(initial_state)

    for i in range(n):
        z = zscores[i]
        if np.isnan(z):
            positions[i] = state
            continue

        # Hard SL (catastrophic widen)
        if state == 1 and z <= -hard_sl_z:
            state = np.int8(0)
            exit_codes[i] = np.int8(2)
            positions[i] = state
            continue
        if state == -1 and z >= hard_sl_z:
            state = np.int8(0)
            exit_codes[i] = np.int8(2)
            positions[i] = state
            continue

        # Zero-cross exit
        if state == 1 and z >= 0.0:
            state = np.int8(0)
            exit_codes[i] = np.int8(1)
            positions[i] = state
            continue
        if state == -1 and z <= 0.0:
            state = np.int8(0)
            exit_codes[i] = np.int8(1)
            positions[i] = state
            continue

        # Entry (only when flat)
        if state == 0:
            if z < -entry_z:
                state = np.int8(1)
            elif z > entry_z:
                state = np.int8(-1)
        positions[i] = state

    return positions, exit_codes


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_pair_daily(
    resid_a_form: pd.Series,
    resid_b_form: pd.Series,
    resid_a_trade: pd.Series,
    resid_b_trade: pd.Series,
    alpha: float,
    beta: float,
    entry_z: float = 2.0,
    z_window: int = 60,
    hard_sl_z: float = 4.0,
    # --- V4 + Week 5 dynamic cost integration (opt-in) ---
    cost_data=None,                  # engine_daily.cost_engine.CostData
    ticker_a: str | None = None,
    ticker_b: str | None = None,
    # --- E2 carry-forward (opt-in; default 0/None matches V4 behavior) ---
    initial_position: int = 0,       # +1, -1, or 0 (V4 default = flat)
    original_entry_date: pd.Timestamp | None = None,  # for borrow accrual on
                                                       # carried positions; if
                                                       # None and initial != 0,
                                                       # uses day-0 of trading
    original_notional: float | None = None,  # frozen sizing for carried positions;
                                              # if None and initial != 0, falls
                                              # back to vol-target recomputed value
    carries_done: int = 0,           # diagnostics: # boundaries survived
    fold_origin: int | None = None,  # diagnostics: original fold of entry
    # --- Regime size dampener (default 1.0 = no scaling, preserves V4) ---
    fold_size_mult: float = 1.0,     # multiplier applied to vol-target notional;
                                     # 0.0 = effectively halt (no trades),
                                     # 0.5 = half size, 1.0 = full size
) -> pd.DataFrame:
    """
    Run V4 daily engine for one pair, one trading window.

    Parameters
    ----------
    resid_a_form, resid_b_form  : daily residual log-price Series, formation window
    resid_a_trade, resid_b_trade: daily residual log-price Series, trading window
    alpha, beta : float, hedge ratio params (alpha refit by Q2, beta from formation)
    entry_z, z_window, hard_sl_z : engine knobs

    Returns
    -------
    bar-level DataFrame indexed by trading days with columns:
      spread, zscore, signal, position, exit_code, daily_pnl, cum_pnl, notional
    """
    # ---- Align inputs ----
    df_form = pd.concat([resid_a_form, resid_b_form], axis=1, join="inner").dropna()
    df_form.columns = ["a", "b"]
    df_trade = pd.concat([resid_a_trade, resid_b_trade], axis=1, join="inner").dropna()
    df_trade.columns = ["a", "b"]

    if len(df_trade) < 5:
        return pd.DataFrame()

    # ---- Spread = resid_a - alpha - beta * resid_b ----
    spread_form = pd.Series(
        df_form["a"].values - alpha - beta * df_form["b"].values,
        index=df_form.index,
    )
    spread_trade = pd.Series(
        df_trade["a"].values - alpha - beta * df_trade["b"].values,
        index=df_trade.index,
    )

    # ---- Rolling Z with formation warmup ----
    z = rolling_z_with_warmup(spread_trade, spread_form, window=z_window)
    z_vals = z.values.astype(np.float64)

    # ---- State machine ----
    # For carried positions, the state machine starts in initial_position (not flat).
    signal, exit_code = _state_machine_daily(
        z_vals, entry_z, hard_sl_z, initial_state=int(initial_position),
    )

    # ---- 1-bar execution lag: position[t] = signal[t-1] ----
    # For carried positions, position[0] = initial_position (already holding from
    # prior fold). For non-carry (V4 default initial_position=0), position[0] = 0.
    position = np.zeros(len(signal), dtype=np.int8)
    position[0] = int(initial_position)
    position[1:] = signal[:-1]

    # ---- Sizing (vol-target on trading-window spread) ----
    # For carried positions, REUSE the original notional (sizing decision is
    # frozen with the position — actual shares don't change at a fold boundary).
    # For new entries, recompute from current spread vol.
    if initial_position != 0 and original_notional is not None and original_notional > 0:
        notional = float(original_notional)
    else:
        notional = compute_vol_target_notional_daily(spread_trade.values)

    # ---- Regime size dampener (applied AFTER vol-target clip) ----
    # fold_size_mult = 0.0 -> notional = 0 -> P&L = 0 -> no costs booked
    # fold_size_mult = 0.5 -> half size, half P&L, half costs
    # fold_size_mult = 1.0 -> unchanged (V4 default behavior)
    notional = notional * float(fold_size_mult)
    if notional <= 0:
        # Skip the pair entirely — no trades, no costs, empty result.
        return pd.DataFrame()

    # ---- Daily P&L: position[t] * spread_change[t] * notional ----
    spread_change = np.zeros(len(spread_trade))
    spread_change[1:] = np.diff(spread_trade.values)
    daily_pnl = position.astype(np.float64) * spread_change * notional

    # ---- Trade costs ----
    # Both paths share the same booking convention: entry-side costs are
    # booked at the entry bar, exit-side costs at the exit bar (or the last
    # bar if open at EOM — force-close convention). Borrow is calendar-day
    # accrual on the short leg, booked lump-sum at the exit bar.
    #
    # DYNAMIC path delegates per-trade cost calc to
    # `engine_daily.cost_engine.compute_pair_trade_cost` (single source of
    # truth — same kappa tiers, same fallback, same calendar-day borrow).
    #
    # FLAT path uses a single flat-bps figure as if it were entry+exit
    # commission. Decomposition columns are zero (flat has no breakdown).
    # pos_prev[0] reflects the position carried in from the prior fold
    # (initial_position). For non-carry default this is 0 -- preserves V4 behavior.
    pos_prev = np.zeros(len(position), dtype=np.int8)
    pos_prev[0] = int(initial_position)
    pos_prev[1:] = position[:-1]
    is_entry = (position != 0) & (pos_prev == 0)  # day-0 carried bar is NOT an entry
    is_exit  = (position == 0) & (pos_prev != 0)
    n_bars = len(position)
    index = df_trade.index

    cost_entry = np.zeros(n_bars, dtype=np.float64)
    cost_exit  = np.zeros(n_bars, dtype=np.float64)
    borrow_cost = np.zeros(n_bars, dtype=np.float64)
    # Decomposition columns — populated only on dynamic path
    cost_spread     = np.zeros(n_bars, dtype=np.float64)
    cost_impact     = np.zeros(n_bars, dtype=np.float64)
    cost_commission = np.zeros(n_bars, dtype=np.float64)

    use_dynamic = (cost_data is not None and ticker_a is not None and ticker_b is not None)

    # Walk through trades once, then per-trade compute costs (path-specific).
    notional_per_leg = notional * 0.5
    short_notional_per_leg = notional_per_leg
    flat_commission_per_side = (_TC_BPS / 10000.0) * notional  # 30bps × full notional
    daily_borrow_per_short = (_BORROW_BPS_YR / 10000.0) / 365.0 * short_notional_per_leg

    i = 0

    # --- Carried-position special path (E2) ---
    # If initial_position != 0, the pair entered in a PRIOR fold and is being
    # continued here. There is NO entry on day 0 (entry cost was booked in the
    # original fold). Find its exit (natural in this fold, or EOM force-close)
    # and book exit-side costs only. Borrow accrues from original_entry_date.
    if initial_position != 0:
        exit_idx = -1
        for j in range(1, n_bars):
            if is_exit[j]:
                exit_idx = j
                break
        if exit_idx == -1:
            exit_idx = n_bars - 1  # carried position still open at EOM

        carry_entry_date = (original_entry_date if original_entry_date is not None
                            else index[0])
        carry_exit_date = index[exit_idx]
        side_a = int(initial_position)

        # Borrow (calendar-day from ORIGINAL entry to exit)
        cal_days = (pd.Timestamp(carry_exit_date).normalize().date()
                    - pd.Timestamp(carry_entry_date).normalize().date()).days
        if cal_days > 0:
            borrow_cost[exit_idx] += daily_borrow_per_short * cal_days

        if use_dynamic:
            from engine_daily.cost_engine import compute_pair_trade_cost
            c = compute_pair_trade_cost(
                cost_data, ticker_a, ticker_b,
                entry_date=carry_entry_date, exit_date=carry_exit_date,
                notional_per_leg=notional_per_leg, side_a=side_a,
                borrow_rate_bps_annual=_BORROW_BPS_YR,
            )
            # NO entry cost (already paid in original fold).
            cost_exit[exit_idx]   += (c["spread_exit_$"]
                                      + c["impact_exit_$"]
                                      + c["commission_exit_$"])
            cost_spread[exit_idx]     += c["spread_exit_$"]
            cost_impact[exit_idx]     += c["impact_exit_$"]
            cost_commission[exit_idx] += c["commission_exit_$"]
        else:
            cost_exit[exit_idx] += flat_commission_per_side  # exit half only

        i = exit_idx + 1

    while i < n_bars:
        if is_entry[i]:
            entry_idx = i
            # find matching exit (or last bar if open at EOM)
            exit_idx = -1
            for j in range(i + 1, n_bars):
                if is_exit[j]:
                    exit_idx = j
                    break
            if exit_idx == -1:
                exit_idx = n_bars - 1  # open at EOM -> force-close at last bar

            entry_date = index[entry_idx]
            exit_date  = index[exit_idx]
            side_a     = int(position[entry_idx])  # +1 long A short B; -1 short A long B

            # ----- Borrow (same calendar-day basis for both paths) -----
            cal_days = (pd.Timestamp(exit_date).normalize().date()
                        - pd.Timestamp(entry_date).normalize().date()).days
            if cal_days > 0:
                borrow_cost[exit_idx] += daily_borrow_per_short * cal_days

            if use_dynamic:
                # ----- DYNAMIC: delegate to cost_engine API -----
                from engine_daily.cost_engine import compute_pair_trade_cost
                c = compute_pair_trade_cost(
                    cost_data, ticker_a, ticker_b,
                    entry_date=entry_date, exit_date=exit_date,
                    notional_per_leg=notional_per_leg, side_a=side_a,
                    borrow_rate_bps_annual=_BORROW_BPS_YR,
                )
                cost_entry[entry_idx] += (c["spread_entry_$"]
                                          + c["impact_entry_$"]
                                          + c["commission_entry_$"])
                cost_exit[exit_idx]   += (c["spread_exit_$"]
                                          + c["impact_exit_$"]
                                          + c["commission_exit_$"])
                # Borrow already booked above (calendar-day) so DO NOT
                # add c["borrow_$"] here — same convention, just don't double-book.
                # Decomposition (booked at exit bar for simplicity — sum is what matters)
                cost_spread[entry_idx]     += c["spread_entry_$"]
                cost_spread[exit_idx]      += c["spread_exit_$"]
                cost_impact[entry_idx]     += c["impact_entry_$"]
                cost_impact[exit_idx]      += c["impact_exit_$"]
                cost_commission[entry_idx] += c["commission_entry_$"]
                cost_commission[exit_idx]  += c["commission_exit_$"]
            else:
                # ----- FLAT: single bps figure, entry + exit -----
                cost_entry[entry_idx] += flat_commission_per_side
                cost_exit[exit_idx]   += flat_commission_per_side
                # No decomposition for flat — leave cost_spread/impact/commission = 0.

            i = exit_idx + 1
        else:
            i += 1

    daily_pnl_net = daily_pnl - cost_entry - cost_exit - borrow_cost
    cum_pnl = np.cumsum(daily_pnl_net)

    df = pd.DataFrame({
        "spread": spread_trade.values,
        "zscore": z_vals,
        "signal": signal,
        "position": position,
        "exit_code": exit_code,
        "daily_pnl_gross": daily_pnl,
        "daily_pnl_net": daily_pnl_net,
        "cum_pnl": cum_pnl,
        "cost_entry": cost_entry,
        "cost_exit": cost_exit,
        "borrow_cost": borrow_cost,
        # Decomposition (dynamic only; flat path leaves these at 0)
        "cost_spread": cost_spread,
        "cost_impact": cost_impact,
        "cost_commission": cost_commission,
    }, index=df_trade.index)
    df.attrs["notional"] = notional
    df.attrs["cost_mode"] = "dynamic" if use_dynamic else "flat"
    # E2 carry-forward provenance (used by orchestrator's gate test next fold)
    df.attrs["initial_position"] = int(initial_position)
    df.attrs["original_entry_date"] = (
        original_entry_date if original_entry_date is not None else None
    )
    df.attrs["carries_done"] = int(carries_done)
    df.attrs["fold_origin"] = fold_origin
    return df


# ---------------------------------------------------------------------------
# Fold-level runner
# ---------------------------------------------------------------------------

def run_fold_daily(
    pairs_df: pd.DataFrame,
    resid_form: pd.DataFrame,   # (T_form, n_tickers) formation residual log-prices
    resid_trade: pd.DataFrame,  # (T_trade, n_tickers) trading residual log-prices
    alpha_lookback: int = 60,
    entry_z: float = 2.0,
    z_window: int = 60,
    hard_sl_z: float = 4.0,
    cost_data=None,             # engine_daily.cost_engine.CostData | None
    # E2 carry-forward (default {} = V4 behavior)
    carry_state_in: dict | None = None,  # dict[(ta,tb)] -> CarryState
    current_fold_n: int | None = None,   # stamped onto fresh entries for diagnostics
    fold_size_mult: float = 1.0,         # regime size dampener (1.0 = no scaling)
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Run V4 engine for all pairs in a fold.

    `alpha_lookback`: passed to recompute_alpha (Q2 refit). beta stays from
    `pairs_df['beta_pca']`; alpha is recomputed per-pair on last `alpha_lookback`
    bars of formation residual.

    E2 carry-forward:
        `carry_state_in` maps (ta, tb) -> CarryState for pairs being carried
        from prior fold. These pairs are INJECTED into the pair list (even if
        not in `pairs_df` -- the discovery may have dropped them via BH-FDR
        but the gate already revalidated cointegration). For carry pairs:
          - beta is OVERRIDDEN with carry_state.beta_original (frozen)
          - alpha is refit on this fold's formation (V4 standard)
          - initial_position is passed to run_pair_daily (so day-0 is not an entry)
          - original_entry_date is passed (for borrow accrual)
          - original_notional is passed (frozen sizing)
    """
    from engine_daily.alpha_refit import recompute_alpha
    from engine_daily.portfolio import apply_ticker_concentration_cap

    if carry_state_in is None:
        carry_state_in = {}

    pairs_capped = apply_ticker_concentration_cap(pairs_df)

    # Build the effective pair list: discovery output + carry pairs not already in it.
    # Carry pairs override beta with their frozen value.
    discovery_keys = set()
    if not pairs_capped.empty:
        discovery_keys = {(r["ticker_a"], r["ticker_b"])
                          for _, r in pairs_capped.iterrows()}

    injected_rows: list[dict] = []
    for key, state in carry_state_in.items():
        if key in discovery_keys:
            continue  # already in pair list; carry-beta override happens in main loop
        injected_rows.append({
            "ticker_a": state.ticker_a, "ticker_b": state.ticker_b,
            "beta_pca": float(state.beta_original),
            "alpha_pca": 0.0,           # placeholder; alpha refit happens per-pair below
            "_carry_injected": True,
        })

    if injected_rows:
        injected_df = pd.DataFrame(injected_rows)
        # Mark non-injected rows for clarity
        if not pairs_capped.empty:
            pairs_capped = pairs_capped.assign(_carry_injected=False)
        pairs_effective = pd.concat(
            [pairs_capped, injected_df], ignore_index=True, sort=False,
        )
    else:
        pairs_effective = pairs_capped.copy()
        if not pairs_effective.empty:
            pairs_effective["_carry_injected"] = False

    if pairs_effective.empty:
        log.info("V4 engine: empty pair list (no discovery + no carry) -- skip")
        return {}

    n_carry = sum(1 for k in carry_state_in if k in {
        (r["ticker_a"], r["ticker_b"]) for _, r in pairs_effective.iterrows()
    })
    log.info(
        "V4 daily engine: %d pairs (%d carried, %d injected; entry_z=%.2f, "
        "hard_sl_z=%.2f, z_win=%d)",
        len(pairs_effective), n_carry, len(injected_rows),
        entry_z, hard_sl_z, z_window,
    )

    results: dict[tuple[str, str], pd.DataFrame] = {}
    for _, row in pairs_effective.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        if ta not in resid_form.columns or tb not in resid_form.columns:
            continue
        if ta not in resid_trade.columns or tb not in resid_trade.columns:
            continue

        ra_form = resid_form[ta].dropna()
        rb_form = resid_form[tb].dropna()
        ra_trade = resid_trade[ta].dropna()
        rb_trade = resid_trade[tb].dropna()
        if len(ra_form) < alpha_lookback or len(rb_form) < alpha_lookback:
            continue
        if len(ra_trade) < 5 or len(rb_trade) < 5:
            continue

        # Carry override: use frozen beta_original if this pair is carried.
        carry = carry_state_in.get((ta, tb))
        if carry is not None:
            beta = float(carry.beta_original)
        else:
            beta = float(row["beta_pca"])

        try:
            alpha_new = recompute_alpha(ra_form, rb_form, beta, n_lookback=alpha_lookback)
        except ValueError as e:
            log.debug("alpha_refit skipped %s/%s: %s", ta, tb, e)
            continue

        if carry is not None:
            pair_df = run_pair_daily(
                resid_a_form=ra_form, resid_b_form=rb_form,
                resid_a_trade=ra_trade, resid_b_trade=rb_trade,
                alpha=alpha_new, beta=beta,
                entry_z=entry_z, z_window=z_window, hard_sl_z=hard_sl_z,
                cost_data=cost_data, ticker_a=ta, ticker_b=tb,
                initial_position=int(carry.position),
                original_entry_date=carry.original_entry_date,
                original_notional=float(carry.notional),
                carries_done=int(carry.carries_done),
                fold_origin=int(carry.fold_origin),
                fold_size_mult=fold_size_mult,
            )
        else:
            pair_df = run_pair_daily(
                resid_a_form=ra_form, resid_b_form=rb_form,
                resid_a_trade=ra_trade, resid_b_trade=rb_trade,
                alpha=alpha_new, beta=beta,
                entry_z=entry_z, z_window=z_window, hard_sl_z=hard_sl_z,
                cost_data=cost_data, ticker_a=ta, ticker_b=tb,
                fold_origin=current_fold_n,
                fold_size_mult=fold_size_mult,
            )
        if not pair_df.empty:
            results[(ta, tb)] = pair_df

    log.info("V4 engine complete: %d pairs ran", len(results))
    return results
